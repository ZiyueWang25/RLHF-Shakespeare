from typing import Sequence
import wandb
from rlhf_sp.config import from_args_to_dict
from rlhf_sp.config import Config
from rlhf_sp import model
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch import nn
import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import os
from rlhf_sp import generate_samples
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_USE_CUDA_DSA"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # obtain an accurate stack trace


def cal_num_same(outputs, labels):
  return (outputs.argmax(axis=-1).reshape(labels.shape) == labels).sum().cpu().item()


def early_stop(valid_losses):
  if len(valid_losses) < 5:
    return False
  for i in range(4):
    if valid_losses[-i - 1] <= valid_losses[-i - 2]:
      return False
  return True


class AttentionScheduler:
  def __init__(self, warmup_steps, d_model, optimizer, lr_mul=1):
    self._optimizer = optimizer
    self.lr_mul = lr_mul
    self.d_model = d_model
    self.warmup_steps = warmup_steps
    self.n_steps = 0

  @property
  def param_groups(self):
    return self._optimizer.param_groups

  def step(self):
    self._update_learning_rate()
    self._optimizer.step()

  def zero_grad(self):
    self._optimizer.zero_grad()

  def _get_lr_scale(self):
    d_model = self.d_model
    n_steps, warmup_steps = self.n_steps, self.warmup_steps
    return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * warmup_steps ** (-1.5))

  def _update_learning_rate(self):
    self.n_steps += 1
    lr = self.lr_mul * self._get_lr_scale()

    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr


def run_epoch(cfg, epoch, data_loader, criterion, net, mask, optimizer, device, train=True):
  if train:
    net.train()
  else:
    net.eval()
  running_loss = 0
  total_num_same = 0
  total_num = 0
  pbar = tqdm(enumerate(data_loader), total=len(data_loader))
  step = epoch * len(data_loader)
  for i, vals in pbar:
    x = vals[0].to(device)
    y = vals[1].to(device)

    if train:
      logits = net(x=x, mask=mask)
    else:
      with torch.no_grad():
        logits = net(x, mask=mask)
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1),)
    if train:
      loss.backward()
      clip_grad_norm_(net.parameters(), 1)
      optimizer.step()
      optimizer.zero_grad()
    num_same = cal_num_same(logits, y)
    acc = num_same / y.view(-1).shape[0]
    total_num_same += num_same
    total_num += y.view(-1).shape[0]
    running_loss += loss.cpu().item()
    if train:
      pbar.set_description(
        f"iter {i}: train loss {loss.item():.5f}, accuracy {acc:.2%}")
      if cfg.use_wandb:
        lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "train_loss": loss.cpu().item(),
            "train_acc": acc,
            "lr": lr,
        }, step=step)
    step += 1
  epoch_loss = running_loss / len(data_loader)
  epoch_acc = total_num_same / total_num
  return epoch_loss, epoch_acc


def train(cfg: Config, tokenizer, train_dl, valid_dl, device, base_model=None, save=True,
          stage="pretrain", name_suffix="", verbose=True, do_early_stop=True):
  if stage == "pretrain":
    epochs = cfg.epochs
    lr = cfg.lr
    lr_mul = cfg.lr_mul
    mask = model.create_forward_mask(cfg.T, cfg.T).to(device)

  elif stage == "reward_train":
    epochs = cfg.reward_epochs
    lr = cfg.reward_lr
    lr_mul = cfg.reward_lr_mul
    mask = model.create_forward_mask(cfg.reward_T, cfg.reward_T).to(device)

  total_steps = epochs * len(train_dl)
  if stage == "pretrain":
    net = model.Model(cfg, device=device).to(device)
  elif stage == "reward_train":
    net = model.RewardModel(cfg, base_model).to(device)
  print("# of parameter:", model.get_num_params(net))
  criterion = nn.CrossEntropyLoss(
    label_smoothing=cfg.label_smoothing, ignore_index=-100)
  optimizer = optim.Adam(net.parameters(), lr=lr,
                         betas=(0.9, 0.98), eps=1e-9)
  if lr_mul is not None:
    warmup_steps = int(total_steps * 0.05)
    optimizer = AttentionScheduler(warmup_steps, cfg.d_model, optimizer, lr_mul)
  name = stage + name_suffix
  if cfg.use_wandb:
    wandb.init(
        project=cfg.wandb_project_name,
        name=name,
        config=from_args_to_dict(cfg)
    )
  best_valid_loss = float("inf")
  valid_losses = []
  for epoch in range(epochs):
    train_loss, train_acc = run_epoch(
        cfg, epoch, train_dl, criterion, net, mask, optimizer, device=device, train=True)
    valid_loss, valid_acc = run_epoch(
        cfg, epoch, valid_dl, criterion, net, mask, optimizer, device=device, train=False)
    if stage == "pretrain" and verbose:
      sample = generate_samples.generate_unconditional_samples(
        tokenizer, net, 1, device, context="\n",
        T=128, gen_size=200, temperature=1.0, greedy=False, top_k=10)
      print(sample[0])
    if cfg.use_wandb:
      wandb.log({
          "train_epoch_loss": train_loss,
          "train_epoch_ppl": np.exp(train_loss),
          "train_epoch_acc": train_acc,
          "valid_epoch_loss": valid_loss,
          "valid_epoch_ppl": np.exp(valid_loss),
          "valid_epoch_acc": valid_acc,
          "epoch": epoch,
      }, step=(epoch + 1) * len(train_dl))
    valid_losses.append(valid_loss)
    if verbose:
      print(f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc :.1%},\
            valid losss {valid_loss:.3f} acc {valid_acc:.1%}")
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      if save:
        path = os.path.join(cfg.save_dir, name, f"best_valid.pt")
        save_model(path, epoch, net, train_loss, valid_loss)
    if save:
      note = "final" if ((epoch == epochs - 1)
                         or early_stop(valid_losses)) else f"{epoch}"
      if (epoch % 3 == 0) or note == "final":
        path = os.path.join(cfg.save_dir, name, f"{note}.pt")
        save_model(path, epoch, net, train_loss, valid_loss)
        if do_early_stop and (epoch % 3 == 0) and epoch - 6 >= 0:
          path = os.path.join(cfg.save_dir, name, f"{epoch - 6}.pt")
          os.remove(path)
    if do_early_stop and early_stop(valid_losses):
      print("Early Stopping")
      break
  print('Finished Training')
  if cfg.use_wandb:
    wandb.finish()
  return net


def compute_ppo_loss(step, cfg, net: model.PPOAgent, mask,
                     batch: Sequence[model.ReplayBufferSamples]) -> torch.Tensor:
  '''Handles learning phase for a single batch. Returns loss function to be minimized.'''
  obs, actions, rewards = batch.obs, batch.actions, batch.rewards
  original_logprobs, old_logprobs = batch.original_logprobs, batch.curr_logprobs
  # add kl divergence to rewards. Ref: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py#L149
  logits = net(obs, mask)
  logprobs = model.get_logprobs(logits, actions)
  ori_logratio = logprobs - original_logprobs
  rewards = rewards - cfg.ppo_beta * ori_logratio
  # calc_clipped_surrogate_objective
  logratio = logprobs - old_logprobs
  ratio = logratio.exp()
  non_clipped = ratio * rewards
  clipped = torch.clip(ratio, 1 - cfg.ppo_clip_coef, 1 +
                       cfg.ppo_clip_coef) * rewards
  loss = - torch.minimum(non_clipped, clipped).mean()

  with torch.inference_mode():
    ori_divergence_score = ori_logratio.mean()
    ori_ratio = ori_logratio.exp()
    ori_approx_kl = ((ori_ratio - 1) - ori_logratio).mean().item()

    approx_kl = ((ratio - 1) - logratio).mean().item()
    clipfrac = ((ratio - 1.0).abs() > cfg.ppo_clip_coef).float().mean().item()

  if cfg.use_wandb:
    wandb.log(batch.log_d, step=step)
    wandb.log(dict(
      ratio=ratio.mean().item(),
      approx_kl=approx_kl,
      ori_approx_kl=ori_approx_kl,
      ori_divergence_score=ori_divergence_score.item(),
      clipfrac=clipfrac,
    ), step=step)

  return loss


def run_ppo_epoch(cfg, epoch, net: model.PPOAgent, mask, optimizer):
  net.rollout_phase()
  batches = net.get_batches()
  step = epoch * len(batches)
  for _, batch in enumerate(batches):
    running_loss = 0
    for _ in range(cfg.ppo_noptepochs):
      optimizer.zero_grad()
      loss = compute_ppo_loss(step, cfg, net, mask, batch)
      loss.backward()
      clip_grad_norm_(net.parameters(), 1)
      optimizer.step()
      running_loss += loss.cpu().item()
    epoch_loss = running_loss / cfg.ppo_noptepochs
    if cfg.use_wandb:
      lr = optimizer.param_groups[0]["lr"]
      wandb.log({
          "train_loss": epoch_loss,
          "lr": lr,
      }, step=step)
    step += 1
  epoch_loss = running_loss / cfg.ppo_batchs_per_epoch
  return epoch_loss, batches


def ppo_eval(net: model.PPOAgent, tokenizer, T, num_sample=2, name="", verbose=True):
  net.eval()
  sample = net.sample(T, num_sample)
  reward_probs = net.critic(sample[:, :-1]).softmax(dim=-1)
  samples = []
  pos_probs = []
  if verbose:
    print(f"{name} evaluation:")
  for i in range(num_sample):
    decode_val = tokenizer.decode(sample[i])
    samples.append(decode_val)
    pos_probs.append(reward_probs[i][0].item())
    if verbose:
      print("-" * 10 + f"sample {i}" + "-" * 10)
      print(decode_val)
      print(f"Positive Prob: {pos_probs[-1]:.1%}\n")
  return samples, pos_probs


def ppo_train(cfg, device, base_net, reward_net, tokenizer, name_suffix="", num_eval_samples=2, save=True, verbose=True):
  name = "ppo_train" + name_suffix
  epochs = cfg.ppo_total_steps
  lr = cfg.ppo_lr
  lr_mul = cfg.ppo_lr_mul
  mask = model.create_forward_mask(cfg.ppo_T, cfg.ppo_T).to(device)
  net = model.PPOAgent(cfg, base_net, reward_net, tokenizer, device)
  optimizer = optim.Adam(net.parameters(), lr=lr,
                         betas=(0.9, 0.98), eps=1e-9)
  total_steps = epochs * cfg.ppo_batchs_per_epoch
  if lr_mul is not None:
    warmup_steps = int(total_steps * 0.05)
    optimizer = AttentionScheduler(warmup_steps, cfg.d_model, optimizer, lr_mul)
  if cfg.use_wandb:
    wandb.init(
        project=cfg.wandb_project_name,
        name=name,
        config=from_args_to_dict(cfg)
    )
  ppo_eval(net, tokenizer, T=128, name="Before Training",
           verbose=verbose, num_sample=num_eval_samples)
  for epoch in tqdm(range(epochs)):
    train_loss = run_ppo_epoch(
      cfg, epoch, net, mask, optimizer)
    samples, pos_probs = ppo_eval(net, tokenizer, T=128,
                                  num_sample=num_eval_samples, name=f"Epoch {epoch}",
                                  verbose=verbose)
    save_samples_and_probs(epoch, samples, pos_probs,
                           os.path.join(cfg.save_dir, name, f"samples.txt"))
    if save and ((epoch % 2 == 0) or (epoch == epochs - 1)):
      note = "final" if (epoch == epochs - 1) else f"{epoch}"
      path = os.path.join(cfg.save_dir, name, f"{note}.pt")
      save_model(path, epoch, net, train_loss, 0.0)
  if cfg.use_wandb:
    wandb.finish()
  return net


def save_samples_and_probs(epoch, samples, probs, path):
  dir = os.path.dirname(path)
  if not os.path.exists(dir):
    os.makedirs(dir)
  with open(path, 'a+') as f:
    f.write(f"Epoch {epoch}\n\n")
    for i, s in enumerate(samples):
      f.write(f"#### Sample {i}, pos prob {probs[i]}:\n{s} \n")


def save_model(path, epoch, net, train_loss, valid_loss):
  dir = os.path.dirname(path)
  if not os.path.exists(dir):
    os.makedirs(dir)
  torch.save({
      'epoch': epoch,
      'model_state_dict': net.state_dict(),
      'train_loss': train_loss,
      'valid_loss': valid_loss,
  }, path)
  return
