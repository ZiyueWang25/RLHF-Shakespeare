import wandb
from rlhf_sp.config import from_args_to_dict
from rlhf_sp.config import Config
from rlhf_sp import model
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import os
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
    return self.optimizer.param_groups

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


def run_epoch(cfg, epoch, data_loader, criterion, model, mask, optimizer, device, train=True):
  if train:
    model.train()
  else:
    model.eval()
  running_loss = 0
  total_num_same = 0
  total_num = 0
  pbar = tqdm(enumerate(data_loader), total=len(data_loader))
  step = epoch * len(data_loader)
  for i, vals in pbar:
    x = vals[0].to(device)
    y = vals[1].to(device)

    if train:
      optimizer.zero_grad()
    if train:
      logits = model(x=x, mask=mask)
    else:
      with torch.no_grad():
        logits = model(x, mask=mask)
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1),)
    if train:
      loss.backward()
      clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
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
            "lr": lr,
        }, step=step)
    step += 1
  epoch_loss = running_loss / len(data_loader)
  epoch_acc = total_num_same / total_num
  return epoch_loss, epoch_acc


def train(cfg: Config, train_dl, valid_dl, device, base_model=None, save=True, stage="pretrain"):
  if stage == "pretrain":
    epochs = cfg.epochs
    lr = cfg.lr
    lr_mul = cfg.lr_mul
    mask = model.create_forward_mask(cfg.T, cfg.T).to(device)

  elif stage == "reward_train":
    epochs = cfg.reward_epochs
    lr = cfg.reward_lr
    lr_mul = cfg.lr_mul
    mask = model.create_forward_mask(cfg.reward_T, cfg.reward_T).to(device)

  total_steps = epochs * len(train_dl)
  if stage == "pretrain":
    net = model.Model(cfg, device=device, used_learned_pe=False).to(device)
  else:
    net = model.RewardModel(cfg, base_model).to(device)
  print("# of parameter:", model.get_num_params(net))
  criterion = nn.CrossEntropyLoss(
    label_smoothing=cfg.label_smoothing, ignore_index=-100)
  optimizer = optim.Adam(net.parameters(), lr=lr,
                         betas=(0.9, 0.98), eps=1e-9)
  if lr_mul is not None:
    warmup_steps = int(total_steps * 0.05)
    optimizer = AttentionScheduler(warmup_steps, cfg.d_model, optimizer, lr_mul)
  if cfg.use_wandb:
    wandb.init(
        project=cfg.wandb_project_name,
        name=stage,
        config=from_args_to_dict(cfg)
    )
  valid_losses = []
  for epoch in range(epochs):
    train_loss, train_acc = run_epoch(
        cfg, epoch, train_dl, criterion, net, mask, optimizer, device=device, train=True)
    valid_loss, valid_acc = run_epoch(
        cfg, epoch, valid_dl, criterion, net, mask, optimizer, device=device, train=False)
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
    print(f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc :.1%},\
           valid losss {valid_loss:.3f} acc {valid_acc:.1%}")
    if save:
      note = "final" if ((epoch == epochs - 1)
                         or early_stop(valid_losses)) else f"{epoch}"
      if (note == f"{epoch}" and (epoch % 3 == 0)) or note == "final":
        path = os.path.join(cfg.save_dir, f"{note}_{stage}.pt")
        save_model(path, epoch, net, optimizer, train_loss, valid_loss)
        if epoch - 6 >= 0:
          os.remove(os.path.join(cfg.save_dir,
                                 f"{epoch - 6}_{stage}.pt"))
    if early_stop(valid_losses):
      print("Early Stopping")
      break
  print('Finished Training')
  if cfg.use_wandb:
    wandb.finish()
  return net


def save_model(path, epoch, net, optimizer, train_loss, valid_loss):
  torch.save({
      'epoch': epoch,
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'train_loss': train_loss,
      'valid_loss': valid_loss,
  }, path)
  return
