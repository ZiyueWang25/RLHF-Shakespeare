from tqdm import tqdm
import os

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

import model
from config import Config

def cal_num_same(outputs, labels):
  return (outputs.argmax(axis=-1) == labels).sum().cpu().item()

def cal_acc(data_loader, num_same, cfg):
  num_total = len(data_loader) * cfg.B * cfg.T
  return num_same / num_total

def run_epoch(cfg, data_loader, criterion, model, mask, optimizer, device, train=True):
  if train:
    model.train()
  else:
    model.eval()
  running_loss = 0
  total_num_same = 0
  pbar = tqdm(enumerate(data_loader), total=len(data_loader))
  for i, (x, y) in pbar:
    x = x.to(device)
    y = y.to(device)
    if train:
      optimizer.zero_grad()
    if train:
      logits = model(x, mask)
    else:
      with torch.no_grad():
        logits = model(x, mask)
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
    if train:
      loss.backward()
      clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
    total_num_same += cal_num_same(logits, y)
    running_loss += loss.cpu().item()
    if train:
      pbar.set_description(f"iter {i}: train loss {loss.item():.5f}")
  return running_loss / len(data_loader), cal_acc(data_loader, total_num_same, cfg)

def early_stop(valid_losses, valid_accs):
  if valid_accs[-1] == 1:
    return True
  if len(valid_losses) < 4:
    return False
  for i in range(4):
    if valid_losses[-i-1] <= valid_losses[-i-2]:
      return False
  return True

class AttentionScheduler:
  def __init__(self, warmup_steps, d_model, optimizer, lr_mul=1):
    self._optimizer = optimizer
    self.lr_mul = lr_mul
    self.d_model = d_model
    self.warmup_steps = warmup_steps
    self.n_steps = 0

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

def pretrain(cfg: Config, train_dl, valid_dl, device):
    # when using scheduler
    total_steps = cfg.epochs * len(train_dl)
    warmup_steps = int(total_steps * 0.05)
    lr_mul=0.5
    net = model.Model(cfg.vocab_size, cfg.T, cfg.N, cfg.d_model, cfg.d_ff, cfg.h, cfg.dropout, used_learned_pe=False).to(device)
    print("# of parameter:", model.get_num_params(net))
    mask = model.create_forward_mask(cfg.T, cfg.T).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    sched = AttentionScheduler(warmup_steps, cfg.d_model, optimizer, lr_mul=lr_mul)

    train_losses = []
    train_ppls = []
    train_accs = []
    lrs = []
    valid_losses = []
    valid_ppls = []
    valid_accs = []
    for epoch in range(cfg.epochs):
        train_loss, train_acc = run_epoch(cfg, train_dl, criterion, net, mask, sched, device=device, train=True)
        valid_loss, valid_acc = run_epoch(cfg, valid_dl, criterion, net, mask, sched, device=device, train=False)
        lrs.append(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        train_ppls.append(np.exp(train_loss))
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_ppls.append(np.exp(valid_loss))
        valid_accs.append(valid_acc)
        if epoch % 3 == 0 or epoch == cfg.epochs - 1:
            for note in [f"{epoch}", "final"]:
                path = os.path.join(cfg.save_dir, f"{note}_state_dict_model.pt")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            }, path)
                if epoch - 6 >= 0 and note == f"{epoch}":
                    # save space
                    os.remove(os.path.join(cfg.save_dir, f"{epoch - 6}_state_dict_model.pt"))
            print(f"epoch {epoch}: lr: {lrs[-1]:.4f} train ppl {np.exp(train_loss):.3f} acc {train_acc :.1%}, valid loss {np.exp(valid_loss):.3f} acc {valid_acc:.1%}")
        if early_stop(valid_losses, valid_accs):
            print("Early Stopping")
            break
    print('Finished Training')