import re
from dataclasses import dataclass
from typing import List
import copy
import time

import numpy as np
from jaxtyping import Float, Int

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical
import wandb

from rlhf_sp.generate_samples import sample


def get_num_params(net):
  return sum(param.numel() for param in net.parameters())


class PE(nn.Module):
  def __init__(self, T, D, device):
    super().__init__()
    pos_vals = torch.log(torch.arange(0, T, 1) +
                         1e-5).reshape(-1, 1).repeat((1, D))
    D_vals = (torch.arange(0, D, 1) // 2 * 2 / D *
              torch.log(torch.tensor(10000))).reshape(1, -1).repeat((T, 1))
    add_pi = (torch.arange(0, D, 1) %
              2 * torch.tensor(np.pi) / 2).reshape(1, -1)
    vals = torch.exp(pos_vals - D_vals) + add_pi
    self.pe_vals = torch.sin(vals).to(device)

  def forward(self, x):
    _, T, D = x.shape
    x += self.pe_vals[:T, :D]
    return x


class LearnedPE(nn.Module):
  def __init__(self, T, D):
    super().__init__()
    self.pos_embedding = nn.Embedding(T, D)

  def forward(self, pos):
    return self.pos_embedding(pos)


def attention(q, k, v, mask=None):
  # Inside mask: 1 for ignoring
  # scale dot product.
  T_q, d_q = q.shape[-2:]
  T_k, d_k = k.shape[-2:]
  T_v, d_v = v.shape[-2:]
  assert d_q == d_k
  assert T_k == T_v
  att_v = q @ k.transpose(-1, -2) / np.sqrt(d_k)
  if mask is not None:
    assert mask.shape == (1, T_q, T_k)
    att_v = att_v.masked_fill(mask == 1, -1e9)
  att = att_v.softmax(dim=-1)
  result = att @ v
  return att, result


def create_forward_mask(T_q, T_k):
  forward_mask = torch.triu(torch.ones((1, T_q, T_k)),
                            diagonal=1).type(torch.uint8)
  return forward_mask == 1


class AttentionBlock(nn.Module):
  def __init__(self, d_model, d_q, d_k, d_v):
    super().__init__()
    self.W_q = nn.Linear(d_model, d_q)
    self.W_k = nn.Linear(d_model, d_k)
    self.W_v = nn.Linear(d_model, d_v)

  def forward(self, x, mask=None):
    q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
    return attention(q, k, v, mask)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, h):
    super().__init__()
    assert d_model % h == 0
    self.h = h
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.linear = nn.Linear(d_model, d_model)

  def forward(self, x, mask=None):
    B, T, D = x.shape
    q = self.W_q(x).view(B, T, self.h, -1).transpose(1, 2)
    k = self.W_k(x).view(B, T, self.h, -1).transpose(1, 2)
    v = self.W_v(x).view(B, T, self.h, -1).transpose(1, 2)
    _, x = attention(q, k, v, mask)
    x = x.transpose(1, 2).contiguous().view(B, T, -1)
    return self.linear(x)


class FNN(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()
    self.W_1 = nn.Linear(d_model, d_ff)
    self.W_2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    return self.W_2(F.relu(self.W_1(x)))


class SublayerConnection(nn.Module):
  def __init__(self, d_model, dropout=0):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.ln = nn.LayerNorm(normalized_shape=d_model)

  def forward(self, prev, curr):
    curr = self.dropout(curr)
    return self.ln(curr + prev)


class Decoder(nn.Module):
  def __init__(self, d_model, d_ff, h, dropout=0):
    super().__init__()
    self.fnn = FNN(d_model, d_ff)
    self.mha = MultiHeadAttention(d_model, h)
    self.sc_1 = SublayerConnection(d_model, dropout)
    self.sc_2 = SublayerConnection(d_model, dropout)

  def forward(self, x, mask=None):
    x = self.sc_1(x, self.mha(x, mask))
    x = self.sc_2(x, self.fnn(x))
    return x


class Model(nn.Module):
  def __init__(self, cfg, device, used_learned_pe=False):
    super().__init__()
    self.cfg = cfg
    self.N = cfg.N
    self.decoders = nn.ModuleList(
      [Decoder(cfg.d_model, cfg.d_ff, cfg.h, cfg.dropout) for _ in range(cfg.N)])
    self.used_learned_pe = used_learned_pe
    self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.pe = LearnedPE(cfg.T, cfg.d_model) if used_learned_pe else PE(
      cfg.T, cfg.d_model, device)
    self.dropout = nn.Dropout(p=cfg.dropout)
    self.linear = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

  def forward(self, x, mask=None, **kwargs):
    # x.shape == (B, T)
    B, T = x.shape
    if self.used_learned_pe:
      pos = torch.arange(T).to(x.device)
      x = self.emb(x) + self.pe(pos)
    else:
      x = self.pe(self.emb(x))
    x = self.dropout(x)
    for i in range(self.N):
      x = self.decoders[i](x, mask)
    x = self.linear(x)
    return x


class RewardModel(nn.Module):
  def __init__(self, cfg, base_model: nn.Module):
    super().__init__()
    self.base_model = copy.deepcopy(base_model.to("cpu"))
    self.base_model.linear = nn.Linear(
        self.base_model.linear.in_features, cfg.reward_num_labels)

  def forward(self, x, mask=None, **kwargs):
    return self.base_model(x, mask)[:, -1]


@dataclass
class ReplayBufferSamples:
  '''
  Samples from the replay buffer, converted to PyTorch for use in neural network training.
  '''
  obs: Float[Tensor, "minibatch_size * obs_shape"]
  actions: Int[Tensor, "minibatch_size * obs_shape"]
  rewards: Float[Tensor, "minibatch_size * obs_shape"]
  original_logprobs: Float[Tensor, "minibatch_size * obs_shape"]
  curr_logprobs: Float[Tensor, "minibatch_size * obs_shape"]
  log_d: dict


class ReplayBuffer:
  def __init__(self, B, T):
    self.B = B
    self.T = T
    self.experiences = []

  def add(self, obs, actions, rewards, original_logprobs, curr_logprobs, log_d) -> None:
    assert obs.shape == (self.B, self.T), obs.shape
    assert obs.shape == actions.shape
    assert obs.shape == rewards.shape
    assert obs.shape == original_logprobs.shape
    assert obs.shape == curr_logprobs.shape

    new_experiences_as_tensors = [
      torch.from_numpy(d) if isinstance(d, np.ndarray) else d.clone()
      for d in (obs, actions, rewards, original_logprobs, curr_logprobs)
    ] + [log_d]
    self.experiences.append(ReplayBufferSamples(*new_experiences_as_tensors))

  def get_batches(self) -> List[ReplayBufferSamples]:
    experiences = self.experiences
    self.experiences = []
    return experiences


class PPOAgent(nn.Module):
  critic: nn.Module
  actor: nn.Module

  def __init__(self, cfg, net, reward_net, tokenizer, device):
    super().__init__()
    self.cfg = cfg

    # Keep track of global number of steps taken by agent
    self.steps = 0
    self.device = device

    # Get actor and critic networks
    self.actor = copy.deepcopy(net.to("cpu")).to(device)
    self.original_actor = net.to(device)
    self.critic = reward_net.to(device)

    self.start_x = torch.tensor(tokenizer.encode(re.split(r"\b", "\n")),
                                dtype=torch.long)[None, ...].to(device).repeat(cfg.ppo_B, 1)
    self.rb = ReplayBuffer(cfg.ppo_B, cfg.ppo_T)
    print(f"start_x.shape {self.start_x.shape}")

    self._set_grad()

  def _set_grad(self):
    self.actor.train()
    for param in self.actor.parameters():
      param.requires_grad = True

    self.original_actor.eval()
    for param in self.original_actor.parameters():
      param.requires_grad = False

    self.critic.eval()
    for param in self.critic.parameters():
      param.requires_grad = False

  def forward(self, x, mask=None, **kwargs):
    return self.actor(x, mask)

  @torch.no_grad
  def sample(self, T, num_sample=None):
    start_x = self.start_x[:num_sample] if num_sample is not None else self.start_x
    return sample(self.actor, start_x, T=T + 1,
                  gen_size=T, temperature=self.cfg.ppo_rollout_temp,
                  greedy=False, top_k=None)

  def step(self):
    t_start = time.time()
    with torch.inference_mode():
      acts = self.sample(self.cfg.ppo_T)
      samples = acts[:, :-1]
      acts = acts[:, 1:]
      reward_logits = self.critic(samples)
      curr_actor_logits = self.actor(samples)
      original_actor_logits = self.original_actor(samples)

    time_total = (time.time() - t_start)
    tokens_per_sec = self.cfg.ppo_B * \
        self.cfg.ppo_T / (time.time() - t_start)
    rewards = get_reward(reward_logits, self.cfg.ppo_T)
    log_d = dict(
        tokens_per_sec=tokens_per_sec,
        avg_reward=rewards.mean().item(),
        rollout_time=time_total,
      )
    self.steps += 1
    original_logprobs = get_logprobs(original_actor_logits, acts)
    curr_logprobs = get_logprobs(curr_actor_logits, acts)
    self.rb.add(samples, acts, rewards, original_logprobs, curr_logprobs, log_d)
    return

  def rollout_phase(self):
    for _ in range(self.cfg.ppo_batchs_per_epoch):
      self.step()

  def get_batches(self):
    # normalize rewards
    batches = self.rb.get_batches()
    all_rewards = torch.cat(tuple(b.rewards for b in batches))
    m, s = all_rewards.mean(), all_rewards.std()
    for b in batches:
      b.rewards = (b.rewards - m) / s
    return batches


def get_reward(logits, T):
  # B, 2 -> B, T
  probs = logits.softmax(dim=-1)
  prob_pos = probs[:, 0][:, None]
  ret = torch.log(prob_pos)
  return ret.repeat(1, T)


def get_logprobs(logits, acts):
  return Categorical(logits=logits).log_prob(acts)
