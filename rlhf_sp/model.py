import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


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
  def __init__(self, vocab_size, T, N, d_model, d_ff, h, dropout, device, used_learned_pe=False):
    super().__init__()
    self.N = N
    self.decoders = nn.ModuleList(
      [Decoder(d_model, d_ff, h, dropout) for _ in range(N)])
    self.used_learned_pe = used_learned_pe
    self.emb = nn.Embedding(vocab_size, d_model)
    self.pe = LearnedPE(T, d_model) if used_learned_pe else PE(
      T, d_model, device)
    self.dropout = nn.Dropout(p=dropout)
    self.linear = nn.Linear(d_model, vocab_size, bias=False)

  def forward(self, x, mask=None):
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
  def __init__(self, base_model: nn.Module):
    self.base_model = base_model

  def forward(self, x, places, mask=None):
    logits = self.base_model(x, mask)
    return logits[places]


def get_num_params(net):
  return sum(param.numel() for param in net.parameters())
