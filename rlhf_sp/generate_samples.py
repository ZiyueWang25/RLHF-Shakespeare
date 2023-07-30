from tqdm import tqdm
import re

import torch
from torch.nn import functional as F


def top_k_logits(logits, k):
  v, ix = torch.topk(logits, k)
  out = logits.clone()
  out[out < v[:, [-1]]] = -float('Inf')
  return out


@torch.no_grad()
def sample(model, x, T=128, gen_size=50, temperature=1.0, greedy=False, top_k=None):
  model.eval()
  for i in range(gen_size):
    x_cond = x if x.size(1) <= T else x[:, -T:]
    T_curr = x_cond.shape[-1]
    out = model(x_cond)
    last_out = out[:, -1, :] / temperature  # only consider the last step
    if top_k is not None:
      last_out = top_k_logits(last_out, top_k)
    probs = F.softmax(last_out, dim=-1)
    if greedy:
      next_ids = probs.argmax(dim=-1)[None, :]
    else:
      next_ids = torch.multinomial(probs, num_samples=1)
    x = torch.cat((x, next_ids), dim=-1)
  return x


def generate_unconditional_samples(tokenizer, net, N, device, context=".", **kwargs):
  unconditional_samples = []
  for _ in tqdm(range(N)):
    x = torch.tensor(tokenizer.encode(re.split(r"\b", context)),
                     dtype=torch.long)[None, ...].to(device)
    y = sample(net, x, **kwargs)[0]
    completion = tokenizer.decode(y)
    unconditional_samples.append(completion[1:])
  return unconditional_samples
