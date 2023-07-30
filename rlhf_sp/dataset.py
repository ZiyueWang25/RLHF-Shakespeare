import re

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pathlib


DATA_PATH = pathlib.Path(
  __file__).parent.resolve().parent / "data" / "100-0.txt"


class ShakeSpeareDataset(Dataset):
  def __init__(self, ids, T):
    self.ids = ids
    self.T = T
    # no overlapping data
    self.idxes = list(range(0, len(self.ids) - self.T, T))

  def __len__(self):
    return len(self.idxes)

  def __getitem__(self, idx):
    idx = self.idxes[idx]
    chunk = self.ids[idx:idx + self.T + 1]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y


class Tokenizer:
  def __init__(self, id_by_token, token_by_id):
    self.id_by_token = id_by_token
    self.token_by_id = token_by_id

  def encode(self, tokens):
    return [self.id_by_token.get(t, self.id_by_token["[UNKNOWN]"]) for t in tokens]

  def decode(self, ids):
    if isinstance(ids, torch.Tensor):
      tokens = [self.token_by_id[i.item()] for i in ids]
    else:
      tokens = [self.token_by_id[i] for i in ids]
    return "".join(tokens)


def get_corpus():
  TRAIN_RATIO = .9
  corpus = open(DATA_PATH, "r", encoding="utf-8-sig").read()
  all_tokens = re.split(r"\b", corpus)
  total_num = len(all_tokens)
  train_corpus = all_tokens[:int(total_num * TRAIN_RATIO)]
  valid_corpus = all_tokens[int(total_num * TRAIN_RATIO):]
  return train_corpus, valid_corpus


def get_tokenizer(train_corpus):
  vocabs = ["[UNKNOWN]"] + sorted(set(train_corpus))
  print(f"train has {len(train_corpus)} words, {len(vocabs)} unique.")
  id_by_token = dict(zip(vocabs, range(len(vocabs))))
  token_by_id = {v: k for k, v in id_by_token.items()}
  return Tokenizer(id_by_token, token_by_id)


def get_dataset(T=128):
  train_corpus, valid_corpus = get_corpus()
  tokenizer = get_tokenizer(train_corpus)
  train_ids = tokenizer.encode(train_corpus)
  valid_ids = tokenizer.encode(valid_corpus)
  train_ds = ShakeSpeareDataset(train_ids, T)
  valid_ds = ShakeSpeareDataset(valid_ids, T)
  return train_ds, valid_ds


def get_dataloader(train_ds, valid_ds, B=128):
  train_dl = DataLoader(train_ds, batch_size=B, pin_memory=True, shuffle=True)
  valid_dl = DataLoader(valid_ds, batch_size=B, pin_memory=True, shuffle=False)
  return train_dl, valid_dl
