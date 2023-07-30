class Config:
  save_dir = "./model/"
  T = 128
  B = 128
  N = 8
  d_model = 512
  d_ff = 2048
  h = 8
  dropout = 0.2
  epochs = 50
  label_smoothing = 0.1
  lr = 6e-4
  use_wandb: bool = False
  wandb_project_name: str = "RLHF_SP"


def from_args_to_dict(args):
  return dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('__'))
