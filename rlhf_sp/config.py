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
  lr_mul = .5
  use_wandb: bool = False
  wandb_project_name: str = "RLHF_SP"

  reward_T = 64
  reward_num_labels = 2
  reward_epochs = 30
  reward_lr = 5e-5
  reward_lr_mul = None

  PPO_B = 64
  PPO_T = 64
  PPO_beta = 0.2


def from_args_to_dict(args):
  return dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('__'))
