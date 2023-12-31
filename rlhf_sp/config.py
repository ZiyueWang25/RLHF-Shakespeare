class Config:
  save_dir = "./model/"
  T = 128
  B = 64
  N = 8
  d_model = 512
  d_ff = 2048
  h = 8
  dropout = 0.2
  epochs = 15
  label_smoothing = .0
  used_learned_pe = False
  lr = 2e-4
  lr_mul = .3
  use_wandb: bool = False
  wandb_project_name: str = "RLHF_SP"

  reward_T = 64
  reward_num_labels = 2
  reward_epochs = 50
  reward_lr = 5e-5
  reward_lr_mul = 5e-3

  ppo_B = 128
  ppo_T = 64
  ppo_reward_normalization = True
  ppo_beta = 0.1
  ppo_total_steps = 40
  ppo_noptepochs = 4
  ppo_lr = 1e-5
  ppo_lr_mul = None
  ppo_batchs_per_epoch = 1
  ppo_clip_coef = .2
  ppo_eps = 1e-8
  ppo_rollout_temp = 1.0


def from_args_to_dict(args):
  return dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('__'))
