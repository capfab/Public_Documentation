from dataclasses import dataclass
from pprint import pprint

import wandb

from .common import *


@dataclass(frozen=True)
class ConfigArgs:
    f: str
    sample_random_times: int
    grad_pen: bool
    noise: bool
    noise_std: float
    lambda_gp: int
    max_clip: float
    num_v_updates: int
    log_loss: bool
    alpha: float
    eval_interval: int
    v_update: str
    clip_threshold: float
    update_Q_inference: bool
    good_reward_coeff: float
    bad_reward_coeff: float


def log_wandb(info, args, step):
    if args.use_wandb:
        wandb.log(info, step=step)
    else:
        pprint(info)
