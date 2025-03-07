# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a agent using JAX on the specified environment."""

import functools
import importlib.resources as pkg_resources
import json
import os
import sys
import time
import warnings
from datetime import datetime

import h5py
import hydra
import jax
import jax.numpy as jp
import mujoco_playground
import wandb
from absl import logging
from etils import epath
from ml_collections import config_dict
from mujoco_playground import registry, wrapper
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter

import vr_learning_algorithms
from vr_learning_algorithms.lfd.algos.BC.bc import BC
from vr_learning_algorithms.lfd.dataset.bc_dataset import Dataset
from vr_learning_algorithms.lfd.utils import ConfigArgs

# Get the absolute path of the config directory
config_path = str(pkg_resources.files(vr_learning_algorithms) / "conf")

print(f"\nconfig_path: {config_path}\n")

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

from mujoco_playground._src import dm_control_suite


def brax_bc_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned Brax BC config for the given environment."""
    env_config = dm_control_suite.get_default_config(env_name)

    rl_config = config_dict.create(
        num_timesteps=5_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.99,
        learning_rate=1e-3,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=8,
        max_replay_size=1048576 * 4,
        min_replay_size=8192,
        network_factory=config_dict.create(
            q_network_layer_norm=True,
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            policy_obs_key="state",
        ),
    )

    if env_name == "PendulumSwingUp":
        rl_config.action_repeat = 4

    if (
        env_name.startswith("Acrobot")
        or env_name.startswith("Swimmer")
        or env_name.startswith("Finger")
        or env_name.startswith("Hopper")
        or env_name in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
    ):
        rl_config.num_timesteps = 10_000_000

    return rl_config


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in mujoco_playground.manipulation._envs:
        raise ValueError(f"BC is not supported for manipulation environments")
    elif env_name in mujoco_playground.locomotion._envs:
        return brax_bc_config(env_name)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        # raise ValueError(f"BC is not supported for DM Control Suite environments")
        return brax_bc_config(env_name)
    else:
        raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def load_file(path, dataset_size):
    if not os.path.isfile(path):
        print(f"File {path} does not exist")
        raise FileNotFoundError(f"File {path} does not exist")

    hdf_trajs = h5py.File(path, "r")
    eps = 1e-5
    lim = 1 - eps
    data = dict(
        observations=jp.array(hdf_trajs["observations"][:dataset_size]),
        actions=jp.array(hdf_trajs["actions"][:dataset_size]).clip(-lim, lim),
        next_observations=jp.array(hdf_trajs["next_observations"][:dataset_size]),
        rewards=jp.array(hdf_trajs["rewards"][:dataset_size]),
        terminals=jp.array(hdf_trajs["terminals"][:dataset_size]),
        timeouts=jp.array(hdf_trajs["timeouts"][:dataset_size]),
    )
    hdf_trajs.close()

    # print(data['observations'].shape)
    # print(data['actions'].shape)
    # print(data['next_observations'].shape)
    # print(data['rewards'].shape)
    # print(data['terminals'].shape)
    # print(data['timeouts'].shape)
    # TODO: correct later, not every episode is of the same lengtfh
    # return_val = data['rewards'].reshape(-1,episode_length).sum(axis=-1)
    # print(f'Return: {jp.mean(return_val)} +- {jp.std(return_val)}')
    return data


# @hydra.main(config_path="conf", config_name="config")
@hydra.main(config_path=config_path, config_name="config")
def main(cfg: DictConfig):
    # args = get_args(cfg)

    # training:
    #   batch_size: 137
    #   num_v_updates: 137
    #   grad_pen: 137
    #   lambda_gp: 137

    print("\n\n==========================\n\n")
    print(f"cfg.training.batch_size: {cfg.training.batch_size}")
    print(f"cfg.training.num_v_updates: {cfg.training.num_v_updates}")
    print(f"cfg.training.grad_pen: {cfg.training.grad_pen}")
    print(f"cfg.training.lambda_gp: {cfg.training.lambda_gp}")
    print("\n\n==========================\n\n")

    # sys.exit(0)

    # Load environment configuration
    env_cfg = registry.get_default_config(cfg.env.env_name)

    algo_params = get_rl_config(cfg.env.env_name)

    env = registry.load(cfg.env.env_name, config=env_cfg)

    print(f"Environment Config:\n{env_cfg}")
    print(f"Algorithm {cfg.training.algo_name} Parameters:\n{algo_params}")

    # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("Date(Y%Y-M%m-D%d)-Time(h%H:m%M:s%S)")
    if cfg.experiment.play_only:
        exp_name = f"__test__"
    else:
        exp_name = f"__test__{cfg.env.env_name}-{cfg.training.algo_name}-{timestamp}"
    if cfg.experiment.suffix is not None:
        exp_name += f"-{cfg.experiment.suffix}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Initialize Weights & Biases if required
    if cfg.logging.use_tb and not cfg.experiment.play_only:
        wandb.login(key="5ea6c6e6651d710fa883225381f0a2f97f35a4c6")
        wandb.init(
            project="test_mujoco",
            entity="hmhuy",
            name=exp_name,
            group=cfg.env.env_name,
            job_type=f"{cfg.training.algo_name}-{cfg.dataset.expert_dataset_size}",
        )
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": cfg.env.env_name})

    # Initialize TensorBoard if required
    if cfg.logging.use_tb and not cfg.experiment.play_only:
        writer = SummaryWriter(logdir)
    else:
        writer = None

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"New checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_json(), fp, indent=4)

    training_params = dict(algo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    if cfg.experiment.domain_randomization:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            cfg.env.env_name
        )

    if cfg.experiment.vision:
        env = wrapper.wrap_for_brax_training(
            env,
            vision=True,
            num_vision_envs=env_cfg.vision_config.render_batch_size,
            episode_length=algo_params.episode_length,
            action_repeat=algo_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

    num_eval_envs = (
        algo_params.num_envs
        if cfg.experiment.vision
        else algo_params.get("num_eval_envs", 128)
    )

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    # Load dataset
    expert_dataset = load_file(
        f"/home/baothach/Downloads/MJP_Huy/IL_dataset/{cfg.env.env_name}-PPO.hdf5",
        algo_params.episode_length * cfg.dataset.expert_dataset_size,
    )

    expert_dataset = Dataset(
        observations=expert_dataset["observations"],
        actions=expert_dataset["actions"],
        rewards=expert_dataset["rewards"],
        masks=1 - expert_dataset["terminals"],
        dones_float=expert_dataset["terminals"],
        next_observations=expert_dataset["next_observations"],
        size=expert_dataset["observations"].shape[0],
    )

    agent_args = ConfigArgs(
        f=cfg.training.get("f", None),
        sample_random_times=cfg.training.sample_random_times,
        grad_pen=cfg.training.grad_pen,
        lambda_gp=cfg.training.lambda_gp,
        noise=cfg.training.noise,
        max_clip=cfg.training.max_clip,
        alpha=cfg.training.alpha,
        num_v_updates=cfg.training.num_v_updates,
        log_loss=cfg.training.get("log_loss", False),
        noise_std=cfg.training.get("noise_std", 0.0),
        eval_interval=cfg.training.eval_interval,
        v_update=cfg.training.get("v_update", False),
        clip_threshold=cfg.training.get("clip_threshold", None),
        update_Q_inference=cfg.training.update_Q_inference,
        good_reward_coeff=cfg.training.good_reward_coeff,
        bad_reward_coeff=cfg.training.bad_reward_coeff,
    )

    if isinstance(env.observation_size, dict):
        obs_size = env.observation_size["state"][0]
    else:
        obs_size = env.observation_size

    agent = BC(
        cfg.general.seed,
        observations=jp.zeros((1, obs_size)),
        actions=jp.zeros((1, env.action_size)),
        max_steps=cfg.agent.max_steps,
        double_q=cfg.agent.double,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
        disc_lr=cfg.agent.disc_lr,
        value_lr=cfg.agent.value_lr,
        hidden_dims=cfg.architecture.hidden_dims,
        discount=cfg.agent.discount,
        expectile=cfg.agent.expectile,
        actor_temperature=cfg.agent.actor_temperature,
        dropout_rate=cfg.agent.dropout_rate,
        layernorm=cfg.agent.layernorm,
        tau=cfg.agent.tau,
        episode_length=cfg.agent.episode_length,
        action_repeat=cfg.agent.action_repeat,
        reward_gap=cfg.agent.reward_gap,
        weight_decay=cfg.agent.weight_decay,
        args=agent_args,
    )

    times = [time.monotonic()]

    # Progress function for logging

    # Load evaluation environment
    eval_env = (
        None
        if cfg.experiment.vision
        else registry.load(cfg.env.env_name, config=env_cfg)
    )
    wrap_env_fn = None if cfg.experiment.vision else wrapper.wrap_for_brax_training

    if wrap_env_fn is not None:
        v_randomization_fn = None
        if training_params.get("randomization_fn") is not None:
            eval_key, agent.rng = jax.random.split(agent.rng)
            v_randomization_fn = functools.partial(
                training_params.get("randomization_fn"),
                rng=jax.random.split(eval_key, num_eval_envs),
            )
        eval_env = wrap_env_fn(
            eval_env,
            episode_length=algo_params.episode_length,
            action_repeat=algo_params.action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    # agent.train_function(expert_dataset,writer,eval_env,args)
    agent.train_function(expert_dataset, writer, eval_env, cfg)


if __name__ == "__main__":
    # app.run(main)
    main()
