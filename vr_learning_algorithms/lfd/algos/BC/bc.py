import os
import time
from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.debug import print

from vr_learning_algorithms.lfd.algos.base_offline_algo import BaseBC
from vr_learning_algorithms.lfd.dataset.bc_dataset import Dataset
from vr_learning_algorithms.lfd.logging.bc_logger import BCLogger
from vr_learning_algorithms.lfd.models import critic, discriminator, policy, value
from vr_learning_algorithms.lfd.utils import (
    Batch,
    InfoDict,
    MixBatch,
    Model,
    Params,
    PRNGKey,
    evaluation,
    target_update,
)


@partial(jax.jit, static_argnames=[])
def update_actor(key: PRNGKey, actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": key},
        )
        log_probs = dist.log_prob(batch.actions).sum(-1)
        actor_loss = -(log_probs.clip(min=-100, max=100)).mean()

        info = {"expert_data/logp": log_probs.mean()}

        return actor_loss, info

    new_actor, grad_info = actor.apply_gradient(actor_loss_fn)
    return new_actor, grad_info


@partial(jax.jit, static_argnames=["args"])
def _update_BC(
    rng: PRNGKey,
    actor: Model,
    expert_batch: Batch,
    args,
) -> Tuple[PRNGKey, Model, InfoDict]:
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, expert_batch)

    return rng, new_actor, {**actor_info}


class BC(BaseBC):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float,
        critic_lr: float,
        value_lr: float,
        disc_lr: float,
        hidden_dims: Sequence[int],
        discount: float,
        expectile: float,
        actor_temperature: float,
        dropout_rate: float,
        layernorm: bool,
        tau: float,
        double_q: bool = True,
        opt_decay_schedule: Optional[str] = "cosine",
        max_steps: Optional[int] = None,
        value_dropout_rate: Optional[float] = None,
        reward_gap: float = 2.0,
        weight_decay: float = 0.0,
        episode_length: int = 1000,
        action_repeat: int = 1,
        args=None,
    ):

        self.expectile = expectile
        self.reward_gap = reward_gap
        self.tau = tau
        self.discount = discount
        self.actor_temperature = actor_temperature
        self.double_q = double_q
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.args = args
        self.train_bad = True
        self.train_mix = True

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        action_dim = actions.shape[1]

        # ---- actor ----#
        print("actor with tanh squash = True")
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=True,
        )

        if opt_decay_schedule == "cosine":
            print("Using cosine decay schedule")
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            print(f"Using AdamW with weight decay {weight_decay}")
            optimiser = optax.adamw(learning_rate=actor_lr, weight_decay=weight_decay)

        actor_net = Model.create(
            actor_def, inputs=[actor_key, observations], tx=optimiser
        )

        self.actor = actor_net
        self.rng = rng
        self.logger = BCLogger(log_dir="logs/", algo_name="BC")

    def train_function(self, expert_dataset: Dataset, writer, eval_env, args):
        times = [time.monotonic()]

        eval_key, self.rng = jax.random.split(self.rng)
        evaluator = evaluation.Evaluator(
            eval_env,
            self.actor.apply_fn,
            num_eval_envs=args.evaluation.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=eval_key,
        )

        def progress(num_steps, metrics):
            times.append(time.monotonic())
            if args.logging.use_wandb and not args.experiment.play_only:
                wandb.log(metrics, step=num_steps)
            if args.logging.use_tb and not args.experiment.play_only:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, num_steps)
                writer.flush()
            percentage = num_steps / args.agent.max_steps * 100
            ret = metrics["eval/episode_reward"]
            time_left = metrics["time_left"] if "time_left" in metrics else 0
            train_time = metrics["train_time"] if "train_time" in metrics else 0
            jax.debug.print(
                "[{p:.2f}%] {n}/{m}: ret={r:.3f}\ttime={t:.1f}s\ttime_left={tl:.1f}s",
                p=percentage,
                n=num_steps,
                m=args.agent.max_steps,
                r=ret,
                t=train_time,
                tl=time_left,
            )

        @jax.jit
        def _train_step(carry, _):
            rng, actor = carry
            key, rng = jax.random.split(rng)
            expert_batch = expert_dataset.sample(
                key=key, batch_size=args.training.batch_size, shift=0, scale=1
            )
            rng, actor, info = _update_BC(rng, actor, expert_batch, args)
            return (rng, actor), info

        # eval before training
        eval_info = evaluator.run_evaluation(self.actor.params, {})
        progress(0, eval_info)

        train_times = []

        for step in range(args.agent.max_steps // args.training.eval_interval + 1):
            train_start = time.monotonic()
            # Thực hiện eval_interval lần cập nhật sử dụng scan
            init_carry = (self.rng, self.actor)
            inner_steps = jnp.arange(args.training.eval_interval)
            final_carry, train_info = jax.lax.scan(_train_step, init_carry, inner_steps)

            self.rng, self.actor = final_carry

            # Tính trung bình các metrics từ train_info
            train_metrics = jax.tree_map(lambda x: x[-1], train_info)

            # Thực hiện đánh giá
            eval_info = evaluator.run_evaluation(self.actor.params, {})

            # Kết hợp train_metrics và eval_info
            combined_info = {**train_metrics, **eval_info}
            train_times.append(time.monotonic() - train_start)
            combined_info["train_time"] = np.mean(train_times)
            combined_info["time_left"] = np.mean(train_times) * (
                (args.agent.max_steps / args.training.eval_interval) - (step + 1)
            )
            progress((step + 1) * args.training.eval_interval, combined_info)

    @partial(jax.jit, static_argnames=["training"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        random_tempurature: float = 1.0,
        training: bool = False,
    ) -> jnp.ndarray:
        rng, actions = policy.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            random_tempurature,
            training=training,
        )
        self.rng = rng
        actions = jnp.asarray(actions)
        return jnp.clip(actions, -1, 1)

    def load(self, save_dir: str):
        if os.path.exists(save_dir):
            print(f"Loading model from {save_dir}")
            self.actor = self.actor.load(os.path.join(save_dir, "actor"))
        else:
            print(f"Model not found in {save_dir}")

    def save(self, save_dir: str):
        print(f"Saving model to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        self.actor.save(os.path.join(save_dir, "actor"))
