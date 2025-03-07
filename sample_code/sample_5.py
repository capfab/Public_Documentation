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
    """_summary_

    :param key: _description_
    :type key: PRNGKey
    :param actor: _description_
    :type actor: Model
    :param batch: _description_
    :type batch: Batch
    :return: _description_
    :rtype: Tuple[Model, InfoDict]
    """

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        """_summary_

        :param actor_params: _description_
        :type actor_params: Params
        :return: _description_
        :rtype: Tuple[jnp.ndarray, InfoDict]
        """
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
