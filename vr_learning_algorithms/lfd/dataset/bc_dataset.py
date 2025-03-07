import collections

import gym
import jax
import jax.numpy as jnp
from tqdm import tqdm

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


class Dataset(object):
    def __init__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        masks: jnp.ndarray,
        dones_float: jnp.ndarray,
        next_observations: jnp.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, key, batch_size: int, shift: float, scale: float) -> Batch:
        indx = jax.random.randint(
            key=key, minval=0, maxval=self.size, shape=(batch_size,)
        )
        return Batch(
            observations=(self.observations[indx] + shift) * scale,
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=(self.next_observations[indx] + shift) * scale,
        )
