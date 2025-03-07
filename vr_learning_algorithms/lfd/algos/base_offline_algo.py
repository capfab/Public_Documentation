import os
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.debug import print

from vr_learning_algorithms.lfd.dataset.base_dataset import Dataset
from vr_learning_algorithms.lfd.utils import Batch, InfoDict, Model, PRNGKey, evaluation


class BaseBC(ABC):
    @abstractmethod
    def train_function(self, expert_dataset: Dataset, writer, eval_env, args):
        """Train the behavior cloning model using expert dataset."""

    @abstractmethod
    def sample_actions(
        self,
        observations: jnp.ndarray,
        random_temperature: float = 1.0,
        training: bool = False,
    ) -> jnp.ndarray:
        """Sample actions given observations."""

    def load(self, save_dir: str):
        """Loads the saved actor model."""
        if os.path.exists(save_dir):
            print(f"Loading model from {save_dir}")
            self.actor = self.actor.load(os.path.join(save_dir, "actor"))
        else:
            print(f"Model not found in {save_dir}")

    def save(self, save_dir: str):
        """Saves the actor model."""
        print(f"Saving model to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        self.actor.save(os.path.join(save_dir, "actor"))
