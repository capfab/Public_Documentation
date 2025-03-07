import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class BaseLogger(ABC):
    # Required metrics that all subclasses MUST log
    REQUIRED_METRICS = {
        "episode_reward/mean",
        "episode_reward/min",
        "episode_reward/max",
        "episode_length/mean",
        "episode_length/min",
        "episode_length/max",
    }

    def __init__(
        self,
        log_dir: str,
        use_tb: bool = True,
        wandb_project: str = None,
        wandb_entity: str = None,
        config: dict = None,
    ):
        """
        Base logger allowing flexible metric tracking.

        Supports:
        - TensorBoard (optional)
        - Weights & Biases (optional)

        Args:
            log_dir (str): Directory for saving TensorBoard logs.
            use_tb (bool): Whether to use TensorBoard logging.
            wandb_project (str, optional): Project name for Weights & Biases.
            wandb_entity (str, optional): Entity for Weights & Biases.
            config (dict, optional): Experiment configuration.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.use_tb = use_tb
        self.use_wandb = wandb_project is not None

        if self.use_tb:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=config,
                sync_tensorboard=self.use_tb,
                dir=self.log_dir,
            )

    def log_metric(self, name: str, value, step: int):
        """Logs a single metric (scalar or histogram)."""
        if name not in self.logged_eval_metrics:
            raise ValueError(
                f"Metric '{name}' is not in the defined used metrics list: {self.logged_eval_metrics}"
            )

        if isinstance(value, (int, float)):  # Scalar
            if self.use_tb:
                self.tb_writer.add_scalar(name, value, step)
            if self.use_wandb:
                wandb.log({name: value, "step": step})
        elif isinstance(value, (list, np.ndarray, torch.Tensor)):  # Histogram
            if self.use_tb:
                self.tb_writer.add_histogram(name, np.array(value), step)
            if self.use_wandb:
                wandb.log({name: wandb.Histogram(np.array(value)), "step": step})
        else:
            raise ValueError(f"Unsupported type {type(value)} for metric {name}")

    def log_metrics(self, metrics: dict, step: int):
        """Logs multiple metrics at once and flushes immediately."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)  # `log_metric` already calls `flush()`
        self.flush()

    def log_wandb_config(self, config: dict):
        """Logs the experiment configuration."""
        if self.use_wandb:
            wandb.config.update(config)

    def flush(self):
        """Ensures that all buffered data is written to disk immediately."""
        if self.use_tb:
            self.tb_writer.flush()

    def close(self):
        """Closes the logger."""
        if self.use_tb:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()

    @abstractmethod
    def log_checkpoints(self):
        """Method to save model checkpoints."""

    @abstractmethod
    def log_config(self):
        """Method to log experiment hydra configurations."""

    @property
    @abstractmethod
    def algo_name(self):
        """An attribute that all subclasses must define"""

    @property
    @abstractmethod
    def logged_eval_metrics(self):
        """A property that subclasses must define to specify tracked metrics."""
