import torch
import wandb

from vr_learning_algorithms.lfd.logging.base_logger import BaseLogger


class BCLogger(BaseLogger):
    def __init__(
        self,
        log_dir,
        algo_name,
        use_tb=True,
        wandb_project=None,
        wandb_entity=None,
        config=None,
    ):
        """
        Reinforcement Learning Logger that extends BaseLogger.

        Args:
            log_dir (str): Directory for saving logs.
            algo_name (str): Name of the RL algorithm (e.g., "DQN", "PPO").
            use_tb (bool): Whether to enable TensorBoard logging.
            wandb_project (str, optional): Project name for Weights & Biases.
            wandb_entity (str, optional): Entity for Weights & Biases.
            config (dict, optional): Experiment configuration.
        """
        self._algo_name = algo_name
        self._logged_eval_metrics = {
            "episode_reward/mean",
            "episode_reward/min",
            "episode_reward/max",
            "episode_length/mean",
            "episode_length/min",
            "episode_length/max",
            "loss",
            "learning_rate",
        }
        super().__init__(log_dir, use_tb, wandb_project, wandb_entity, config)

    def log_checkpoints(self, model: torch.nn.Module, step: int):
        """Saves model checkpoints to disk and logs to W&B if enabled."""
        checkpoint_path = f"{self.log_dir}/checkpoint_step_{step}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        if self.use_wandb:
            wandb.save(checkpoint_path)
            print("Checkpoint uploaded to Weights & Biases.")

    def log_config(self, config: dict):
        """Logs the experiment configuration."""
        if self.use_wandb:
            wandb.config.update(config)
        print("Configuration logged.")

    @property
    def algo_name(self):
        """Returns the algorithm name."""
        return self._algo_name

    @property
    def logged_eval_metrics(self):
        """Defines the set of metrics this logger will track."""
        return self._logged_eval_metrics
