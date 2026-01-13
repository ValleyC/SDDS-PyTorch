"""
Base class for trainers in SDDS.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time


class BaseTrainer(ABC):
    """
    Abstract base class for training discrete diffusion models.

    This class provides the common infrastructure for different training
    algorithms (REINFORCE, PPO, Forward KL, etc.).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        energy_class: Any,
        noise_class: Any,
        device: torch.device = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
            model: The diffusion model to train
            energy_class: Energy function class for the CO problem
            noise_class: Noise distribution class
            device: Device to use for training
        """
        self.config = config
        self.model = model
        self.energy_class = energy_class
        self.noise_class = noise_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Extract common config values
        self.n_diffusion_steps = config.get("n_diffusion_steps", 10)
        self.n_basis_states = config.get("n_basis_states", 8)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.eval_step_factor = config.get("eval_step_factor", 1)
        self.n_sampling_rounds = config.get("n_sampling_rounds", 1)
        self.sampling_temp = config.get("sampling_temp", 1.0)

        # Problem-specific settings
        self.problem_name = config.get("problem_name", "TSP")
        self.dataset_name = config.get("dataset_name", "")

        # For binary problems (TSP with edge representation)
        self.n_bernoulli_features = 2

        # Get noise-related functions
        self.beta_arr = self.noise_class.beta_arr.to(self.device)
        self.noise_func = self.noise_class.calc_noise_loss
        self.combine_losses = self.noise_class.combine_losses

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Logging
        self.log_dict = {}

    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Setup the optimizer.

        Returns:
            Configured optimizer
        """
        optimizer_name = self.config.get("optimizer", "adam")
        weight_decay = self.config.get("weight_decay", 0.0)

        if optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )

    @abstractmethod
    def get_loss(
        self,
        batch: Dict[str, torch.Tensor],
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the loss for a batch.

        Args:
            batch: Dictionary containing batch data
            key: Random generator

        Returns:
            Tuple of (loss, log_dict)
        """
        pass

    @abstractmethod
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        n_samples: int = 1,
        key: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the model.

        Args:
            batch: Dictionary containing batch data (coordinates, etc.)
            n_samples: Number of samples per instance
            key: Random generator

        Returns:
            Dictionary containing samples and metrics
        """
        pass

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform one training step.

        Args:
            batch: Dictionary containing batch data
            key: Random generator

        Returns:
            Tuple of (loss, log_dict)
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, log_dict = self.get_loss(batch, key)

        loss.backward()

        # Gradient clipping
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        return loss, log_dict

    def evaluation_step(
        self,
        batch: Dict[str, torch.Tensor],
        key: Optional[torch.Generator] = None
    ) -> Dict[str, Any]:
        """
        Perform one evaluation step.

        Args:
            batch: Dictionary containing batch data
            key: Random generator

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            start_time = time.time()

            # Sample from the model
            result = self.sample(batch, n_samples=self.n_basis_states, key=key)

            end_time = time.time()

            result["time"] = {
                "forward_pass": end_time - start_time
            }

        return result

    def _reverse_kl_loss(
        self,
        log_q_0_T: torch.Tensor,
        log_p_0_T: torch.Tensor,
        diff_step_axis: int = 0
    ) -> torch.Tensor:
        """
        Compute reverse KL divergence loss.

        Args:
            log_q_0_T: Log probabilities from model q
            log_p_0_T: Log probabilities from target p
            diff_step_axis: Axis along which to sum diffusion steps

        Returns:
            Reverse KL loss
        """
        loss = torch.mean(
            torch.sum(log_q_0_T, dim=diff_step_axis) -
            torch.sum(log_p_0_T, dim=diff_step_axis)
        )
        return loss

    def _forward_kl_loss(
        self,
        log_q_0_T: torch.Tensor,
        log_p_0_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward KL divergence loss with importance weighting.

        Args:
            log_q_0_T: Log probabilities from model q
            log_p_0_T: Log probabilities from target p

        Returns:
            Forward KL loss
        """
        weights = self._compute_importance_weights(log_q_0_T, log_p_0_T)
        forward_kl_per_graph = -torch.sum(
            weights * torch.sum(log_q_0_T, dim=0),
            dim=-1
        )
        forward_kl = torch.mean(forward_kl_per_graph)
        return forward_kl

    def _compute_importance_weights(
        self,
        log_q_0_T: torch.Tensor,
        log_p_0_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights for forward KL.

        Args:
            log_q_0_T: Log probabilities from model q
            log_p_0_T: Log probabilities from target p

        Returns:
            Normalized importance weights
        """
        weights = torch.softmax(
            torch.sum(log_p_0_T - log_q_0_T, dim=0),
            dim=-1
        )
        return weights

    def diffusion_step(
        self,
        x_t: torch.Tensor,
        t: int,
        coords: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one diffusion step (reverse process).

        Args:
            x_t: Current state
            t: Current timestep
            coords: Node coordinates
            edge_index: Edge indices for sparse graphs
            key: Random generator

        Returns:
            Tuple of (x_{t-1}, log_prob, logits)
        """
        # Get model prediction (use integer timestep for consistency)
        timestep = torch.full((coords.shape[0],), t, device=self.device, dtype=torch.long)
        logits = self.model(coords, x_t, timestep, edge_index)

        # Sample next state using corrected noise schedule index
        # t is the diffusion timestep (T-1 to 0), noise_t_idx maps to beta_arr
        noise_t_idx = self.n_diffusion_steps - 1 - t
        x_prev, log_prob = self.noise_class.calc_noise_step(logits, x_t, noise_t_idx, key)

        return x_prev, log_prob, logits

    def compute_energy(
        self,
        coords: torch.Tensor,
        x_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute energy for the given solution.

        Args:
            coords: Node coordinates
            x_0: Solution (edge adjacency matrix)

        Returns:
            Tuple of (energy, violations_per_node, constraint_violations)
        """
        return self.energy_class.calculate_Energy(coords, x_0)

    def get_scheduler(self, num_training_steps: int) -> Optional[Any]:
        """
        Get learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps

        Returns:
            Learning rate scheduler or None
        """
        scheduler_type = self.config.get("scheduler", None)

        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config.get("min_lr", 1e-6)
            )
        elif scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=num_training_steps,
                pct_start=0.1
            )
        else:
            return None

    def save_checkpoint(self, path: str, epoch: int, additional_info: Dict = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            additional_info: Additional information to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
