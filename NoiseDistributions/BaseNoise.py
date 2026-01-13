"""
Base class for noise distributions in discrete diffusion.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class BaseNoiseDistribution(ABC):
    """
    Abstract base class for noise distributions in discrete diffusion models.

    This class defines the interface for noise schedules used in the forward
    and reverse diffusion processes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the noise distribution.

        Args:
            config: Configuration dictionary containing:
                - n_diffusion_steps: Number of diffusion steps
                - diff_schedule: Type of noise schedule ('DiffUCO', 'exp', 'own', etc.)
                - beta_factor: Scaling factor for beta (optional)
        """
        self.config = config
        self.n_diffusion_steps = config.get("n_diffusion_steps", 10)
        self.diff_schedule = config.get("diff_schedule", "DiffUCO")
        self.beta_factor = config.get("beta_factor", 1.0)

        # Pre-compute beta values for all timesteps
        beta_list = []
        for t in range(self.n_diffusion_steps):
            beta_list.append(self.beta_t_func(t, self.n_diffusion_steps, self.beta_factor))

        # Flip so that beta_arr[0] corresponds to t=T (most noisy)
        # and beta_arr[-1] corresponds to t=0 (clean)
        self.beta_arr = torch.flip(torch.tensor(beta_list, dtype=torch.float32), dims=[0])

        print(f"Noise schedule: {self.diff_schedule}")
        print(f"Beta values: {self.beta_arr}")

    def beta_t_func(self, t: int, n_diffusion_steps: int, k: float = 0.0) -> float:
        """
        Compute beta (noise level) at timestep t.

        Args:
            t: Current timestep (0 to n_diffusion_steps-1)
            n_diffusion_steps: Total number of diffusion steps
            k: Additional scaling factor

        Returns:
            Beta value at timestep t
        """
        if self.diff_schedule == "own":
            tau = 30
            beta = 1 / (tau * ((n_diffusion_steps - t - 1) / n_diffusion_steps) + 2)
        elif self.diff_schedule == "exp":
            tau = 6
            x = (n_diffusion_steps - t - 1) / n_diffusion_steps
            beta = (2 ** (-tau * x)) * 0.5
        elif self.diff_schedule == "DiffUCO":
            beta = 1 / (n_diffusion_steps - t + 1)
        elif self.diff_schedule == "Ho":
            beta = (1 - (n_diffusion_steps - (t + 1)) / (n_diffusion_steps - t)) / 2
            beta = max(beta, 0.01)
        elif self.diff_schedule == "Campbell":
            b = 100
            a = 0.01
            Tau = n_diffusion_steps
            beta = 0.5 * (1 - np.exp(Tau * a * (b ** (t / Tau) - b ** ((t + 1) / Tau))))
            beta = max(beta, 0.01)
        else:
            # Default schedule
            beta = 1 / (n_diffusion_steps - t + 2)

        return beta

    def get_gamma_t(self, t_idx: int) -> torch.Tensor:
        """
        Get gamma (flip probability) at timestep index.

        Args:
            t_idx: Timestep index

        Returns:
            Gamma value at the given timestep
        """
        return self.beta_arr[t_idx]

    @abstractmethod
    def sample_forward_diff_process(
        self,
        x0: torch.Tensor,
        t: int,
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from the forward diffusion process q(x_t | x_0).

        Args:
            x0: Clean data at t=0
            t: Target timestep
            key: Random generator for reproducibility

        Returns:
            Tuple of (x_t, log_prob, additional_info)
        """
        pass

    @abstractmethod
    def get_log_p_T_0(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Compute log probability log p(x_t | x_0) for the forward process.

        Args:
            xt: Noisy data at timestep t
            x0: Clean data at t=0
            t: Current timestep

        Returns:
            Log probability tensor
        """
        pass

    @abstractmethod
    def calc_noise_loss(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Compute the noise-related loss term.

        Args:
            logits: Model output logits
            xt: Noisy data at timestep t
            t: Current timestep

        Returns:
            Noise loss tensor
        """
        pass

    @abstractmethod
    def calc_noise_step(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        t: int,
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse diffusion process.

        Args:
            logits: Model output logits for x_0 prediction
            xt: Current noisy state
            t: Current timestep
            key: Random generator

        Returns:
            Tuple of (x_{t-1}, log_prob)
        """
        pass

    @abstractmethod
    def combine_losses(
        self,
        noise_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        entropy_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine different loss components.

        Args:
            noise_loss: Loss from noise prediction
            energy_loss: Loss from energy/reward
            entropy_loss: Entropy regularization loss

        Returns:
            Combined loss
        """
        pass

    @abstractmethod
    def calculate_noise_distr_reward(
        self,
        xt: torch.Tensor,
        x_prev: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Calculate the reward contribution from the noise distribution.

        Args:
            xt: State at timestep t
            x_prev: State at timestep t-1
            t: Current timestep

        Returns:
            Reward tensor
        """
        pass
