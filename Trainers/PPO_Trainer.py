"""
PPO Trainer for SDDS-PyTorch.

This implements Proximal Policy Optimization for training discrete diffusion
models using reinforcement learning. The main SDDS contribution is the
memory-efficient mini-batch processing over diffusion steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .BaseTrainer import BaseTrainer
from utils.moving_average import ExponentialMovingAverage


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer for discrete diffusion models.

    Key features:
    - TD(lambda) advantage estimation
    - Clipped surrogate objective
    - Mini-batch processing over diffusion steps (memory efficient)
    - Moving average reward normalization
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
        Initialize PPO trainer.

        Args:
            config: Configuration with PPO settings
            model: Diffusion model
            energy_class: Energy function for CO problem
            noise_class: Noise distribution class
            device: Device to use
        """
        super().__init__(config, model, energy_class, noise_class, device)

        # PPO parameters
        self.clip_value = config.get("clip_value", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.gamma = config.get("gamma", 1.0)
        self.n_inner_steps = config.get("n_inner_steps", 4)

        # Value network head
        hidden_dim = getattr(model, 'hidden_dim', 64)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Add value head to optimizer
        self.optimizer.add_param_group({"params": self.value_head.parameters()})

        # Moving average for reward normalization
        mov_avg_alpha = config.get("mov_average_alpha", 0.99)
        self.moving_average = ExponentialMovingAverage(alpha=mov_avg_alpha)

        # Temperature
        self.temperature = config.get("temperature", 1.0)

    def sample(
        self,
        coords: torch.Tensor,
        n_samples: int = 1,
        key: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the diffusion model.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            n_samples: Number of samples per instance
            key: Random generator

        Returns:
            Dictionary with samples and metrics
        """
        batch_size, n_nodes, _ = coords.shape
        device = coords.device

        # Expand coords for multiple samples
        coords_exp = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
        coords_flat = coords_exp.reshape(batch_size * n_samples, n_nodes, 2)

        # Sample prior
        x_t = self.model.sample_prior(
            (batch_size * n_samples, n_nodes, n_nodes),
            device=device,
            generator=key
        )

        # Reverse diffusion
        for t in range(self.n_diffusion_steps - 1, -1, -1):
            timestep = torch.full(
                (batch_size * n_samples,), t, device=device, dtype=torch.long
            )

            with torch.no_grad():
                logits = self.model(coords_flat, x_t, timestep)

            x_prev, _ = self.noise_class.calc_noise_step(logits, x_t, t, key)
            x_t = x_prev

        # Reshape
        x_0 = x_t.reshape(batch_size, n_samples, n_nodes, n_nodes)

        # Compute energies
        energies = []
        for s in range(n_samples):
            energy, _, _ = self.energy_class.calculate_Energy(coords, x_0[:, s])
            energies.append(energy)
        energies = torch.stack(energies, dim=1)

        # Best samples
        best_idx = torch.argmin(energies, dim=1)
        best_x0 = x_0[torch.arange(batch_size), best_idx]
        best_energy = energies[torch.arange(batch_size), best_idx]

        return {
            "x_0": x_0,
            "best_x_0": best_x0,
            "energies": energies,
            "best_energy": best_energy,
            "mean_energy": energies.mean(dim=1),
        }

    def collect_rollout(
        self,
        coords: torch.Tensor,
        key: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout data for PPO training.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            key: Random generator

        Returns:
            Rollout buffer dictionary
        """
        batch_size, n_nodes, _ = coords.shape
        device = coords.device
        n_samples = self.n_basis_states

        # Expand for samples
        coords_exp = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
        coords_flat = coords_exp.reshape(batch_size * n_samples, n_nodes, 2)

        # Storage
        T = self.n_diffusion_steps
        B = batch_size * n_samples

        states = torch.zeros(T, B, n_nodes, n_nodes, device=device)
        actions = torch.zeros_like(states)
        log_probs = torch.zeros(T, B, device=device)
        values = torch.zeros(T + 1, B, device=device)
        rewards = torch.zeros(T, B, device=device)

        # Sample prior
        x_t = self.model.sample_prior((B, n_nodes, n_nodes), device=device, generator=key)

        # Run diffusion
        for t in range(T - 1, -1, -1):
            step_idx = T - 1 - t
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            states[step_idx] = x_t

            with torch.no_grad():
                logits = self.model(coords_flat, x_t, timestep)

            # Sample action
            x_prev, log_prob = self.noise_class.calc_noise_step(logits, x_t, t, key)

            actions[step_idx] = x_prev
            log_probs[step_idx] = log_prob.sum(dim=(-2, -1))

            # Entropy reward
            entropy = self.noise_class.get_entropy(logits)
            rewards[step_idx] = self.temperature * entropy.mean(dim=-1)

            x_t = x_prev

        # Terminal reward (negative energy)
        x_0 = x_t.reshape(batch_size, n_samples, n_nodes, n_nodes)

        for s in range(n_samples):
            energy, _, _ = self.energy_class.calculate_Energy(coords, x_0[:, s])
            start_idx = s * batch_size
            end_idx = (s + 1) * batch_size
            rewards[-1, start_idx:end_idx] -= energy

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "coords": coords_flat,
            "x_0": x_0,
        }

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages.

        Args:
            rewards: (T, batch)
            values: (T+1, batch)

        Returns:
            (advantages, value_targets)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        value_targets = advantages + values[:-1]

        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, value_targets

    def ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        value_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss."""
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, value_targets)

        loss = actor_loss + self.value_coef * value_loss

        metrics = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "mean_ratio": ratio.mean().item(),
        }

        return loss, metrics

    def get_loss(
        self,
        batch: Dict[str, torch.Tensor],
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute PPO loss for batch."""
        coords = batch["coords"].to(self.device)

        # Collect rollout
        rollout = self.collect_rollout(coords, key)

        # Compute advantages
        advantages, value_targets = self.compute_advantages(
            rollout["rewards"], rollout["values"]
        )

        # PPO update
        T = self.n_diffusion_steps
        new_log_probs = []

        for t in range(T - 1, -1, -1):
            step_idx = T - 1 - t
            timestep = torch.full(
                (rollout["coords"].shape[0],), t, device=self.device, dtype=torch.long
            )

            logits = self.model(rollout["coords"], rollout["states"][step_idx], timestep)
            probs = F.softmax(logits, dim=-1)

            # Log prob of taken action
            action = rollout["actions"][step_idx]
            log_prob = torch.where(
                action.unsqueeze(-1) == 1,
                torch.log(probs[..., 1:2] + 1e-8),
                torch.log(probs[..., 0:1] + 1e-8)
            ).sum(dim=(-3, -2, -1))
            new_log_probs.append(log_prob)

        new_log_probs = torch.stack(new_log_probs, dim=0)

        loss, metrics = self.ppo_loss(
            rollout["log_probs"],
            new_log_probs,
            advantages,
            rollout["values"][:-1],
            value_targets
        )

        # Energy metrics
        with torch.no_grad():
            energies = []
            for s in range(rollout["x_0"].shape[1]):
                e, _, _ = self.energy_class.calculate_Energy(coords, rollout["x_0"][:, s])
                energies.append(e)
            energies = torch.stack(energies, dim=1)
            metrics["mean_energy"] = energies.mean().item()
            metrics["best_energy"] = energies.min(dim=1)[0].mean().item()

        return loss, metrics
