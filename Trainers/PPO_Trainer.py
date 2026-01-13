"""
PPO Trainer for SDDS-PyTorch with permutation-based TSP.

This implements Proximal Policy Optimization for training discrete diffusion
models using reinforcement learning with the categorical position formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union

from .BaseTrainer import BaseTrainer
from utils.moving_average import ExponentialMovingAverage


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer for discrete diffusion models with categorical positions.

    Key features:
    - TD(lambda) advantage estimation
    - Clipped surrogate objective
    - Multiple PPO epochs per rollout (n_inner_steps)
    - Value function baseline
    - Entropy regularization
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
            model: Diffusion model (outputs position logits)
            energy_class: Energy function for TSP (permutation-based)
            noise_class: Noise distribution class (categorical)
            device: Device to use
        """
        super().__init__(config, model, energy_class, noise_class, device)

        # Get n_nodes for position representation
        self.n_nodes = config.get("n_nodes", 20)

        # PPO parameters
        self.clip_value = config.get("clip_value", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.gamma = config.get("gamma", 1.0)
        self.n_inner_steps = config.get("n_inner_steps", 4)

        # Value network - simple MLP that takes state statistics as input
        # Input: [position distribution stats, timestep_normalized, coord_stats]
        value_input_dim = 8
        hidden_dim = config.get("hidden_dim", 64)
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Add value head to optimizer
        self.optimizer.add_param_group({"params": self.value_head.parameters()})

        # Moving average for reward normalization
        mov_avg_alpha = config.get("mov_average_alpha", 0.99)
        self.moving_average = ExponentialMovingAverage(alpha=mov_avg_alpha)

        # Temperature for entropy reward (default 0 = no entropy reward)
        self.temperature = config.get("temperature", 0.0)

    def _get_value(
        self,
        coords: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value estimate for state.

        Args:
            coords: Node coordinates (B, N, 2)
            x_t: Current position assignments (B, N) with values 0 to N-1
            timestep: Current timestep (B,)

        Returns:
            Value estimate (B,)
        """
        B = coords.shape[0]
        N = coords.shape[1]

        # Handle extra dimension
        if x_t.dim() == 3:
            x_t = x_t.squeeze(-1)

        # Convert to one-hot to get distribution statistics
        x_one_hot = F.one_hot(x_t.long(), num_classes=N).float()  # (B, N, N)

        # Position distribution statistics
        # Row sum tells us how many positions are assigned to each node (should be 1)
        # Col sum tells us how many nodes are assigned to each position (should be 1)
        row_sum = x_one_hot.sum(dim=2)  # (B, N)
        col_sum = x_one_hot.sum(dim=1)  # (B, N)

        row_violation = (row_sum - 1).abs().mean(dim=1)  # (B,)
        col_violation = (col_sum - 1).abs().mean(dim=1)  # (B,)

        # Timestep normalized
        t_norm = timestep.float() / self.n_diffusion_steps  # (B,)

        # Coordinate statistics
        coord_mean = coords.mean(dim=1)  # (B, 2)
        coord_std = coords.std(dim=1) + 1e-8  # (B, 2)

        # Concatenate features
        features = torch.stack([
            row_violation, col_violation,
            t_norm, t_norm ** 2,  # Quadratic time feature
            coord_mean[:, 0], coord_mean[:, 1],
            coord_std[:, 0], coord_std[:, 1]
        ], dim=1)  # (B, 8)

        value = self.value_head(features).squeeze(-1)
        return value

    def _compute_entropy(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy of the policy.

        Args:
            logits: Model output (B, N, N_positions)

        Returns:
            Entropy per sample (B,)
        """
        logits = torch.clamp(logits, min=-50, max=50)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Entropy: -sum(p * log(p)) per node
        entropy_per_node = -torch.sum(probs * log_probs, dim=-1)  # (B, N)

        # Sum over all nodes for per-sample entropy
        return entropy_per_node.sum(dim=-1)  # (B,)

    def sample(
        self,
        coords_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        n_samples: int = 1,
        key: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the diffusion model.

        Args:
            coords_or_batch: Either node coordinates (batch, n_nodes, 2) or batch dict
            n_samples: Number of samples per instance
            key: Random generator

        Returns:
            Dictionary with samples and metrics
        """
        # Handle both coords tensor and batch dict for compatibility
        if isinstance(coords_or_batch, dict):
            coords = coords_or_batch["coords"].to(self.device)
        else:
            coords = coords_or_batch.to(self.device)

        batch_size, n_nodes, _ = coords.shape

        # Expand coords for multiple samples
        coords_exp = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
        coords_flat = coords_exp.reshape(batch_size * n_samples, n_nodes, 2)

        # Sample prior (uniform random positions)
        x_t = self.noise_class.sample_prior(
            (batch_size * n_samples, n_nodes),
            device=self.device,
            generator=key
        )

        T = self.n_diffusion_steps

        # Reverse diffusion: t goes from T-1 to 0
        for t in range(T - 1, -1, -1):
            timestep = torch.full(
                (batch_size * n_samples,), t, device=self.device, dtype=torch.long
            )

            with torch.no_grad():
                logits = self.model(coords_flat, x_t, timestep)

            # Use corrected noise schedule index
            noise_t_idx = T - 1 - t
            x_prev, _ = self.noise_class.calc_noise_step(logits, x_t, noise_t_idx, key)
            x_t = x_prev

        # Reshape: x_0 is (batch_size * n_samples, n_nodes) position indices
        x_0 = x_t.reshape(batch_size, n_samples, n_nodes)

        # Compute energies
        energies = []
        for s in range(n_samples):
            energy, _, _ = self.energy_class.calculate_Energy(coords, x_0[:, s])
            energies.append(energy)
        energies = torch.stack(energies, dim=1)

        # Best samples
        best_idx = torch.argmin(energies, dim=1)
        batch_indices = torch.arange(batch_size, device=self.device)
        best_x0 = x_0[batch_indices, best_idx]
        best_energy = energies[batch_indices, best_idx]

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
        n_samples = self.n_basis_states

        # Expand for samples
        coords_exp = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
        coords_flat = coords_exp.reshape(batch_size * n_samples, n_nodes, 2)

        T = self.n_diffusion_steps
        B = batch_size * n_samples

        # Get n_random_features from model (default 5)
        n_random_features = getattr(self.model, 'n_random_features',
                                    getattr(self.model.model, 'n_random_features', 5))

        # Storage - states are now (T, B, n_nodes) position indices
        states = torch.zeros(T, B, n_nodes, dtype=torch.long, device=self.device)
        actions = torch.zeros_like(states)
        log_probs = torch.zeros(T, B, device=self.device)
        values = torch.zeros(T + 1, B, device=self.device)
        rewards = torch.zeros(T, B, device=self.device)
        entropies = torch.zeros(T, B, device=self.device)
        # IMPORTANT: Cache random features for PPO consistency
        rand_features_all = torch.zeros(T, B, n_nodes, n_random_features, device=self.device)

        # Sample prior (uniform random positions)
        x_t = self.noise_class.sample_prior((B, n_nodes), device=self.device, generator=key)

        # Run reverse diffusion
        for t in range(T - 1, -1, -1):
            step_idx = T - 1 - t  # step_idx=0 at t=T-1, step_idx=T-1 at t=0
            timestep = torch.full((B,), t, device=self.device, dtype=torch.long)

            states[step_idx] = x_t

            # Generate and cache random features for this step
            rand_features = torch.rand(B, n_nodes, n_random_features, device=self.device)
            rand_features_all[step_idx] = rand_features

            # Get model prediction (no grad for rollout collection)
            with torch.no_grad():
                logits = self.model(coords_flat, x_t, timestep, rand_features=rand_features)

                # Compute value estimate
                values[step_idx] = self._get_value(coords_flat, x_t, timestep)

            # Sample action using corrected noise schedule index
            noise_t_idx = T - 1 - t
            x_prev, action_log_prob = self.noise_class.calc_noise_step(logits, x_t, noise_t_idx, key)

            actions[step_idx] = x_prev

            # Use log prob from calc_noise_step
            log_probs[step_idx] = action_log_prob

            # Compute entropy
            entropies[step_idx] = self._compute_entropy(logits)

            # Entropy reward (per sample)
            if self.temperature > 0:
                rewards[step_idx] = self.temperature * entropies[step_idx]

            x_t = x_prev

        # Terminal value (bootstrap from final state)
        with torch.no_grad():
            timestep_final = torch.zeros(B, device=self.device, dtype=torch.long)
            values[T] = self._get_value(coords_flat, x_t, timestep_final)

        # Terminal reward (negative energy, normalized)
        x_0 = x_t.reshape(batch_size, n_samples, n_nodes)

        terminal_energies = []
        for s in range(n_samples):
            energy, _, _ = self.energy_class.calculate_Energy(coords, x_0[:, s])
            terminal_energies.append(energy)
            start_idx = s * batch_size
            end_idx = (s + 1) * batch_size
            # Normalize energy reward
            normalized_energy = energy / n_nodes
            rewards[-1, start_idx:end_idx] -= normalized_energy

        terminal_energies = torch.stack(terminal_energies, dim=1)

        # Update moving average for tracking
        batch_reward_mean = rewards[-1].mean().item()
        self.moving_average.update(batch_reward_mean)

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "entropies": entropies,
            "coords": coords_flat,
            "rand_features": rand_features_all,  # Cached for PPO optimization
            "x_0": x_0,
            "terminal_energies": terminal_energies,
        }

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages.

        Args:
            rewards: (T, B)
            values: (T+1, B)

        Returns:
            (advantages, value_targets) both shape (T, B)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1], device=rewards.device)

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        value_targets = advantages + values[:-1]

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std

        return advantages, value_targets

    def ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        new_values: torch.Tensor,
        value_targets: torch.Tensor,
        entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with entropy regularization.

        Args:
            old_log_probs: (T, B)
            new_log_probs: (T, B)
            advantages: (T, B)
            new_values: (T, B)
            value_targets: (T, B)
            entropy: (T, B)

        Returns:
            (loss, metrics)
        """
        # Policy loss with clipping
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20, max=20)  # Prevent overflow
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, value_targets.detach())

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Clip fraction for logging
        clip_fraction = (torch.abs(ratio - 1.0) > self.clip_value).float().mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "mean_ratio": ratio.mean().item(),
            "clip_fraction": clip_fraction.item(),
        }

        return loss, metrics

    def get_loss(
        self,
        batch: Dict[str, torch.Tensor],
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute PPO loss for batch with multiple inner update steps.
        """
        coords = batch["coords"].to(self.device)

        # Collect rollout (model in eval mode, no grad)
        self.model.eval()
        with torch.no_grad():
            rollout = self.collect_rollout(coords, key)

        # Compute advantages using collected values
        advantages, value_targets = self.compute_advantages(
            rollout["rewards"], rollout["values"]
        )

        # Detach old log probs for ratio computation
        old_log_probs = rollout["log_probs"].detach()

        T = self.n_diffusion_steps
        total_loss = None
        all_metrics = {}

        # Multiple PPO epochs
        self.model.train()
        for ppo_epoch in range(self.n_inner_steps):
            new_log_probs = []
            new_values = []
            new_entropies = []

            for t in range(T - 1, -1, -1):
                step_idx = T - 1 - t
                timestep = torch.full(
                    (rollout["coords"].shape[0],), t, device=self.device, dtype=torch.long
                )

                # Forward pass (with gradients) - IMPORTANT: use cached rand_features
                logits = self.model(
                    rollout["coords"],
                    rollout["states"][step_idx],
                    timestep,
                    rand_features=rollout["rand_features"][step_idx]
                )

                # Compute new log prob using same posterior distribution as sampling
                noise_t_idx = T - 1 - t
                log_prob = self.noise_class.compute_action_log_prob(
                    logits,
                    rollout["states"][step_idx],
                    rollout["actions"][step_idx],
                    noise_t_idx
                )
                new_log_probs.append(log_prob)

                # Compute new value
                value = self._get_value(rollout["coords"], rollout["states"][step_idx], timestep)
                new_values.append(value)

                # Compute entropy
                entropy = self._compute_entropy(logits)
                new_entropies.append(entropy)

            new_log_probs = torch.stack(new_log_probs, dim=0)
            new_values = torch.stack(new_values, dim=0)
            new_entropies = torch.stack(new_entropies, dim=0)

            # Compute PPO loss
            loss, metrics = self.ppo_loss(
                old_log_probs,
                new_log_probs,
                advantages,
                new_values,
                value_targets,
                new_entropies
            )

            # Only backprop on final epoch (or could accumulate)
            if ppo_epoch == self.n_inner_steps - 1:
                total_loss = loss
                all_metrics = metrics

        # Add energy metrics
        with torch.no_grad():
            all_metrics["mean_energy"] = rollout["terminal_energies"].mean().item()
            all_metrics["best_energy"] = rollout["terminal_energies"].min(dim=1)[0].mean().item()

        return total_loss, all_metrics
