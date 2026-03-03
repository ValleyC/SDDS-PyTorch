"""
PPO Trainer - DiffUCO Faithful Implementation

This module implements PPO training exactly as in DiffUCO:
- TD(lambda) advantage computation (_calc_traces)
- Moving average reward normalization (MovingAverages)
- PPO clipped loss (PPO_loss)

Reference: DIffUCO/Trainers/PPO_Trainer.py, DIffUCO/utils/MovingAverages.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from .noise_schedule import CategoricalNoiseSchedule, GaussianNoiseSchedule
    from .step_model import DiffusionStepModel, scatter_sum
    from .trajectory import TrajectoryBuffer, collect_trajectory, collect_continuous_trajectory
    from .continuous_step_model import ContinuousDiffusionStepModel
except ImportError:
    from noise_schedule import CategoricalNoiseSchedule, GaussianNoiseSchedule
    from step_model import DiffusionStepModel, scatter_sum
    from trajectory import TrajectoryBuffer, collect_trajectory, collect_continuous_trajectory
    from continuous_step_model import ContinuousDiffusionStepModel


class MovingAverage:
    """
    Moving average for reward normalization.

    Matches DiffUCO's MovingAverages.py exactly.
    """

    def __init__(self, alpha: float, beta: float):
        """
        Args:
            alpha: Smoothing factor for mean (0 < alpha <= 1)
            beta: Smoothing factor for std (0 < alpha <= 1)
        """
        self.alpha = alpha
        self.beta = beta
        self.step_count = 0
        self.mean_value = 0.0
        self.std_value = 0.0

    def update_mov_averages(self, data: torch.Tensor) -> Tuple[float, float]:
        """
        Update moving averages with new data.

        Args:
            data: New data tensor

        Returns:
            (mean_value, std_value)
        """
        if self.alpha == -1:
            return 0.0, 1.0

        mean_data = data.mean().item()
        # Use unbiased=False to match JAX/NumPy's default (population std)
        std_data = data.std(unbiased=False).item()

        if self.step_count == 0:
            self.mean_value = mean_data
            self.std_value = std_data
        else:
            self.mean_value = self.alpha * mean_data + (1 - self.alpha) * self.mean_value
            self.std_value = self.beta * std_data + (1 - self.beta) * self.std_value

        self.step_count += 1
        return self.mean_value, self.std_value

    def calculate_average(
        self,
        rewards: torch.Tensor,
        mov_average_reward: float,
        mov_std_reward: float
    ) -> torch.Tensor:
        """
        Normalize rewards using moving average.

        From MovingAverages.py line 44-45:
            normed_rewards = (rewards - mov_average_reward)/(mov_std_reward + 10**-10)
        """
        return (rewards - mov_average_reward) / (mov_std_reward + 1e-10)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE (Generalized Advantage Estimation) - TD(lambda).

    Matches DiffUCO's _calc_traces exactly (PPO_Trainer.py lines 491-501):
        for t in range(max_steps):
            idx = max_steps - t - 1
            delta = rewards[idx] + gamma * values[idx + 1] - values[idx]
            advantage[idx] = delta + gamma * lam * advantage[idx + 1]
        value_target = advantage + values

    Args:
        rewards: Rewards tensor (n_steps, n_graphs, n_basis_states)
        values: Values tensor (n_steps+1, n_graphs, n_basis_states)
                Note: values includes one extra step for bootstrapping

    Returns:
        value_target: (n_steps, n_graphs, n_basis_states)
        advantages: (n_steps, n_graphs, n_basis_states)
    """
    n_steps = rewards.size(0)
    device = rewards.device

    # Initialize advantage with zeros (same shape as rewards)
    advantage = torch.zeros_like(rewards)

    # Compute advantages in reverse order
    for t in range(n_steps):
        idx = n_steps - t - 1

        # TD error: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        if idx == n_steps - 1:
            # For last step, next value is values[n_steps] (the extra bootstrap value)
            next_value = values[idx + 1] if values.size(0) > idx + 1 else torch.zeros_like(values[idx])
        else:
            next_value = values[idx + 1]

        delta = rewards[idx] + gamma * next_value - values[idx]

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        if idx == n_steps - 1:
            next_advantage = torch.zeros_like(delta)
        else:
            next_advantage = advantage[idx + 1]

        advantage[idx] = delta + gamma * lam * next_advantage

    # Value target = advantage + values (for critic training)
    value_target = advantage + values[:n_steps]

    return value_target, advantage


def normalize_advantages(advantages: torch.Tensor, exclude_last: bool = True) -> torch.Tensor:
    """
    Normalize advantages.

    From PPO_Trainer.py lines 523-527:
        unpadded_adv = advantages[:,:,:-1]
        normed_advantages = (advantages - mean(unpadded_adv))/(std(unpadded_adv)+10**-10)

    DiffUCO uses [:,:,:-1] to exclude last element in dimension 2 (basis states).

    Args:
        advantages: (n_steps, n_graphs, n_basis_states)
        exclude_last: If True, exclude last basis state for normalization stats
    """
    if exclude_last and advantages.shape[2] > 1:
        # Match DiffUCO exactly: advantages[:,:,:-1] excludes last in dim 2
        unpadded_adv = advantages[:, :, :-1]
        mean = unpadded_adv.mean()
        # Use population std (unbiased=False) to match JAX/NumPy
        std = unpadded_adv.std(unbiased=False)
    else:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)

    return (advantages - mean) / (std + 1e-10)


class PPOTrainer:
    """
    PPO Trainer matching DiffUCO's implementation.

    Key parameters (from PPO_Trainer.py lines 67-77):
        - gamma = 1.0
        - lambda = exp(-log(TD_k) / time_horizon)
        - clip_value = 0.2
        - c1 (value_weighting) = 0.65

    Minibatching (from PPO_Trainer.py lines 74-76, 249-269):
        - minib_diff_steps: minibatch size for diffusion steps
        - minib_basis_states: minibatch size for basis states
        - Total updates = inner_loop_steps * (n_steps * n_basis) / (minib_diff * minib_basis)
    """

    def __init__(
        self,
        model: DiffusionStepModel,
        noise_schedule: CategoricalNoiseSchedule,
        lr: float = 3e-4,
        gamma: float = 1.0,
        TD_k: float = 3.0,
        clip_value: float = 0.2,
        value_weighting: float = 0.65,
        inner_loop_steps: int = 4,
        mov_average_alpha: float = 0.2,
        minib_diff_steps: Optional[int] = None,
        minib_basis_states: Optional[int] = None,
    ):
        """
        Args:
            model: DiffusionStepModel
            noise_schedule: CategoricalNoiseSchedule
            lr: Learning rate
            gamma: Discount factor (DiffUCO uses 1.0)
            TD_k: TD(lambda) parameter, lambda = exp(-log(TD_k)/T)
            clip_value: PPO clip epsilon
            value_weighting: c1 in loss = (1-c1)*actor + c1*critic
            inner_loop_steps: Number of PPO epochs per trajectory batch
            mov_average_alpha: Smoothing factor for reward normalization
            minib_diff_steps: Minibatch size for diffusion steps (None = use all)
            minib_basis_states: Minibatch size for basis states (None = use all)
        """
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma = gamma
        self.clip_value = clip_value
        self.value_weighting = value_weighting
        self.inner_loop_steps = inner_loop_steps
        self.minib_diff_steps = minib_diff_steps
        self.minib_basis_states = minib_basis_states

        # Compute lambda from TD_k (PPO_Trainer.py line 68)
        time_horizon = noise_schedule.n_steps
        self.lam = np.exp(-np.log(TD_k) / time_horizon)

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Moving average for reward normalization
        self.moving_avg = MovingAverage(mov_average_alpha, mov_average_alpha)

    def compute_ppo_loss_minibatch(
        self,
        model: DiffusionStepModel,
        states: torch.Tensor,
        actions: torch.Tensor,
        rand_node_features: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        time_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss on a minibatch.

        This is called with already-indexed (minibatch) data from select_time_idxs.

        Args:
            states: (minib_steps, n_nodes, minib_basis)
            actions: (minib_steps, n_nodes, minib_basis)
            rand_node_features: (minib_steps, n_nodes, minib_basis, n_rand)
            old_log_probs: (minib_steps, n_graphs, minib_basis)
            advantages: (minib_steps, n_graphs, minib_basis)
            value_targets: (minib_steps, n_graphs, minib_basis)
            time_indices: (minib_steps, minib_basis) - per-sample time indices

        Matches DiffUCO's PPO_loss (PPO_Trainer.py lines 532-569).
        """
        n_minib_steps, n_nodes, n_minib_basis = states.shape
        device = states.device

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        all_ratios = []

        for mb_t in range(n_minib_steps):
            for mb_b in range(n_minib_basis):
                # Get state and action for this minibatch sample
                X_t = states[mb_t, :, mb_b]  # (n_nodes,)
                X_next = actions[mb_t, :, mb_b]  # (n_nodes,)
                rand_nodes = rand_node_features[mb_t, :, mb_b, :]  # (n_nodes, n_rand)

                # Get the actual time index for this sample
                t_idx = time_indices[mb_t, mb_b].item()

                # Forward pass with stored random features and time index
                spin_log_probs, values, _ = model(
                    X_t, t_idx, edge_index, edge_attr, node_graph_idx, n_graphs,
                    rand_nodes=rand_nodes
                )

                # Compute new log prob for the action taken
                new_state_log_prob = model.get_state_log_prob(
                    spin_log_probs, X_next, node_graph_idx, n_graphs
                )  # (n_graphs,)

                # Get old log prob and advantages for this sample
                old_state_log_prob = old_log_probs[mb_t, :, mb_b]  # (n_graphs,)
                adv = advantages[mb_t, :, mb_b]  # (n_graphs,)
                v_target = value_targets[mb_t, :, mb_b]  # (n_graphs,)

                # PPO ratio
                ratios = torch.exp(new_state_log_prob - old_state_log_prob)
                all_ratios.append(ratios)

                # Clipped surrogate objective
                # DiffUCO excludes last graph (padding): [:,:-1]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.clip_value, 1 + self.clip_value) * adv
                # Exclude last graph if n_graphs > 1 (padding convention)
                if n_graphs > 1:
                    actor_loss = -torch.min(surr1, surr2)[:-1].mean()
                    critic_loss = F.mse_loss(values[:-1], v_target[:-1])
                else:
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, v_target)

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss

        # Average over minibatch
        n_total = n_minib_steps * n_minib_basis
        total_actor_loss /= n_total
        total_critic_loss /= n_total

        # Combined loss (PPO_Trainer.py line 559)
        overall_loss = (1 - self.value_weighting) * total_actor_loss + \
                       self.value_weighting * total_critic_loss

        # Compute ratio statistics
        all_ratios = torch.cat(all_ratios)
        clip_fraction = ((all_ratios < 1 - self.clip_value) |
                         (all_ratios > 1 + self.clip_value)).float().mean()

        loss_dict = {
            'actor_loss': total_actor_loss.item(),
            'critic_loss': total_critic_loss.item(),
            'overall_loss': overall_loss.item(),
            'max_ratios': all_ratios.max().item(),
            'min_ratios': all_ratios.min().item(),
            'mean_ratios': all_ratios.mean().item(),
            'clip_fraction': clip_fraction.item(),
        }

        return overall_loss, loss_dict

    def select_minibatch(
        self,
        buffer_dict: Dict[str, torch.Tensor],
        diff_step_indices: torch.Tensor,
        basis_indices: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Select a minibatch from the trajectory buffer using random indices.

        Matches DiffUCO's select_time_idxs (PPO_Trainer.py lines 13-55).

        Args:
            buffer_dict: Dict with keys: states, actions, rand_node_features,
                         policies, advantages, value_targets, time_index_per_node
            diff_step_indices: (minib_diff_steps,) - which timesteps to select
            basis_indices: (minib_basis_states,) - which basis states to select
            node_graph_idx: (n_nodes,) - graph index for each node
            n_graphs: Number of graphs

        Returns:
            minibatch_dict with same keys but indexed by minibatch indices
        """
        minib_diff = len(diff_step_indices)
        minib_basis = len(basis_indices)

        # Create index grids for advanced indexing
        # For node-level data: (n_steps, n_nodes, n_basis, ...)
        # For graph-level data: (n_steps, n_graphs, n_basis)

        out_dict = {}

        # Index node-level data: states, actions, rand_node_features
        # These have shape (n_steps, n_nodes, n_basis, ...) or (n_steps, n_nodes, n_basis)
        for key in ['states', 'actions', 'rand_node_features']:
            data = buffer_dict[key]
            # Index: [diff_step_indices, :, basis_indices, ...]
            # Use advanced indexing
            indexed = data[diff_step_indices][:, :, basis_indices]
            out_dict[key] = indexed

        # Index graph-level data: policies, advantages, value_targets
        # These have shape (n_steps, n_graphs, n_basis)
        for key in ['policies', 'advantages', 'value_targets']:
            data = buffer_dict[key]
            indexed = data[diff_step_indices][:, :, basis_indices]
            out_dict[key] = indexed

        # Extract time indices: (minib_diff, minib_basis)
        # From time_index_per_node which is (n_steps, n_nodes, n_basis)
        # All nodes in the same step/basis have the same time index
        time_indices = buffer_dict['time_index_per_node'][diff_step_indices][:, 0, :][:, basis_indices]
        out_dict['time_indices'] = time_indices

        return out_dict

    def compute_ppo_loss(
        self,
        model: DiffusionStepModel,
        states: torch.Tensor,
        actions: torch.Tensor,
        rand_node_features: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        time_index_per_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss (full batch version for backwards compatibility).

        Matches DiffUCO's PPO_loss (PPO_Trainer.py lines 532-569):
            ratios = exp(new_log_prob - old_log_prob)
            surr1 = ratios * advantages
            surr2 = clip(ratios, 1-eps, 1+eps) * advantages
            actor_loss = -mean(min(surr1, surr2)[:,:-1])  # Exclude last (padded) graph
            critic_loss = mean((values - value_target)[:,:-1] ** 2)
            overall_loss = (1-c1) * actor_loss + c1 * critic_loss
        """
        n_steps, n_nodes, n_basis = states.shape

        # Create time indices tensor from time_index_per_node
        # All nodes have the same time index per step/basis
        time_indices = time_index_per_node[:, 0, :]  # (n_steps, n_basis)

        return self.compute_ppo_loss_minibatch(
            model, states, actions, rand_node_features,
            old_log_probs, advantages, value_targets,
            time_indices, edge_index, edge_attr,
            node_graph_idx, n_graphs
        )

    def train_step(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        n_basis_states: int,
        T_temperature: float,
        energy_fn=None,
    ) -> Dict:
        """
        One PPO training step.

        1. Collect trajectory
        2. Compute advantages with reward normalization
        3. Run PPO updates with minibatching (DiffUCO-style)

        From PPO_Trainer.py lines 249-269:
            for i in range(self.inner_loop_steps):
                perm_diff_array, perm_state_array = shuffle(...)
                for each minibatch:
                    batch_dict = select_time_idxs(...)
                    loss, params = ppo_loss_backward(...)

        Returns:
            Dictionary with loss statistics
        """
        self.model.train()
        device = edge_index.device
        n_diffusion_steps = self.noise_schedule.n_steps

        # === Phase 1: Collect trajectory ===
        with torch.no_grad():
            buffer = collect_trajectory(
                model=self.model,
                noise_schedule=self.noise_schedule,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_graph_idx=node_graph_idx,
                n_graphs=n_graphs,
                n_basis_states=n_basis_states,
                T_temperature=T_temperature,
                energy_fn=energy_fn,
                device=device,
            )

        # === Phase 2: Compute advantages ===
        # Reward normalization using moving average (PPO_Trainer.py lines 505-509)
        # DiffUCO: reduced_rewards = rewards[:,:,:-1] excludes last element in dim 2
        # rewards shape: (n_steps, n_graphs, n_basis_states)
        # [:,:,:-1] excludes last basis state for stat computation
        # (In jraph batching, the last graph is padding, but DiffUCO slices axis 2)
        reduced_rewards = buffer.rewards[:, :, :-1]  # Match DiffUCO exactly: [:,:,:-1]
        mov_mean, mov_std = self.moving_avg.update_mov_averages(reduced_rewards)
        # Apply normalization to full rewards
        normed_rewards = self.moving_avg.calculate_average(buffer.rewards, mov_mean, mov_std)

        # Append bootstrap value (zeros for terminal state)
        values_with_bootstrap = torch.cat([
            buffer.values,
            torch.zeros(1, *buffer.values.shape[1:], device=device)
        ], dim=0)

        # Compute GAE
        value_targets, advantages = compute_gae(
            normed_rewards, values_with_bootstrap,
            gamma=self.gamma, lam=self.lam
        )

        # Normalize advantages
        normed_advantages = normalize_advantages(advantages)

        # === Phase 3: PPO updates with minibatching ===
        # Determine minibatch sizes (PPO_Trainer.py lines 74-75)
        minib_diff = self.minib_diff_steps if self.minib_diff_steps else n_diffusion_steps
        minib_basis = self.minib_basis_states if self.minib_basis_states else n_basis_states
        minib_diff = min(minib_diff, n_diffusion_steps)
        minib_basis = min(minib_basis, n_basis_states)

        n_diff_batches = n_diffusion_steps // minib_diff
        n_basis_batches = n_basis_states // minib_basis

        # Prepare buffer dict for minibatching
        buffer_dict = {
            'states': buffer.states,
            'actions': buffer.actions,
            'rand_node_features': buffer.rand_node_features,
            'policies': buffer.policies,
            'advantages': normed_advantages,
            'value_targets': value_targets,
            'time_index_per_node': buffer.time_index_per_node,
        }

        total_loss_dict = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'overall_loss': 0.0,
            'max_ratios': 0.0,
            'min_ratios': float('inf'),
            'mean_ratios': 0.0,
            'clip_fraction': 0.0,
        }

        n_updates = 0

        for epoch in range(self.inner_loop_steps):
            # Shuffle indices (PPO_Trainer.py lines 251-253)
            perm_diff = torch.randperm(n_diffusion_steps, device=device)
            perm_basis = torch.randperm(n_basis_states, device=device)

            # Split into minibatches
            diff_batches = perm_diff.split(minib_diff)
            basis_batches = perm_basis.split(minib_basis)

            for diff_idx in diff_batches:
                for basis_idx in basis_batches:
                    # Select minibatch
                    minibatch = self.select_minibatch(
                        buffer_dict, diff_idx, basis_idx,
                        node_graph_idx, n_graphs
                    )

                    self.optimizer.zero_grad()

                    loss, loss_dict = self.compute_ppo_loss_minibatch(
                        self.model,
                        minibatch['states'],
                        minibatch['actions'],
                        minibatch['rand_node_features'],
                        minibatch['policies'],
                        minibatch['advantages'],
                        minibatch['value_targets'],
                        minibatch['time_indices'],
                        edge_index,
                        edge_attr,
                        node_graph_idx,
                        n_graphs,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Accumulate statistics
                    for key in ['actor_loss', 'critic_loss', 'overall_loss', 'mean_ratios', 'clip_fraction']:
                        total_loss_dict[key] += loss_dict[key]
                    total_loss_dict['max_ratios'] = max(total_loss_dict['max_ratios'], loss_dict['max_ratios'])
                    total_loss_dict['min_ratios'] = min(total_loss_dict['min_ratios'], loss_dict['min_ratios'])

                    n_updates += 1

        # Average over all updates
        if n_updates > 0:
            for key in ['actor_loss', 'critic_loss', 'overall_loss', 'mean_ratios', 'clip_fraction']:
                total_loss_dict[key] /= n_updates

        # Add trajectory statistics
        total_loss_dict['noise_reward_sum'] = buffer.noise_rewards.sum().item()
        total_loss_dict['entropy_reward_sum'] = buffer.entropy_rewards.sum().item()
        total_loss_dict['energy_reward_mean'] = buffer.energy_rewards.mean().item()
        total_loss_dict['mov_avg_mean'] = mov_mean
        total_loss_dict['mov_avg_std'] = mov_std
        total_loss_dict['n_ppo_updates'] = n_updates

        # Advantage statistics (for debugging)
        total_loss_dict['advantage_mean'] = advantages.mean().item()
        total_loss_dict['advantage_std'] = advantages.std().item()
        total_loss_dict['advantage_max'] = advantages.max().item()
        total_loss_dict['advantage_min'] = advantages.min().item()
        total_loss_dict['reward_mean'] = buffer.rewards.mean().item()
        total_loss_dict['reward_std'] = buffer.rewards.std().item()

        return total_loss_dict


##############################################################################
# Continuous PPO Trainer for Chip Placement
##############################################################################

class ContinuousPPOTrainer:
    """
    PPO Trainer for continuous (Gaussian) diffusion.

    Same structure as PPOTrainer but uses:
    - ContinuousDiffusionStepModel (outputs mean, log_var)
    - GaussianNoiseSchedule
    - collect_continuous_trajectory (with per-step energy)
    """

    def __init__(
        self,
        model: ContinuousDiffusionStepModel,
        noise_schedule: GaussianNoiseSchedule,
        lr: float = 1e-4,
        gamma: float = 1.0,
        TD_k: float = 3.0,
        clip_value: float = 0.2,
        value_weighting: float = 0.65,
        inner_loop_steps: int = 2,
        mov_average_alpha: float = 0.0009,
        per_step_energy: bool = True,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma = gamma
        self.clip_value = clip_value
        self.value_weighting = value_weighting
        self.inner_loop_steps = inner_loop_steps
        self.per_step_energy = per_step_energy
        self.grad_clip = grad_clip

        time_horizon = noise_schedule.n_steps
        self.lam = np.exp(-np.log(TD_k) / time_horizon)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.moving_avg = MovingAverage(mov_average_alpha, mov_average_alpha)

    def compute_ppo_loss_continuous(
        self,
        model: ContinuousDiffusionStepModel,
        states: torch.Tensor,
        actions: torch.Tensor,
        rand_node_features: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        time_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        node_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss for continuous states.

        Args:
            states: (minib_steps, n_nodes, minib_basis, continuous_dim)
            actions: (minib_steps, n_nodes, minib_basis, continuous_dim)
            rand_node_features: (minib_steps, n_nodes, minib_basis, n_rand)
            old_log_probs: (minib_steps, n_graphs, minib_basis)
            advantages: (minib_steps, n_graphs, minib_basis)
            value_targets: (minib_steps, n_graphs, minib_basis)
            time_indices: (minib_steps, minib_basis)
            node_features: (n_nodes, node_feature_dim)
        """
        n_minib_steps = states.shape[0]
        n_minib_basis = states.shape[2]

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        all_ratios = []

        for mb_t in range(n_minib_steps):
            for mb_b in range(n_minib_basis):
                X_t = states[mb_t, :, mb_b, :]     # (n_nodes, continuous_dim)
                X_next = actions[mb_t, :, mb_b, :]  # (n_nodes, continuous_dim)
                rand_nodes = rand_node_features[mb_t, :, mb_b, :]
                t_idx = time_indices[mb_t, mb_b].item()

                # Forward pass
                mean, log_var, values, _ = model(
                    X_t, t_idx, edge_index, edge_attr,
                    node_graph_idx, n_graphs, node_features,
                    rand_nodes=rand_nodes,
                )

                # New log prob for the action taken
                new_state_log_prob = model.get_state_log_prob(
                    mean, log_var, X_next, node_graph_idx, n_graphs
                )

                old_state_log_prob = old_log_probs[mb_t, :, mb_b]
                adv = advantages[mb_t, :, mb_b]
                v_target = value_targets[mb_t, :, mb_b]

                # PPO ratio
                ratios = torch.exp(new_state_log_prob - old_state_log_prob)
                all_ratios.append(ratios)

                # Clipped surrogate
                surr1 = ratios * adv
                surr2 = torch.clamp(
                    ratios, 1 - self.clip_value, 1 + self.clip_value
                ) * adv

                # No [:-1] slicing — our chip batch has no padding graphs
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, v_target)

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss

        n_total = n_minib_steps * n_minib_basis
        total_actor_loss /= n_total
        total_critic_loss /= n_total

        overall_loss = (
            (1 - self.value_weighting) * total_actor_loss
            + self.value_weighting * total_critic_loss
        )

        all_ratios = torch.cat(all_ratios)
        clip_fraction = (
            (all_ratios < 1 - self.clip_value) | (all_ratios > 1 + self.clip_value)
        ).float().mean()

        loss_dict = {
            'actor_loss': total_actor_loss.item(),
            'critic_loss': total_critic_loss.item(),
            'overall_loss': overall_loss.item(),
            'max_ratios': all_ratios.max().item(),
            'min_ratios': all_ratios.min().item(),
            'mean_ratios': all_ratios.mean().item(),
            'clip_fraction': clip_fraction.item(),
        }

        return overall_loss, loss_dict

    def select_minibatch_continuous(
        self,
        buffer_dict: Dict[str, torch.Tensor],
        diff_step_indices: torch.Tensor,
        basis_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Select minibatch from continuous trajectory buffer.

        Handles 4D state tensors (steps, nodes, basis, continuous_dim).
        """
        out_dict = {}

        # 4D node-level: (steps, nodes, basis, dim) or (steps, nodes, basis, n_rand)
        for key in ['states', 'actions', 'rand_node_features']:
            data = buffer_dict[key]
            indexed = data[diff_step_indices][:, :, basis_indices]
            out_dict[key] = indexed

        # 3D graph-level: (steps, graphs, basis)
        for key in ['policies', 'advantages', 'value_targets']:
            data = buffer_dict[key]
            indexed = data[diff_step_indices][:, :, basis_indices]
            out_dict[key] = indexed

        # Time indices: (steps, nodes, basis) -> (minib_steps, minib_basis)
        time_data = buffer_dict['time_index_per_node']
        time_indices = time_data[diff_step_indices][:, 0, :][:, basis_indices]
        out_dict['time_indices'] = time_indices

        return out_dict

    def train_step(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        n_basis_states: int,
        T_temperature: float,
        node_features: torch.Tensor,
        energy_fn=None,
    ) -> Dict:
        """
        One PPO training step for continuous diffusion.

        1. Collect trajectory with per-step energy
        2. Compute advantages with reward normalization
        3. Run PPO updates
        """
        self.model.train()
        device = edge_index.device
        n_diffusion_steps = self.noise_schedule.n_steps

        # === Phase 1: Collect trajectory ===
        with torch.no_grad():
            buffer = collect_continuous_trajectory(
                model=self.model,
                noise_schedule=self.noise_schedule,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_graph_idx=node_graph_idx,
                n_graphs=n_graphs,
                n_basis_states=n_basis_states,
                T_temperature=T_temperature,
                node_features=node_features,
                energy_fn=energy_fn,
                per_step_energy=self.per_step_energy,
                device=device,
            )

        # === Phase 2: Compute advantages ===
        reduced_rewards = buffer.rewards[:, :, :-1]
        mov_mean, mov_std = self.moving_avg.update_mov_averages(reduced_rewards)
        normed_rewards = self.moving_avg.calculate_average(
            buffer.rewards, mov_mean, mov_std
        )

        values_with_bootstrap = torch.cat([
            buffer.values,
            torch.zeros(1, *buffer.values.shape[1:], device=device)
        ], dim=0)

        value_targets, advantages = compute_gae(
            normed_rewards, values_with_bootstrap,
            gamma=self.gamma, lam=self.lam
        )
        normed_advantages = normalize_advantages(advantages)

        # === Phase 3: PPO updates ===
        buffer_dict = {
            'states': buffer.states,
            'actions': buffer.actions,
            'rand_node_features': buffer.rand_node_features,
            'policies': buffer.policies,
            'advantages': normed_advantages,
            'value_targets': value_targets,
            'time_index_per_node': buffer.time_index_per_node,
        }

        total_loss_dict = {
            'actor_loss': 0.0, 'critic_loss': 0.0, 'overall_loss': 0.0,
            'max_ratios': 0.0, 'min_ratios': float('inf'),
            'mean_ratios': 0.0, 'clip_fraction': 0.0,
        }
        n_updates = 0

        for epoch in range(self.inner_loop_steps):
            perm_diff = torch.randperm(n_diffusion_steps, device=device)
            perm_basis = torch.randperm(n_basis_states, device=device)

            # Use full batch (no minibatching for simplicity)
            minibatch = self.select_minibatch_continuous(
                buffer_dict, perm_diff, perm_basis
            )

            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_ppo_loss_continuous(
                self.model,
                minibatch['states'],
                minibatch['actions'],
                minibatch['rand_node_features'],
                minibatch['policies'],
                minibatch['advantages'],
                minibatch['value_targets'],
                minibatch['time_indices'],
                edge_index, edge_attr, node_graph_idx, n_graphs,
                node_features,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

            for key in ['actor_loss', 'critic_loss', 'overall_loss',
                        'mean_ratios', 'clip_fraction']:
                total_loss_dict[key] += loss_dict[key]
            total_loss_dict['max_ratios'] = max(
                total_loss_dict['max_ratios'], loss_dict['max_ratios']
            )
            total_loss_dict['min_ratios'] = min(
                total_loss_dict['min_ratios'], loss_dict['min_ratios']
            )
            n_updates += 1

        if n_updates > 0:
            for key in ['actor_loss', 'critic_loss', 'overall_loss',
                        'mean_ratios', 'clip_fraction']:
                total_loss_dict[key] /= n_updates

        # Add trajectory stats
        total_loss_dict['noise_reward_sum'] = buffer.noise_rewards.sum().item()
        total_loss_dict['entropy_reward_sum'] = buffer.entropy_rewards.sum().item()
        total_loss_dict['energy_reward_mean'] = buffer.energy_rewards.mean().item()
        total_loss_dict['mov_avg_mean'] = mov_mean
        total_loss_dict['mov_avg_std'] = mov_std
        total_loss_dict['n_ppo_updates'] = n_updates
        total_loss_dict['advantage_mean'] = advantages.mean().item()
        total_loss_dict['advantage_std'] = advantages.std().item()
        total_loss_dict['reward_mean'] = buffer.rewards.mean().item()
        total_loss_dict['reward_std'] = buffer.rewards.std().item()

        return total_loss_dict


def test_gae_computation():
    """
    Test GAE computation against DiffUCO's _calc_traces.
    """
    print("=" * 60)
    print("Testing GAE (TD-lambda) Computation")
    print("=" * 60)

    # Test parameters
    n_steps = 5
    n_graphs = 2
    n_basis = 3
    gamma = 1.0
    TD_k = 3.0
    lam = np.exp(-np.log(TD_k) / n_steps)

    print(f"\nTest parameters:")
    print(f"  n_steps: {n_steps}")
    print(f"  gamma: {gamma}")
    print(f"  TD_k: {TD_k}")
    print(f"  lambda (computed): {lam:.6f}")

    # Create test data
    torch.manual_seed(42)
    rewards = torch.randn(n_steps, n_graphs, n_basis)
    values = torch.randn(n_steps + 1, n_graphs, n_basis)  # +1 for bootstrap

    print(f"\nInput shapes:")
    print(f"  rewards: {rewards.shape}")
    print(f"  values: {values.shape}")

    # Our computation
    value_target, advantages = compute_gae(rewards, values, gamma=gamma, lam=lam)

    print(f"\nOutput shapes:")
    print(f"  value_target: {value_target.shape}")
    print(f"  advantages: {advantages.shape}")

    # Manual verification (DiffUCO's algorithm)
    manual_advantage = torch.zeros_like(rewards)
    for t in range(n_steps):
        idx = n_steps - t - 1
        delta = rewards[idx] + gamma * values[idx + 1] - values[idx]
        if idx == n_steps - 1:
            manual_advantage[idx] = delta
        else:
            manual_advantage[idx] = delta + gamma * lam * manual_advantage[idx + 1]

    manual_value_target = manual_advantage + values[:n_steps]

    # Compare
    adv_match = torch.allclose(advantages, manual_advantage, atol=1e-6)
    vt_match = torch.allclose(value_target, manual_value_target, atol=1e-6)

    print(f"\n[{'OK' if adv_match else 'FAIL'}] Advantages match manual computation: {adv_match}")
    print(f"[{'OK' if vt_match else 'FAIL'}] Value targets match manual computation: {vt_match}")

    # Verify specific values
    print(f"\nSample values (step=0, graph=0, basis=0):")
    print(f"  Reward: {rewards[0, 0, 0].item():.6f}")
    print(f"  Value[0]: {values[0, 0, 0].item():.6f}")
    print(f"  Value[1]: {values[1, 0, 0].item():.6f}")
    print(f"  Advantage (ours): {advantages[0, 0, 0].item():.6f}")
    print(f"  Advantage (manual): {manual_advantage[0, 0, 0].item():.6f}")

    return adv_match and vt_match


def test_moving_average():
    """
    Test moving average computation.
    """
    print("\n" + "=" * 60)
    print("Testing Moving Average")
    print("=" * 60)

    alpha = 0.2
    mov_avg = MovingAverage(alpha, alpha)

    print(f"\nalpha = {alpha}")

    # Simulate several updates
    data_sequence = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([7.0, 8.0, 9.0]),
    ]

    print("\nUpdating with data sequence:")
    for i, data in enumerate(data_sequence):
        mean, std = mov_avg.update_mov_averages(data)
        print(f"  Step {i}: data_mean={data.mean().item():.4f}, "
              f"mov_mean={mean:.4f}, mov_std={std:.4f}")

    # Test normalization
    test_rewards = torch.tensor([5.0, 6.0, 7.0])
    normed = mov_avg.calculate_average(test_rewards, mean, std)

    print(f"\nNormalization test:")
    print(f"  Rewards: {test_rewards.tolist()}")
    print(f"  Normalized: {normed.tolist()}")

    # Verify: normed = (rewards - mean) / (std + 1e-10)
    expected_normed = (test_rewards - mean) / (std + 1e-10)
    normed_match = torch.allclose(normed, expected_normed, atol=1e-6)

    print(f"[{'OK' if normed_match else 'FAIL'}] Normalization correct: {normed_match}")

    return normed_match


def test_ppo_trainer():
    """
    Test PPO trainer end-to-end.
    """
    print("\n" + "=" * 60)
    print("Testing PPO Trainer")
    print("=" * 60)

    # Test parameters
    n_classes = 4
    n_nodes = 20
    n_edges = 40
    n_graphs = 2
    n_diffusion_steps = 3
    n_basis_states = 2
    T_temperature = 1.0

    print(f"\nTest parameters:")
    print(f"  n_classes: {n_classes}")
    print(f"  n_nodes: {n_nodes}")
    print(f"  n_graphs: {n_graphs}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  n_basis_states: {n_basis_states}")

    # Create model and noise schedule
    model = DiffusionStepModel(
        n_classes=n_classes,
        edge_dim=2,
        hidden_dim=32,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=2,
        n_random_features=5
    )

    noise_schedule = CategoricalNoiseSchedule(
        n_steps=n_diffusion_steps,
        n_classes=n_classes,
        schedule='diffuco'
    )

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        noise_schedule=noise_schedule,
        lr=1e-4,
        gamma=1.0,
        TD_k=3.0,
        clip_value=0.2,
        value_weighting=0.65,
        inner_loop_steps=2,
        mov_average_alpha=0.2,
    )

    print(f"\nTrainer config:")
    print(f"  gamma: {trainer.gamma}")
    print(f"  lambda: {trainer.lam:.6f}")
    print(f"  clip_value: {trainer.clip_value}")
    print(f"  value_weighting: {trainer.value_weighting}")

    # Create test graph
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 2)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), n_nodes // n_graphs
    )

    # Run training step
    print("\nRunning training step...")
    loss_dict = trainer.train_step(
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
    )

    print(f"\nTraining step results:")
    for key, value in loss_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Verify loss values are reasonable
    actor_loss_ok = loss_dict['actor_loss'] < 100  # Should be bounded
    critic_loss_ok = loss_dict['critic_loss'] < 1000
    ratio_ok = 0 < loss_dict['mean_ratios'] < 10  # Ratios should be around 1

    all_ok = actor_loss_ok and critic_loss_ok and ratio_ok

    print(f"\n[{'OK' if all_ok else 'FAIL'}] Training step completed successfully")

    # Test gradient flow
    print("\nVerifying gradient flow...")
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    grad_ok = params_with_grad == total_params

    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"[{'OK' if grad_ok else 'FAIL'}] All parameters have gradients")

    return all_ok and grad_ok


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# PPO Trainer Tests")
    print("#" * 60)

    results = {}

    results['gae'] = test_gae_computation()
    results['moving_average'] = test_moving_average()
    results['ppo_trainer'] = test_ppo_trainer()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")

    if all(results.values()):
        print("\nAll PPO trainer tests passed!")


if __name__ == "__main__":
    main()
