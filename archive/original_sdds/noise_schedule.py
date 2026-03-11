"""
Categorical Noise Schedule - DiffUCO Faithful Implementation

This module implements the categorical noise distribution exactly as in DiffUCO:
- BaseNoise.py: beta schedule with flip (reversal)
- CategoricalNoise.py: transition kernel and noise reward computation

Reference: DIffUCO/NoiseDistributions/BaseNoise.py, CategoricalNoise.py
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CategoricalNoiseSchedule:
    """
    Categorical noise schedule matching DiffUCO exactly.

    From BaseNoise.py:
        - beta_t = beta_t_func(t, n_diffusion_steps)
        - beta_arr is FLIPPED (reversed) after construction

    From CategoricalNoise.py:
        - beta_t = 2 * gamma_t (where gamma_t = beta_arr[t])
        - p_change = beta_t / K
        - p_stay = 1 - beta_t + p_change
    """

    def __init__(self, n_steps: int, n_classes: int, schedule: str = 'diffuco'):
        """
        Args:
            n_steps: Number of diffusion steps T
            n_classes: Number of categories K (n_bernoulli_features in DiffUCO)
            schedule: 'diffuco' for 1/(T-t+1) schedule
        """
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.schedule = schedule

        # Compute gamma schedule (called beta_list in DiffUCO before flip)
        gamma_list = []
        for t in range(n_steps):
            gamma_list.append(self._beta_t_func(t, n_steps, schedule))

        # CRITICAL: DiffUCO flips the array (BaseNoise.py line 16)
        # self.beta_arr = jnp.flip(jnp.array(self.beta_list), axis=-1)
        gamma_arr = torch.tensor(gamma_list, dtype=torch.float32)
        self.gamma_arr = torch.flip(gamma_arr, dims=[0])

        # In CategoricalNoise, beta_t = 2 * gamma_t
        self.beta_arr = 2 * self.gamma_arr

    def _beta_t_func(self, t: int, n_diffusion_steps: int, schedule: str) -> float:
        """
        Beta schedule function from BaseNoise.py lines 54-80.

        DiffUCO schedule: beta = 1 / (n_diffusion_steps - t + 1)
        """
        if schedule == 'diffuco':
            # From BaseNoise.py line 64
            beta = 1.0 / (n_diffusion_steps - t + 1)
        elif schedule == 'linear':
            beta = 0.01 + (0.25 - 0.01) * t / max(n_diffusion_steps - 1, 1)
        else:
            # Default schedule from line 79
            beta = 1.0 / (n_diffusion_steps - t + 2)
        return beta

    def get_transition_probs(self, t: int, n_classes_effective: int = None) -> Tuple[float, float]:
        """
        Get transition probabilities for step t.

        From CategoricalNoise.py lines 57-59:
            gamma_t = self.beta_arr[model_step_idx]
            beta_t = 2 * gamma_t
            p_change_value = beta_t / (self.n_bernoilli_features)
            p_stay_value = 1 - beta_t + p_change_value

        Note: In our implementation, gamma_arr already contains the flipped values,
        and beta_arr = 2 * gamma_arr, so we just use beta_arr[t] directly.

        Args:
            t: Timestep index
            n_classes_effective: Number of valid classes (K_instance). If None, uses self.n_classes.

        Returns:
            p_stay: Probability of staying in same state
            p_change: Probability of changing to any other state
        """
        if n_classes_effective is None:
            n_classes_effective = self.n_classes
        beta_t = self.beta_arr[t].item()
        p_change = beta_t / n_classes_effective
        p_stay = 1 - beta_t + p_change
        return p_stay, p_change

    def compute_forward_log_prob(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t: int,
        n_classes_effective: int = None
    ) -> torch.Tensor:
        """
        Compute log P_forward(X_next | X_prev) for the noise process.

        From CategoricalNoise.py lines 61-65:
            log_p_i = jnp.where(X_next == X_prev, jnp.log(p_stay_value), jnp.log(p_change_value))
            noise_per_node = jnp.sum(log_p_i, axis=-1)

        Args:
            X_prev: Previous state (n,) or (n, basis_states)
            X_next: Next state, same shape as X_prev
            t: Timestep index (into the flipped beta_arr)
            n_classes_effective: Number of valid classes (K_instance). If None, uses self.n_classes.

        Returns:
            log_prob: Log probability per node, same shape as input
        """
        p_stay, p_change = self.get_transition_probs(t, n_classes_effective)
        device = X_prev.device

        # Compute log probabilities with numerical stability
        eps = 1e-10
        log_p_stay = torch.log(torch.tensor(max(p_stay, eps), device=device))
        log_p_change = torch.log(torch.tensor(max(p_change, eps), device=device))

        # Where X_next == X_prev, use log_p_stay; else log_p_change
        same_mask = (X_next == X_prev)
        log_p = torch.where(same_mask, log_p_stay, log_p_change)

        return log_p

    def compute_noise_reward(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t: int,
        T_temperature: float,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        n_classes_effective: int = None
    ) -> torch.Tensor:
        """
        Compute noise reward for a step, matching CategoricalNoise.calc_noise_step.

        From CategoricalNoise.py lines 54-68:
            log_p_i = jnp.where(X_next == X_prev, jnp.log(p_stay_value), jnp.log(p_change_value))
            noise_per_node = jnp.sum(log_p_i, axis=-1)
            noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)
            noise_step_value = -T*noise_per_graph

        The reward stored is: noise_rewards[t] -= noise_step_value
                            = noise_rewards[t] + T * noise_per_graph

        Args:
            X_prev: Previous state (n,) or (n, basis_states)
            X_next: Next state
            t: Timestep index
            T_temperature: Temperature parameter
            node_graph_idx: Graph index for each node (n,)
            n_graphs: Number of graphs
            n_classes_effective: Number of valid classes (K_instance). If None, uses self.n_classes.

        Returns:
            noise_reward_per_graph: (n_graphs,) or (n_graphs, basis_states)
        """
        log_p_per_node = self.compute_forward_log_prob(X_prev, X_next, t, n_classes_effective)

        # Sum over nodes per graph (segment_sum equivalent)
        if log_p_per_node.dim() == 1:
            # Single basis state
            noise_per_graph = torch.zeros(n_graphs, device=X_prev.device)
            noise_per_graph.scatter_add_(0, node_graph_idx, log_p_per_node)
        else:
            # Multiple basis states: (n, basis_states)
            n_basis = log_p_per_node.size(1)
            noise_per_graph = torch.zeros(n_graphs, n_basis, device=X_prev.device)
            for b in range(n_basis):
                noise_per_graph[:, b].scatter_add_(0, node_graph_idx, log_p_per_node[:, b])

        # noise_step_value = -T * noise_per_graph
        # reward contribution = -noise_step_value = T * noise_per_graph
        noise_reward = T_temperature * noise_per_graph

        return noise_reward

    def calculate_combined_reward(
        self,
        noise_reward: torch.Tensor,
        entropy_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine noise and entropy rewards.

        From CategoricalNoise.py lines 15-16:
            def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
                return -(noise_distr_step - entropy_reward)

        In PPO_Trainer.py line 381:
            combined_reward = self.NoiseDistrClass.calculate_noise_distr_reward(-noise_rewards, entropy_rewards)

        So: combined = -(-noise_rewards - entropy_rewards) = noise_rewards + entropy_rewards

        Args:
            noise_reward: Noise reward (already T * log_p_forward)
            entropy_reward: Entropy reward (T * (-state_log_prob))

        Returns:
            combined_reward
        """
        return noise_reward + entropy_reward


class GaussianNoiseSchedule:
    """
    Gaussian noise schedule for continuous diffusion, matching DiffUCO exactly.

    Forward process: X_t = sqrt(alpha_t) * X_{t-1} + sqrt(beta_t) * epsilon
    where epsilon ~ N(0, I)

    Key difference from CategoricalNoiseSchedule:
        - Categorical: beta_arr = 2 * gamma_arr
        - Gaussian: beta_arr = gamma_arr (no 2x factor)
        - Both use the same underlying gamma schedule (1/(T-t+1), flipped)

    Reference: DIffUCO/NoiseDistributions/GaussianNoise.py
    """

    def __init__(self, n_steps: int, schedule: str = 'diffuco', **kwargs):
        """
        Args:
            n_steps: Number of diffusion steps T
            schedule: 'diffuco' for 1/(T-t+1) schedule
        """
        self.n_steps = n_steps
        self.schedule = schedule

        # Compute gamma schedule (same base schedule as categorical)
        gamma_list = []
        for t in range(n_steps):
            gamma_list.append(self._beta_t_func(t, n_steps, schedule))

        # CRITICAL: DiffUCO flips the array (BaseNoise.py line 16)
        gamma_arr = torch.tensor(gamma_list, dtype=torch.float32)
        self.gamma_arr = torch.flip(gamma_arr, dims=[0])

        # For Gaussian: beta_arr = gamma_arr directly (NO 2x factor)
        # In CategoricalNoise: beta_t = 2 * gamma_t (for transition kernel math)
        # In GaussianNoise: beta_t = gamma_t (used directly as variance)
        self.beta_arr = self.gamma_arr.clone()

        # Precompute alpha values: alpha_t = 1 - beta_t
        self.alpha_arr = 1.0 - self.beta_arr

        # Precompute cumulative products: alpha_bar_t = prod_{i=0}^{t} alpha_i
        self.alpha_bar_arr = torch.cumprod(self.alpha_arr, dim=0)

    def to(self, device):
        """Move all schedule arrays to the specified device."""
        self.gamma_arr = self.gamma_arr.to(device)
        self.beta_arr = self.beta_arr.to(device)
        self.alpha_arr = self.alpha_arr.to(device)
        self.alpha_bar_arr = self.alpha_bar_arr.to(device)
        return self

    def _beta_t_func(self, t: int, n_diffusion_steps: int, schedule: str) -> float:
        """Beta schedule function from BaseNoise.py."""
        if schedule == 'diffuco':
            beta = 1.0 / (n_diffusion_steps - t + 1)
        elif schedule == 'linear':
            beta = 0.01 + (0.25 - 0.01) * t / max(n_diffusion_steps - 1, 1)
        else:
            beta = 1.0 / (n_diffusion_steps - t + 2)
        return beta

    def sample_forward(self, X_prev: torch.Tensor, t: int) -> torch.Tensor:
        """
        Sample from forward diffusion: X_t ~ p(X_t | X_{t-1}).

        X_t = sqrt(alpha_t) * X_{t-1} + sqrt(beta_t) * epsilon

        Args:
            X_prev: Previous state (n_components, continuous_dim) or (..., continuous_dim)
            t: Timestep index

        Returns:
            X_t: Next state, same shape as X_prev
        """
        alpha_t = self.alpha_arr[t]
        beta_t = self.beta_arr[t]

        mean = torch.sqrt(alpha_t) * X_prev
        std = torch.sqrt(beta_t)

        epsilon = torch.randn_like(X_prev)
        X_t = mean + std * epsilon

        return X_t

    def compute_forward_log_prob(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute log P_forward(X_next | X_prev) for Gaussian process.

        p(X_t | X_{t-1}) = N(X_t; sqrt(alpha_t) * X_{t-1}, beta_t * I)

        log p = -0.5 * sum_d [(X_t^d - mu^d)^2 / var + log(2*pi*var)]

        From GaussianNoise.get_log_p_T_0_per_component (lines 99-136).

        Args:
            X_prev: Previous state (..., continuous_dim)
            X_next: Next state (..., continuous_dim)
            t: Timestep index

        Returns:
            log_prob_per_component: Log probability per component (summed over dims).
                Shape: all dims except last removed, i.e., (...,)
        """
        beta_t = self.beta_arr[t]
        alpha_t = self.alpha_arr[t]

        # Forward process: mean = sqrt(alpha_t) * X_prev, var = beta_t
        mean = torch.sqrt(alpha_t) * X_prev
        var = beta_t

        # Gaussian log probability per dimension
        diff = X_next - mean
        log_prob_per_dim = -0.5 * (diff ** 2 / var + torch.log(torch.tensor(2.0 * math.pi) * var))

        # Sum over continuous dimensions → per-component log prob
        log_prob_per_component = log_prob_per_dim.sum(dim=-1)

        return log_prob_per_component

    def compute_noise_reward(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t: int,
        T_temperature: float,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """
        Compute noise reward for a step, matching GaussianNoise.calc_noise_step.

        From GaussianNoise.py calc_noise_step (lines 284-328):
            noise_step_value = -T * log_prob_per_graph
            noise_rewards[t] -= noise_step_value  (i.e., += T * log_prob_per_graph)

        Args:
            X_prev: Previous state (n_components, continuous_dim) or (n_components, n_basis, continuous_dim)
            X_next: Next state, same shape
            t: Timestep index
            T_temperature: Temperature parameter
            node_graph_idx: Graph index per component (n_components,)
            n_graphs: Number of graphs

        Returns:
            noise_reward_per_graph: (n_graphs,) or (n_graphs, n_basis)
        """
        log_p_per_component = self.compute_forward_log_prob(X_prev, X_next, t)

        device = X_prev.device

        if log_p_per_component.dim() == 1:
            # Single basis state: (n_components,)
            noise_per_graph = torch.zeros(n_graphs, device=device)
            noise_per_graph.scatter_add_(0, node_graph_idx, log_p_per_component)
        else:
            # Multiple basis states: (n_components, n_basis)
            n_basis = log_p_per_component.size(1)
            noise_per_graph = torch.zeros(n_graphs, n_basis, device=device)
            for b in range(n_basis):
                noise_per_graph[:, b].scatter_add_(
                    0, node_graph_idx, log_p_per_component[:, b]
                )

        # noise_reward = T * log_prob_per_graph
        noise_reward = T_temperature * noise_per_graph

        return noise_reward

    def calculate_combined_reward(
        self,
        noise_reward: torch.Tensor,
        entropy_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine noise and entropy rewards.

        Same as categorical: combined = noise_reward + entropy_reward

        From GaussianNoise.py calculate_noise_distr_reward (line 63):
            return -(noise_distr_step - entropy_reward)
        With negation from PPO_Trainer: combined = noise_rewards + entropy_rewards
        """
        return noise_reward + entropy_reward


def test_against_diffuco():
    """
    Test our implementation against DiffUCO's expected values.
    """
    print("=" * 60)
    print("Testing CategoricalNoiseSchedule against DiffUCO")
    print("=" * 60)

    # Test parameters
    n_steps = 5
    n_classes = 4  # K = 4 categories

    schedule = CategoricalNoiseSchedule(n_steps, n_classes, schedule='diffuco')

    # Print gamma/beta arrays
    print(f"\nn_steps={n_steps}, n_classes={n_classes}")
    print(f"\nGamma array (after flip):")
    print(f"  {schedule.gamma_arr.numpy()}")
    print(f"\nBeta array (2 * gamma):")
    print(f"  {schedule.beta_arr.numpy()}")

    # Verify: DiffUCO computes gamma_t = 1/(T-t+1) for t in 0..T-1, then flips
    # For T=5: gamma_list = [1/6, 1/5, 1/4, 1/3, 1/2] (before flip)
    # After flip: [1/2, 1/3, 1/4, 1/5, 1/6]
    expected_gamma = torch.tensor([1/2, 1/3, 1/4, 1/5, 1/6])
    expected_beta = 2 * expected_gamma

    print(f"\nExpected gamma (from manual calculation):")
    print(f"  {expected_gamma.numpy()}")
    print(f"\nExpected beta:")
    print(f"  {expected_beta.numpy()}")

    gamma_match = torch.allclose(schedule.gamma_arr, expected_gamma, atol=1e-6)
    beta_match = torch.allclose(schedule.beta_arr, expected_beta, atol=1e-6)

    print(f"\n[OK] Gamma matches: {gamma_match}")
    print(f"[OK] Beta matches: {beta_match}")

    # Test transition probabilities at each step
    print(f"\nTransition probabilities at each step:")
    print(f"{'Step':<6} {'beta_t':<10} {'p_stay':<10} {'p_change':<10}")
    print("-" * 40)

    for t in range(n_steps):
        p_stay, p_change = schedule.get_transition_probs(t)
        beta_t = schedule.beta_arr[t].item()

        # Verify: p_change = beta_t / K, p_stay = 1 - beta_t + p_change
        expected_p_change = beta_t / n_classes
        expected_p_stay = 1 - beta_t + expected_p_change

        print(f"{t:<6} {beta_t:<10.4f} {p_stay:<10.4f} {p_change:<10.4f}")

        assert abs(p_change - expected_p_change) < 1e-6, f"p_change mismatch at t={t}"
        assert abs(p_stay - expected_p_stay) < 1e-6, f"p_stay mismatch at t={t}"

    print("\n[OK] All transition probabilities correct")

    # Test forward log probability computation
    print("\nTesting forward log probability computation:")

    n_nodes = 10
    X_prev = torch.randint(0, n_classes, (n_nodes,))
    X_next = X_prev.clone()
    X_next[0] = (X_next[0] + 1) % n_classes  # Change first node
    X_next[5] = (X_next[5] + 2) % n_classes  # Change sixth node

    t = 2  # Test at step 2
    log_p = schedule.compute_forward_log_prob(X_prev, X_next, t)

    p_stay, p_change = schedule.get_transition_probs(t)
    expected_log_p = torch.where(
        X_prev == X_next,
        torch.log(torch.tensor(p_stay)),
        torch.log(torch.tensor(p_change))
    )

    log_p_match = torch.allclose(log_p, expected_log_p, atol=1e-6)
    print(f"  X_prev: {X_prev.numpy()}")
    print(f"  X_next: {X_next.numpy()}")
    print(f"  log_p:  {log_p.numpy()}")
    print(f"[OK] Forward log prob matches: {log_p_match}")

    # Test noise reward with graph aggregation
    print("\nTesting noise reward with graph aggregation:")

    n_nodes = 12
    n_graphs = 3
    nodes_per_graph = 4
    node_graph_idx = torch.repeat_interleave(torch.arange(n_graphs), nodes_per_graph)

    X_prev = torch.randint(0, n_classes, (n_nodes,))
    X_next = X_prev.clone()
    X_next[::2] = (X_next[::2] + 1) % n_classes  # Change every other node

    T_temperature = 1.5
    noise_reward = schedule.compute_noise_reward(
        X_prev, X_next, t=1, T_temperature=T_temperature,
        node_graph_idx=node_graph_idx, n_graphs=n_graphs
    )

    print(f"  n_nodes={n_nodes}, n_graphs={n_graphs}")
    print(f"  node_graph_idx: {node_graph_idx.numpy()}")
    print(f"  T_temperature: {T_temperature}")
    print(f"  noise_reward per graph: {noise_reward.numpy()}")

    # Verify manually
    log_p = schedule.compute_forward_log_prob(X_prev, X_next, t=1)
    expected_noise_per_graph = torch.zeros(n_graphs)
    expected_noise_per_graph.scatter_add_(0, node_graph_idx, log_p)
    expected_noise_reward = T_temperature * expected_noise_per_graph

    noise_reward_match = torch.allclose(noise_reward, expected_noise_reward, atol=1e-6)
    print(f"[OK] Noise reward matches: {noise_reward_match}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_against_diffuco()
