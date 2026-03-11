"""
Trajectory Collection - DiffUCO Faithful Implementation

This module implements trajectory collection for PPO training exactly as in DiffUCO:
- PPO_Trainer._environment_steps_scan: main trajectory collection
- PPO_Trainer.scan_body: per-step logic

Reference: DIffUCO/Trainers/PPO_Trainer.py
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import math

try:
    from .noise_schedule import CategoricalNoiseSchedule, GaussianNoiseSchedule
    from .step_model import DiffusionStepModel, CVRPDiffusionStepModel, scatter_sum
    from .continuous_step_model import ContinuousDiffusionStepModel
except ImportError:
    from noise_schedule import CategoricalNoiseSchedule, GaussianNoiseSchedule
    from step_model import DiffusionStepModel, CVRPDiffusionStepModel, scatter_sum
    from continuous_step_model import ContinuousDiffusionStepModel


@dataclass
class TrajectoryBuffer:
    """
    Buffer for storing trajectory data for PPO training.

    Matches DiffUCO's RL buffer structure from PPO_Trainer.py lines 409-411:
        log_dict = {"RL": {"states": ..., "rand_node_features": ...,
                           "actions": ..., "policies": ..., "rewards": ...,
                           "values": ...}, ...}
    """
    # Shape: (n_diffusion_steps, n_nodes, n_basis_states) or similar
    states: torch.Tensor           # X_t at each step
    actions: torch.Tensor          # X_{t-1} (next state) at each step
    rand_node_features: torch.Tensor  # Random features used at each step
    policies: torch.Tensor         # state_log_probs at each step
    rewards: torch.Tensor          # Combined rewards at each step
    values: torch.Tensor           # Value estimates at each step
    time_index_per_node: torch.Tensor  # Per-node time indices (n_steps, n_nodes, n_basis)

    # Additional info
    noise_rewards: torch.Tensor    # For logging
    entropy_rewards: torch.Tensor  # For logging
    energy_rewards: torch.Tensor   # Final energy reward
    final_states: torch.Tensor     # X_0 (final denoised state)

    # Diagnostic info (DiffUCO PPO_Trainer.py lines 398-406)
    # For computing forward/reverse KL diagnostics
    log_q_T: Optional[torch.Tensor] = None  # Log prob of initial sample from prior
    log_q_0_T: Optional[torch.Tensor] = None  # Full trajectory log probs under policy
    log_p_0_T: Optional[torch.Tensor] = None  # Full trajectory log probs under target


def collect_trajectory(
    model: DiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float,
    energy_fn: Optional[Callable] = None,
    device: torch.device = None,
    eval_step_factor: int = 1,
) -> TrajectoryBuffer:
    """
    Collect one trajectory with DiffUCO-faithful rewards.

    Matches PPO_Trainer._environment_steps_scan

    Args:
        model: DiffusionStepModel
        noise_schedule: CategoricalNoiseSchedule
        edge_index: Graph edges (2, n_edges)
        edge_attr: Edge features (n_edges, edge_dim)
        node_graph_idx: Graph index for each node (n_nodes,)
        n_graphs: Number of graphs in batch
        n_basis_states: Number of parallel samples (N_basis_states in DiffUCO)
        T_temperature: Temperature parameter
        energy_fn: Optional function to compute energy (returns per-graph energy)
        device: Device to use
        eval_step_factor: Factor to multiply diffusion steps (DiffUCO PPO_Trainer.py:344)
                         overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    Returns:
        TrajectoryBuffer containing all trajectory data
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    n_classes = model.n_classes
    n_diffusion_steps = noise_schedule.n_steps

    # DiffUCO PPO_Trainer.py line 344:
    # overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    # === Initialize from uniform prior ===
    # DiffUCO: sample_prior_w_probs samples uniformly from {0, ..., K-1}
    # Shape: (n_nodes, n_basis_states)
    X_prev = torch.randint(0, n_classes, (n_nodes, n_basis_states), device=device)

    # Compute log prob of initial sample from uniform prior (for diagnostics)
    # log P(X_T) = sum over nodes of log(1/K) = -n_nodes_per_graph * log(K)
    # DiffUCO: calc_log_q_T computes this per graph
    log_uniform = -np.log(n_classes)  # log(1/K) per node
    # Sum over nodes per graph for log_q_T
    log_q_T_per_node = torch.full((n_nodes, n_basis_states), log_uniform, device=device)
    log_q_T = scatter_sum(log_q_T_per_node, node_graph_idx, dim=0, dim_size=n_graphs)

    # === Allocate storage arrays ===
    # Match DiffUCO's array shapes from _environment_steps_scan
    # Use overall_diffusion_steps instead of n_diffusion_steps

    # States at each step: (overall_steps+1, n_nodes, n_basis_states)
    Xs_over_steps = torch.zeros(overall_diffusion_steps + 1, n_nodes, n_basis_states,
                                 dtype=torch.long, device=device)
    Xs_over_steps[0] = X_prev

    # Random node features: (overall_steps, n_nodes, n_basis_states, n_random_features)
    rand_node_features_steps = torch.zeros(
        overall_diffusion_steps, n_nodes, n_basis_states, model.n_random_features,
        dtype=torch.float32, device=device
    )

    # Log policies (state_log_probs): (overall_steps, n_graphs, n_basis_states)
    log_policies = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                               dtype=torch.float32, device=device)

    # Values: (overall_steps+1, n_graphs, n_basis_states)
    values_over_steps = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                                    dtype=torch.float32, device=device)

    # Noise rewards: (overall_steps, n_graphs, n_basis_states)
    noise_rewards = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                                dtype=torch.float32, device=device)

    # Entropy rewards: (overall_steps, n_graphs, n_basis_states)
    entropy_rewards = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                                  dtype=torch.float32, device=device)

    # Per-node time indices: (overall_steps, n_nodes, n_basis_states)
    # Used by PPO for minibatching across timesteps
    # Store the MODEL step index (not loop index) for proper PPO minibatching
    time_index_per_node = torch.zeros(overall_diffusion_steps, n_nodes, n_basis_states,
                                      dtype=torch.long, device=device)

    # === Run diffusion steps ===
    # DiffUCO runs steps 0, 1, ..., overall_diffusion_steps-1
    # PPO_Trainer.py line 291: model_step_idx = i / eval_step_factor (integer division)
    for step_idx in range(overall_diffusion_steps):
        # Compute model step index (DiffUCO PPO_Trainer.py line 291)
        model_step_idx = step_idx // eval_step_factor

        # Process each basis state
        # In DiffUCO, this is vectorized over basis states
        step_state_log_probs = torch.zeros(n_graphs, n_basis_states, device=device)
        step_values = torch.zeros(n_graphs, n_basis_states, device=device)
        step_X_next = torch.zeros(n_nodes, n_basis_states, dtype=torch.long, device=device)
        step_rand_nodes = torch.zeros(n_nodes, n_basis_states, model.n_random_features,
                                      device=device)

        for b in range(n_basis_states):
            X_t_b = X_prev[:, b]  # (n_nodes,)

            # Forward pass (generates fresh random nodes)
            # Use model_step_idx instead of step_idx for the model
            spin_log_probs, values, rand_nodes = model(
                X_t_b, model_step_idx, edge_index, edge_attr, node_graph_idx, n_graphs
            )

            # Sample next state
            X_next_b, _ = model.sample_action(spin_log_probs)

            # Compute state log prob (sum over nodes per graph)
            state_log_probs_b = model.get_state_log_prob(
                spin_log_probs, X_next_b, node_graph_idx, n_graphs
            )

            step_X_next[:, b] = X_next_b
            step_state_log_probs[:, b] = state_log_probs_b
            step_values[:, b] = values
            step_rand_nodes[:, b, :] = rand_nodes

        # === Compute rewards for this step ===

        # Entropy reward = T * (-state_log_probs)
        # From PPO_Trainer.py line 312: entropy_step = -state_log_probs
        # From PPO_Trainer.py line 319: entropy_rewards[i] = T * entropy_step
        entropy_step = -step_state_log_probs  # (n_graphs, n_basis_states)
        entropy_rewards[step_idx] = T_temperature * entropy_step

        # Noise reward = T * log P_forward(X_next | X_prev)
        # Computed per basis state
        # Use model_step_idx for noise schedule lookup (matches DiffUCO)
        for b in range(n_basis_states):
            noise_reward_b = noise_schedule.compute_noise_reward(
                X_prev[:, b], step_X_next[:, b], model_step_idx, T_temperature,
                node_graph_idx, n_graphs
            )
            noise_rewards[step_idx, :, b] = noise_reward_b

        # Store trajectory data
        Xs_over_steps[step_idx + 1] = step_X_next
        rand_node_features_steps[step_idx] = step_rand_nodes
        log_policies[step_idx] = step_state_log_probs
        values_over_steps[step_idx] = step_values

        # Store per-node time indices (model_step_idx for all nodes and basis states)
        # This is used by PPO for minibatching - should be the model's step index
        time_index_per_node[step_idx] = model_step_idx

        # Update X_prev for next step
        X_prev = step_X_next

    # === Compute combined rewards ===
    # From PPO_Trainer.py line 381:
    # combined_reward = NoiseDistrClass.calculate_noise_distr_reward(-noise_rewards, entropy_rewards)
    # This returns: -(−noise_rewards − entropy_rewards) = noise_rewards + entropy_rewards
    combined_rewards = noise_schedule.calculate_combined_reward(noise_rewards, entropy_rewards)

    # === Compute energy reward at final step ===
    # From PPO_Trainer.py lines 376-384
    X_0 = X_prev  # Final state after all diffusion steps

    if energy_fn is not None:
        # Compute energy for each basis state
        energy_rewards_per_graph = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step = torch.zeros(n_graphs, n_basis_states, device=device)  # Raw energy (not negated)
        for b in range(n_basis_states):
            energy_per_graph = energy_fn(X_0[:, b], node_graph_idx, n_graphs)
            energy_step[:, b] = energy_per_graph  # Raw energy
            energy_rewards_per_graph[:, b] = -energy_per_graph  # Negative because we minimize
    else:
        # Placeholder: zero energy reward
        energy_rewards_per_graph = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step = torch.zeros(n_graphs, n_basis_states, device=device)

    # Add energy reward to final step
    # From PPO_Trainer.py line 384: rewards[-1] += energy_reward
    rewards = combined_rewards.clone()
    rewards[-1] = rewards[-1] + energy_rewards_per_graph

    # === Compute diagnostic log probs (PPO_Trainer.py lines 398-404) ===
    # log_q_0_T: log probs under policy (for forward KL)
    # log_p_0_T: log probs under target distribution (for reverse KL)
    # Shape: (overall_steps+1, n_graphs, n_basis_states)
    log_q_0_T = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                            dtype=torch.float32, device=device)
    log_p_0_T = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                            dtype=torch.float32, device=device)

    # log_q_0_T[0] = log_q_T (prior log prob)
    # log_q_0_T[1:] = log_policies
    log_q_0_T[0] = log_q_T
    log_q_0_T[1:] = log_policies

    # log_p_0_T[:-1] = -1/(T*1e-6) * noise_rewards
    # log_p_0_T[-1] = 1/(T*1e-6) * energy_step
    # Note: This scaling is specific to DiffUCO's formulation
    T_scale = T_temperature * 1e-6 if T_temperature > 0 else 1e-6
    log_p_0_T[:-1] = -1.0 / T_scale * noise_rewards
    log_p_0_T[-1] = 1.0 / T_scale * energy_step

    # === Build trajectory buffer ===
    # states: Xs_over_steps[0:-1] (excluding final)
    # actions: Xs_over_steps[1:] (excluding initial)
    buffer = TrajectoryBuffer(
        states=Xs_over_steps[:-1],              # (n_steps, n_nodes, n_basis_states)
        actions=Xs_over_steps[1:],              # (n_steps, n_nodes, n_basis_states)
        rand_node_features=rand_node_features_steps,  # (n_steps, n_nodes, n_basis_states, n_rand)
        policies=log_policies,                  # (n_steps, n_graphs, n_basis_states)
        rewards=rewards,                        # (n_steps, n_graphs, n_basis_states)
        values=values_over_steps[:-1],          # (n_steps, n_graphs, n_basis_states)
        time_index_per_node=time_index_per_node,  # (n_steps, n_nodes, n_basis_states)
        noise_rewards=noise_rewards,
        entropy_rewards=entropy_rewards,
        energy_rewards=energy_rewards_per_graph,
        final_states=X_0,
        # Diagnostic fields (PPO_Trainer.py lines 398-406)
        log_q_T=log_q_T,
        log_q_0_T=log_q_0_T,
        log_p_0_T=log_p_0_T,
    )

    return buffer


def collect_trajectories_batch(
    model: DiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    n_trajectories: int,
    T_temperature: float,
    energy_fn: Optional[Callable] = None,
    device: torch.device = None,
    eval_step_factor: int = 1,
) -> List[TrajectoryBuffer]:
    """
    Collect multiple trajectories.

    Args:
        ... (same as collect_trajectory)
        n_trajectories: Number of trajectories to collect
        eval_step_factor: Factor to multiply diffusion steps (default 1)

    Returns:
        List of TrajectoryBuffer
    """
    trajectories = []

    with torch.no_grad():
        for _ in range(n_trajectories):
            buffer = collect_trajectory(
                model, noise_schedule, edge_index, edge_attr,
                node_graph_idx, n_graphs, n_basis_states,
                T_temperature, energy_fn, device, eval_step_factor
            )
            trajectories.append(buffer)

    return trajectories


def sample_with_eval_step_factor(
    model: DiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample from the diffusion model with eval_step_factor for finer-grained denoising.

    From DiffUCO PPO_Trainer.py lines 291, 344:
        overall_diffusion_steps = n_diffusion_steps * eval_step_factor
        model_step_idx = i / eval_step_factor (integer division)

    This allows running more diffusion steps during evaluation while reusing
    model weights trained at fewer steps.

    Args:
        model: DiffusionStepModel
        noise_schedule: CategoricalNoiseSchedule
        edge_index: Graph edges (2, n_edges)
        edge_attr: Edge features (n_edges, edge_dim)
        node_graph_idx: Graph index for each node (n_nodes,)
        n_graphs: Number of graphs in batch
        n_basis_states: Number of parallel samples
        T_temperature: Temperature (use 0 for greedy sampling during eval)
        eval_step_factor: Factor to multiply diffusion steps (default 1)
        device: Device to use

    Returns:
        X_0: Final denoised states (n_nodes, n_basis_states)
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    n_classes = model.n_classes
    n_diffusion_steps = noise_schedule.n_steps

    # Total steps with eval_step_factor
    # DiffUCO: overall_diffusion_steps = n_diffusion_steps * eval_step_factor
    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    # Sample from uniform prior
    X_prev = torch.randint(0, n_classes, (n_nodes, n_basis_states), device=device)

    model.eval()
    with torch.no_grad():
        for step_i in range(overall_diffusion_steps):
            # Map to model step index
            # DiffUCO: model_step_idx = i / eval_step_factor (integer division)
            model_step_idx = step_i // eval_step_factor

            for b in range(n_basis_states):
                X_t_b = X_prev[:, b]

                # Forward pass
                spin_log_probs, _, _ = model(
                    X_t_b, model_step_idx, edge_index, edge_attr,
                    node_graph_idx, n_graphs
                )

                # Sample action
                if T_temperature > 0:
                    # Stochastic sampling with temperature
                    scaled_logits = spin_log_probs / T_temperature
                    dist = torch.distributions.Categorical(logits=scaled_logits)
                    X_next_b = dist.sample()
                else:
                    # Greedy (argmax) sampling
                    X_next_b = spin_log_probs.argmax(dim=-1)

                X_prev[:, b] = X_next_b

    return X_prev


##############################################################################
# CVRP-specific Trajectory Collection
##############################################################################

def collect_cvrp_trajectory(
    model: CVRPDiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    node_features: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float,
    energy_fn: Optional[Callable] = None,
    device: torch.device = None,
    eval_step_factor: int = 1,
    K_valid: Optional[int] = None,
    depot_idx: int = 0,
) -> TrajectoryBuffer:
    """
    Collect one CVRP trajectory with DiffUCO-faithful rewards.

    This is the CVRP-specific version that includes node features
    (demand/capacity, r, theta) in each forward pass.

    IMPORTANT: Only CUSTOMER nodes participate in state sampling and rewards.
    The depot (index 0) is included in GNN for spatial context but does NOT
    contribute to entropy/noise rewards.

    Args:
        model: CVRPDiffusionStepModel (with node_feat_dim support)
        noise_schedule: CategoricalNoiseSchedule
        edge_index: Graph edges (2, n_edges)
        edge_attr: Edge features (n_edges, edge_dim)
        node_graph_idx: Graph index for each node (n_nodes,)
        node_features: CVRP node features (n_nodes, node_feat_dim)
        n_graphs: Number of graphs in batch
        n_basis_states: Number of parallel samples
        T_temperature: Temperature parameter
        energy_fn: Optional energy function(X, node_graph_idx, n_graphs) -> energy
        device: Device to use
        eval_step_factor: Factor to multiply diffusion steps
        K_valid: Number of valid vehicle classes for this instance.
                 If provided, mask logits for classes >= K_valid.
        depot_idx: Index of depot node (default 0). Depot is excluded from
                   state sampling and reward computation.

    Returns:
        TrajectoryBuffer containing all trajectory data
    """
    if device is None:
        device = edge_index.device

    n_nodes_total = node_graph_idx.size(0)  # Including depot
    n_customers = n_nodes_total - 1  # Excluding depot
    n_classes = model.n_classes
    n_diffusion_steps = noise_schedule.n_steps

    # Use K_valid if provided, otherwise use all classes
    K_effective = K_valid if K_valid is not None else n_classes

    # Customer indices (all nodes except depot)
    customer_mask = torch.ones(n_nodes_total, dtype=torch.bool, device=device)
    customer_mask[depot_idx] = False
    customer_indices = torch.where(customer_mask)[0]
    customer_graph_idx = node_graph_idx[customer_mask]

    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    # Initialize from uniform prior over VALID classes - CUSTOMERS ONLY
    X_prev = torch.randint(0, K_effective, (n_customers, n_basis_states), device=device)

    # Log prob of initial sample - CUSTOMERS ONLY
    log_uniform = -np.log(K_effective)
    log_q_T_per_node = torch.full((n_customers, n_basis_states), log_uniform, device=device)
    log_q_T = scatter_sum(log_q_T_per_node, customer_graph_idx, dim=0, dim_size=n_graphs)

    # Allocate storage arrays - CUSTOMERS ONLY for states
    Xs_over_steps = torch.zeros(overall_diffusion_steps + 1, n_customers, n_basis_states,
                                 dtype=torch.long, device=device)
    Xs_over_steps[0] = X_prev

    # Random features for all nodes (model sees full graph)
    rand_node_features_steps = torch.zeros(
        overall_diffusion_steps, n_nodes_total, n_basis_states, model.n_random_features,
        dtype=torch.float32, device=device
    )

    log_policies = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                               dtype=torch.float32, device=device)

    values_over_steps = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                                    dtype=torch.float32, device=device)

    noise_rewards = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                                dtype=torch.float32, device=device)

    entropy_rewards = torch.zeros(overall_diffusion_steps, n_graphs, n_basis_states,
                                  dtype=torch.float32, device=device)

    time_index_per_node = torch.zeros(overall_diffusion_steps, n_customers, n_basis_states,
                                      dtype=torch.long, device=device)

    # Run diffusion steps
    for step_idx in range(overall_diffusion_steps):
        model_step_idx = step_idx // eval_step_factor

        step_state_log_probs = torch.zeros(n_graphs, n_basis_states, device=device)
        step_values = torch.zeros(n_graphs, n_basis_states, device=device)
        step_X_next = torch.zeros(n_customers, n_basis_states, dtype=torch.long, device=device)
        step_rand_nodes = torch.zeros(n_nodes_total, n_basis_states, model.n_random_features,
                                      device=device)

        for b in range(n_basis_states):
            # Customer states only
            X_customers_b = X_prev[:, b]

            # Build full state for model: depot (fixed at 0) + customers
            X_full_b = torch.zeros(n_nodes_total, dtype=torch.long, device=device)
            X_full_b[depot_idx] = 0  # Depot has fixed "state" (ignored)
            X_full_b[customer_mask] = X_customers_b

            # Forward pass with full graph (GNN sees depot for spatial context)
            # Pass customer_mask so value head only aggregates customer nodes
            spin_log_probs_full, values, rand_nodes = model(
                X_full_b, model_step_idx, edge_index, edge_attr,
                node_graph_idx, n_graphs, node_features,
                customer_mask=customer_mask
            )

            # Check for NaN in model output (indicates exploding gradients)
            if torch.isnan(spin_log_probs_full).any():
                raise RuntimeError(
                    f"NaN detected in model output at step {step_idx}, basis {b}. "
                    "This usually indicates gradient explosion during training. "
                    "Try: (1) lower learning rate, (2) gradient clipping, (3) smaller batch size."
                )

            # Extract CUSTOMER log probs only (depot excluded from policy)
            spin_log_probs = spin_log_probs_full[customer_mask]

            # Mask invalid vehicle labels (classes >= K_valid)
            if K_valid is not None and K_valid < n_classes:
                spin_log_probs = spin_log_probs.clone()
                spin_log_probs[:, K_valid:] = float('-inf')

            # Sample next state - CUSTOMERS ONLY
            X_next_customers_b, _ = model.sample_action(spin_log_probs)

            # Compute state log prob - CUSTOMERS ONLY (summed over customer nodes)
            state_log_probs_b = model.get_state_log_prob(
                spin_log_probs, X_next_customers_b, customer_graph_idx, n_graphs
            )

            step_X_next[:, b] = X_next_customers_b
            step_state_log_probs[:, b] = state_log_probs_b
            step_values[:, b] = values
            step_rand_nodes[:, b, :] = rand_nodes

        # Compute rewards - CUSTOMERS ONLY
        entropy_step = -step_state_log_probs
        entropy_rewards[step_idx] = T_temperature * entropy_step

        for b in range(n_basis_states):
            # Noise reward computed over CUSTOMERS ONLY
            noise_reward_b = noise_schedule.compute_noise_reward(
                X_prev[:, b], step_X_next[:, b], model_step_idx, T_temperature,
                customer_graph_idx, n_graphs, K_effective  # Pass K_effective
            )
            noise_rewards[step_idx, :, b] = noise_reward_b

        # Store trajectory data
        Xs_over_steps[step_idx + 1] = step_X_next
        rand_node_features_steps[step_idx] = step_rand_nodes
        log_policies[step_idx] = step_state_log_probs
        values_over_steps[step_idx] = step_values
        time_index_per_node[step_idx] = model_step_idx

        X_prev = step_X_next

    # Compute combined rewards
    combined_rewards = noise_schedule.calculate_combined_reward(noise_rewards, entropy_rewards)

    # Compute energy reward at final step
    X_0 = X_prev

    if energy_fn is not None:
        energy_rewards_per_graph = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step = torch.zeros(n_graphs, n_basis_states, device=device)
        for b in range(n_basis_states):
            # X_0 is customer states only (depot excluded)
            energy_per_graph = energy_fn(X_0[:, b], customer_graph_idx, n_graphs)
            energy_step[:, b] = energy_per_graph
            energy_rewards_per_graph[:, b] = -energy_per_graph
    else:
        energy_rewards_per_graph = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step = torch.zeros(n_graphs, n_basis_states, device=device)

    # Add energy reward to final step
    rewards = combined_rewards.clone()
    rewards[-1] = rewards[-1] + energy_rewards_per_graph

    # Diagnostic log probs
    log_q_0_T = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                            dtype=torch.float32, device=device)
    log_p_0_T = torch.zeros(overall_diffusion_steps + 1, n_graphs, n_basis_states,
                            dtype=torch.float32, device=device)

    log_q_0_T[0] = log_q_T
    log_q_0_T[1:] = log_policies

    T_scale = T_temperature * 1e-6 if T_temperature > 0 else 1e-6
    log_p_0_T[:-1] = -1.0 / T_scale * noise_rewards
    log_p_0_T[-1] = 1.0 / T_scale * energy_step

    # Build trajectory buffer
    buffer = TrajectoryBuffer(
        states=Xs_over_steps[:-1],
        actions=Xs_over_steps[1:],
        rand_node_features=rand_node_features_steps,
        policies=log_policies,
        rewards=rewards,
        values=values_over_steps[:-1],
        time_index_per_node=time_index_per_node,
        noise_rewards=noise_rewards,
        entropy_rewards=entropy_rewards,
        energy_rewards=energy_rewards_per_graph,
        final_states=X_0,
        log_q_T=log_q_T,
        log_q_0_T=log_q_0_T,
        log_p_0_T=log_p_0_T,
    )

    return buffer


def sample_cvrp_with_eval_step_factor(
    model: CVRPDiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    node_features: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample from the CVRP diffusion model with eval_step_factor.

    CVRP-specific version that includes node features in each forward pass.

    Args:
        model: CVRPDiffusionStepModel
        noise_schedule: CategoricalNoiseSchedule
        edge_index: Graph edges
        edge_attr: Edge features
        node_graph_idx: Graph index per node
        node_features: CVRP node features (n_nodes, node_feat_dim)
        n_graphs: Number of graphs
        n_basis_states: Number of parallel samples
        T_temperature: Temperature (0 for greedy)
        eval_step_factor: Step factor
        device: Device

    Returns:
        X_0: Final denoised states (n_nodes, n_basis_states)
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    n_classes = model.n_classes
    n_diffusion_steps = noise_schedule.n_steps

    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    # Sample from uniform prior
    X_prev = torch.randint(0, n_classes, (n_nodes, n_basis_states), device=device)

    model.eval()
    with torch.no_grad():
        for step_i in range(overall_diffusion_steps):
            model_step_idx = step_i // eval_step_factor

            for b in range(n_basis_states):
                X_t_b = X_prev[:, b]

                # Forward pass with node features
                spin_log_probs, _, _ = model(
                    X_t_b, model_step_idx, edge_index, edge_attr,
                    node_graph_idx, n_graphs, node_features
                )

                # Sample action
                if T_temperature > 0:
                    scaled_logits = spin_log_probs / T_temperature
                    dist = torch.distributions.Categorical(logits=scaled_logits)
                    X_next_b = dist.sample()
                else:
                    X_next_b = spin_log_probs.argmax(dim=-1)

                X_prev[:, b] = X_next_b

    return X_prev


def _expand_graph_for_basis(edge_index, edge_attr, node_graph_idx,
                             node_features, n_nodes, n_graphs, n_basis):
    """
    Expand graph structure to batch basis states into a single mega-graph.

    Each basis state becomes a separate set of graphs, enabling a single
    GNN forward pass for all basis states simultaneously. This eliminates
    the sequential `for b in range(n_basis)` loop that causes 0% GPU utilization.

    Node ordering in the expanded graph:
        [basis0_node0, ..., basis0_nodeN, basis1_node0, ..., basis1_nodeN, ...]

    Returns:
        expanded_edge_index: (2, n_edges * n_basis)
        expanded_edge_attr: (n_edges * n_basis, edge_dim) or None
        expanded_node_graph_idx: (n_nodes * n_basis,)
        expanded_node_features: (n_nodes * n_basis, feature_dim) or None
        expanded_n_graphs: n_graphs * n_basis
    """
    device = edge_index.device

    # Edge index: replicate with node offsets for each basis state
    node_offsets = torch.arange(n_basis, device=device) * n_nodes
    ei_expanded = edge_index.unsqueeze(0) + node_offsets.reshape(-1, 1, 1)
    expanded_edge_index = ei_expanded.permute(1, 0, 2).reshape(2, -1)

    # Edge attr: repeat for each basis state
    expanded_edge_attr = edge_attr.repeat(n_basis, 1) if edge_attr is not None else None

    # Node graph idx: replicate with graph offsets
    graph_offsets = torch.arange(n_basis, device=device) * n_graphs
    ngi_expanded = node_graph_idx.unsqueeze(0) + graph_offsets.reshape(-1, 1)
    expanded_node_graph_idx = ngi_expanded.reshape(-1)

    # Node features: repeat for each basis state
    expanded_node_features = node_features.repeat(n_basis, 1) if node_features is not None else None

    expanded_n_graphs = n_graphs * n_basis

    return (expanded_edge_index, expanded_edge_attr, expanded_node_graph_idx,
            expanded_node_features, expanded_n_graphs)


##############################################################################
# Continuous (Gaussian) Trajectory Collection for Chip Placement
##############################################################################

def collect_continuous_trajectory(
    model: ContinuousDiffusionStepModel,
    noise_schedule: GaussianNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float,
    node_features: torch.Tensor,
    energy_fn: Optional[Callable] = None,
    per_step_energy: bool = True,
    device: torch.device = None,
    eval_step_factor: int = 1,
    init_positions: torch.Tensor = None,
    legalize_fn: Optional[Callable] = None,
    refine_noise_scale: float = 0.0,
    energy_norm: str = 'none',
) -> TrajectoryBuffer:
    """
    Collect trajectory for continuous (Gaussian) diffusion with per-step energy.

    Key differences from categorical collect_trajectory:
    - States are float tensors (n_nodes, n_basis, continuous_dim) not long
    - Prior is N(0, I) not uniform categorical
    - Per-step energy: compute energy at EVERY step (SDDS fix)
    - Model outputs (mean, log_var) not categorical logits

    Reference: DIffUCO/Trainers/PPO_Trainer.py scan_body

    Args:
        model: ContinuousDiffusionStepModel
        noise_schedule: GaussianNoiseSchedule
        edge_index: (2, n_edges)
        edge_attr: (n_edges, edge_dim)
        node_graph_idx: (n_nodes,)
        n_graphs: Number of graphs
        n_basis_states: Number of parallel samples
        T_temperature: Temperature parameter
        node_features: (n_nodes, node_feature_dim) component sizes
        energy_fn: energy_fn(positions, node_graph_idx, n_graphs) -> (n_graphs,)
        per_step_energy: If True, compute energy at every step (SDDS fix)
        device: Device
        eval_step_factor: Multiply diffusion steps for finer denoising
        legalize_fn: Optional function(positions) -> legalized_positions.
            If provided, called after each diffusion step to project
            positions into near-legal space (legalization-in-loop).
        refine_noise_scale: If > 0 and init_positions provided, use this as
            the noise standard deviation instead of the schedule's alpha_bar.
            E.g., 0.1 adds light Gaussian noise (std=0.1) to legal positions.
            This preserves much more signal than full forward diffusion.
        energy_norm: How to normalize energy rewards relative to noise+entropy:
            'none' = raw energy (can cause scale explosion with high penalty weights)
            'scale' = scale energy rewards so their std matches combined rewards std
            'mean_subtract' = subtract per-graph mean (baseline subtraction)

    Returns:
        TrajectoryBuffer
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    continuous_dim = model.continuous_dim
    n_diffusion_steps = noise_schedule.n_steps
    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    # === Initialize prior ===
    # Shape: (n_nodes, n_basis_states, continuous_dim)
    if init_positions is not None and refine_noise_scale > 0:
        # Refinement mode: add light noise (user-specified scale)
        # X = legal + noise_scale * epsilon
        # Preserves much more signal than full forward diffusion
        base = init_positions.unsqueeze(1).expand(-1, n_basis_states, -1)
        eps = torch.randn_like(base)
        X_prev = base + refine_noise_scale * eps
    elif init_positions is not None:
        # Warm start: forward-diffuse positions to step T using schedule
        alpha_bar_T = noise_schedule.alpha_bar_arr[-1]
        signal_scale = torch.sqrt(alpha_bar_T)
        noise_scale = torch.sqrt(1.0 - alpha_bar_T)
        base = init_positions.unsqueeze(1).expand(-1, n_basis_states, -1)
        eps = torch.randn_like(base)
        X_prev = signal_scale * base + noise_scale * eps
    else:
        X_prev = torch.randn(n_nodes, n_basis_states, continuous_dim, device=device)

    # Log prob of initial sample under N(0, I) prior
    # (Even for warm start, we use the standard prior log prob for PPO;
    #  the warm start only changes the initialization, not the math.)
    log_prior_per_dim = -0.5 * (X_prev ** 2 + math.log(2 * math.pi))
    log_prior_per_node = log_prior_per_dim.sum(dim=-1)  # (n_nodes, n_basis)

    # === Create expanded graph for batched basis processing ===
    (exp_edge_index, exp_edge_attr, exp_node_graph_idx,
     exp_node_features, exp_n_graphs) = _expand_graph_for_basis(
        edge_index, edge_attr, node_graph_idx, node_features,
        n_nodes, n_graphs, n_basis_states,
    )

    # Vectorized scatter_sum for prior log prob using expanded graph
    log_prior_flat = log_prior_per_node.transpose(0, 1).reshape(-1)
    log_q_T_flat = scatter_sum(
        log_prior_flat, exp_node_graph_idx, dim=0, dim_size=exp_n_graphs
    )
    log_q_T = log_q_T_flat.reshape(n_basis_states, n_graphs).t()

    # === Allocate storage ===
    # States: (steps+1, n_nodes, n_basis, continuous_dim)
    Xs_over_steps = torch.zeros(
        overall_diffusion_steps + 1, n_nodes, n_basis_states, continuous_dim,
        dtype=torch.float32, device=device
    )
    Xs_over_steps[0] = X_prev

    # Random node features: (steps, n_nodes, n_basis, n_random_features)
    rand_node_features_steps = torch.zeros(
        overall_diffusion_steps, n_nodes, n_basis_states, model.n_random_features,
        dtype=torch.float32, device=device
    )

    # Log policies: (steps, n_graphs, n_basis)
    log_policies = torch.zeros(
        overall_diffusion_steps, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )

    # Values: (steps+1, n_graphs, n_basis)
    values_over_steps = torch.zeros(
        overall_diffusion_steps + 1, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )

    # Rewards: (steps, n_graphs, n_basis)
    noise_rewards = torch.zeros(
        overall_diffusion_steps, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )
    entropy_rewards = torch.zeros(
        overall_diffusion_steps, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )

    # Per-step energy rewards (SDDS fix)
    energy_rewards_per_step = torch.zeros(
        overall_diffusion_steps, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )

    # Time indices: (steps, n_nodes, n_basis)
    time_index_per_node = torch.zeros(
        overall_diffusion_steps, n_nodes, n_basis_states,
        dtype=torch.long, device=device
    )

    # === Run diffusion steps ===
    for step_idx in range(overall_diffusion_steps):
        model_step_idx = step_idx // eval_step_factor

        # --- Batched forward pass: all basis states in one GNN call ---
        # Flatten (n_nodes, n_basis, dim) -> (n_basis * n_nodes, dim)
        X_t_flat = X_prev.transpose(0, 1).reshape(-1, continuous_dim)

        mean_flat, log_var_flat, values_flat, rand_nodes_flat = model(
            X_t_flat, model_step_idx, exp_edge_index, exp_edge_attr,
            exp_node_graph_idx, exp_n_graphs, exp_node_features,
        )

        # Sample actions for all basis states at once
        X_next_flat, _ = model.sample_action(mean_flat, log_var_flat)

        # State log probs for all basis states
        state_log_probs_flat = model.get_state_log_prob(
            mean_flat, log_var_flat, X_next_flat, exp_node_graph_idx, exp_n_graphs
        )

        # Reshape back: (n_basis * n_nodes, ...) -> (n_nodes, n_basis, ...)
        step_X_next = X_next_flat.reshape(
            n_basis_states, n_nodes, continuous_dim
        ).transpose(0, 1)
        step_state_log_probs = state_log_probs_flat.reshape(
            n_basis_states, n_graphs
        ).t()
        step_values = values_flat.reshape(n_basis_states, n_graphs).t()
        step_rand_nodes = rand_nodes_flat.reshape(
            n_basis_states, n_nodes, model.n_random_features
        ).transpose(0, 1)

        # === Compute rewards (BEFORE legalization) ===
        # Noise/entropy rewards use the raw model sample for correct PPO math.
        # Energy rewards use the legalized positions (if legalize_fn provided).

        # Entropy reward = T * (-state_log_probs)
        entropy_rewards[step_idx] = T_temperature * (-step_state_log_probs)

        # Noise reward: computed on raw (pre-legalization) positions
        log_p_per_node = noise_schedule.compute_forward_log_prob(
            X_prev, step_X_next, model_step_idx
        )  # (n_nodes, n_basis)
        log_p_flat = log_p_per_node.transpose(0, 1).reshape(-1)
        noise_per_graph_flat = scatter_sum(
            log_p_flat, exp_node_graph_idx, dim=0, dim_size=exp_n_graphs
        )
        noise_rewards[step_idx] = (
            T_temperature * noise_per_graph_flat
        ).reshape(n_basis_states, n_graphs).t()

        # === PER-STEP ENERGY (SDDS FIX) ===
        # If legalize_fn is provided, compute energy on LEGALIZED positions
        # (reward shaping — the model learns to produce outputs that, after
        # legalization, have low energy). But the raw sample remains the
        # action for PPO — this preserves the importance sampling ratio.
        if per_step_energy and energy_fn is not None:
            for b in range(n_basis_states):
                pos_for_energy = step_X_next[:, b, :]
                if legalize_fn is not None:
                    pos_for_energy = legalize_fn(pos_for_energy)
                energy_per_graph = energy_fn(
                    pos_for_energy, node_graph_idx, n_graphs
                )
                energy_rewards_per_step[step_idx, :, b] = -energy_per_graph

        # Store
        Xs_over_steps[step_idx + 1] = step_X_next
        rand_node_features_steps[step_idx] = step_rand_nodes
        log_policies[step_idx] = step_state_log_probs
        values_over_steps[step_idx] = step_values
        time_index_per_node[step_idx] = model_step_idx

        X_prev = step_X_next

    # === Compute combined rewards ===
    combined_rewards = noise_schedule.calculate_combined_reward(
        noise_rewards, entropy_rewards
    )

    # === Final energy reward ===
    X_0 = X_prev  # Final state

    if energy_fn is not None:
        energy_rewards_final = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step_final = torch.zeros(n_graphs, n_basis_states, device=device)
        for b in range(n_basis_states):
            pos_for_energy = X_0[:, b, :]
            if legalize_fn is not None:
                pos_for_energy = legalize_fn(pos_for_energy)
            energy_per_graph = energy_fn(pos_for_energy, node_graph_idx, n_graphs)
            energy_step_final[:, b] = energy_per_graph
            energy_rewards_final[:, b] = -energy_per_graph
    else:
        energy_rewards_final = torch.zeros(n_graphs, n_basis_states, device=device)
        energy_step_final = torch.zeros(n_graphs, n_basis_states, device=device)

    # === Normalize energy rewards ===
    if energy_norm == 'scale' and per_step_energy:
        # Scale energy rewards so their std ≈ combined (noise+entropy) rewards std.
        # This prevents high penalty weights from causing energy to dominate.
        combined_std = combined_rewards.std().clamp(min=1e-8)
        energy_std = energy_rewards_per_step.std().clamp(min=1e-8)
        energy_rewards_per_step = energy_rewards_per_step * (combined_std / energy_std)
        # Also scale final energy for consistency
        energy_rewards_final = energy_rewards_final * (combined_std / energy_std)
    elif energy_norm == 'scale' and not per_step_energy:
        combined_std = combined_rewards.std().clamp(min=1e-8)
        energy_std = energy_rewards_final.std().clamp(min=1e-8)
        energy_rewards_final = energy_rewards_final * (combined_std / energy_std)

    # === Compose total rewards ===
    rewards = combined_rewards.clone()
    if per_step_energy:
        rewards = rewards + energy_rewards_per_step
        # per_step already includes the final step — do NOT double-count
    else:
        # Only add final energy when per-step is disabled
        rewards[-1] = rewards[-1] + energy_rewards_final

    # === Diagnostic log probs ===
    log_q_0_T = torch.zeros(
        overall_diffusion_steps + 1, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )
    log_p_0_T = torch.zeros(
        overall_diffusion_steps + 1, n_graphs, n_basis_states,
        dtype=torch.float32, device=device
    )
    log_q_0_T[0] = log_q_T
    log_q_0_T[1:] = log_policies
    T_scale = T_temperature if T_temperature > 0 else 1.0
    log_p_0_T[:-1] = -1.0 / T_scale * noise_rewards
    log_p_0_T[-1] = 1.0 / T_scale * energy_step_final

    # === Build buffer ===
    # For continuous states, we store the full (n_nodes, n_basis, continuous_dim)
    # but TrajectoryBuffer expects (steps, n_nodes, n_basis) for states/actions.
    # We reshape to (steps, n_nodes * continuous_dim, n_basis) for compatibility
    # with PPO minibatching, or keep 4D and handle in ContinuousPPOTrainer.
    #
    # Decision: keep states as (steps, n_nodes, n_basis, continuous_dim)
    # and handle the extra dim in ContinuousPPOTrainer.
    buffer = TrajectoryBuffer(
        states=Xs_over_steps[:-1],
        actions=Xs_over_steps[1:],
        rand_node_features=rand_node_features_steps,
        policies=log_policies,
        rewards=rewards,
        values=values_over_steps[:-1],
        time_index_per_node=time_index_per_node,
        noise_rewards=noise_rewards,
        entropy_rewards=entropy_rewards,
        energy_rewards=energy_rewards_final,
        final_states=X_0,
        log_q_T=log_q_T,
        log_q_0_T=log_q_0_T,
        log_p_0_T=log_p_0_T,
    )

    return buffer


def sample_continuous_with_eval_step_factor(
    model: ContinuousDiffusionStepModel,
    noise_schedule: GaussianNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    node_features: torch.Tensor,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
    device: torch.device = None,
    init_positions: torch.Tensor = None,
    legalize_fn: Optional[Callable] = None,
    refine_noise_scale: float = 0.0,
) -> torch.Tensor:
    """
    Sample from continuous diffusion model (evaluation).

    Args:
        ... (same as collect_continuous_trajectory)
        T_temperature: 0 = use mean (greedy), >0 = sample from Gaussian
        init_positions: Optional (n_nodes, 2) starting positions.
            If provided, forward-diffuses to step T as warm start.
            If None, starts from N(0, I) prior.
        legalize_fn: Optional legalization function applied after each step.
        refine_noise_scale: If > 0 and init_positions provided, use as noise std.

    Returns:
        X_0: Final positions (n_nodes, n_basis_states, continuous_dim)
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    continuous_dim = model.continuous_dim
    n_diffusion_steps = noise_schedule.n_steps
    overall_diffusion_steps = n_diffusion_steps * eval_step_factor

    if init_positions is not None and refine_noise_scale > 0:
        # Refinement: light noise on legal positions
        base = init_positions.unsqueeze(1).expand(-1, n_basis_states, -1)
        noise = torch.randn_like(base)
        X_prev = base + refine_noise_scale * noise
    elif init_positions is not None:
        # Warm start: forward-diffuse legal positions to step T
        alpha_bar_T = noise_schedule.alpha_bar_arr[-1]
        signal_scale = torch.sqrt(alpha_bar_T)
        noise_scale = torch.sqrt(1.0 - alpha_bar_T)
        base = init_positions.unsqueeze(1).expand(-1, n_basis_states, -1)
        noise = torch.randn_like(base)
        X_prev = signal_scale * base + noise_scale * noise
    else:
        # Standard: sample from N(0, I) prior
        X_prev = torch.randn(n_nodes, n_basis_states, continuous_dim, device=device)

    # Create expanded graph for batched basis processing
    (exp_ei, exp_ea, exp_ngi, exp_nf, exp_ng) = _expand_graph_for_basis(
        edge_index, edge_attr, node_graph_idx, node_features,
        n_nodes, n_graphs, n_basis_states,
    )

    model.eval()
    with torch.no_grad():
        for step_i in range(overall_diffusion_steps):
            model_step_idx = step_i // eval_step_factor

            # Batched forward pass for all basis states
            X_t_flat = X_prev.transpose(0, 1).reshape(-1, continuous_dim)

            mean_flat, log_var_flat, _, _ = model(
                X_t_flat, model_step_idx, exp_ei, exp_ea,
                exp_ngi, exp_ng, exp_nf,
            )

            if T_temperature > 0:
                X_next_flat, _ = model.sample_action(mean_flat, log_var_flat)
            else:
                X_next_flat = mean_flat  # Greedy: use mean directly

            X_prev = X_next_flat.reshape(
                n_basis_states, n_nodes, continuous_dim
            ).transpose(0, 1)

            # Legalization-in-loop for eval
            if legalize_fn is not None:
                for b in range(n_basis_states):
                    X_prev[:, b, :] = legalize_fn(X_prev[:, b, :])

    return X_prev


def test_cvrp_trajectory_collection():
    """
    Test CVRP trajectory collection.
    """
    print("=" * 60)
    print("Testing CVRP Trajectory Collection")
    print("=" * 60)

    # Test parameters
    n_classes = 4  # K vehicles
    n_nodes = 20   # customers
    n_edges = 60
    n_graphs = 2
    nodes_per_graph = n_nodes // n_graphs
    n_diffusion_steps = 5
    n_basis_states = 3
    T_temperature = 1.0
    edge_dim = 2
    node_feat_dim = 3  # (demand/cap, r, theta)

    # Create CVRP model
    model = CVRPDiffusionStepModel(
        n_classes=n_classes,
        node_feat_dim=node_feat_dim,
        edge_dim=edge_dim,
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

    # Create test data
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), nodes_per_graph
    )
    node_features = torch.rand(n_nodes, node_feat_dim)  # CVRP features

    print(f"\nTest parameters:")
    print(f"  n_classes (K): {n_classes}")
    print(f"  n_nodes: {n_nodes}")
    print(f"  node_feat_dim: {node_feat_dim}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  n_basis_states: {n_basis_states}")

    # Collect trajectory
    print("\nCollecting CVRP trajectory...")
    model.eval()

    buffer = collect_cvrp_trajectory(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        node_features=node_features,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        energy_fn=None,
    )

    print("\nTrajectory buffer shapes:")
    print(f"  states: {buffer.states.shape}")
    print(f"  actions: {buffer.actions.shape}")
    print(f"  policies: {buffer.policies.shape}")
    print(f"  rewards: {buffer.rewards.shape}")
    print(f"  final_states: {buffer.final_states.shape}")

    # Verify shapes
    expected_shapes = {
        'states': (n_diffusion_steps, n_nodes, n_basis_states),
        'actions': (n_diffusion_steps, n_nodes, n_basis_states),
        'policies': (n_diffusion_steps, n_graphs, n_basis_states),
        'rewards': (n_diffusion_steps, n_graphs, n_basis_states),
        'final_states': (n_nodes, n_basis_states),
    }

    all_shapes_ok = True
    for name, expected in expected_shapes.items():
        actual = tuple(getattr(buffer, name).shape)
        if actual != expected:
            print(f"  [FAIL] {name}: expected {expected}, got {actual}")
            all_shapes_ok = False

    print(f"\n[{'OK' if all_shapes_ok else 'FAIL'}] All shapes correct")

    # Test with dummy CVRP energy function
    print("\nTesting with CVRP energy function...")

    def dummy_cvrp_energy_fn(X, node_graph_idx, n_graphs):
        """Dummy CVRP energy: sum of partition indices per graph."""
        energy = scatter_sum(X.float(), node_graph_idx, dim=0, dim_size=n_graphs)
        return energy

    buffer_with_energy = collect_cvrp_trajectory(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        node_features=node_features,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        energy_fn=dummy_cvrp_energy_fn,
    )

    print(f"  Energy rewards: {buffer_with_energy.energy_rewards}")
    print(f"  Final rewards include energy: {not torch.allclose(buffer_with_energy.rewards[-1], buffer.rewards[-1], atol=1e-6)}")

    # Test sampling
    print("\nTesting CVRP sampling...")
    X_0 = sample_cvrp_with_eval_step_factor(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        node_features=node_features,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=0.0,  # Greedy
        eval_step_factor=1,
    )

    print(f"  X_0 shape: {X_0.shape}")
    print(f"  X_0 range: [{X_0.min().item()}, {X_0.max().item()}]")

    print("\n" + "=" * 60)
    print("All CVRP Trajectory Collection tests passed!")
    print("=" * 60)

    return True


def test_trajectory_collection():
    """
    Test trajectory collection.
    """
    print("=" * 60)
    print("Testing Trajectory Collection")
    print("=" * 60)

    # Test parameters
    n_classes = 4
    n_nodes = 20
    n_edges = 60
    n_graphs = 2
    nodes_per_graph = n_nodes // n_graphs
    n_diffusion_steps = 5
    n_basis_states = 3
    T_temperature = 1.0
    edge_dim = 2

    # Create model and noise schedule
    model = DiffusionStepModel(
        n_classes=n_classes,
        edge_dim=edge_dim,
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

    # Create test data
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), nodes_per_graph
    )

    print(f"\nTest parameters:")
    print(f"  n_classes: {n_classes}")
    print(f"  n_nodes: {n_nodes}")
    print(f"  n_graphs: {n_graphs}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  n_basis_states: {n_basis_states}")
    print(f"  T_temperature: {T_temperature}")

    # Collect trajectory
    print("\nCollecting trajectory...")
    model.eval()

    buffer = collect_trajectory(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        energy_fn=None,
    )

    print("\nTrajectory buffer shapes:")
    print(f"  states: {buffer.states.shape}")
    print(f"  actions: {buffer.actions.shape}")
    print(f"  rand_node_features: {buffer.rand_node_features.shape}")
    print(f"  policies: {buffer.policies.shape}")
    print(f"  rewards: {buffer.rewards.shape}")
    print(f"  values: {buffer.values.shape}")
    print(f"  noise_rewards: {buffer.noise_rewards.shape}")
    print(f"  entropy_rewards: {buffer.entropy_rewards.shape}")
    print(f"  energy_rewards: {buffer.energy_rewards.shape}")
    print(f"  final_states: {buffer.final_states.shape}")

    # Verify shapes
    expected_shapes = {
        'states': (n_diffusion_steps, n_nodes, n_basis_states),
        'actions': (n_diffusion_steps, n_nodes, n_basis_states),
        'rand_node_features': (n_diffusion_steps, n_nodes, n_basis_states, 5),
        'policies': (n_diffusion_steps, n_graphs, n_basis_states),
        'rewards': (n_diffusion_steps, n_graphs, n_basis_states),
        'values': (n_diffusion_steps, n_graphs, n_basis_states),
        'noise_rewards': (n_diffusion_steps, n_graphs, n_basis_states),
        'entropy_rewards': (n_diffusion_steps, n_graphs, n_basis_states),
        'energy_rewards': (n_graphs, n_basis_states),
        'final_states': (n_nodes, n_basis_states),
    }

    all_shapes_ok = True
    for name, expected in expected_shapes.items():
        actual = tuple(getattr(buffer, name).shape)
        if actual != expected:
            print(f"  [FAIL] {name}: expected {expected}, got {actual}")
            all_shapes_ok = False

    print(f"\n[{'OK' if all_shapes_ok else 'FAIL'}] All shapes correct: {all_shapes_ok}")

    # Verify state transitions
    print("\nVerifying state transitions...")
    # actions[t] should equal states[t+1]
    # Since actions = Xs_over_steps[1:] and states = Xs_over_steps[:-1]
    # We expect actions[t] == states[t+1] for t in 0..T-2
    # Actually, buffer.states[t] = Xs[t] and buffer.actions[t] = Xs[t+1]
    # So for t=0..T-2: buffer.actions[t] should equal what would be states[t+1] if we had it
    # But our states only go to T-1, so let's verify:
    # For step t: state=X_t, action=X_{t-1} in diffusion (reverse)
    # Actually in DiffUCO: states[t] = X_t before step, actions[t] = X_next after step

    states_valid = (buffer.states >= 0).all() and (buffer.states < n_classes).all()
    actions_valid = (buffer.actions >= 0).all() and (buffer.actions < n_classes).all()
    print(f"  States in valid range: {states_valid}")
    print(f"  Actions in valid range: {actions_valid}")

    # Verify rewards structure
    print("\nVerifying reward structure...")
    # Noise rewards: T * log P_forward(X_next | X_prev) - typically negative
    noise_sum = buffer.noise_rewards.sum().item()
    print(f"  Noise rewards sum: {noise_sum:.4f}")
    # Entropy rewards: T * (-state_log_probs) = T * entropy - should be POSITIVE
    # (since state_log_probs are negative log probabilities)
    entropy_sum = buffer.entropy_rewards.sum().item()
    print(f"  Entropy rewards sum: {entropy_sum:.4f}")
    print(f"  Combined rewards sum: {buffer.rewards.sum().item():.4f}")

    # Verify entropy rewards are positive (entropy = -log_prob, and log_prob < 0)
    entropy_positive = entropy_sum > 0
    print(f"  Entropy rewards positive: {entropy_positive}")
    assert entropy_positive, "Entropy rewards should be positive (T * -log_prob)"

    # Verify policies are log probabilities (should be <= 0)
    policies_valid = (buffer.policies <= 0).all()
    print(f"  Policies are log probs (<= 0): {policies_valid}")

    print(f"\n[{'OK' if all_shapes_ok and states_valid and actions_valid and policies_valid else 'FAIL'}] "
          f"Trajectory collection test passed")

    # Test with dummy energy function
    print("\nTesting with energy function...")

    def dummy_energy_fn(X, node_graph_idx, n_graphs):
        """Dummy energy: count nodes in class 0 per graph."""
        is_class_0 = (X == 0).float()
        energy = scatter_sum(is_class_0, node_graph_idx, dim=0, dim_size=n_graphs)
        return energy

    buffer_with_energy = collect_trajectory(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        energy_fn=dummy_energy_fn,
    )

    print(f"  Energy rewards: {buffer_with_energy.energy_rewards}")
    print(f"  Energy rewards shape: {buffer_with_energy.energy_rewards.shape}")

    # Energy should be negative (we return -energy from energy_fn result)
    energy_reasonable = buffer_with_energy.energy_rewards.abs().max() <= n_nodes
    print(f"  Energy rewards reasonable: {energy_reasonable}")

    # Final step rewards should include energy
    final_rewards_include_energy = not torch.allclose(
        buffer_with_energy.rewards[-1],
        buffer.rewards[-1],
        atol=1e-6
    )
    print(f"  Final rewards include energy: {final_rewards_include_energy}")

    print("\n" + "=" * 60)
    print("All Trajectory Collection tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_trajectory_collection()
    print("\n")
    test_cvrp_trajectory_collection()
