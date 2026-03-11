"""
SDDS Trajectory Collection — No Legalization

Modified from archive/original_sdds/trajectory.py:
- Removed legalize_fn entirely
- Removed init_positions / refine_noise_scale (always N(0,I))
- Removed energy_norm (handled externally)
- Energy computed on raw model samples at every step
"""

import sys
import os
import torch
import math
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archive', 'original_sdds'))
from noise_schedule import GaussianNoiseSchedule

from gnn_layers import scatter_sum
from sdds_step_model import SDDSStepModel


@dataclass
class TrajectoryBuffer:
    """Buffer for storing trajectory data for PPO training."""
    states: torch.Tensor
    actions: torch.Tensor
    rand_node_features: torch.Tensor
    policies: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    time_index_per_node: torch.Tensor
    noise_rewards: torch.Tensor
    entropy_rewards: torch.Tensor
    energy_rewards: torch.Tensor
    final_states: torch.Tensor
    log_q_T: Optional[torch.Tensor] = None
    log_q_0_T: Optional[torch.Tensor] = None
    log_p_0_T: Optional[torch.Tensor] = None


def _expand_graph_for_basis(edge_index, edge_attr, node_graph_idx,
                            node_features, n_nodes, n_graphs, n_basis):
    """Expand graph structure to batch basis states into a single mega-graph."""
    device = edge_index.device
    node_offsets = torch.arange(n_basis, device=device) * n_nodes
    ei_expanded = edge_index.unsqueeze(0) + node_offsets.reshape(-1, 1, 1)
    expanded_edge_index = ei_expanded.permute(1, 0, 2).reshape(2, -1)
    expanded_edge_attr = edge_attr.repeat(n_basis, 1) if edge_attr is not None else None
    graph_offsets = torch.arange(n_basis, device=device) * n_graphs
    ngi_expanded = node_graph_idx.unsqueeze(0) + graph_offsets.reshape(-1, 1)
    expanded_node_graph_idx = ngi_expanded.reshape(-1)
    expanded_node_features = node_features.repeat(n_basis, 1) if node_features is not None else None
    expanded_n_graphs = n_graphs * n_basis
    return (expanded_edge_index, expanded_edge_attr, expanded_node_graph_idx,
            expanded_node_features, expanded_n_graphs)


def collect_sdds_trajectory(
    model: SDDSStepModel,
    noise_schedule: GaussianNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_basis_states: int,
    T_temperature: float,
    node_features: torch.Tensor,
    energy_fn: Optional[Callable] = None,
    device: torch.device = None,
) -> TrajectoryBuffer:
    """
    Collect trajectory for SDDS chip placement.

    No legalization. No warm-start. Always starts from N(0,I).
    Per-step energy on raw positions.

    Args:
        model: SDDSStepModel
        noise_schedule: GaussianNoiseSchedule
        edge_index: (2, E)
        edge_attr: (E, 4)
        node_graph_idx: (V,)
        n_graphs: int
        n_basis_states: int
        T_temperature: float
        node_features: (V, 14)
        energy_fn: energy_fn(positions, node_graph_idx, n_graphs) -> (n_graphs,)
        device: torch device

    Returns:
        TrajectoryBuffer
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    continuous_dim = model.continuous_dim
    n_diffusion_steps = noise_schedule.n_steps

    # Initialize from N(0, I)
    X_prev = torch.randn(n_nodes, n_basis_states, continuous_dim, device=device)

    # Log prob under N(0, I) prior
    log_prior_per_dim = -0.5 * (X_prev ** 2 + math.log(2 * math.pi))
    log_prior_per_node = log_prior_per_dim.sum(dim=-1)  # (V, n_basis)

    # Expanded graph for batched basis processing
    (exp_ei, exp_ea, exp_ngi, exp_nf, exp_ng) = _expand_graph_for_basis(
        edge_index, edge_attr, node_graph_idx, node_features,
        n_nodes, n_graphs, n_basis_states,
    )

    # Prior log prob per graph
    log_prior_flat = log_prior_per_node.transpose(0, 1).reshape(-1)
    log_q_T_flat = scatter_sum(log_prior_flat, exp_ngi, dim=0, dim_size=exp_ng)
    log_q_T = log_q_T_flat.reshape(n_basis_states, n_graphs).t()

    # Allocate storage
    Xs = torch.zeros(n_diffusion_steps + 1, n_nodes, n_basis_states, continuous_dim,
                     dtype=torch.float32, device=device)
    Xs[0] = X_prev

    rand_features = torch.zeros(n_diffusion_steps, n_nodes, n_basis_states,
                                model.n_random_features, dtype=torch.float32, device=device)
    log_policies = torch.zeros(n_diffusion_steps, n_graphs, n_basis_states,
                               dtype=torch.float32, device=device)
    values_steps = torch.zeros(n_diffusion_steps + 1, n_graphs, n_basis_states,
                               dtype=torch.float32, device=device)
    noise_rewards = torch.zeros(n_diffusion_steps, n_graphs, n_basis_states,
                                dtype=torch.float32, device=device)
    entropy_rewards = torch.zeros(n_diffusion_steps, n_graphs, n_basis_states,
                                  dtype=torch.float32, device=device)
    energy_rewards = torch.zeros(n_diffusion_steps, n_graphs, n_basis_states,
                                 dtype=torch.float32, device=device)
    time_indices = torch.zeros(n_diffusion_steps, n_nodes, n_basis_states,
                               dtype=torch.long, device=device)

    # Diffusion loop
    for step in range(n_diffusion_steps):
        # Flatten for batched GNN
        X_t_flat = X_prev.transpose(0, 1).reshape(-1, continuous_dim)

        mean_flat, log_var_flat, vals_flat, rand_flat = model(
            X_t_flat, step, exp_ei, exp_ea, exp_ngi, exp_ng, exp_nf,
        )

        # Sample
        X_next_flat, _ = model.sample_action(mean_flat, log_var_flat)

        # State log probs
        slp_flat = model.get_state_log_prob(
            mean_flat, log_var_flat, X_next_flat, exp_ngi, exp_ng,
        )

        # Reshape back
        step_X = X_next_flat.reshape(n_basis_states, n_nodes, continuous_dim).transpose(0, 1)
        step_slp = slp_flat.reshape(n_basis_states, n_graphs).t()
        step_vals = vals_flat.reshape(n_basis_states, n_graphs).t()
        step_rand = rand_flat.reshape(n_basis_states, n_nodes, model.n_random_features).transpose(0, 1)

        # Entropy reward
        entropy_rewards[step] = T_temperature * (-step_slp)

        # Noise reward
        log_p = noise_schedule.compute_forward_log_prob(X_prev, step_X, step)
        log_p_flat = log_p.transpose(0, 1).reshape(-1)
        noise_pg = scatter_sum(log_p_flat, exp_ngi, dim=0, dim_size=exp_ng)
        noise_rewards[step] = (T_temperature * noise_pg).reshape(n_basis_states, n_graphs).t()

        # Per-step energy on RAW positions (no legalization)
        if energy_fn is not None:
            for b in range(n_basis_states):
                e_per_graph = energy_fn(step_X[:, b, :], node_graph_idx, n_graphs)
                energy_rewards[step, :, b] = -e_per_graph

        # Store
        Xs[step + 1] = step_X
        rand_features[step] = step_rand
        log_policies[step] = step_slp
        values_steps[step] = step_vals
        time_indices[step] = step

        X_prev = step_X

    # Combined rewards
    combined = noise_schedule.calculate_combined_reward(noise_rewards, entropy_rewards)
    rewards = combined + energy_rewards

    # Final energy for logging
    X_0 = X_prev
    final_energy = torch.zeros(n_graphs, n_basis_states, device=device)
    if energy_fn is not None:
        for b in range(n_basis_states):
            final_energy[:, b] = energy_fn(X_0[:, b, :], node_graph_idx, n_graphs)

    # Diagnostics
    log_q_0_T = torch.zeros(n_diffusion_steps + 1, n_graphs, n_basis_states,
                             dtype=torch.float32, device=device)
    log_p_0_T = torch.zeros(n_diffusion_steps + 1, n_graphs, n_basis_states,
                             dtype=torch.float32, device=device)
    log_q_0_T[0] = log_q_T
    log_q_0_T[1:] = log_policies
    T_s = T_temperature if T_temperature > 0 else 1.0
    log_p_0_T[:-1] = -1.0 / T_s * noise_rewards
    log_p_0_T[-1] = 1.0 / T_s * final_energy

    return TrajectoryBuffer(
        states=Xs[:-1],
        actions=Xs[1:],
        rand_node_features=rand_features,
        policies=log_policies,
        rewards=rewards,
        values=values_steps[:-1],
        time_index_per_node=time_indices,
        noise_rewards=noise_rewards,
        entropy_rewards=entropy_rewards,
        energy_rewards=-final_energy,
        final_states=X_0,
        log_q_T=log_q_T,
        log_q_0_T=log_q_0_T,
        log_p_0_T=log_p_0_T,
    )


def sample_sdds(
    model: SDDSStepModel,
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
) -> torch.Tensor:
    """
    Sample from trained SDDS model (evaluation only).

    Args:
        T_temperature: 0 = greedy (mean), >0 = stochastic

    Returns:
        X_0: (V, n_basis_states, 2) final positions
    """
    if device is None:
        device = edge_index.device

    n_nodes = node_graph_idx.size(0)
    continuous_dim = model.continuous_dim
    n_steps = noise_schedule.n_steps
    overall_steps = n_steps * eval_step_factor

    X_prev = torch.randn(n_nodes, n_basis_states, continuous_dim, device=device)

    (exp_ei, exp_ea, exp_ngi, exp_nf, exp_ng) = _expand_graph_for_basis(
        edge_index, edge_attr, node_graph_idx, node_features,
        n_nodes, n_graphs, n_basis_states,
    )

    model.eval()
    with torch.no_grad():
        for step in range(overall_steps):
            model_step = step // eval_step_factor
            X_t_flat = X_prev.transpose(0, 1).reshape(-1, continuous_dim)

            mean_flat, log_var_flat, _, _ = model(
                X_t_flat, model_step, exp_ei, exp_ea, exp_ngi, exp_ng, exp_nf,
            )

            if T_temperature > 0:
                X_next_flat, _ = model.sample_action(mean_flat, log_var_flat)
            else:
                X_next_flat = mean_flat

            X_prev = X_next_flat.reshape(
                n_basis_states, n_nodes, continuous_dim
            ).transpose(0, 1)

    return X_prev


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _test():
    """Quick smoke test: trajectory + sampling shapes."""
    print("=== Testing sdds_trajectory.py ===\n")

    V, E, n_graphs = 30, 80, 1
    T = 10
    n_basis = 2

    model = SDDSStepModel(n_diffusion_steps=T)
    ns = GaussianNoiseSchedule(T)

    edge_index = torch.randint(0, V, (2, E))
    edge_attr = torch.randn(E, 4)
    ngi = torch.zeros(V, dtype=torch.long)
    nf = torch.randn(V, 14)

    # Simple energy fn
    def energy_fn(pos, ngi, ng):
        return pos.abs().sum(dim=1).mean().unsqueeze(0)

    print("--- Trajectory collection ---")
    buf = collect_sdds_trajectory(
        model, ns, edge_index, edge_attr, ngi, n_graphs,
        n_basis, T_temperature=0.5, node_features=nf, energy_fn=energy_fn,
    )
    print(f"  states: {buf.states.shape}")
    print(f"  actions: {buf.actions.shape}")
    print(f"  rewards: {buf.rewards.shape}")
    print(f"  values: {buf.values.shape}")
    print(f"  final_states: {buf.final_states.shape}")
    print(f"  energy_rewards (final): {buf.energy_rewards.shape}")
    print(f"  noise_reward sum: {buf.noise_rewards.sum():.4f}")
    print(f"  entropy_reward sum: {buf.entropy_rewards.sum():.4f}")

    assert buf.states.shape == (T, V, n_basis, 2)
    assert buf.rewards.shape == (T, n_graphs, n_basis)

    print("\n--- Sampling ---")
    X_0 = sample_sdds(
        model, ns, edge_index, edge_attr, ngi, n_graphs,
        n_basis, nf, T_temperature=0.0,
    )
    print(f"  X_0: {X_0.shape}")
    print(f"  X_0 range: [{X_0.min():.3f}, {X_0.max():.3f}]")
    assert X_0.shape == (V, n_basis, 2)

    print("\nAll checks passed!")


if __name__ == '__main__':
    _test()
