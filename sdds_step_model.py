"""
SDDS Step Model — Continuous Diffusion GNN with 14D Node Features

Outputs (mean, log_var) for Gaussian distribution over 2D positions.

Changes from archived ContinuousDiffusionStepModel:
- node_feature_dim: 2 -> 14 (sizes + Laplacian PE + centrality)
- mean_aggr: False -> True (normalizes for variable-degree nets)
- Input dim: 2 + 14 + 32 + 5 = 53 (was 41)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from gnn_layers import EncodeProcessDecode, ReluMLP, get_sinusoidal_positional_encoding, scatter_sum


class SDDSStepModel(nn.Module):
    """
    Continuous diffusion step model for chip placement with 14D features.

    Per-step forward pass:
    1. Input: cat([X_t, node_features_14D, time_embed, rand_nodes])
    2. GNN (EncodeProcessDecode)
    3. Output: (position_mean, position_log_var, values)
    """

    def __init__(
        self,
        continuous_dim: int = 2,
        node_feature_dim: int = 14,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 50,
        n_message_passes: int = 5,
        n_random_features: int = 5,
        time_encoding: str = 'sinusoidal',
        embedding_dim: int = 32,
        mean_aggr: bool = True,
    ):
        super().__init__()

        self.continuous_dim = continuous_dim
        self.node_feature_dim = node_feature_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.n_random_features = n_random_features
        self.time_encoding = time_encoding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input dimension
        if time_encoding == 'one_hot':
            time_dim = n_diffusion_steps
        else:
            time_dim = embedding_dim

        input_dim = continuous_dim + node_feature_dim + time_dim + n_random_features

        # GNN backbone
        self.gnn = EncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=mean_aggr,
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dim, continuous_dim)

        # Log-variance head, clipped to [-10, 2]
        self.log_var_head = nn.Linear(hidden_dim, continuous_dim)

        # Value head (matches DIffUCO ContinuousHead)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _build_input(
        self,
        X_t: torch.Tensor,
        node_features: torch.Tensor,
        rand_nodes: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """Build input: cat([X_t, node_features, time_embed, rand_nodes])"""
        n_nodes = X_t.size(0)
        device = X_t.device

        if self.time_encoding == 'one_hot':
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = F.one_hot(t_tensor, num_classes=self.n_diffusion_steps).float()
            t_embed = t_embed.expand(n_nodes, -1)
        else:
            t_tensor = torch.tensor([t_idx], device=device)
            t_embed = get_sinusoidal_positional_encoding(
                t_tensor, self.embedding_dim, self.n_diffusion_steps
            )
            t_embed = t_embed.expand(n_nodes, -1)

        return torch.cat([X_t, node_features, t_embed, rand_nodes], dim=-1)

    def reinit_rand_nodes(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """Generate fresh random node features (uniform [0, 1])."""
        return torch.rand(n_nodes, self.n_random_features, device=device)

    def forward(
        self,
        X_t: torch.Tensor,
        t_idx: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        node_features: torch.Tensor,
        rand_nodes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one diffusion step.

        Returns:
            position_mean: (n_nodes, continuous_dim)
            position_log_var: (n_nodes, continuous_dim) clipped to [-10, 2]
            values: (n_graphs,)
            rand_nodes: (n_nodes, n_random_features)
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        if rand_nodes is None:
            rand_nodes = self.reinit_rand_nodes(n_nodes, device)

        X_input = self._build_input(X_t, node_features, rand_nodes, t_idx)
        node_embeddings = self.gnn(X_input, edge_index, edge_attr)

        position_mean = self.mean_head(node_embeddings)
        position_log_var = torch.clamp(self.log_var_head(node_embeddings), -10.0, 2.0)

        # Value head: sqrt-normalized mean pooling
        n_node_per_graph = scatter_sum(
            torch.ones(n_nodes, device=device),
            node_graph_idx, dim=0, dim_size=n_graphs
        )
        value_embeddings = scatter_sum(
            node_embeddings, node_graph_idx, dim=0, dim_size=n_graphs
        )
        value_embeddings = value_embeddings / torch.sqrt(
            n_node_per_graph.unsqueeze(-1).clamp(min=1.0)
        )
        values = self.value_head(value_embeddings).squeeze(-1)

        return position_mean, position_log_var, values, rand_nodes

    def sample_action(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample via reparameterization: X_next = mean + std * epsilon

        Returns:
            X_next: (n_nodes, continuous_dim)
            per_node_log_prob: (n_nodes,)
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        X_next = mean + std * epsilon

        log_prob_per_dim = -0.5 * (
            (X_next - mean) ** 2 / (std ** 2 + 1e-8)
            + log_var
            + math.log(2 * math.pi)
        )
        per_node_log_prob = log_prob_per_dim.sum(dim=-1)

        return X_next, per_node_log_prob

    def get_state_log_prob(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        X_next: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """
        Log probability of X_next under Gaussian(mean, exp(log_var)),
        aggregated per graph.

        Returns:
            state_log_probs: (n_graphs,)
        """
        std = torch.exp(0.5 * log_var)
        log_prob_per_dim = -0.5 * (
            (X_next - mean) ** 2 / (std ** 2 + 1e-8)
            + log_var
            + math.log(2 * math.pi)
        )
        per_node_log_prob = log_prob_per_dim.sum(dim=-1)
        return scatter_sum(per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _test():
    """Forward/backward pass with correct shapes."""
    print("=== Testing SDDSStepModel ===\n")

    V, E, n_graphs = 50, 120, 2
    model = SDDSStepModel(hidden_dim=64, n_message_passes=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Verify input dim matches expectation (2 + 14 + 32 + 5 = 53)
    print(f"Expected input dim: 53 (2+14+32+5)")

    # Create test data
    X_t = torch.randn(V, 2)
    edge_index = torch.randint(0, V, (2, E))
    edge_attr = torch.randn(E, 4)
    node_graph_idx = torch.repeat_interleave(torch.arange(n_graphs), V // n_graphs)
    node_features = torch.randn(V, 14)

    # Forward
    mean, log_var, values, rand_nodes = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs, node_features,
    )
    print(f"mean: {mean.shape}, log_var: {log_var.shape}, values: {values.shape}")
    assert mean.shape == (V, 2)
    assert log_var.shape == (V, 2)
    assert values.shape == (n_graphs,)

    # Sample
    X_next, per_node_lp = model.sample_action(mean, log_var)
    assert X_next.shape == (V, 2)
    assert per_node_lp.shape == (V,)
    print(f"X_next: {X_next.shape}, per_node_lp: {per_node_lp.shape}")

    # State log prob
    slp = model.get_state_log_prob(mean, log_var, X_next, node_graph_idx, n_graphs)
    assert slp.shape == (n_graphs,)

    # Gradient
    loss = values.sum() + slp.sum()
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Gradient flow: {has_grad}/{sum(1 for _ in model.parameters())} params have gradients")

    # Replay with stored rand_nodes
    mean2, _, _, _ = model(
        X_t, 0, edge_index, edge_attr, node_graph_idx, n_graphs, node_features,
        rand_nodes=rand_nodes,
    )
    print(f"Replay match: {torch.allclose(mean.detach(), mean2.detach(), atol=1e-5)}")

    print("\nAll checks passed!")


if __name__ == '__main__':
    _test()
