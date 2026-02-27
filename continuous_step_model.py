"""
Continuous Diffusion Step Model for Chip Placement

Outputs (mean, log_var) for Gaussian distribution over continuous positions,
instead of categorical logits. Reuses EncodeProcessDecode GNN backbone.

Reference: DIffUCO/Networks/Modules/HeadModules/ContinuousHead.py
           DIffUCO/Networks/DiffModel.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional

try:
    from .step_model import (
        EncodeProcessDecode, ReluMLP, ValueMLP,
        get_sinusoidal_positional_encoding, scatter_sum
    )
except ImportError:
    from step_model import (
        EncodeProcessDecode, ReluMLP, ValueMLP,
        get_sinusoidal_positional_encoding, scatter_sum
    )


class ContinuousDiffusionStepModel(nn.Module):
    """
    Continuous diffusion step model for chip placement.

    Per-step forward pass:
    1. Input: cat([X_t, node_features, time_embed, rand_nodes])
    2. GNN (EncodeProcessDecode)
    3. Output: (position_mean, position_log_var, values)

    Key differences from DiffusionStepModel:
    - Input uses raw continuous positions (not one_hot)
    - Output is (mean, log_var) for Gaussian (not categorical logits)
    - Node features (component sizes) are part of input
    """

    def __init__(
        self,
        continuous_dim: int = 2,
        node_feature_dim: int = 2,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 50,
        n_message_passes: int = 5,
        n_random_features: int = 5,
        time_encoding: str = 'sinusoidal',
        embedding_dim: int = 32,
        mean_aggr: bool = False,
    ):
        """
        Args:
            continuous_dim: Dimension of continuous state (2 for x,y positions)
            node_feature_dim: Dimension of node features (2 for component w,h)
            edge_dim: Edge feature dimension (4 for terminal offsets)
            hidden_dim: GNN hidden dimension
            n_diffusion_steps: Total diffusion steps T
            n_message_passes: Number of GNN message passing layers
            n_random_features: Number of random node features
            time_encoding: 'sinusoidal' or 'one_hot'
            embedding_dim: Time embedding dimension (if sinusoidal)
            mean_aggr: Use mean aggregation instead of sum in GNN
        """
        super().__init__()

        self.continuous_dim = continuous_dim
        self.node_feature_dim = node_feature_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.n_random_features = n_random_features
        self.time_encoding = time_encoding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input dimension: positions + node_features + time_embed + random
        if time_encoding == 'one_hot':
            time_dim = n_diffusion_steps
        else:
            time_dim = embedding_dim

        input_dim = continuous_dim + node_feature_dim + time_dim + n_random_features

        # GNN backbone (reuse from step_model.py)
        self.gnn = EncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=mean_aggr,
        )

        # Mean head: unbounded output for position mean
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_dim),
        )

        # Log-variance head: clipped to [-10, 2]
        self.log_var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_dim),
        )

        # Value head (same architecture as DiffusionStepModel)
        value_features = [hidden_dim, 120, 64, 1]
        self.value_head = ValueMLP(value_features)

    def _build_input(
        self,
        X_t: torch.Tensor,
        node_features: torch.Tensor,
        rand_nodes: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        Build input features: cat([X_t, node_features, time_embed, rand_nodes])

        Unlike categorical model which uses one_hot(X_t), we use raw positions.

        Args:
            X_t: Current positions (n_nodes, continuous_dim)
            node_features: Component sizes (n_nodes, node_feature_dim)
            rand_nodes: Random features (n_nodes, n_random_features)
            t_idx: Timestep index

        Returns:
            X_input: (n_nodes, input_dim)
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        # Time embedding
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

        X_input = torch.cat([X_t, node_features, t_embed, rand_nodes], dim=-1)
        return X_input

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

        Args:
            X_t: Current positions (n_nodes, continuous_dim)
            t_idx: Timestep index
            edge_index: (2, n_edges)
            edge_attr: (n_edges, edge_dim)
            node_graph_idx: (n_nodes,)
            n_graphs: Number of graphs
            node_features: Component sizes (n_nodes, node_feature_dim)
            rand_nodes: Optional pre-generated random features

        Returns:
            position_mean: (n_nodes, continuous_dim)
            position_log_var: (n_nodes, continuous_dim) clipped to [-10, 2]
            values: (n_graphs,)
            rand_nodes: (n_nodes, n_random_features) for storage
        """
        n_nodes = X_t.size(0)
        device = X_t.device

        if rand_nodes is None:
            rand_nodes = self.reinit_rand_nodes(n_nodes, device)

        # Build input and run GNN
        X_input = self._build_input(X_t, node_features, rand_nodes, t_idx)
        node_embeddings = self.gnn(X_input, edge_index, edge_attr)  # (n_nodes, hidden_dim)

        # Position mean (unbounded)
        position_mean = self.mean_head(node_embeddings)  # (n_nodes, continuous_dim)

        # Position log variance (clipped to [-10, 2] matching DIffUCO ContinuousHead)
        position_log_var = self.log_var_head(node_embeddings)
        position_log_var = torch.clamp(position_log_var, -10.0, 2.0)

        # Value head
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
        values = self.value_head(value_embeddings).squeeze(-1)  # (n_graphs,)

        return position_mean, position_log_var, values, rand_nodes

    def sample_action(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from Gaussian using reparameterization trick.

        X_next = mean + sqrt(exp(log_var)) * epsilon

        Args:
            mean: (n_nodes, continuous_dim)
            log_var: (n_nodes, continuous_dim)

        Returns:
            X_next: (n_nodes, continuous_dim)
            per_node_log_prob: (n_nodes,) log prob summed over dims
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        X_next = mean + std * epsilon

        # Log probability: sum over continuous dimensions
        log_prob_per_dim = -0.5 * (
            (X_next - mean) ** 2 / (std ** 2 + 1e-8)
            + log_var
            + np.log(2 * np.pi)
        )
        per_node_log_prob = log_prob_per_dim.sum(dim=-1)  # (n_nodes,)

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
        Compute log probability of X_next under Gaussian(mean, exp(log_var)),
        aggregated per graph.

        Args:
            mean: (n_nodes, continuous_dim)
            log_var: (n_nodes, continuous_dim)
            X_next: (n_nodes, continuous_dim)
            node_graph_idx: (n_nodes,)
            n_graphs: number of graphs

        Returns:
            state_log_probs: (n_graphs,) summed over nodes per graph
        """
        std = torch.exp(0.5 * log_var)

        log_prob_per_dim = -0.5 * (
            (X_next - mean) ** 2 / (std ** 2 + 1e-8)
            + log_var
            + np.log(2 * np.pi)
        )
        per_node_log_prob = log_prob_per_dim.sum(dim=-1)  # (n_nodes,)

        state_log_probs = scatter_sum(
            per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs
        )

        return state_log_probs


def test_continuous_step_model():
    """Test the continuous diffusion step model."""
    print("=" * 60)
    print("Testing ContinuousDiffusionStepModel")
    print("=" * 60)

    # Test parameters
    n_nodes = 20
    n_edges = 60
    n_graphs = 2
    continuous_dim = 2
    node_feature_dim = 2
    edge_dim = 4
    n_diffusion_steps = 10

    model = ContinuousDiffusionStepModel(
        continuous_dim=continuous_dim,
        node_feature_dim=node_feature_dim,
        edge_dim=edge_dim,
        hidden_dim=32,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=3,
        n_random_features=5,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Create test data
    X_t = torch.randn(n_nodes, continuous_dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs), n_nodes // n_graphs
    )
    node_features = torch.rand(n_nodes, node_feature_dim) * 0.5  # Component sizes

    # Forward pass
    mean, log_var, values, rand_nodes = model(
        X_t, t_idx=0, edge_index=edge_index, edge_attr=edge_attr,
        node_graph_idx=node_graph_idx, n_graphs=n_graphs,
        node_features=node_features,
    )

    print(f"\nForward pass shapes:")
    print(f"  mean: {mean.shape}")
    print(f"  log_var: {log_var.shape}")
    print(f"  values: {values.shape}")
    print(f"  rand_nodes: {rand_nodes.shape}")

    # Verify log_var clipping
    log_var_ok = (log_var >= -10.0).all() and (log_var <= 2.0).all()
    print(f"\nLog var range: [{log_var.min().item():.3f}, {log_var.max().item():.3f}] {'(OK)' if log_var_ok else '(FAIL)'}")

    # Test sampling
    X_next, per_node_log_prob = model.sample_action(mean, log_var)
    print(f"\nSampling:")
    print(f"  X_next: {X_next.shape}")
    print(f"  per_node_log_prob: {per_node_log_prob.shape}")

    # Test state log prob
    state_log_probs = model.get_state_log_prob(
        mean, log_var, X_next, node_graph_idx, n_graphs
    )
    print(f"  state_log_probs: {state_log_probs.shape}")
    print(f"  state_log_probs values: {state_log_probs.detach().numpy()}")

    # Verify state_log_probs equals sum of per_node_log_prob per graph
    manual_state_log_probs = scatter_sum(
        per_node_log_prob, node_graph_idx, dim=0, dim_size=n_graphs
    )
    match = torch.allclose(state_log_probs, manual_state_log_probs, atol=1e-5)
    print(f"  State log prob aggregation: {'OK' if match else 'FAIL'}")

    # Test gradient flow
    loss = values.sum() + state_log_probs.sum()
    loss.backward()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"\nGradient flow: {params_with_grad}/{total_params} parameters have gradients")

    # Test with stored rand_nodes (PPO replay)
    mean2, log_var2, values2, _ = model(
        X_t, t_idx=0, edge_index=edge_index, edge_attr=edge_attr,
        node_graph_idx=node_graph_idx, n_graphs=n_graphs,
        node_features=node_features, rand_nodes=rand_nodes,
    )
    replay_match = torch.allclose(mean, mean2, atol=1e-5)
    print(f"  Replay with stored rand_nodes: {'OK' if replay_match else 'FAIL'}")

    print("\n" + "=" * 60)
    print("ContinuousDiffusionStepModel tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_continuous_step_model()
