"""
Heatmap-based Chip Placement Model

GLOP-inspired architecture:
  - GNN (EncodeProcessDecode) processes netlist graph → per-component embeddings
  - Heatmap head: embedding → logits over G×G grid cells
  - Gated occupancy MLP: decode-time spatial bias for sequential placement
  - Imitation loss: cross-entropy on reference placement cell labels (warm-start)

The GNN runs once (non-autoregressive). The occupancy MLP is called per-step
by the greedy placer to condition logits on current placement state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .step_model import EncodeProcessDecode, ReluMLP
except ImportError:
    from step_model import EncodeProcessDecode, ReluMLP


class ChipHeatmapModel(nn.Module):
    """GNN + grid heatmap decoder + gated occupancy conditioning."""

    def __init__(
        self,
        node_feature_dim: int = 2,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        grid_size: int = 32,
        n_message_passes: int = 5,
        alpha_max: float = 2.0,
    ):
        """
        Args:
            node_feature_dim: Input node features (component w, h).
            edge_dim: Edge features (pin offsets, 4D).
            hidden_dim: GNN hidden dimension.
            grid_size: G for G×G placement grid.
            n_message_passes: Number of GNN message passing layers.
            alpha_max: Maximum value for gated occupancy bias.
        """
        super().__init__()
        self.grid_size = grid_size
        self.alpha_max = alpha_max
        g2 = grid_size * grid_size

        # GNN backbone (single forward pass, non-autoregressive)
        self.gnn = EncodeProcessDecode(
            input_dim=node_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=False,
        )

        # Heatmap head: per-component logits over G² cells
        self.heatmap_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, g2),
        )
        # Init final linear layer with small weights for uniform-ish start
        nn.init.xavier_normal_(self.heatmap_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.heatmap_head[-1].bias)

        # Occupancy conditioning MLP: decode-time spatial bias
        # Input: G² area-weighted occupancy values
        # Output: G² bias added to logits at each decode step
        self.occ_mlp = nn.Sequential(
            ReluMLP([g2, hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, g2),
        )
        nn.init.xavier_normal_(self.occ_mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.occ_mlp[-1].bias)

        # Gated alpha: learnable scalar, clipped to [0, alpha_max]
        # sigmoid(0) = 0.5, so initial alpha = 0.5 * alpha_max
        self.occ_alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        """
        Single GNN forward pass over the netlist graph.

        Args:
            node_features: (N, node_feature_dim) component sizes (w, h)
            edge_index: (2, E) netlist edges
            edge_attr: (E, edge_dim) pin offsets

        Returns:
            embeddings: (N, hidden_dim) per-component GNN embeddings
            logits: (N, G²) raw heatmap logits per component
        """
        embeddings = self.gnn(node_features, edge_index, edge_attr)
        logits = self.heatmap_head(embeddings)
        return embeddings, logits

    def compute_occ_bias(self, occupancy_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute gated occupancy bias for decode-time conditioning.

        Called at each sequential placement step by the greedy placer.

        Args:
            occupancy_flat: (G²,) area-weighted occupancy grid (flattened)

        Returns:
            bias: (G²,) spatial bias to add to raw logits
        """
        alpha = torch.sigmoid(self.occ_alpha_raw) * self.alpha_max
        raw_bias = self.occ_mlp(occupancy_flat)
        return alpha * raw_bias

    def imitation_loss(
        self,
        logits: torch.Tensor,
        reference_cells: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for warm-start pretraining on reference placements.

        Args:
            logits: (N, G²) raw heatmap logits per component
            reference_cells: (N,) int — grid cell index of reference placement

        Returns:
            loss: scalar cross-entropy loss
        """
        return F.cross_entropy(logits, reference_cells)

    @property
    def current_alpha(self) -> float:
        """Current occupancy gate value (for logging)."""
        with torch.no_grad():
            return (torch.sigmoid(self.occ_alpha_raw) * self.alpha_max).item()
