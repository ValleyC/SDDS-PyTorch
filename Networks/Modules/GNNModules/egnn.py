"""
E(n) Equivariant Graph Neural Network encoder.
Adapted from EDISCO for SDDS-PyTorch.

This module implements E(2)-equivariant message passing layers that
preserve rotational and translational symmetry of the input coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn_utils import timestep_embedding


class EGNNLayerDense(nn.Module):
    """
    Dense E(n) Equivariant GNN layer for small to medium graphs.

    This layer computes messages between all pairs of nodes using
    dense matrix operations, suitable for graphs with up to ~500 nodes.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        """
        Initialize dense EGNN layer.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for MLPs
            coord_dim: Coordinate dimension (2 for 2D)
            coord_update_alpha: Learning rate for coordinate updates
            weight_temp: Temperature for coordinate weight softmax
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_update_alpha = coord_update_alpha
        self.weight_temp = weight_temp

        # Message network: combines source node, target node, edge, and distance
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate network (no bias to preserve equivariance)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update network
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h, x, e):
        """
        Forward pass with dense adjacency matrix.

        Args:
            h: Node features (batch, n_nodes, node_dim)
            x: Node coordinates (batch, n_nodes, coord_dim)
            e: Edge features (batch, n_nodes, n_nodes, edge_dim)

        Returns:
            Tuple of (h_new, x_new, e_new)
        """
        batch_size, n_nodes, _ = h.shape

        # Compute pairwise coordinate differences and distances
        x_i = x.unsqueeze(2)  # (batch, n, 1, coord_dim)
        x_j = x.unsqueeze(1)  # (batch, 1, n, coord_dim)
        x_diff = x_j - x_i    # (batch, n, n, coord_dim)
        distances = torch.norm(x_diff, dim=-1, keepdim=True)  # (batch, n, n, 1)

        # Prepare node features for all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, node_dim)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, node_dim)

        # Compute messages
        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)
        messages = self.message_mlp(msg_input)

        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(messages)
        coord_weights = torch.tanh(coord_weights / self.weight_temp)
        x_update = coord_weights * x_diff / (distances + 1e-8)
        x_agg = x_update.sum(dim=2)
        x_new = x + self.coord_update_alpha * x_agg

        # Update nodes
        h_agg = messages.sum(dim=2)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        # Update edges
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, messages], dim=-1)))

        return h_new, x_new, e_new


class EGNNLayerSparse(nn.Module):
    """
    Sparse E(n) Equivariant GNN layer for large graphs.

    This layer uses edge indices for sparse message passing,
    suitable for graphs with 1000+ nodes.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        """
        Initialize sparse EGNN layer.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for MLPs
            coord_dim: Coordinate dimension (2 for 2D)
            coord_update_alpha: Learning rate for coordinate updates
            weight_temp: Temperature for coordinate weight softmax
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_update_alpha = coord_update_alpha
        self.weight_temp = weight_temp

        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h, x, e, edge_index):
        """
        Forward pass with sparse edge index.

        Args:
            h: Node features (n_nodes, node_dim)
            x: Node coordinates (n_nodes, coord_dim)
            e: Edge features (n_edges, edge_dim)
            edge_index: Edge indices (2, n_edges)

        Returns:
            Tuple of (h_new, x_new, e_new)
        """
        n_nodes = h.shape[0]
        row, col = edge_index[0], edge_index[1]

        # Ensure e is 2D (n_edges, edge_dim)
        if e.dim() == 3:
            e = e.reshape(-1, e.shape[-1])
        elif e.dim() == 1:
            e = e.unsqueeze(-1)
            if e.shape[-1] != self.edge_dim:
                raise ValueError(f"Edge features have wrong dimension: {e.shape[-1]} vs expected {self.edge_dim}")

        # Compute coordinate differences and distances
        x_diff = x[col] - x[row]
        distances = torch.norm(x_diff, dim=-1, keepdim=True)

        # Prepare features
        h_i = h[row]
        h_j = h[col]

        # Compute messages
        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)
        messages = self.message_mlp(msg_input)

        # Update coordinates
        coord_weights = self.coord_mlp(messages)
        coord_weights = torch.tanh(coord_weights / self.weight_temp)
        x_update = coord_weights * x_diff / (distances + 1e-8)

        # Aggregate coordinate updates
        x_agg = torch.zeros(n_nodes, self.coord_dim, device=x.device)
        x_agg.index_add_(0, row, x_update)
        x_new = x + self.coord_update_alpha * x_agg

        # Aggregate messages for nodes
        h_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        h_agg.index_add_(0, row, messages)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        # Update edge features
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, messages], dim=-1)))

        return h_new, x_new, e_new


class EGNNEncoder(nn.Module):
    """
    E(2)-Equivariant encoder for combinatorial optimization.

    This encoder processes graphs while maintaining equivariance to
    rotations and translations of the input coordinates.
    """

    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2, sparse=False,
                 dense_only=False, use_activation_checkpoint=False,
                 coord_update_alpha=0.1, weight_temp=10.0, **kwargs):
        """
        Initialize EGNN encoder.

        Args:
            n_layers: Number of EGNN layers
            hidden_dim: Hidden dimension for MLPs
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            time_dim: Time embedding dimension
            coord_dim: Coordinate dimension (2 for 2D)
            out_channels: Output channels (2 for binary classification)
            sparse: Use sparse layers
            dense_only: Force dense-only mode
            use_activation_checkpoint: Enable gradient checkpointing
            coord_update_alpha: Learning rate for coordinate updates
            weight_temp: Temperature for coordinate weights
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels

        # Set execution mode at initialization
        self.sparse = sparse and not dense_only
        self.dense_only = dense_only or not sparse
        self.use_activation_checkpoint = use_activation_checkpoint

        # Initial embeddings
        self.node_embed = nn.Linear(coord_dim, node_dim)
        self.edge_embed = nn.Linear(1, edge_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )

        # Create layers based on execution mode
        if self.dense_only:
            self.layers = nn.ModuleList([
                EGNNLayerDense(node_dim, edge_dim, hidden_dim, coord_dim,
                              coord_update_alpha, weight_temp)
                for _ in range(n_layers)
            ])
        elif self.sparse:
            self.layers = nn.ModuleList([
                EGNNLayerSparse(node_dim, edge_dim, hidden_dim, coord_dim,
                               coord_update_alpha, weight_temp)
                for _ in range(n_layers)
            ])
        else:
            # Flexible mode - use sparse layers
            self.layers = nn.ModuleList([
                EGNNLayerSparse(node_dim, edge_dim, hidden_dim, coord_dim,
                               coord_update_alpha, weight_temp)
                for _ in range(n_layers)
            ])

        # Time injection layers
        self.time_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_dim, edge_dim),
                nn.SiLU()
            ) for _ in range(n_layers)
        ])

        # Output head
        self.out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_channels)
        )

        # Initialize output to zero for stability
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """
        Forward pass that routes to the appropriate implementation.

        Args:
            coords: Node coordinates (batch, n_nodes, 2) or (n_nodes, 2)
            adj_matrix: Adjacency matrix or edge features
            timesteps: Diffusion timesteps
            edge_index: Edge indices for sparse mode (2, n_edges)

        Returns:
            Edge logits for classification
        """
        if self.dense_only:
            return self._forward_dense_only(coords, adj_matrix, timesteps)
        elif self.sparse:
            return self._forward_sparse_only(coords, adj_matrix, timesteps, edge_index)
        else:
            return self._forward_flexible(coords, adj_matrix, timesteps, edge_index)

    def _forward_dense_only(self, coords, adj_matrix, timesteps):
        """Dense-only forward pass."""
        batch_size, n_nodes, _ = coords.shape

        # Initialize features
        h = self.node_embed(coords)
        x = coords.clone()

        # Prepare adjacency matrix
        adj_input = adj_matrix.unsqueeze(-1) if adj_matrix.dim() == 3 else adj_matrix
        e = self.edge_embed(adj_input)

        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))

        # Apply layers
        for layer, time_layer in zip(self.layers, self.time_layers):
            # Time modulation
            time_mod = time_layer(t_emb).view(batch_size, 1, 1, -1)
            e_with_time = e * (1 + time_mod)

            # Apply layer
            h, x, e = layer(h, x, e_with_time)

        # Output
        return self.out(e)

    def _forward_sparse_only(self, coords, adj_matrix, timesteps, edge_index):
        """Optimized sparse-only forward."""
        n_nodes = coords.shape[0]
        n_edges = edge_index.shape[1]

        # Validate edge_index bounds
        if edge_index.max() >= n_nodes:
            raise ValueError(f"Edge index {edge_index.max().item()} >= num_nodes {n_nodes}")
        if edge_index.min() < 0:
            raise ValueError(f"Edge index {edge_index.min().item()} < 0")

        # Initialize features
        h = self.node_embed(coords)
        x = coords.clone()

        # Prepare edge features
        if adj_matrix.dim() == 1:
            adj_input = adj_matrix.unsqueeze(-1)
        elif adj_matrix.dim() == 2:
            if adj_matrix.shape[0] == n_edges and adj_matrix.shape[1] == 1:
                adj_input = adj_matrix
            elif adj_matrix.shape[0] == n_edges:
                adj_input = adj_matrix[:, 0:1] if adj_matrix.shape[1] > 1 else adj_matrix
            else:
                adj_input = adj_matrix.reshape(-1, 1)
                if adj_input.shape[0] != n_edges:
                    raise ValueError(f"Cannot reshape adj_matrix {adj_matrix.shape} to match n_edges {n_edges}")
        else:
            raise ValueError(f"Unexpected adj_matrix dimensions: {adj_matrix.dim()}, shape: {adj_matrix.shape}")

        e = self.edge_embed(adj_input)

        # Time embedding
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        t_emb_base = self.time_embed(timestep_embedding(timesteps, self.time_dim))

        # Expand time embedding to match number of edges
        if t_emb_base.shape[0] == 1:
            t_emb = t_emb_base.expand(n_edges, -1)
        elif t_emb_base.shape[0] == n_edges:
            t_emb = t_emb_base
        else:
            row = edge_index[0]
            if t_emb_base.shape[0] <= row.max():
                t_emb = t_emb_base[0:1].expand(n_edges, -1)
            else:
                t_emb = t_emb_base[row]

        # Apply layers
        for layer, time_layer in zip(self.layers, self.time_layers):
            time_mod = time_layer(t_emb)
            e_with_time = e * (1 + time_mod)
            h, x, e = layer(h, x, e_with_time, edge_index)

        return self.out(e)

    def _forward_flexible(self, coords, adj_matrix, timesteps, edge_index=None):
        """Fallback flexible implementation with runtime checks."""
        if edge_index is not None and coords.dim() == 2:
            return self._forward_sparse_only(coords, adj_matrix, timesteps, edge_index)
        else:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
                if adj_matrix.dim() == 2:
                    adj_matrix = adj_matrix.unsqueeze(0)
                if timesteps.dim() == 0:
                    timesteps = timesteps.unsqueeze(0)

            if self.dense_only:
                return self._forward_dense_only(coords, adj_matrix, timesteps)
            else:
                batch_size, n_nodes, _ = coords.shape
                if batch_size != 1:
                    raise NotImplementedError("Flexible mode with batched dense data not supported")

                # Create full graph edge_index for dense computation
                edge_index = torch.combinations(torch.arange(n_nodes, device=coords.device), r=2).T
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                self_loops = torch.arange(n_nodes, device=coords.device).unsqueeze(0).repeat(2, 1)
                edge_index = torch.cat([edge_index, self_loops], dim=1)

                coords_flat = coords[0]
                adj_flat = adj_matrix[0].reshape(-1)
                n_edges = edge_index.shape[1]
                adj_edge = adj_flat[:n_edges] if adj_flat.numel() >= n_edges else torch.zeros(n_edges, device=adj_flat.device)

                return self._forward_sparse_only(coords_flat, adj_edge, timesteps, edge_index)


class EGNNEncoderDense(nn.Module):
    """
    Pure dense-only implementation for maximum performance.
    No sparse support, no runtime checks.
    """

    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 time_dim=128, coord_dim=2, out_channels=2,
                 use_activation_checkpoint=False,
                 coord_update_alpha=0.1, weight_temp=10.0, **kwargs):
        """
        Initialize dense-only EGNN encoder.

        Args:
            n_layers: Number of EGNN layers
            hidden_dim: Hidden dimension for MLPs
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            time_dim: Time embedding dimension
            coord_dim: Coordinate dimension (2 for 2D)
            out_channels: Output channels (2 for binary classification)
            use_activation_checkpoint: Enable gradient checkpointing
            coord_update_alpha: Learning rate for coordinate updates
            weight_temp: Temperature for coordinate weights
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels

        # Initial embeddings
        self.node_embed = nn.Linear(coord_dim, node_dim)
        self.edge_embed = nn.Linear(1, edge_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )

        # Dense-only layers
        self.layers = nn.ModuleList([
            EGNNLayerDense(node_dim, edge_dim, hidden_dim, coord_dim,
                          coord_update_alpha, weight_temp)
            for _ in range(n_layers)
        ])

        # Time injection layers
        self.time_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_dim, edge_dim),
                nn.SiLU()
            ) for _ in range(n_layers)
        ])

        # Output head
        self.out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_channels)
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, coords, adj_matrix, timesteps, edge_index=None):
        """
        Direct dense forward - no checks, edge_index ignored.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            adj_matrix: Adjacency matrix (batch, n_nodes, n_nodes)
            timesteps: Diffusion timesteps (batch,)
            edge_index: Ignored in dense mode

        Returns:
            Edge logits (batch, n_nodes, n_nodes, out_channels)
        """
        batch_size, n_nodes, _ = coords.shape

        # Direct processing
        h = self.node_embed(coords)
        x = coords
        e = self.edge_embed(adj_matrix.unsqueeze(-1))

        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))

        # Apply layers
        for layer, time_layer in zip(self.layers, self.time_layers):
            time_mod = time_layer(t_emb).view(batch_size, 1, 1, -1)
            e_with_time = e * (1 + time_mod)
            h, x, e = layer(h, x, e_with_time)

        return self.out(e)
