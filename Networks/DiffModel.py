"""
Diffusion Model wrapper for SDDS-PyTorch.

This module implements the diffusion model for TSP using permutation matrix
representation, following the DiffUCO formulation.

The model predicts position logits for each node:
- Input: Current state X_t where X_t[i] is node i's position (0 to N-1)
- Output: Logits for each node's predicted position (batch, n_nodes, n_positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .Modules.nn_utils import timestep_embedding


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer with coordinate-aware attention.
    """

    def __init__(self, node_dim: int, hidden_dim: int, coord_dim: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism
        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim, hidden_dim)
        self.value = nn.Linear(node_dim, hidden_dim)

        # Distance embedding
        self.dist_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.SiLU(),
        )

        # Layer norm
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Node features (batch, n_nodes, node_dim)
            coords: Node coordinates (batch, n_nodes, 2)

        Returns:
            Updated node features (batch, n_nodes, node_dim)
        """
        batch_size, n_nodes, _ = h.shape

        # Compute queries, keys, values
        q = self.query(h)  # (batch, n_nodes, hidden_dim)
        k = self.key(h)    # (batch, n_nodes, hidden_dim)
        v = self.value(h)  # (batch, n_nodes, hidden_dim)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # (batch, n_nodes, n_nodes)

        # Add distance-based bias
        dist = torch.sqrt(
            torch.sum((coords.unsqueeze(2) - coords.unsqueeze(1)) ** 2, dim=-1, keepdim=True) + 1e-8
        )  # (batch, n_nodes, n_nodes, 1)
        dist_bias = self.dist_embed(dist).squeeze(-1)  # (batch, n_nodes, n_nodes)
        scores = scores + dist_bias

        # Softmax
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.bmm(attn, v)  # (batch, n_nodes, hidden_dim)
        out = self.out_proj(out)  # (batch, n_nodes, node_dim)

        # Residual connection with layer norm
        return self.norm(h + out)


class DiffusionModelTSP(nn.Module):
    """
    Diffusion model for TSP with permutation matrix representation.

    Each node has a categorical variable indicating its position in the tour.
    The model predicts the probability distribution over positions for each node.
    """

    def __init__(
        self,
        n_nodes: int,
        n_layers: int = 4,
        hidden_dim: int = 64,
        node_dim: int = 64,
        time_dim: int = 64,
        coord_dim: int = 2,
        n_random_features: int = 5,
        **kwargs
    ):
        """
        Initialize the TSP diffusion model.

        Args:
            n_nodes: Number of nodes (also number of positions)
            n_layers: Number of message passing layers
            hidden_dim: Hidden dimension for MLPs
            node_dim: Node feature dimension
            time_dim: Time embedding dimension
            coord_dim: Coordinate dimension (2 for 2D)
            n_random_features: Number of random features for symmetry breaking
        """
        super().__init__()

        self.n_nodes = n_nodes
        self.n_positions = n_nodes  # Number of positions = number of nodes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.n_random_features = n_random_features

        # Input embedding: one-hot position + coordinates + random features
        input_dim = n_nodes + coord_dim + n_random_features

        # Node encoder: from input features to hidden representation
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
        )

        # Time to node modulation
        self.time_to_node = nn.Linear(time_dim, node_dim)

        # Message passing layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GraphAttentionLayer(node_dim, hidden_dim, coord_dim)
            )

        # Output head: from node features to position logits
        self.output_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_nodes),  # Output logits for each position
        )

        # Initialize output to near-uniform
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        coords: torch.Tensor,
        X_t: torch.Tensor,
        timesteps: torch.Tensor,
        rand_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict position logits.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            X_t: Current position assignments (batch, n_nodes) with values 0 to n_nodes-1
            timesteps: Timestep indices (batch,)
            rand_features: Random features for symmetry breaking (batch, n_nodes, n_random_features)

        Returns:
            Position logits (batch, n_nodes, n_positions)
        """
        batch_size = coords.shape[0]
        n_nodes = coords.shape[1]
        device = coords.device

        # Handle extra dimension
        if X_t.dim() == 3:
            X_t = X_t.squeeze(-1)

        # Convert position indices to one-hot
        X_t_long = X_t.long()
        X_one_hot = F.one_hot(X_t_long, num_classes=self.n_nodes).float()  # (batch, n_nodes, n_positions)

        # Generate random features if not provided
        if rand_features is None:
            rand_features = torch.rand(batch_size, n_nodes, self.n_random_features, device=device)

        # Concatenate one-hot position with coordinates and random features
        node_input = torch.cat([X_one_hot, coords, rand_features], dim=-1)

        # Encode nodes
        h = self.node_encoder(node_input)  # (batch, n_nodes, node_dim)

        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))  # (batch, time_dim)
        t_node = self.time_to_node(t_emb)  # (batch, node_dim)

        # Add time modulation to node features
        h = h * (1 + t_node.unsqueeze(1))

        # Apply message passing layers
        for layer in self.layers:
            h = layer(h, coords)

        # Output position logits
        logits = self.output_head(h)  # (batch, n_nodes, n_positions)

        return logits

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample positions from predicted logits.

        Args:
            logits: Position logits (batch, n_nodes, n_positions)
            generator: Random generator

        Returns:
            Tuple of (sampled_positions, log_probs)
        """
        batch_size, n_nodes, n_positions = logits.shape

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        samples = torch.multinomial(
            probs.view(-1, n_positions),
            num_samples=1,
            generator=generator
        ).view(batch_size, n_nodes)

        # Compute log probability of samples
        log_probs = F.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand(-1, n_nodes)
        node_idx = torch.arange(n_nodes, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        sample_log_probs = log_probs[batch_idx, node_idx, samples]

        return samples, sample_log_probs

    def sample_prior(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Sample from the prior distribution (uniform over positions).

        Args:
            shape: (batch_size, n_nodes) or (batch_size, n_nodes, _)
            device: Device to create tensor on
            generator: Random generator

        Returns:
            Sampled position indices (batch, n_nodes)
        """
        if len(shape) >= 2:
            batch_size, n_nodes = shape[0], shape[1]
        else:
            raise ValueError(f"Invalid shape: {shape}")

        # Uniform random positions
        return torch.randint(
            0, self.n_positions,
            (batch_size, n_nodes),
            device=device,
            generator=generator
        )


class DiffusionModelDense(nn.Module):
    """
    Dense diffusion model for TSP.

    This wraps DiffusionModelTSP to match the expected interface.
    """

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 64,
        node_dim: int = 64,
        edge_dim: int = 64,
        time_dim: int = 64,
        n_nodes: int = 20,
        n_random_features: int = 5,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.n_random_features = n_random_features

        self.model = DiffusionModelTSP(
            n_nodes=n_nodes,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            time_dim=time_dim,
            n_random_features=n_random_features,
            **kwargs
        )

    def forward(
        self,
        coords: torch.Tensor,
        X_t: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        rand_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning position logits."""
        return self.model(coords, X_t, timesteps, rand_features=rand_features)

    def sample_prior(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Sample from prior (uniform over positions)."""
        return self.model.sample_prior(shape, device, generator)

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample positions from logits."""
        return self.model.sample_from_logits(logits, generator)


# Backward compatibility aliases
class DiffusionModel(DiffusionModelTSP):
    """Alias for backward compatibility."""
    pass


class DiffusionModelSparse(DiffusionModelDense):
    """Sparse mode (falls back to dense for now)."""
    pass
