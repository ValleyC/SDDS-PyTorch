"""
Diffusion Model wrapper for SDDS-PyTorch.

This module wraps the EGNN encoder with diffusion-specific functionality,
including edge probability prediction, sampling, and log probability computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .Modules.GNNModules import EGNNEncoder, EGNNEncoderDense


class DiffusionModel(nn.Module):
    """
    Diffusion model for discrete edge prediction.

    This wraps an EGNN encoder to predict edge probabilities at each diffusion step.
    Supports both dense and sparse graph representations.
    """

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 64,
        node_dim: int = 64,
        edge_dim: int = 64,
        time_dim: int = 64,
        coord_dim: int = 2,
        out_channels: int = 2,
        use_dense: bool = True,
        **kwargs
    ):
        """
        Initialize the diffusion model.

        Args:
            n_layers: Number of EGNN layers
            hidden_dim: Hidden dimension for EGNN layers
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            time_dim: Time embedding dimension
            coord_dim: Coordinate dimension (2 for 2D)
            out_channels: Output channels (2 for binary classification)
            use_dense: Whether to use dense mode
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.coord_dim = coord_dim
        self.out_channels = out_channels
        self.use_dense = use_dense

        # EGNN encoder (handles time embedding internally)
        if use_dense:
            self.encoder = EGNNEncoderDense(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                coord_dim=coord_dim,
                out_channels=out_channels,
                **kwargs
            )
        else:
            self.encoder = EGNNEncoder(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                coord_dim=coord_dim,
                out_channels=out_channels,
                sparse=True,
                **kwargs
            )

    def forward(
        self,
        coords: torch.Tensor,
        adj_matrix: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict edge logits.

        Args:
            coords: Node coordinates, shape (batch, N, 2) for dense or (N, 2) for sparse
            adj_matrix: Current edge values, shape (batch, N, N) for dense
            timesteps: Timestep indices, shape (batch_size,)
            edge_index: Edge indices for sparse mode, shape (2, E)

        Returns:
            Edge logits of shape (batch, N, N, 2) for dense or (E, 2) for sparse
        """
        return self.encoder(coords, adj_matrix, timesteps, edge_index)

    def get_edge_probs(
        self,
        coords: torch.Tensor,
        adj_matrix: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get edge probabilities (softmax of logits).

        Returns:
            Probabilities of shape (batch, N, N, 2) for dense or (E, 2) for sparse
        """
        logits = self.forward(coords, adj_matrix, timesteps, edge_index)
        return F.softmax(logits, dim=-1)

    def get_edge_prediction(
        self,
        coords: torch.Tensor,
        adj_matrix: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary edge predictions.

        Returns:
            Binary predictions of shape (batch, N, N) for dense
        """
        probs = self.get_edge_probs(coords, adj_matrix, timesteps, edge_index)
        # Take probability of edge presence (class 1)
        return (probs[..., 1] > threshold).float()

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample edges from predicted logits.

        Args:
            logits: Edge logits of shape (..., 2)
            generator: Random generator

        Returns:
            Tuple of (sampled_edges, log_probs)
        """
        probs = F.softmax(logits, dim=-1)
        # Sample from Bernoulli using probability of edge (class 1)
        p_edge = probs[..., 1]
        samples = torch.bernoulli(p_edge, generator=generator)

        # Compute log probability of samples
        log_probs = torch.where(
            samples == 1,
            torch.log(torch.clamp(p_edge, min=1e-8)),
            torch.log(torch.clamp(1 - p_edge, min=1e-8))
        )

        return samples, log_probs

    def sample_prior(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Sample from the prior distribution (uniform Bernoulli).

        Args:
            shape: Shape of edges to sample
            device: Device to create tensor on
            generator: Random generator

        Returns:
            Sampled binary edges
        """
        return torch.bernoulli(
            torch.full(shape, 0.5, device=device),
            generator=generator
        )


class DiffusionModelDense(DiffusionModel):
    """Convenience class for dense mode diffusion model."""

    def __init__(self, **kwargs):
        kwargs['use_dense'] = True
        super().__init__(**kwargs)


class DiffusionModelSparse(DiffusionModel):
    """Convenience class for sparse mode diffusion model."""

    def __init__(self, **kwargs):
        kwargs['use_dense'] = False
        super().__init__(**kwargs)
