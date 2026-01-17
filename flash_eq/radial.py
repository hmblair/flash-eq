"""Radial weight networks for distance-dependent equivariant operations.

This module provides neural networks that map edge features (typically distances
or radial basis functions) to tensor product weights for equivariant operations.

Classes:
    RadialWeight: General radial weight network (edge features → weights)
    BinnedRadialWeight: Binned radial weights with interpolation (memory efficient)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RadialWeight(nn.Module):
    """Compute tensor product weights from invariant edge features.

    A neural network that maps edge features to weights for tensor product
    contractions. Used in equivariant message passing networks.

    For flash-eq, the output is a weight matrix applied to block-diagonal
    features. The num_basis corresponds to weight_dim (number of independent
    block-diagonal weight parameters).

    Args:
        edge_dim: Dimension of input edge features.
        hidden_dim: Hidden layer dimension.
        num_basis: Number of weight elements (weight_dim for block-diagonal).
        in_mult: Input multiplicity (channels_in).
        out_mult: Output multiplicity (channels_out).
        num_layers: Number of hidden layers (default: 2).
        dropout: Dropout probability.

    Example:
        >>> weight_net = RadialWeight(16, 32, num_basis=44, in_mult=8, out_mult=8)
        >>> edge_features = torch.randn(100, 16)
        >>> weights = weight_net(edge_features)  # (100, 8, 8, 44)
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        num_basis: int,
        in_mult: int,
        out_mult: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.in_mult = in_mult
        self.out_mult = out_mult

        # Build MLP: edge_dim -> hidden -> ... -> (out_mult * in_mult * num_basis)
        output_dim = out_mult * in_mult * num_basis
        layers = [nn.Linear(edge_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weights from edge features.

        Args:
            x: Edge features of shape (..., edge_dim).

        Returns:
            Weights of shape (..., out_mult, in_mult, num_basis).
        """
        *batch_dims, _ = x.shape
        out = self.mlp(x)
        return out.view(*batch_dims, self.out_mult, self.in_mult, self.num_basis)


class BinnedRadialWeight(nn.Module):
    """Binned radial weights with interpolation for memory efficiency.

    Instead of computing weights per-edge, this module precomputes weights
    at bin edges and interpolates at runtime. This reduces memory from
    O(edges) to O(bins) while maintaining accuracy.

    The forward pass returns the weight table at bin edges. Use
    interpolate_weights() to get per-edge weights.

    Args:
        hidden_dim: Hidden layer dimension.
        num_basis: Number of weight elements (weight_dim for block-diagonal).
        in_mult: Input multiplicity (channels_in).
        out_mult: Output multiplicity (channels_out).
        num_bins: Number of distance bins (default: 100).
        min_dist: Minimum distance (default: 0.0).
        max_dist: Maximum distance (default: 10.0).
        num_layers: Number of hidden layers (default: 2).

    Example:
        >>> weight_net = BinnedRadialWeight(32, num_basis=44, in_mult=8, out_mult=8)
        >>> weight_table = weight_net()  # (101, 8, 8, 44)
        >>> distances = torch.rand(500) * 5.0
        >>> per_edge_weights = weight_net.interpolate(weight_table, distances)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_basis: int,
        in_mult: int,
        out_mult: int,
        num_bins: int = 100,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.in_mult = in_mult
        self.out_mult = out_mult
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist

        # Precompute bin edges
        self.register_buffer(
            'bin_edges',
            torch.linspace(min_dist, max_dist, num_bins + 1)
        )

        # Build MLP: distance (1-dim) -> weights
        output_dim = out_mult * in_mult * num_basis
        layers = [nn.Linear(1, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(_init_weights)

    def forward(self) -> torch.Tensor:
        """Compute weight table at bin edges.

        Returns:
            Weight table of shape (num_bins+1, out_mult, in_mult, num_basis).
        """
        # Evaluate MLP at bin edges
        weights = self.mlp(self.bin_edges.unsqueeze(-1))
        return weights.view(
            self.num_bins + 1,
            self.out_mult,
            self.in_mult,
            self.num_basis
        )

    def interpolate(
        self,
        weight_table: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate weights from table at given distances.

        Args:
            weight_table: (num_bins+1, out_mult, in_mult, num_basis) from forward()
            distances: (num_edges,) edge distances

        Returns:
            Interpolated weights of shape (num_edges, out_mult, in_mult, num_basis).
        """
        # Normalize distances to [0, num_bins]
        bin_width = (self.max_dist - self.min_dist) / self.num_bins
        normalized = (distances - self.min_dist) / bin_width

        # Clamp to valid range
        normalized = normalized.clamp(0, self.num_bins - 1e-6)

        # Get bin indices and interpolation weights
        bin_idx = normalized.floor().long()
        alpha = (normalized - bin_idx.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Linear interpolation: w = (1-alpha) * w[bin] + alpha * w[bin+1]
        w0 = weight_table[bin_idx]
        w1 = weight_table[bin_idx + 1]

        return w0 + alpha * (w1 - w0)

    def get_bin_info(self, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get bin indices and interpolation weights for sorting optimization.

        Args:
            distances: (num_edges,) edge distances

        Returns:
            bin_indices: (num_edges,) integer bin indices
            alphas: (num_edges,) interpolation weights in [0, 1)
        """
        bin_width = (self.max_dist - self.min_dist) / self.num_bins
        normalized = (distances - self.min_dist) / bin_width
        normalized = normalized.clamp(0, self.num_bins - 1e-6)

        bin_indices = normalized.floor().long()
        alphas = normalized - bin_indices.float()

        return bin_indices, alphas
