"""
SO(3)-equivariant edgewise linear layer with distance-dependent weights.

This module provides the main public API for flash-eq: a memory-efficient
equivariant linear layer where weights depend on pairwise distances.

The layer uses binned interpolation for O(num_bins) memory instead of O(batch),
enabling training on large molecular systems.
"""

import torch
import torch.nn as nn
from typing import Optional

from .representations import Repr, ProductRepr
from .block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_binned_interp_cuda,
)


class EquivariantEdgewiseLinear(nn.Module):
    """SO(3)-equivariant linear layer with distance-dependent weights.

    Applies a block-diagonal linear transformation where weights are
    determined by pairwise distances. Uses binned interpolation for
    memory efficiency (O(num_bins) instead of O(batch)).

    The layer includes a radial MLP that learns to map distances to
    block-diagonal weights. Weights are precomputed at bin edges and
    interpolated at runtime.

    Args:
        in_repr: Input representation (Repr object with lvals and mult).
        out_repr: Output representation.
        num_bins: Number of bins for distance interpolation (default: 100).
        min_dist: Minimum distance in Angstroms (default: 0.0).
        max_dist: Maximum distance in Angstroms (default: 10.0).
        radial_hidden: Hidden dimension for radial MLP (default: 64).
        radial_layers: Number of hidden layers in radial MLP (default: 2).

    Example:
        >>> in_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr)
        >>>
        >>> # features: (batch, channels_in, dim_in)
        >>> # distances: (batch,)
        >>> features = torch.randn(1000, 32, 9)  # 9 = 1 + 3 + 5
        >>> distances = torch.rand(1000) * 5.0
        >>> output = layer(features, distances)
        >>> output.shape
        torch.Size([1000, 32, 9])

    Memory usage:
        - Standard approach: O(batch * cout * cin * weight_dim)
        - This layer: O(num_bins * cout * cin * weight_dim)
        - Typical reduction: 5-50x for large batch sizes
    """

    def __init__(
        self,
        in_repr: Repr,
        out_repr: Repr,
        num_bins: int = 100,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        radial_hidden: int = 64,
        radial_layers: int = 2,
    ):
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist

        # Store as submodules so buffers transfer with .to()
        self.add_module('_in_repr', in_repr)
        self.add_module('_out_repr', out_repr)

        # Compute weight dimension from representation product
        product = ProductRepr(in_repr, out_repr)
        self.weight_dim = product.weight_dim()
        self.channels_in = in_repr.mult
        self.channels_out = out_repr.mult

        # Build radial MLP: distance -> weights
        output_dim = self.channels_out * self.channels_in * self.weight_dim
        layers = [nn.Linear(1, radial_hidden), nn.SiLU()]
        for _ in range(radial_layers - 1):
            layers.extend([nn.Linear(radial_hidden, radial_hidden), nn.SiLU()])
        layers.append(nn.Linear(radial_hidden, output_dim))
        self.radial_mlp = nn.Sequential(*layers)

        # Precompute bin edges
        self.register_buffer(
            'bin_edges',
            torch.linspace(min_dist, max_dist, num_bins + 1)
        )

        # Metadata will be built lazily (needs device)
        self._metadata: Optional[tuple] = None
        self._metadata_device: Optional[torch.device] = None

    def _get_metadata(self, device: torch.device):
        """Get or build CUDA metadata for the given device."""
        if self._metadata is None or self._metadata_device != device:
            self._metadata = build_block_metadata(
                self.in_repr.lvals, self.out_repr.lvals, device
            )
            self._metadata_device = device
        return self._metadata

    def forward(
        self,
        features: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            features: (batch, channels_in, dim_in) input features in diagonal basis.
            distances: (batch,) pairwise distances.

        Returns:
            output: (batch, channels_out, dim_out) transformed features.
        """
        device = features.device
        dtype = features.dtype

        # Get metadata for this device
        metadata = self._get_metadata(device)

        # Evaluate radial MLP at bin edges -> (num_bins+1, cout, cin, weight_dim)
        radial_table = self.radial_mlp(self.bin_edges.unsqueeze(-1))
        radial_table = radial_table.view(
            self.num_bins + 1,
            self.channels_out,
            self.channels_in,
            self.weight_dim
        ).to(dtype)

        # Apply block-diagonal multiplication with binned interpolation
        return block_diagonal_binned_interp_cuda(
            features, radial_table, distances, metadata,
            self.min_dist, self.max_dist
        )

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_bins={self.num_bins}, "
            f"dist=[{self.min_dist}, {self.max_dist}]"
        )
