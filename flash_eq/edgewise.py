"""
SO(3)-equivariant edgewise linear layer with distance-dependent weights.

This module provides the main public API for flash-eq: a memory-efficient
equivariant linear layer where weights depend on pairwise distances.

The layer uses binned interpolation for O(num_bins) memory instead of O(batch),
enabling training on large molecular systems.

Supports two usage modes:
1. Diagonal basis: Pass features already in m-diagonalized basis (default)
2. Standard basis: Pass features in standard (ℓ,m) basis with pre-computed P, Q

For multi-layer networks, use WignerDBasis to compute P, Q once and share:

    basis = WignerDBasis(repr_in, repr_out)
    P, Q = basis(directions)

    out1 = layer1(features, distances, P=P, Q=Q)
    out2 = layer2(out1, distances, P=P, Q=Q)
"""

import torch
import torch.nn as nn
from typing import Optional

from .representations import Repr, ProductRepr
from .block_diagonal_cuda import block_diagonal_binned_interp_cuda


class EquivariantEdgewiseLinear(nn.Module):
    """SO(3)-equivariant linear layer with distance-dependent weights.

    Applies a block-diagonal linear transformation where weights are
    determined by pairwise distances. Uses binned interpolation for
    memory efficiency (O(num_bins) instead of O(batch)).

    The layer includes a radial MLP that learns to map distances to
    block-diagonal weights. Weights are precomputed at bin edges and
    interpolated at runtime.

    Supports two modes:
    1. **Diagonal basis mode** (P, Q not provided): Features are assumed
       to already be in the m-diagonalized basis. Output is also in
       diagonal basis.

    2. **Standard basis mode** (P, Q provided): Features are in standard
       spherical harmonic basis. The layer transforms to diagonal basis,
       applies weights, and transforms back.

    For multi-layer networks, compute P, Q once using WignerDBasis and
    pass to all layers to avoid redundant computation.

    Args:
        in_repr: Input representation (Repr object with lvals and mult).
        out_repr: Output representation.
        num_bins: Number of bins for distance interpolation (default: 100).
        min_dist: Minimum distance in Angstroms (default: 0.0).
        max_dist: Maximum distance in Angstroms (default: 10.0).
        radial_hidden: Hidden dimension for radial MLP (default: 64).
        radial_layers: Number of hidden layers in radial MLP (default: 2).

    Example (diagonal basis):
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr)
        >>> features_diag = torch.randn(1000, 32, 9)  # Already in diagonal basis
        >>> output_diag = layer(features_diag, distances)

    Example (standard basis with shared Wigner-D):
        >>> from flash_eq import WignerDBasis
        >>> basis = WignerDBasis(in_repr, out_repr)
        >>> layer1 = EquivariantEdgewiseLinear(in_repr, out_repr)
        >>> layer2 = EquivariantEdgewiseLinear(in_repr, out_repr)
        >>>
        >>> P, Q = basis(directions)  # Compute once
        >>> out1 = layer1(features, distances, P=P, Q=Q)
        >>> out2 = layer2(out1, distances, P=P, Q=Q)

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

        # Compute structure from representation product
        self._product = ProductRepr(in_repr, out_repr)
        self.weight_dim = self._product.weight_dim()
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
            self._metadata = self._product.build_block_metadata(device)
            self._metadata_device = device
        return self._metadata

    def forward(
        self,
        features: torch.Tensor,
        distances: torch.Tensor,
        P: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            features: (batch, channels_in, dim_in) input features.
                If P, Q provided: features in standard (ℓ,m) basis.
                If P, Q not provided: features in m-diagonalized basis.
            distances: (batch,) pairwise distances.
            P: (batch, dim_in, dim_in) input basis matrix from WignerDBasis.
                If provided, transforms features from standard to diagonal basis.
            Q: (batch, dim_out, dim_out) output basis matrix from WignerDBasis.
                If provided, transforms output from diagonal to standard basis.

        Returns:
            output: (batch, channels_out, dim_out) transformed features.
                In same basis as input (standard if P, Q provided, diagonal otherwise).

        Note:
            P and Q must both be provided or both be None.
        """
        if (P is None) != (Q is None):
            raise ValueError("P and Q must both be provided or both be None")

        device = features.device
        dtype = features.dtype

        # Get metadata for this device
        metadata = self._get_metadata(device)

        # Transform to diagonal basis if P provided
        if P is not None:
            # features: (batch, channels, dim) -> need to apply P^T per channel
            # P: (batch, dim, dim)
            # f_diag = P^T @ f for each channel
            f_diag = torch.einsum('bji,bci->bcj', P, features)
        else:
            f_diag = features

        # Evaluate radial MLP at bin edges -> (num_bins+1, cout, cin, weight_dim)
        radial_table = self.radial_mlp(self.bin_edges.unsqueeze(-1))
        radial_table = radial_table.view(
            self.num_bins + 1,
            self.channels_out,
            self.channels_in,
            self.weight_dim
        ).to(dtype)

        # Apply block-diagonal multiplication with binned interpolation
        out_diag = block_diagonal_binned_interp_cuda(
            f_diag, radial_table, distances, metadata,
            self.min_dist, self.max_dist
        )

        # Transform back to standard basis if Q provided
        if Q is not None:
            # out_diag: (batch, channels, dim_out)
            # Q: (batch, dim_out, dim_out)
            # output = Q @ out_diag for each channel
            return torch.einsum('bij,bcj->bci', Q, out_diag)
        else:
            return out_diag

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_bins={self.num_bins}, "
            f"dist=[{self.min_dist}, {self.max_dist}]"
        )
