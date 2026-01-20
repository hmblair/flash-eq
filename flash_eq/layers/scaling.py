"""Distance-dependent scaling for equivariant features.

This module provides scaling operations that follow the solid harmonic
structure, where higher angular momentum components are naturally suppressed
at short distances.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr


class SolidHarmonicScaling(nn.Module):
    """Scale features by (r / (r + scale))^l following solid harmonic structure.

    The solid harmonics r^l Y_l^m naturally vanish at r=0 for l>0. This module
    applies analogous distance-dependent scaling to equivariant features:
    - l=0 (scalars): unchanged (weight^0 = 1)
    - l>0: smoothly suppressed as distance decreases

    This provides:
    - Smooth, differentiable behavior as points pass close to each other
    - Principled handling of near-zero edge vectors (e.g., during diffusion)
    - No discontinuities or special-case logic

    Args:
        repr: The representation defining the l-value structure.
        scale: Length scale parameter. At distance=scale, the weight is 0.5.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> scaling = SolidHarmonicScaling(repr, scale=1.0)
        >>> features = torch.randn(100, 8, 9)  # (edges, mult, dim)
        >>> distances = torch.rand(100) * 5    # (edges,)
        >>> scaled = scaling(features, distances)
    """

    def __init__(self, repr: Repr, scale: float = 1.0) -> None:
        super().__init__()
        self.repr = repr
        self.scale = scale
        # Cache l-values per dimension for fast vectorized scaling
        self.register_buffer('_l_indices', repr.l_indices())

    def forward(
        self,
        features: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Apply distance-dependent scaling to features.

        Args:
            features: (..., mult, dim) equivariant features.
            distances: (...,) distances (same leading dims as features[:-2]).

        Returns:
            Scaled features with same shape as input.
        """
        # Compute weight in [0, 1): approaches 1 for large distances
        weight = distances / (distances + self.scale)

        # Raise to power of l for each component: weight^l
        # Shape: (..., dim)
        scaling = weight[..., None].pow(self._l_indices)

        # Apply scaling: (..., mult, dim) * (..., 1, dim)
        return features * scaling[..., None, :]

    def extra_repr(self) -> str:
        return f"repr={self.repr}, scale={self.scale}"
