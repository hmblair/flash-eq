"""Equivariant normalization layers for spherical tensors.

This module provides normalization components for SO(3)-equivariant networks:
- RepNorm: Compute rotation-invariant norms per irrep
- EquivariantLayerNorm: Equivariant layer normalization
- SeparableEquivariantLayerNorm: Separable normalization (EquiformerV2)

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr
from ..utils import get_epsilon


class RepNorm(nn.Module):
    """Compute norms of spherical tensor components.

    For a spherical tensor with multiple irrep components, computes
    the norm of each component separately. This produces rotation-invariant
    features that can be used for gating or as input to invariant networks.

    Args:
        repr: The representation specifying the tensor structure.
        epsilon: Small constant for numerical stability. If None, uses
            dtype-aware default.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> norm = RepNorm(repr)
        >>> st = torch.randn(32, 9)
        >>> norms = norm(st)  # shape: (32, 3)
    """

    def __init__(self, repr: Repr, epsilon: float | None = None) -> None:
        super().__init__()
        self.num_reps = repr.nreps()
        self._epsilon = epsilon

        # Create indices mapping each dimension to its irrep index
        cdims = repr.cumdims()
        indices = []
        for i in range(self.num_reps):
            size = cdims[i + 1] - cdims[i]
            indices.extend([i] * size)
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))

    def forward(self, st: torch.Tensor) -> torch.Tensor:
        """Compute the norm of each irrep component.

        Uses vectorized scatter_add for GPU efficiency. Adds epsilon before
        sqrt to avoid NaN gradients when components are zero.

        Args:
            st: Spherical tensor of shape (..., dim).

        Returns:
            Norms of shape (..., nreps).
        """
        epsilon = self._epsilon if self._epsilon is not None else get_epsilon(st.dtype)
        sq = st * st

        result = torch.zeros(
            *sq.shape[:-1], self.num_reps,
            device=sq.device, dtype=sq.dtype
        )
        ix = self.indices.expand(sq.shape)  # type: ignore[operator]
        result.scatter_add_(-1, ix, sq)

        return (result + epsilon).sqrt()


class _ChannelLayerNorm(nn.Module):
    """LayerNorm for scalar components, normalizing over channel dimension.

    Standard LayerNorm normalizes over the last dim. This variant normalizes
    over dim=-2 (channel/multiplicity), suitable for scalar irrep components.
    """

    def __init__(self, mult: int, nscalar: int, epsilon: float | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(mult, nscalar))
        self.bias = nn.Parameter(torch.zeros(mult, nscalar))
        self._epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = self._epsilon if self._epsilon is not None else get_epsilon(x.dtype)
        mu = x.mean(dim=-2, keepdim=True)
        sigma = x.std(dim=-2, keepdim=True, unbiased=False)
        return (x - mu) / (sigma + epsilon) * self.weight + self.bias


class _EquivariantRMSNorm(nn.Module):
    """RMS normalization for higher-degree components with per-channel scale.

    Computes a single RMS value across all channels and spatial dimensions,
    then applies per-channel learnable scaling. This preserves relative
    magnitudes between different L>0 degrees.
    """

    def __init__(self, mult: int, epsilon: float | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(mult))
        self._epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = self._epsilon if self._epsilon is not None else get_epsilon(x.dtype)
        rms = torch.sqrt((x ** 2).mean(dim=(-2, -1), keepdim=True) + epsilon)
        return x * (self.weight[..., None] / rms)


class EquivariantLayerNorm(nn.Module):
    """Equivariant layer normalization.

    Normalizes spherical tensors by their irrep norms while
    preserving equivariance. Applies standard LayerNorm to
    the norms across the multiplicity dimension.

    Args:
        repr: The representation of input tensors.
        epsilon: Small constant for numerical stability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> ln = EquivariantLayerNorm(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = ln(x)
    """

    def __init__(
        self,
        repr: Repr,
        epsilon: float | None = None,
    ) -> None:
        super().__init__()

        self.mult = repr.mult
        self.norm = RepNorm(repr)
        # LayerNorm(1) is degenerate, so skip it for mult=1
        self.lnorm = nn.LayerNorm(repr.mult) if repr.mult > 1 else None
        self._epsilon = epsilon  # None means dtype-aware default

        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant layer normalization.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Normalized tensor of shape (..., mult, dim).
        """
        epsilon = self._epsilon if self._epsilon is not None else get_epsilon(f.dtype)

        # Compute norms: shape (..., mult, nreps)
        norms = self.norm(f)

        # For mult=1, just normalize by the norm
        if self.lnorm is None:
            norms_r = 1.0 / (norms + epsilon)
            return f * norms_r[..., self.ix]

        # LayerNorm over mult dimension
        norms_t = norms.transpose(-2, -1)  # (..., nreps, mult)
        lnorms_t = self.lnorm(norms_t)
        lnorms = lnorms_t.transpose(-2, -1)  # (..., mult, nreps)

        # Renormalize features
        norms_r = lnorms / (norms + epsilon)
        return f * norms_r[..., self.ix]


class SeparableEquivariantLayerNorm(nn.Module):
    """Separable equivariant layer normalization (EquiformerV2).

    Unlike standard EquivariantLayerNorm which normalizes each degree
    independently, this version:
    - For L=0 (scalars): applies standard LayerNorm with mean/std and
      learnable scale (γ) and bias (β)
    - For L>0 (higher degrees): normalizes by a shared RMS computed across
      all higher degrees, preserving relative magnitudes between degrees

    This design preserves the relative importance of different degrees,
    which improves force predictions where angular information matters.

    Reference: EquiformerV2 (Liao et al., ICLR 2024), Section 3.4

    Args:
        repr: The representation of input tensors.
        epsilon: Small constant for numerical stability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> ln = SeparableEquivariantLayerNorm(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = ln(x)
    """

    def __init__(
        self,
        repr: Repr,
        epsilon: float | None = None,
    ) -> None:
        super().__init__()

        self.repr = repr

        # Find scalar and higher-degree components
        nscalar, scalar_locs = repr.find_scalar()
        self.scalar_locs = scalar_locs
        self.nscalar = nscalar

        higher_mask = torch.ones(repr.dim(), dtype=torch.bool)
        higher_mask[scalar_locs] = False
        self.register_buffer('higher_mask', higher_mask)
        self.nhigher = repr.nreps() - nscalar

        # Create normalization modules
        if nscalar > 0:
            self.scalar_norm = _ChannelLayerNorm(repr.mult, nscalar, epsilon)

        if self.nhigher > 0:
            self.higher_norm = _EquivariantRMSNorm(repr.mult, epsilon)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply separable equivariant layer normalization.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Normalized tensor of shape (..., mult, dim).
        """
        out = f.clone()

        if self.nscalar > 0:
            out[..., self.scalar_locs] = self.scalar_norm(f[..., self.scalar_locs])

        if self.nhigher > 0:
            out[..., self.higher_mask] = self.higher_norm(f[..., self.higher_mask])  # type: ignore[index]

        return out

    def extra_repr(self) -> str:
        return f"nscalar={self.nscalar}, nhigher={self.nhigher}"
