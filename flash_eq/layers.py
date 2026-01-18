"""Equivariant neural network layers for spherical tensors.

This module provides basic building blocks for SO(3)-equivariant networks:
- RepNorm: Compute rotation-invariant norms per irrep
- EquivariantLinear: Linear layer preserving lvals, changing multiplicity
- EquivariantGating: Norm-based gating nonlinearity
- EquivariantLayerNorm: Equivariant layer normalization

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .representations import Repr
from .utils import get_epsilon


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RepNorm(nn.Module):
    """Compute norms of spherical tensor components.

    For a spherical tensor with multiple irrep components, computes
    the norm of each component separately. This produces rotation-invariant
    features that can be used for gating or as input to invariant networks.

    Args:
        repr: The representation specifying the tensor structure.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> norm = RepNorm(repr)
        >>> st = torch.randn(32, 9)
        >>> norms = norm(st)  # shape: (32, 3)
    """

    def __init__(self, repr: Repr) -> None:
        super().__init__()
        self.num_reps = repr.nreps()

        # Create indices mapping each dimension to its irrep index
        cdims = repr.cumdims()
        indices = []
        for i in range(self.num_reps):
            size = cdims[i + 1] - cdims[i]
            indices.extend([i] * size)
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))

    def forward(self, st: torch.Tensor) -> torch.Tensor:
        """Compute the norm of each irrep component.

        Uses vectorized scatter_add for GPU efficiency.

        Args:
            st: Spherical tensor of shape (..., dim).

        Returns:
            Norms of shape (..., nreps).
        """
        sq = st * st

        result = torch.zeros(
            *sq.shape[:-1], self.num_reps,
            device=sq.device, dtype=sq.dtype
        )
        ix = self.indices.expand(sq.shape)
        result.scatter_add_(-1, ix, sq)

        return result.sqrt()


class EquivariantLinear(nn.Module):
    """Linear layer that preserves spherical tensor structure.

    Applies separate linear transformations to each irrep degree,
    preserving SO(3) equivariance. Only degree-0 (scalar) components
    can have bias terms.

    Args:
        in_repr: Input representation.
        out_repr: Output representation (must have same lvals as in_repr).
        bias: Whether to include bias for scalar components.
        activation: Activation function (applied only to scalars).

    Raises:
        ValueError: If in_repr and out_repr have different lvals.

    Example:
        >>> repr_in = Repr(lvals=[0, 1, 2], mult=8)
        >>> repr_out = Repr(lvals=[0, 1, 2], mult=16)
        >>> layer = EquivariantLinear(repr_in, repr_out)
        >>> x = torch.randn(32, 8, 9)
        >>> y = layer(x)  # shape: (32, 16, 9)
    """

    def __init__(
        self,
        in_repr: Repr,
        out_repr: Repr,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if not torch.equal(in_repr.lvals, out_repr.lvals):
            raise ValueError(
                "EquivariantLinear cannot modify the degrees of a representation. "
                f"Got input lvals={in_repr.lvals.tolist()}, output lvals={out_repr.lvals.tolist()}"
            )

        self.in_repr = in_repr
        self.out_repr = out_repr

        # Weight matrix for linear transformation
        self.weight = nn.Parameter(
            torch.empty(in_repr.nreps() * out_repr.mult, in_repr.mult)
        )
        nn.init.xavier_uniform_(self.weight)

        # Indices for gathering correct degrees
        indices = torch.tensor(in_repr.indices(), dtype=torch.long)
        self.register_buffer('indices', indices)

        self.expanddims = (1, out_repr.mult, in_repr.dim())
        self.outdims = (in_repr.nreps(), out_repr.mult, in_repr.dim())

        # Bias only for scalar (degree-0) components
        nscalar, scalar_locs = in_repr.find_scalar()
        self.scalar_locs = scalar_locs

        if nscalar > 0 and bias:
            self.bias = nn.Parameter(
                torch.zeros(out_repr.mult, nscalar),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Transformed tensor of shape (..., out_mult, dim).
        """
        GATHER_DIM = -3

        *b, _, _ = f.shape

        # Apply linear transformation
        out = (self.weight @ f).view(*b, *self.outdims)

        # Gather components for each degree
        ix = self.indices.expand(*b, *self.expanddims)
        out = out.gather(dim=GATHER_DIM, index=ix).squeeze(GATHER_DIM)

        # Add bias and activation to scalar components
        if self.bias is not None:
            out = out.clone()
            scalar_out = self.activation(out[..., self.scalar_locs] + self.bias)
            out[..., self.scalar_locs] = scalar_out.to(out.dtype)

        return out


class EquivariantGating(nn.Module):
    """Norm-based gating for spherical tensors.

    Computes norms of each irrep component, processes them through
    a linear layer and sigmoid, then uses the result to gate the
    original tensor. This provides a learnable, equivariant nonlinearity.

    Args:
        repr: The representation of input tensors.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> gate = EquivariantGating(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = gate(x)  # shape: (32, 8, 9)
    """

    def __init__(self, repr: Repr) -> None:
        super().__init__()

        self.repr = repr
        self.norm = RepNorm(repr)

        # Linear layer for processing norms
        self.linear = nn.Linear(
            repr.nreps() * repr.mult,
            repr.nreps() * repr.mult,
        )

        # Indices mapping norms back to full dimension
        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

        self.outdims = (repr.mult, repr.nreps())
        self.activation = nn.Sigmoid()

        self.apply(_init_weights)

    def forward(self, st: torch.Tensor) -> torch.Tensor:
        """Apply gating to spherical tensor.

        Args:
            st: Spherical tensor of shape (..., mult, dim).

        Returns:
            Gated tensor of shape (..., mult, dim).
        """
        # Compute norms
        norms = self.norm(st)

        *b, _, _ = norms.size()
        norms = norms.flatten(-2, -1)

        # Process through linear layer
        norms = self.linear(norms).view(*b, *self.outdims)

        # Apply activation
        norms = self.activation(norms)

        # Gate the input
        return st * norms[..., self.ix]


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
