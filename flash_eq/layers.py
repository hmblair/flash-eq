"""Equivariant neural network layers for spherical tensors.

This module provides basic building blocks for SO(3)-equivariant networks:
- RepNorm: Compute rotation-invariant norms per irrep
- EquivariantLinear: Linear layer preserving lvals, changing multiplicity
- EquivariantGating: Norm-based gating nonlinearity
- EquivariantLayerNorm: Equivariant layer normalization
- GraphPooling: Aggregate edge features to nodes (sum/mean/max)

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
        ix = self.indices.expand(sq.shape)  # type: ignore[operator]
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
        self.bias: nn.Parameter | None

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
        ix = self.indices.expand(*b, *self.expanddims)  # type: ignore[operator]
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


def _expand_indices(dst_indices: torch.Tensor, channels: int, dim: int) -> torch.Tensor:
    """Expand (num_edges,) indices to (num_edges, channels, dim) for scatter."""
    return dst_indices.view(-1, 1, 1).expand(-1, channels, dim)


def _pool_sum(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sum pooling: aggregate edge features to nodes via summation."""
    _, channels, dim = edge_features.shape
    idx = _expand_indices(dst_indices, channels, dim)
    out = torch.zeros(num_nodes, channels, dim, device=edge_features.device, dtype=edge_features.dtype)
    return out.scatter_add_(0, idx, edge_features)


def _pool_mean(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean pooling: aggregate edge features to nodes via averaging."""
    out = _pool_sum(edge_features, dst_indices, num_nodes)

    # Count edges per node and divide
    counts = torch.zeros(num_nodes, device=out.device, dtype=out.dtype)
    counts.scatter_add_(0, dst_indices, torch.ones_like(dst_indices, dtype=out.dtype))
    return out / counts.clamp(min=1).view(-1, 1, 1)


def _pool_max(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Max pooling: aggregate edge features to nodes via maximum."""
    _, channels, dim = edge_features.shape
    idx = _expand_indices(dst_indices, channels, dim)
    out = torch.full((num_nodes, channels, dim), float('-inf'), device=edge_features.device, dtype=edge_features.dtype)
    out.scatter_reduce_(0, idx, edge_features, reduce='amax')
    return out.nan_to_num(neginf=0.0)


_POOL_FUNCTIONS = {
    'sum': _pool_sum,
    'mean': _pool_mean,
    'max': _pool_max,
}


class GraphPooling(nn.Module):
    """Aggregate edge features to nodes.

    Supports sum, mean, and max pooling. Uses PyTorch's scatter operations
    for GPU-efficient aggregation without external graph library dependencies.

    Args:
        reduce: Aggregation method ('sum', 'mean', or 'max').

    Example:
        >>> pool = GraphPooling(reduce='sum')
        >>> edge_features = torch.randn(1000, 32, 9)  # 1000 edges, 32 channels, dim=9
        >>> dst_indices = torch.randint(0, 100, (1000,))  # 100 destination nodes
        >>> node_features = pool(edge_features, dst_indices, num_nodes=100)
        >>> node_features.shape
        torch.Size([100, 32, 9])
    """

    def __init__(self, reduce: str = 'sum') -> None:
        super().__init__()
        if reduce not in _POOL_FUNCTIONS:
            raise ValueError(f"reduce must be one of {list(_POOL_FUNCTIONS.keys())}, got '{reduce}'")
        self.reduce = reduce
        self._pool_fn = _POOL_FUNCTIONS[reduce]

    def forward(
        self,
        edge_features: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate edge features to destination nodes.

        Args:
            edge_features: (num_edges, channels, dim) edge feature tensor.
            dst_indices: (num_edges,) destination node index for each edge.
            num_nodes: Total number of nodes in the graph.

        Returns:
            node_features: (num_nodes, channels, dim) aggregated node features.
        """
        return self._pool_fn(edge_features, dst_indices, num_nodes)

    def extra_repr(self) -> str:
        return f"reduce='{self.reduce}'"
