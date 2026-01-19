"""Equivariant graph attention layers.

Implements attention from EquiformerV2 (Liao et al., ICLR 2024) Section 3.2.
Operates on edge features in standard spherical harmonic basis.

Pipeline:
    edge_features = EquivariantEdgewiseLinear(...)  # (E, mult, dim)
    edge_features = EquivariantEdgeAttention(...)   # (E, mult, dim)
    node_features = GraphPooling(...)               # (N, mult, dim)

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr
from .linear import EquivariantEdgewiseLinear
from .pooling import GraphPooling


class EquivariantEdgeAttention(nn.Module):
    """Attention mechanism for equivariant edge features.

    Computes scalar attention weights from l=0 components and applies
    them to full equivariant features. The softmax normalizes over all
    edges with the same destination node (i.e., over neighbors).

    Features:
        - Attention re-normalization (LayerNorm on scalars before softmax)
        - Multi-head attention (splits multiplicity dimension)
        - Operates in standard SH basis (after Q transform)

    Args:
        repr: Representation of edge features (must include l=0).
        num_heads: Number of attention heads. Must divide repr.mult.
        use_layer_norm: Apply LayerNorm to scalars before attention (recommended).
        dropout: Dropout rate for attention weights.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> attn = EquivariantEdgeAttention(repr, num_heads=8)
        >>>
        >>> # edge_features from EquivariantEdgewiseLinear
        >>> edge_features = torch.randn(1000, 32, 9)  # (E, mult, dim)
        >>> dst_indices = torch.randint(0, 100, (1000,))
        >>>
        >>> weighted = attn(edge_features, dst_indices, num_nodes=100)
    """

    _scalar_locs: torch.Tensor

    def __init__(
        self,
        repr: Repr,
        num_heads: int = 1,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if repr.mult % num_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must divide mult ({repr.mult})")

        self.repr = repr
        self.num_heads = num_heads
        self.head_dim = repr.mult // num_heads

        # Locate scalar (l=0) components
        nscalar, scalar_locs = repr.find_scalar()
        if nscalar == 0:
            raise ValueError(
                f"Representation must include l=0 for attention. "
                f"Got lvals={repr.lvals.tolist()}"
            )
        self.nscalar = nscalar
        self.register_buffer('_scalar_locs', torch.tensor(scalar_locs, dtype=torch.long))

        # Scalar feature dimension: mult * nscalar
        scalar_dim = repr.mult * nscalar
        head_scalar_dim = self.head_dim * nscalar

        # Attention re-normalization (Section 3.2)
        self.layer_norm = nn.LayerNorm(scalar_dim) if use_layer_norm else nn.Identity()

        # Attention projection: scalars -> logit
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_proj = nn.Linear(head_scalar_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_proj.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        edge_features: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Apply attention to edge features.

        Args:
            edge_features: (num_edges, mult, dim) edge features in standard SH basis.
            dst_indices: (num_edges,) destination node index for each edge.
            num_nodes: Total number of destination nodes.

        Returns:
            Attention-weighted edge features, shape (num_edges, mult, dim).
        """
        E, mult, dim = edge_features.shape

        # Extract scalar (l=0) features: (E, mult, nscalar)
        scalars = edge_features[..., self._scalar_locs]

        # Flatten for LayerNorm: (E, mult * nscalar)
        scalars_flat = scalars.reshape(E, -1)

        # Attention re-normalization
        scalars_flat = self.layer_norm(scalars_flat)

        # Reshape for multi-head: (E, num_heads, head_dim * nscalar)
        scalars_heads = scalars_flat.view(E, self.num_heads, -1)

        # Compute attention logits
        logits = self.attn_proj(self.leaky_relu(scalars_heads))  # (E, H, 1)
        logits = logits.squeeze(-1)  # (E, H)

        # Softmax over neighbors (edges to same destination)
        attn_weights = self._neighbor_softmax(logits, dst_indices, num_nodes)
        attn_weights = self.dropout(attn_weights)  # (E, H)

        # Apply attention weights to features
        return self._apply_attention(edge_features, attn_weights)

    def _neighbor_softmax(
        self,
        logits: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Softmax over edges sharing the same destination node.

        Args:
            logits: (E, H) attention logits per edge per head.
            dst_indices: (E,) destination node for each edge.
            num_nodes: Total number of nodes.

        Returns:
            Normalized attention weights, shape (E, H).
        """
        E, H = logits.shape
        idx = dst_indices.unsqueeze(-1).expand(E, H)

        # Subtract max for numerical stability
        max_logits = torch.full(
            (num_nodes, H), float('-inf'),
            device=logits.device, dtype=logits.dtype
        )
        max_logits.scatter_reduce_(0, idx, logits, reduce='amax', include_self=False)
        logits = logits - max_logits.gather(0, idx)

        # Exponentiate and sum
        exp_logits = torch.exp(logits)
        sum_exp = torch.zeros(num_nodes, H, device=exp_logits.device, dtype=exp_logits.dtype)
        sum_exp.scatter_add_(0, idx, exp_logits)

        # Normalize
        return exp_logits / (sum_exp.gather(0, idx) + 1e-8)

    def _apply_attention(
        self,
        features: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply features by attention weights.

        Args:
            features: (E, mult, dim) edge features.
            attn_weights: (E, H) attention weights per head.

        Returns:
            Weighted features, shape (E, mult, dim).
        """
        E, mult, dim = features.shape

        if self.num_heads == 1:
            # Single head: broadcast weight over all channels
            return features * attn_weights.view(E, 1, 1)

        # Multi-head: each head weights its portion of channels
        features = features.view(E, self.num_heads, self.head_dim, dim)
        attn_weights = attn_weights.view(E, self.num_heads, 1, 1)
        return (features * attn_weights).view(E, mult, dim)

    def extra_repr(self) -> str:
        return (
            f"repr={self.repr}, num_heads={self.num_heads}, "
            f"nscalar={self.nscalar}"
        )


class EquivariantAttention(nn.Module):
    """SO(3)-equivariant graph attention layer.

    Combines the full message-passing pipeline:
        1. EquivariantEdgewiseLinear: Transform node features to edge features
        2. EquivariantEdgeAttention: Apply attention weighting on edges
        3. GraphPooling: Aggregate edge features back to nodes

    This is the primary building block for equivariant graph neural networks,
    implementing the attention mechanism from EquiformerV2 (Liao et al., 2024).

    Args:
        in_repr: Input representation (Repr with lvals and mult).
        out_repr: Output representation.
        num_heads: Number of attention heads. Must divide out_repr.mult.
        num_bins: Number of distance bins for radial weight interpolation.
        num_bases: Number of radial basis functions. If None, uses independent
            weights per bin. If set (e.g., 16), uses radial basis functions
            for parameter efficiency (recommended for high L).
        min_dist: Minimum distance in Angstroms.
        max_dist: Maximum distance in Angstroms.
        log_bins: If True, use logarithmic bin spacing (density ~ 1/r).
        sigma: Gaussian smoothing kernel width for radial weights.
            Only used when num_bases=None.
        use_layer_norm: Apply LayerNorm to scalars before attention.
        dropout: Dropout rate for attention weights.
        reduce: Pooling reduction method ('sum', 'mean', or 'max').

    Example:
        >>> from flash_eq import Repr, WignerDBasis, EquivariantAttention
        >>>
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> layer = EquivariantAttention(repr, repr, num_heads=4).cuda()
        >>> basis = WignerDBasis([repr, repr]).cuda()
        >>>
        >>> # Graph data
        >>> P, Q = basis(directions)
        >>> output = layer(P, Q, node_features, distances, src, dst, num_nodes)
    """

    def __init__(
        self,
        in_repr: Repr,
        out_repr: Repr,
        num_heads: int = 1,
        num_bins: int = 100,
        num_bases: int | None = None,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        log_bins: bool = False,
        sigma: float = 1.0,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        reduce: str = 'sum',
    ):
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr

        # Edgewise linear transformation
        self.linear = EquivariantEdgewiseLinear(
            in_repr=in_repr,
            out_repr=out_repr,
            num_bins=num_bins,
            num_bases=num_bases,
            min_dist=min_dist,
            max_dist=max_dist,
            log_bins=log_bins,
            sigma=sigma,
        )

        # Edge attention
        self.attention = EquivariantEdgeAttention(
            repr=out_repr,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # Aggregation
        self.pool = GraphPooling(reduce=reduce)

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        node_features: torch.Tensor,
        distances: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Apply equivariant attention layer.

        Args:
            P: (num_edges, dim_in, dim_in) input basis from WignerDBasis.
            Q: (num_edges, dim_out, dim_out) output basis from WignerDBasis.
            node_features: (num_nodes, channels_in, dim_in) node features.
            distances: (num_edges,) edge distances.
            src_indices: (num_edges,) source node for each edge.
            dst_indices: (num_edges,) destination node for each edge.
            num_nodes: Total number of nodes.

        Returns:
            output: (num_nodes, channels_out, dim_out) updated node features.
        """
        # Transform to edge features
        edge_features = self.linear(P, Q, node_features, distances, src_indices)

        # Apply attention weighting
        edge_features = self.attention(edge_features, dst_indices, num_nodes)

        # Aggregate to nodes
        return self.pool(edge_features, dst_indices, num_nodes)

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_heads={self.attention.num_heads}, reduce='{self.pool.reduce}'"
        )
