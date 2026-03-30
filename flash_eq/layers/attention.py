"""Equivariant graph attention layers.

Implements attention from EquiformerV2 (Liao et al., ICLR 2024) Section 3.2.
Uses full equivariant Q/K projections - dot product of equivariant features is invariant.

Pipeline:
    Q = EquivariantLinear(norm(node_features))[dst]
    K = EquivariantLinear(norm(node_features))[src]
    V = EquivariantEdgewiseLinear(node_features[src])
    output = GraphPooling(attention(Q, K, V))

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..graph import Graph
from ..representations import Repr
from .linear import EquivariantLinear, EquivariantEdgewiseLinear
from .norm import EquivariantLayerNorm
from .pooling import GraphPooling


class EquivariantEdgeAttention(nn.Module):
    """Low-level attention mechanism for pre-computed Q, K, V tensors.

    Takes Q, K, V tensors and applies scaled dot-product attention with
    neighbor softmax (normalizing over edges to the same destination).

    Args:
        num_heads: Number of attention heads.
        qk_dim: Dimension of Q/K per head (for scaling).
        dropout: Dropout rate for attention weights.
        distance_decay_scale: If provided, add distance decay bias to attention
            logits: logits -= distance / scale. This encourages the model to
            attend more to nearby neighbors. Typical values: 2.0-5.0 Angstroms.

    Example:
        >>> attn = EquivariantEdgeAttention(num_heads=4, qk_dim=36)
        >>> graph = Graph.random(num_nodes=100, num_edges=1000)
        >>> Q = torch.randn(1000, 4, 36)  # (E, H, qk_dim)
        >>> K = torch.randn(1000, 4, 36)  # (E, H, qk_dim)
        >>> V = torch.randn(1000, 32, 9)  # (E, mult, dim)
        >>> output = attn(Q, K, V, graph)
    """

    def __init__(
        self,
        num_heads: int,
        qk_dim: int,
        dropout: float = 0.0,
        distance_decay_scale: float | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.distance_decay_scale = distance_decay_scale

        # Learnable temperature for attention (stored as log for numerical stability)
        init_scale = qk_dim ** -0.5
        self.log_scale = nn.Parameter(torch.tensor(init_scale).log())

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        graph: Graph,
        distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention to edge features.

        Args:
            Q: (num_edges, num_heads, qk_dim) query features from dst nodes.
            K: (num_edges, num_heads, qk_dim) key features from src nodes.
            V: (num_edges, mult, dim) value features (edge features).
            graph: Graph containing edge indices and node count.
            distances: Optional (num_edges,) edge distances for distance decay.

        Returns:
            Attention-weighted edge features, shape (num_edges, mult, dim).
        """
        # Scaled dot-product attention
        logits = (Q * K).sum(-1) * self.scale  # (E, H)

        # Distance decay bias: logits -= distance / scale
        # Equivalent to attention *= exp(-distance / scale)
        if self.distance_decay_scale is not None and distances is not None:
            decay_bias = distances.unsqueeze(-1) / self.distance_decay_scale  # (E, 1)
            logits = logits - decay_bias

        # Softmax over neighbors (edges to same destination)
        attn_weights = self._neighbor_softmax(logits, graph.dst, graph.num_nodes)
        attn_weights = self.dropout(attn_weights)  # (E, H)

        # Apply attention weights to values
        return self._apply_attention(V, attn_weights)

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
        V: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply values by attention weights.

        Args:
            V: (E, mult, dim) edge features.
            attn_weights: (E, H) attention weights per head.

        Returns:
            Weighted features, shape (E, mult, dim).
        """
        E, mult, dim = V.shape

        if self.num_heads == 1:
            # Single head: broadcast weight over all channels
            return V * attn_weights.view(E, 1, 1)

        # Multi-head: each head weights its portion of channels
        head_dim = mult // self.num_heads
        V = V.view(E, self.num_heads, head_dim, dim)
        attn_weights = attn_weights.view(E, self.num_heads, 1, 1)
        return (V * attn_weights).view(E, mult, dim)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, qk_dim={self.qk_dim}"


class EquivariantAttention(nn.Module):
    """SO(3)-equivariant graph attention layer.

    Combines equivariant Q/K projections with the full message-passing pipeline:
        1. Q, K: EquivariantLinear projections from node features (gathered to edges)
        2. V: EquivariantEdgewiseLinear transformation of edge features
        3. Attention: Scaled dot-product with neighbor softmax
        4. Pooling: Aggregate weighted edges back to nodes

    The key insight is that the dot product of equivariant features is invariant
    under rotation, so we can use full equivariant Q/K features (not just scalars).

    Args:
        in_repr: Input representation (Repr with lvals and mult).
        out_repr: Output representation.
        num_heads: Number of attention heads. Must divide both in_repr.mult and out_repr.mult.
        num_bins: Number of distance bins for radial weight interpolation.
        rank: Number of channel mixing patterns for radial weights (default 4).
        hidden_dim: Hidden dimension for radial MLP (default 64).
        min_dist: Minimum distance in Angstroms.
        max_dist: Maximum distance in Angstroms.
        log_bins: If True, use logarithmic bin spacing (density ~ 1/r).
        dropout: Dropout rate for attention weights.
        reduce: Pooling reduction method ('sum', 'mean', or 'max').
        distance_decay_scale: If provided, add distance decay bias to attention
            logits: logits -= distance / scale. Encourages attending to nearby
            neighbors. Typical values: 2.0-5.0 Angstroms.

    Example:
        >>> from flash_eq import Repr, Graph, WignerDBasis, EquivariantAttention
        >>>
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> layer = EquivariantAttention(repr, repr, num_heads=4).cuda()
        >>> basis = WignerDBasis([repr, repr]).cuda()
        >>>
        >>> # Graph data
        >>> graph = Graph.random(num_nodes=100, num_edges=1000).to('cuda')
        >>> P, Q = basis(directions)
        >>> output = layer(P, Q, node_features, distances, graph)
    """

    def __init__(
        self,
        in_repr: Repr,
        out_repr: Repr,
        num_heads: int = 1,
        num_bins: int = 100,
        rank: int = 4,
        hidden_dim: int = 64,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        log_bins: bool = False,
        dropout: float = 0.0,
        reduce: str = 'sum',
        distance_decay_scale: float | None = None,
    ) -> None:
        super().__init__()

        if in_repr.mult % num_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must divide in_repr.mult ({in_repr.mult})")
        if out_repr.mult % num_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must divide out_repr.mult ({out_repr.mult})")

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.num_heads = num_heads

        # Fused Q/K projection: single norm + linear with 2x output mult, then split
        self.norm_qk = EquivariantLayerNorm(in_repr)
        qk_repr = Repr(lvals=in_repr.lvals, mult=in_repr.mult * 2)
        self.linear_qk = EquivariantLinear(in_repr, qk_repr, bias=False)

        # Q/K dimension per head: (mult / H) * dim
        qk_dim = (in_repr.mult // num_heads) * in_repr.dim()

        # Low-level attention mechanism
        self.attention = EquivariantEdgeAttention(
            num_heads=num_heads,
            qk_dim=qk_dim,
            dropout=dropout,
            distance_decay_scale=distance_decay_scale,
        )

        # V projection: in_repr -> out_repr (can change lvals via edgewise linear)
        self.linear_v = EquivariantEdgewiseLinear(
            in_repr=in_repr,
            out_repr=out_repr,
            num_bins=num_bins,
            rank=rank,
            hidden_dim=hidden_dim,
            min_dist=min_dist,
            max_dist=max_dist,
            log_bins=log_bins,
        )

        # Aggregation
        self.pool = GraphPooling(reduce=reduce)

    def forward(
        self,
        P: torch.Tensor,
        Q_basis: torch.Tensor,
        node_features: torch.Tensor,
        distances: torch.Tensor,
        graph: Graph,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant attention layer.

        Args:
            P: (num_edges, dim_in, dim_in) input basis from WignerDBasis.
            Q_basis: (num_edges, dim_out, dim_out) output basis from WignerDBasis.
            node_features: (num_nodes, mult_in, dim_in) node features.
            distances: (num_edges,) edge distances.
            graph: Graph containing edge indices and node count.
            edge_features: Optional (num_edges, mult_in, dim_in) edge features
                to add to gathered node features before the V projection.

        Returns:
            output: (num_nodes, mult_out, dim_out) updated node features.
        """
        E = graph.num_edges
        mult = self.in_repr.mult

        # Fused Q/K: normalize once, project to 2x mult, then split and gather
        QK = self.linear_qk(self.norm_qk(node_features))  # (N, 2*mult, dim)
        Q = QK[:, :mult][graph.dst]  # (E, mult, dim)
        K = QK[:, mult:][graph.src]  # (E, mult, dim)

        # Reshape for multi-head: (E, H, qk_dim) where qk_dim = (mult/H) * dim
        Q = Q.view(E, self.num_heads, -1)
        K = K.view(E, self.num_heads, -1)

        # V from source nodes through edgewise linear
        gathered = node_features[graph.src]
        if edge_features is not None:
            gathered = gathered + edge_features
        V = self.linear_v(P, Q_basis, gathered, distances)  # (E, mult_out, dim_out)

        # Apply attention and aggregate
        weighted = self.attention(Q, K, V, graph, distances)
        return self.pool(weighted, graph)

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_heads={self.num_heads}, reduce='{self.pool.reduce}'"
        )
