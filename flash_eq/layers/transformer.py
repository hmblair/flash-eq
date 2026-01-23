"""SO(3)-equivariant transformer blocks.

This module implements equivariant transformer blocks that combine
attention, normalization, and feed-forward layers while preserving
SO(3) equivariance.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..graph import Graph
from ..representations import Repr, WignerDBasis
from .attention import EquivariantAttention, GeometricEquivariantAttention
from .linear import EquivariantLinear
from .norm import SeparableEquivariantLayerNorm
from .gating import EquivariantGating
from .s2_activation import SeparableS2Activation


class LayerScale(nn.Module):
    """Learnable scaling for training stability in deep networks.

    Introduced in CaiT (Touvron et al., 2021). Initializes to small values
    so deeper layers contribute less initially, improving optimization.

    Args:
        init_value: Initial scale value (default 1e-4 for deep networks).
    """

    def __init__(self, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth / drop path for regularization in deep networks.

    Randomly drops entire sublayers during training. For GNNs, this makes
    a single binary decision per forward pass (drop or keep the whole sublayer),
    rather than per-node decisions which would be inconsistent.

    Args:
        drop_prob: Probability of dropping the path (default 0.0).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        # Single binary decision: drop entire sublayer or keep it
        if torch.rand(1, device=x.device).item() < self.drop_prob:
            return torch.zeros_like(x)
        # Scale to maintain expected value
        return x / (1 - self.drop_prob)


class EquivariantTransformerBlock(nn.Module):
    """SO(3)-equivariant transformer block.

    Combines self-attention with a feed-forward network while preserving
    SO(3) equivariance. Supports different input and output representations,
    allowing the block to change both multiplicity and angular momentum.

    Architecture:
        x_norm = LayerNorm(x)
        attn_out = Attention(x_norm, graph)
        x = x + attn_out  (residual, only if in_repr == out_repr)

        x_norm = LayerNorm(x)
        mlp_out = MLP(x_norm)
        x = x + mlp_out  (residual)

    When in_repr != out_repr:
        - The attention layer transforms from in_repr to out_repr
        - No residual connection around attention (shapes don't match)
        - MLP operates on out_repr with residual

    Args:
        in_repr: Input representation.
        out_repr: Output representation.
        num_heads: Number of attention heads. Must divide out_repr.mult.
        num_bins: Number of distance bins for radial weights.
        rank: Number of channel mixing patterns for radial weights (default 4).
        hidden_dim: Hidden dimension for radial MLP (default 64).
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        log_bins: Use logarithmic bin spacing.
        mlp_ratio: Hidden dimension multiplier for MLP (default 2).
        dropout: Dropout rate for attention weights.
        use_gating: Use EquivariantGating in MLP (default True). Ignored if
            use_s2_activation is True.
        use_s2_activation: Use SeparableS2Activation instead of gating (default False).
            This applies S² nonlinearity to higher degrees and SiLU to scalars,
            following EquiformerV2. Requires l=0 in out_repr.
        s2_precision: Lebedev precision for S² grid (default 47, 770 points).
        use_geometric_attention: Use GeometricEquivariantAttention with distance-aware
            Q/K projections (default False). When True, Q and K are computed via
            EdgewiseLinear, allowing attention to depend on edge distances.

    Example:
        >>> from flash_eq import Repr, Graph, WignerDBasis
        >>> from flash_eq.layers import EquivariantTransformerBlock
        >>>
        >>> in_repr = Repr(lvals=[0, 1], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=64)
        >>> block = EquivariantTransformerBlock(in_repr, out_repr, num_heads=8).cuda()
        >>> basis = WignerDBasis([in_repr, out_repr]).cuda()
        >>>
        >>> graph = Graph.random(num_nodes=100, num_edges=1000).to('cuda')
        >>> P, Q = basis(directions)
        >>> output = block(P, Q, node_features, distances, graph)
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
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float | None = 1e-4,
        use_gating: bool = True,
        use_s2_activation: bool = False,
        s2_precision: int = 47,
        use_geometric_attention: bool = False,
    ) -> None:
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.use_s2_activation = use_s2_activation
        self.use_geometric_attention = use_geometric_attention

        # Check if we can use residual connections
        self._use_attn_residual = (
            torch.equal(in_repr.lvals, out_repr.lvals) and
            in_repr.mult == out_repr.mult
        )

        # Pre-attention normalization (only used for standard attention)
        if not use_geometric_attention:
            self.norm1 = SeparableEquivariantLayerNorm(in_repr)
        else:
            self.norm1 = None  # GeometricEquivariantAttention has its own norms

        # Attention layer (transforms in_repr -> out_repr)
        AttentionClass = GeometricEquivariantAttention if use_geometric_attention else EquivariantAttention
        self.attn = AttentionClass(
            in_repr=in_repr,
            out_repr=out_repr,
            num_heads=num_heads,
            num_bins=num_bins,
            rank=rank,
            hidden_dim=hidden_dim,
            min_dist=min_dist,
            max_dist=max_dist,
            log_bins=log_bins,
            dropout=dropout,
            reduce='sum',
        )

        # Pre-MLP normalization (operates on out_repr)
        self.norm2 = SeparableEquivariantLayerNorm(out_repr)

        # MLP: out_repr -> expanded -> out_repr
        mlp_hidden_mult = out_repr.mult * mlp_ratio
        mlp_hidden_repr = Repr(lvals=out_repr.lvals, mult=mlp_hidden_mult)

        self.mlp_up = EquivariantLinear(out_repr, mlp_hidden_repr, bias=True)

        # Choose activation: S² activation (EquiformerV2) or gating
        self.mlp_act: nn.Module
        if use_s2_activation:
            self.mlp_act = SeparableS2Activation(
                mlp_hidden_repr,
                hidden_mult=2,
                use_gate=True,
                precision=s2_precision,
            )
        elif use_gating:
            self.mlp_act = EquivariantGating(mlp_hidden_repr)
        else:
            self.mlp_act = nn.Identity()

        self.mlp_down = EquivariantLinear(mlp_hidden_repr, out_repr, bias=True)

        # LayerScale for training stability (scales sublayer outputs before residual)
        if layer_scale_init is not None:
            self.layer_scale_attn = LayerScale(layer_scale_init)
            self.layer_scale_mlp = LayerScale(layer_scale_init)
        else:
            self.layer_scale_attn = nn.Identity()
            self.layer_scale_mlp = nn.Identity()

        # DropPath for regularization (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        node_features: torch.Tensor,
        distances: torch.Tensor,
        graph: Graph,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer block.

        Args:
            P: (num_edges, dim_in, dim_in) input basis matrix from WignerDBasis.
            Q: (num_edges, dim_out, dim_out) output basis matrix from WignerDBasis.
            node_features: (num_nodes, mult_in, dim_in) input node features.
            distances: (num_edges,) edge distances.
            graph: Graph containing edge indices and node count.
            edge_features: Optional (num_edges, mult_in, dim_in) edge features
                to add to gathered node features before the linear transformation.
                Useful for injecting edge-level information like positional encodings.

        Returns:
            (num_nodes, mult_out, dim_out) transformed node features.
        """
        # Self-attention block
        if self.norm1 is not None:
            # Standard attention: normalize then attend
            x_norm = self.norm1(node_features)
            attn_out = self.attn(
                P, Q, x_norm, distances, graph,
                edge_features=edge_features,
            )
        else:
            # Geometric attention: has its own internal norms
            attn_out = self.attn(
                P, Q, node_features, distances, graph,
                edge_features=edge_features,
            )

        if self._use_attn_residual:
            x = node_features + self.drop_path(self.layer_scale_attn(attn_out))
        else:
            x = attn_out

        # MLP block (always has residual since shapes match)
        x_norm = self.norm2(x)
        mlp_out = self.mlp_down(self.mlp_act(self.mlp_up(x_norm)))
        x = x + self.drop_path(self.layer_scale_mlp(mlp_out))

        return x

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"attn_residual={self._use_attn_residual}, "
            f"geometric_attn={self.use_geometric_attention}"
        )


class EquivariantTransformer(nn.Module):
    """Stack of equivariant transformer blocks.

    A full equivariant transformer with input projection, multiple transformer
    blocks, and output projection. Computes Wigner-D basis matrices internally
    from coordinates and graph structure.

    Args:
        in_repr: Input representation.
        hidden_repr: Hidden representation for intermediate layers.
        out_repr: Output representation.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_bins: Number of distance bins for radial weights.
        rank: Number of channel mixing patterns for radial weights (default 4).
        hidden_dim: Hidden dimension for radial MLP (default 64).
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        log_bins: Use logarithmic bin spacing.
        mlp_ratio: Hidden dimension multiplier for MLP.
        dropout: Dropout rate for attention.
        use_s2_activation: Use SeparableS2Activation instead of gating (default False).
        s2_precision: Lebedev precision for S² grid (default 47).
        use_geometric_attention: Use GeometricEquivariantAttention with distance-aware
            Q/K projections (default False). Enables attention to depend on distances.

    Example:
        >>> from flash_eq import Repr, Graph
        >>> from flash_eq.layers import EquivariantTransformer
        >>>
        >>> in_repr = Repr(lvals=[0, 1], mult=32)
        >>> hidden_repr = Repr(lvals=[0, 1, 2], mult=64)
        >>> out_repr = Repr(lvals=[0], mult=1)
        >>>
        >>> model = EquivariantTransformer(
        ...     in_repr, hidden_repr, out_repr,
        ...     num_layers=6, num_heads=8
        ... ).cuda()
        >>>
        >>> # Forward pass with coordinates - basis computed internally
        >>> graph = Graph.random(num_nodes=100, num_edges=1000).to('cuda')
        >>> output = model(coordinates, node_features, graph)
    """

    def __init__(
        self,
        in_repr: Repr,
        hidden_repr: Repr,
        out_repr: Repr,
        num_layers: int = 6,
        num_heads: int = 1,
        num_bins: int = 100,
        rank: int = 4,
        hidden_dim: int = 64,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        log_bins: bool = False,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float | None = 1e-4,
        use_s2_activation: bool = False,
        s2_precision: int = 47,
        use_geometric_attention: bool = False,
    ) -> None:
        super().__init__()

        self.in_repr = in_repr
        self.hidden_repr = hidden_repr
        self.out_repr = out_repr
        self.num_layers = num_layers
        self.use_geometric_attention = use_geometric_attention

        # Wigner-D basis for computing rotation matrices
        self._basis_reprs = [in_repr, hidden_repr, out_repr]
        self.basis = WignerDBasis(self._basis_reprs)

        # Linearly increasing drop path rates (0 at first layer, drop_path at last)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]

        # Build layers list with their repr configurations
        # Layer 0: in_repr -> hidden_repr
        # Layers 1 to num_layers-2: hidden_repr -> hidden_repr
        # Layer num_layers-1: hidden_repr -> out_repr
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Determine input repr for this layer
            if i == 0:
                layer_in = in_repr
            else:
                layer_in = hidden_repr

            # Determine output repr for this layer
            if i == num_layers - 1:
                layer_out = out_repr
            elif i == 0 and num_layers > 1:
                layer_out = hidden_repr
            else:
                layer_out = hidden_repr

            self.layers.append(
                EquivariantTransformerBlock(
                    in_repr=layer_in,
                    out_repr=layer_out,
                    num_heads=num_heads,
                    num_bins=num_bins,
                    rank=rank,
                    hidden_dim=hidden_dim,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    log_bins=log_bins,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[i],
                    layer_scale_init=layer_scale_init,
                    use_s2_activation=use_s2_activation,
                    s2_precision=s2_precision,
                    use_geometric_attention=use_geometric_attention,
                )
            )

        # Final normalization
        self.final_norm = SeparableEquivariantLayerNorm(out_repr)

    def forward(
        self,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        graph: Graph,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            coordinates: (num_nodes, 3) node coordinates.
            node_features: (num_nodes, mult_in, dim_in) input features.
            graph: Graph containing edge indices and node count.

        Returns:
            (num_nodes, mult_out, dim_out) transformed features.
        """
        # Compute edge vectors and distances
        edge_vectors = coordinates[graph.src] - coordinates[graph.dst]
        distances = edge_vectors.norm(dim=-1)

        # Compute basis matrices (WignerDBasis normalizes directions internally)
        M_in, M_hidden, M_out = self.basis(edge_vectors)

        x = node_features

        for i, layer in enumerate(self.layers):
            # Select P based on input repr
            if i == 0:
                P = M_in
            else:
                P = M_hidden

            # Select Q based on output repr
            if i == self.num_layers - 1:
                Q = M_out
            elif i == 0 and self.num_layers > 1:
                Q = M_hidden
            else:
                Q = M_hidden

            x = layer(P, Q, x, distances, graph)

        return self.final_norm(x)

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, hidden_repr={self.hidden_repr}, "
            f"out_repr={self.out_repr}, num_layers={self.num_layers}"
        )
