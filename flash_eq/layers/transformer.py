"""SO(3)-equivariant transformer blocks.

This module implements equivariant transformer blocks that combine
attention, normalization, and feed-forward layers while preserving
SO(3) equivariance.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr, WignerDBasis
from .attention import EquivariantAttention
from .linear import EquivariantLinear
from .norm import SeparableEquivariantLayerNorm
from .gating import EquivariantGating
from .s2_activation import SeparableS2Activation


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
        num_bases: Number of radial basis functions. If None, uses independent
            weights per bin. If set (e.g., 16), uses radial basis functions
            for parameter efficiency (recommended for high L).
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        log_bins: Use logarithmic bin spacing.
        sigma: Gaussian smoothing for radial weights (only used when num_bases=None).
        mlp_ratio: Hidden dimension multiplier for MLP (default 2).
        dropout: Dropout rate for attention weights.
        use_gating: Use EquivariantGating in MLP (default True). Ignored if
            use_s2_activation is True.
        use_s2_activation: Use SeparableS2Activation instead of gating (default False).
            This applies S² nonlinearity to higher degrees and SiLU to scalars,
            following EquiformerV2. Requires l=0 in out_repr.
        s2_precision: Lebedev precision for S² grid (default 47, 770 points).

    Example:
        >>> from flash_eq import Repr, WignerDBasis
        >>> from flash_eq.layers import EquivariantTransformerBlock
        >>>
        >>> in_repr = Repr(lvals=[0, 1], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=64)
        >>>
        >>> # Parameter-efficient version with radial basis functions
        >>> block = EquivariantTransformerBlock(
        ...     in_repr, out_repr, num_heads=8, num_bases=16
        ... ).cuda()
        >>> basis = WignerDBasis([in_repr, out_repr]).cuda()
        >>>
        >>> P, Q = basis(directions)
        >>> output = block(P, Q, node_features, distances, src, dst, num_nodes)
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
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        use_gating: bool = True,
        use_s2_activation: bool = False,
        s2_precision: int = 47,
    ) -> None:
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.use_s2_activation = use_s2_activation

        # Check if we can use residual connections
        self._use_attn_residual = (
            torch.equal(in_repr.lvals, out_repr.lvals) and
            in_repr.mult == out_repr.mult
        )

        # Pre-attention normalization
        self.norm1 = SeparableEquivariantLayerNorm(in_repr)

        # Attention layer (transforms in_repr -> out_repr)
        self.attn = EquivariantAttention(
            in_repr=in_repr,
            out_repr=out_repr,
            num_heads=num_heads,
            num_bins=num_bins,
            num_bases=num_bases,
            min_dist=min_dist,
            max_dist=max_dist,
            log_bins=log_bins,
            sigma=sigma,
            use_layer_norm=True,
            dropout=dropout,
            reduce='sum',
        )

        # Pre-MLP normalization (operates on out_repr)
        self.norm2 = SeparableEquivariantLayerNorm(out_repr)

        # MLP: out_repr -> expanded -> out_repr
        mlp_hidden_mult = out_repr.mult * mlp_ratio
        mlp_hidden_repr = Repr(lvals=out_repr.lvals.tolist(), mult=mlp_hidden_mult)

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

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        node_features: torch.Tensor,
        distances: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer block.

        Args:
            P: (num_edges, dim_in, dim_in) input basis matrix from WignerDBasis.
            Q: (num_edges, dim_out, dim_out) output basis matrix from WignerDBasis.
            node_features: (num_nodes, mult_in, dim_in) input node features.
            distances: (num_edges,) edge distances.
            src_indices: (num_edges,) source node index for each edge.
            dst_indices: (num_edges,) destination node index for each edge.
            num_nodes: Total number of nodes.
            edge_features: Optional (num_edges, mult_in, dim_in) edge features
                to add to gathered node features before the linear transformation.
                Useful for injecting edge-level information like positional encodings.

        Returns:
            (num_nodes, mult_out, dim_out) transformed node features.
        """
        # Self-attention block
        x_norm = self.norm1(node_features)
        attn_out = self.attn(
            P, Q, x_norm, distances, src_indices, dst_indices, num_nodes,
            edge_features=edge_features,
        )

        if self._use_attn_residual:
            x = node_features + attn_out
        else:
            x = attn_out

        # MLP block (always has residual since shapes match)
        x_norm = self.norm2(x)
        mlp_out = self.mlp_down(self.mlp_act(self.mlp_up(x_norm)))
        x = x + mlp_out

        return x

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"attn_residual={self._use_attn_residual}"
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
        num_bases: Number of radial basis functions. If None, uses independent
            weights per bin. If set (e.g., 16), uses radial basis functions
            for parameter efficiency (recommended for high L).
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        log_bins: Use logarithmic bin spacing.
        sigma: Gaussian smoothing for radial weights (only used when num_bases=None).
        mlp_ratio: Hidden dimension multiplier for MLP.
        dropout: Dropout rate for attention.
        use_s2_activation: Use SeparableS2Activation instead of gating (default False).
        s2_precision: Lebedev precision for S² grid (default 47).

    Example:
        >>> from flash_eq import Repr
        >>> from flash_eq.layers import EquivariantTransformer
        >>>
        >>> in_repr = Repr(lvals=[0, 1], mult=32)
        >>> hidden_repr = Repr(lvals=[0, 1, 2], mult=64)
        >>> out_repr = Repr(lvals=[0], mult=1)
        >>>
        >>> model = EquivariantTransformer(
        ...     in_repr, hidden_repr, out_repr,
        ...     num_layers=6, num_heads=8, num_bases=16
        ... ).cuda()
        >>>
        >>> # Forward pass with coordinates - basis computed internally
        >>> output = model(coordinates, node_features, src, dst)
    """

    def __init__(
        self,
        in_repr: Repr,
        hidden_repr: Repr,
        out_repr: Repr,
        num_layers: int = 6,
        num_heads: int = 1,
        num_bins: int = 100,
        num_bases: int | None = None,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        log_bins: bool = False,
        sigma: float = 1.0,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        use_s2_activation: bool = False,
        s2_precision: int = 47,
    ) -> None:
        super().__init__()

        self.in_repr = in_repr
        self.hidden_repr = hidden_repr
        self.out_repr = out_repr
        self.num_layers = num_layers

        # Wigner-D basis for computing rotation matrices
        self._basis_reprs = [in_repr, hidden_repr, out_repr]
        self.basis = WignerDBasis(self._basis_reprs)

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
                    num_bases=num_bases,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    log_bins=log_bins,
                    sigma=sigma,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_s2_activation=use_s2_activation,
                    s2_precision=s2_precision,
                )
            )

        # Final normalization
        self.final_norm = SeparableEquivariantLayerNorm(out_repr)

    def forward(
        self,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            coordinates: (num_nodes, 3) node coordinates.
            node_features: (num_nodes, mult_in, dim_in) input features.
            src_indices: (num_edges,) source node index for each edge.
            dst_indices: (num_edges,) destination node index for each edge.

        Returns:
            (num_nodes, mult_out, dim_out) transformed features.
        """
        num_nodes = coordinates.shape[0]

        # Compute edge vectors and distances
        edge_vectors = coordinates[src_indices] - coordinates[dst_indices]
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

            x = layer(P, Q, x, distances, src_indices, dst_indices, num_nodes)

        return self.final_norm(x)

    def extra_repr(self) -> str:
        return (
            f"in_repr={self.in_repr}, hidden_repr={self.hidden_repr}, "
            f"out_repr={self.out_repr}, num_layers={self.num_layers}"
        )
