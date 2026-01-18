"""
SO(3)-equivariant edgewise linear layers.

This module implements the core equivariant layer:

    out = Q @ Λ(r) @ P^T @ f

where:
    f: input node features in standard SH basis
    P: Wigner-D matrix transforming to m-first diagonal basis
    Λ(r): block-diagonal radial weights
    Q: Wigner-D matrix transforming back to standard SH basis
    out: output edge features in standard SH basis

The computation is split across modules:
    - Gather (PyTorch): edge_f = node_f[src_indices]
    - P^T transform (PyTorch): f_diag = P^T @ edge_f
    - Block multiply (CUDA kernel): out_diag = Λ(r) @ f_diag
    - Q transform (PyTorch): out = Q @ out_diag

Classes:
    EquivariantEdgewiseLinear: Full layer with learned radial MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .representations import Repr, ProductRepr
from .block_diagonal_cuda import block_diagonal_cuda
from .radial import RadialMLP, BinnedModule


class EquivariantEdgewiseLinear(nn.Module):
    """SO(3)-equivariant linear layer with distance-dependent weights.

    Computes: out = Q @ Λ(r) @ P^T @ f

    where P, Q are Wigner-D basis matrices (from WignerDBasis) and Λ(r) is
    a block-diagonal weight matrix depending on edge distance.

    The radial weights are parameterized by a small MLP and stored in a
    binned lookup table for memory efficiency.

    Args:
        in_repr: Input representation (Repr with lvals and mult).
        out_repr: Output representation.
        num_bins: Number of distance bins for weight interpolation (default 100).
        min_dist: Minimum distance in Angstroms (default 0.0).
        max_dist: Maximum distance in Angstroms (default 10.0).
        radial_hidden: Hidden dimension for radial MLP (default 64).
        radial_layers: Number of hidden layers in radial MLP (default 2).

    Example:
        >>> from flash_eq import Repr, WignerDBasis, EquivariantEdgewiseLinear
        >>>
        >>> in_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>>
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr).cuda()
        >>> basis = WignerDBasis(in_repr, out_repr).cuda()
        >>>
        >>> # Compute basis matrices from edge directions
        >>> P, Q = basis(directions)  # directions: (num_edges, 3)
        >>>
        >>> # Apply layer
        >>> output = layer(P, Q, node_features, distances, src_indices)
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

        # Compute weight structure from representation product
        self.product_repr = ProductRepr(in_repr, out_repr)
        self.weight_dim = self.product_repr.nreps()
        self.channels_in = in_repr.mult
        self.channels_out = out_repr.mult

        # Radial weight network with binning
        mlp = RadialMLP(
            hidden_dim=radial_hidden,
            num_basis=self.weight_dim,
            in_mult=self.channels_in,
            out_mult=self.channels_out,
            num_layers=radial_layers,
            r_max=max_dist,
        )
        self.radial_weights = BinnedModule(
            mlp,
            num_bins=num_bins,
            min_val=min_dist,
            max_val=max_dist,
        )

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        node_features: torch.Tensor,
        distances: torch.Tensor,
        src_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SO(3)-equivariant linear transformation.

        Computes: out = Q @ Λ(r) @ P^T @ node_features[src_indices]

        Args:
            P: (num_edges, dim_in, dim_in)
                Input basis matrix from WignerDBasis.
                Columns are permuted to m-first order.
            Q: (num_edges, dim_out, dim_out)
                Output basis matrix from WignerDBasis.
                Columns are permuted to m-first order.
            node_features: (num_nodes, channels_in, dim_in)
                Node features in standard SH basis.
            distances: (num_edges,)
                Edge distances for radial weight interpolation.
            src_indices: (num_edges,)
                Source node index for each edge.

        Returns:
            output: (num_edges, channels_out, dim_out)
                Edge features in standard SH basis.
        """
        # =====================================================================
        # Step 1: Gather node features to edges (PyTorch)
        # =====================================================================
        edge_features = node_features[src_indices]  # (num_edges, channels, dim_in)

        # =====================================================================
        # Step 2: Transform to m-first diagonal basis (PyTorch matmul)
        # f_diag = P^T @ edge_features, equivalent to edge_features @ P
        # =====================================================================
        f_diag = torch.bmm(edge_features, P)

        # =====================================================================
        # Step 3: Block-diagonal multiply with radial weights (CUDA kernel)
        # out_diag = Λ(r) @ f_diag
        # =====================================================================
        bin_lo, interp_weight = self.radial_weights.bin_indices(distances)
        out_diag = block_diagonal_cuda(
            f_diag,
            self.radial_weights(),
            bin_lo,
            interp_weight,
            self.product_repr,
        )

        # =====================================================================
        # Step 4: Transform back to standard SH basis (PyTorch matmul)
        # out = Q @ out_diag, equivalent to out_diag @ Q^T
        # =====================================================================
        return torch.bmm(out_diag, Q.mT)

    def extra_repr(self) -> str:
        rw = self.radial_weights
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_bins={rw.num_bins}, dist=[{rw.min_val}, {rw.max_val}]"
        )
