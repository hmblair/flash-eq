"""SO(3)-equivariant linear layers.

This module implements equivariant linear transformations:
- EquivariantEdgewiseLinear: Edgewise linear with distance-dependent weights
- EquivariantLinear: Linear layer preserving spherical tensor structure

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr, ProductRepr
from ..cuda.block_diagonal import block_diagonal_cuda
from ..radial import BinnedModule


class EquivariantEdgewiseLinear(nn.Module):
    """SO(3)-equivariant linear layer with distance-dependent weights.

    Computes: out = Q @ Λ(r) @ P^T @ f

    where P, Q are Wigner-D basis matrices (from WignerDBasis) and Λ(r) is
    a block-diagonal weight matrix depending on edge distance.

    The radial weights are stored in a learnable lookup table with Gaussian
    smoothing for regularization. Smoothing spreads gradients to neighboring
    bins during backprop, encouraging smooth radial functions.

    Args:
        in_repr: Input representation (Repr with lvals and mult).
        out_repr: Output representation.
        num_bins: Number of distance bins for weight interpolation (default 100).
        min_dist: Minimum distance in Angstroms (default 0.0).
        max_dist: Maximum distance in Angstroms (default 10.0).
        log_bins: If True, use logarithmic bin spacing (density ~ 1/r).
            Useful for molecular data where short-range interactions vary more.
            Requires min_dist > 0.
        sigma: Gaussian smoothing kernel width in bin units (default 1.0).
            Larger values = more smoothing/regularization.

    Example:
        >>> from flash_eq import Repr, WignerDBasis, EquivariantEdgewiseLinear
        >>>
        >>> in_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>>
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr).cuda()
        >>> basis = WignerDBasis([in_repr, out_repr]).cuda()
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
        log_bins: bool = False,
        sigma: float = 1.0,
    ):
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr

        # Compute weight structure from representation product
        self.product_repr = ProductRepr(in_repr, out_repr)
        self.weight_dim = self.product_repr.nreps()
        self.channels_in = in_repr.mult
        self.channels_out = out_repr.mult

        # Learnable radial weight table with smoothing
        self.radial_weights = BinnedModule(
            num_bins=num_bins,
            shape=(self.channels_out, self.channels_in, self.weight_dim),
            min_val=min_dist,
            max_val=max_dist,
            log=log_bins,
            sigma=sigma,
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
        spacing = "log" if rw.log else "linear"
        return (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_bins={rw.num_bins}, dist=[{rw.min_val}, {rw.max_val}], "
            f"spacing={spacing}, sigma={rw.sigma}"
        )


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
