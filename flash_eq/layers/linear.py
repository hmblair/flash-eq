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
from .radial import BinnedModule, BinnedRadialBasis


class EquivariantEdgewiseLinear(nn.Module):
    """SO(3)-equivariant linear layer with distance-dependent weights.

    Computes: out = Q @ Λ(r) @ P^T @ f

    where P, Q are Wigner-D basis matrices (from WignerDBasis) and Λ(r) is
    a block-diagonal weight matrix depending on edge distance.

    Supports two parameterizations for radial weights:
    - BinnedModule (num_bases=None): Independent weights per bin with Gaussian
      smoothing. Parameters: O(num_bins × out × in × weight_dim).
    - BinnedRadialBasis (num_bases=K): K radial basis functions parameterize
      the weights. Parameters: O(K × out × in × weight_dim). More efficient
      for large weight_dim (high angular momentum).

    Args:
        in_repr: Input representation (Repr with lvals and mult).
        out_repr: Output representation.
        num_bins: Number of distance bins for weight interpolation (default 100).
        num_bases: Number of radial basis functions. If None, uses BinnedModule
            with independent weights per bin. If set (e.g., 16), uses
            BinnedRadialBasis for parameter efficiency.
        min_dist: Minimum distance in Angstroms (default 0.0).
        max_dist: Maximum distance in Angstroms (default 10.0).
        log_bins: If True, use logarithmic bin spacing (density ~ 1/r).
            Useful for molecular data where short-range interactions vary more.
            Requires min_dist > 0.
        sigma: Gaussian smoothing kernel width in bin units (default 1.0).
            Only used when num_bases=None.

    Example:
        >>> from flash_eq import Repr, WignerDBasis, EquivariantEdgewiseLinear
        >>>
        >>> in_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> out_repr = Repr(lvals=[0, 1, 2], mult=32)
        >>>
        >>> # Standard binned weights
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr).cuda()
        >>>
        >>> # Parameter-efficient radial basis (recommended for high L)
        >>> layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bases=16).cuda()
        >>>
        >>> basis = WignerDBasis([in_repr, out_repr]).cuda()
        >>> P, Q = basis(directions)
        >>> edge_features = node_features[src_indices]  # gather to edges
        >>> output = layer(P, Q, edge_features, distances)
    """

    radial_weights: BinnedModule | BinnedRadialBasis

    def __init__(
        self,
        in_repr: Repr,
        out_repr: Repr,
        num_bins: int = 100,
        num_bases: int | None = None,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        log_bins: bool = False,
        sigma: float = 1.0,
    ):
        super().__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.num_bases = num_bases

        # Compute weight structure from representation product
        self.product_repr = ProductRepr(in_repr, out_repr)
        self.weight_dim = self.product_repr.nreps()
        self.channels_in = in_repr.mult
        self.channels_out = out_repr.mult

        # Learnable radial weights
        weight_shape = (self.channels_out, self.channels_in, self.weight_dim)

        if num_bases is not None:
            # Radial basis function parameterization (parameter-efficient)
            self.radial_weights = BinnedRadialBasis(
                num_bins=num_bins,
                shape=weight_shape,
                num_bases=num_bases,
                min_val=min_dist,
                max_val=max_dist,
                log=log_bins,
            )
        else:
            # Independent weights per bin with Gaussian smoothing
            self.radial_weights = BinnedModule(
                num_bins=num_bins,
                shape=weight_shape,
                min_val=min_dist,
                max_val=max_dist,
                log=log_bins,
                sigma=sigma,
            )

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        edge_features: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SO(3)-equivariant linear transformation.

        Computes: out = Q @ Λ(r) @ P^T @ edge_features

        This is an edge-to-edge transformation. Callers are responsible for
        gathering node features to edges if needed (e.g., edge_features = node_features[src_indices]).

        Args:
            P: (num_edges, dim_in, dim_in)
                Input basis matrix from WignerDBasis.
                Columns are permuted to m-first order.
            Q: (num_edges, dim_out, dim_out)
                Output basis matrix from WignerDBasis.
                Columns are permuted to m-first order.
            edge_features: (num_edges, channels_in, dim_in)
                Edge features in standard SH basis.
            distances: (num_edges,)
                Edge distances for radial weight interpolation.

        Returns:
            output: (num_edges, channels_out, dim_out)
                Edge features in standard SH basis.
        """

        # =====================================================================
        # Step 1: Transform to m-first diagonal basis (PyTorch matmul)
        # f_diag = P^T @ edge_features, equivalent to edge_features @ P
        # =====================================================================
        f_diag = torch.bmm(edge_features, P)

        # =====================================================================
        # Step 2: Block-diagonal multiply with radial weights (CUDA kernel)
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
        # Step 3: Transform back to standard SH basis (PyTorch matmul)
        # out = Q @ out_diag, equivalent to out_diag @ Q^T
        # =====================================================================
        return torch.bmm(out_diag, Q.mT)

    def extra_repr(self) -> str:
        rw = self.radial_weights
        spacing = "log" if rw.log else "linear"
        base = (
            f"in_repr={self.in_repr}, out_repr={self.out_repr}, "
            f"num_bins={rw.num_bins}, dist=[{rw.min_val}, {rw.max_val}], "
            f"spacing={spacing}"
        )
        if self.num_bases is not None:
            return f"{base}, num_bases={self.num_bases}"
        else:
            return f"{base}, sigma={rw.sigma}"


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
