"""Wigner-D basis matrices for SO(3)-equivariant layers.

This module provides the WignerDBasis class for computing Wigner-D matrices
from direction vectors. These matrices transform between the standard spherical
harmonic basis and the m-diagonalized basis used by equivariant layers.

The key insight is that Wigner-D matrices depend only on directions, not on
layer-specific parameters. By computing them once and sharing across layers,
we avoid redundant computation in multi-layer networks.

Reference: docs/theory.tex, Section 2
"""

import torch
import torch.nn as nn

from .representations import Repr


def _build_m_order_permutation(lvals: list[int]) -> torch.Tensor:
    """Build permutation from standard (ℓ,m) order to m-first order.

    Standard order: [l=0,m=0], [l=1,m=-1,0,1], [l=2,m=-2,-1,0,1,2], ...
    M-first order: [m=0 components] [m=1 reals] [m=1 imags] [m=2 reals] ...

    Args:
        lvals: List of angular momentum values in the representation

    Returns:
        Permutation tensor of shape (dim,)
    """
    perm = []
    lmax = max(lvals) if lvals else 0

    def std_pos(l_idx, m):
        return sum(2 * lvals[i] + 1 for i in range(l_idx)) + lvals[l_idx] + m

    # m=0: one component per l
    for l_idx, l in enumerate(lvals):
        perm.append(std_pos(l_idx, 0))

    # m>0: pair +m/-m for each l to get contiguous 2x2 blocks
    for m in range(1, lmax + 1):
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, m))   # +m
                perm.append(std_pos(l_idx, -m))  # -m

    return torch.tensor(perm, dtype=torch.long)


class WignerDBasis(nn.Module):
    """Compute Wigner-D basis matrices from direction vectors.

    This class computes the P and Q matrices that transform between the
    standard spherical harmonic basis and the m-diagonalized basis:

        f_diag = P^T @ f_standard
        f_standard = Q @ f_diag

    The matrices P and Q are Wigner-D matrices with columns permuted to
    m-first ordering. They depend only on the direction vectors, not on
    any learnable parameters.

    For efficiency, compute the basis once and pass to multiple layers:

        basis = WignerDBasis(in_repr, out_repr)
        P, Q = basis(directions)

        out1 = layer1(features, distances, P=P, Q=Q)
        out2 = layer2(out1, distances, P=P, Q=Q)

    Args:
        repr_in: Input representation (determines P matrix structure)
        repr_out: Output representation (determines Q matrix structure)

    Example:
        >>> repr_in = Repr(lvals=[0, 1, 2], mult=32)
        >>> repr_out = Repr(lvals=[0, 1, 2], mult=32)
        >>> basis = WignerDBasis(repr_in, repr_out)
        >>>
        >>> directions = torch.randn(1000, 3)
        >>> P, Q = basis(directions)
        >>> P.shape, Q.shape
        (torch.Size([1000, 9, 9]), torch.Size([1000, 9, 9]))
    """

    def __init__(self, repr_in: Repr, repr_out: Repr) -> None:
        super().__init__()

        self.repr_in = repr_in
        self.repr_out = repr_out

        # Store as submodules so buffers transfer with .to()
        self.add_module('_repr_in', repr_in)
        self.add_module('_repr_out', repr_out)

        # Build m-order permutations
        self.register_buffer('_perm_in', _build_m_order_permutation(repr_in.lvals))
        self.register_buffer('_perm_out', _build_m_order_permutation(repr_out.lvals))

    def forward(
        self,
        directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Wigner-D basis matrices for given directions.

        Args:
            directions: (batch, 3) direction vectors in Cartesian coordinates
                        (need not be normalized)

        Returns:
            P: (batch, dim_in, dim_in) input basis matrix D(g_x)
            Q: (batch, dim_out, dim_out) output basis matrix D(g_x)

        Note:
            These are the Wigner-D matrices for the rotation g_x that takes
            e_z to the direction x. The equivariant layer computes:
                out = Q @ W @ P^T @ f
            where W is block-diagonal in the m-basis.
        """
        # rot_to_ez returns D(g_x^{-1}) where g_x takes e_z to x
        # We need D(g_x), so we transpose: D(g_x) = D(g_x^{-1})^T
        # cartesian=True because directions are in Cartesian (x,y,z) coordinates
        P = self.repr_in.rot_to_ez(directions).mT
        Q = self.repr_out.rot_to_ez(directions).mT

        return P, Q

    def extra_repr(self) -> str:
        return f"repr_in={self.repr_in}, repr_out={self.repr_out}"
