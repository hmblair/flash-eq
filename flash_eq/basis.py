"""Wigner-D basis matrices for SO(3)-equivariant layers.

This module provides the WignerDBasis class for computing Wigner-D matrices
from direction vectors. These matrices transform between the standard spherical
harmonic basis and the m-diagonalized basis used by equivariant layers.

The key insight is that Wigner-D matrices depend only on directions, not on
layer-specific parameters. By computing them once and sharing across layers,
we avoid redundant computation in multi-layer networks.

Reference: docs/theory.tex, Section 2
"""

from typing import Sequence

import torch
import torch.nn as nn

from .representations import Repr, WignerD


def _build_m_order_permutation(lvals: torch.Tensor) -> torch.Tensor:
    """Build permutation from standard (ℓ,m) order to m-first order.

    Standard order: [l=0,m=0], [l=1,m=-1,0,1], [l=2,m=-2,-1,0,1,2], ...
    M-first order: [m=0 components] [m=1 reals] [m=1 imags] [m=2 reals] ...

    Args:
        lvals: Tensor of angular momentum values in the representation

    Returns:
        Permutation tensor of shape (dim,)
    """
    perm = []
    lmax = int(lvals.max()) if len(lvals) > 0 else 0

    def std_pos(l_idx, m):
        return sum(2 * int(lvals[i]) + 1 for i in range(l_idx)) + int(lvals[l_idx]) + m

    # m=0: one component per l
    for l_idx, l in enumerate(lvals):
        perm.append(std_pos(l_idx, 0))

    # m>0: pair +m/-m for each l to get contiguous 2x2 blocks
    for m in range(1, lmax + 1):
        for l_idx, l in enumerate(lvals):
            if int(l) >= m:
                perm.append(std_pos(l_idx, m))   # +m
                perm.append(std_pos(l_idx, -m))  # -m

    return torch.tensor(perm, dtype=torch.long)


class WignerDBasis(nn.Module):
    """Compute Wigner-D basis matrices from direction vectors.

    This class computes Wigner-D matrices that transform between the standard
    spherical harmonic basis and the m-diagonalized basis used by equivariant
    layers.

    Takes a sequence of Repr objects and returns one matrix per repr. Internally
    deduplicates by lvals (since multiplicity doesn't affect the matrix) for
    computational efficiency.

    For efficiency, compute the basis once and pass to multiple layers:

        basis = WignerDBasis([repr_in, repr_hidden, repr_out])
        M_in, M_hidden, M_out = basis(directions)

        # Layer 1: in -> hidden
        out1 = layer1(features, distances, P=M_in, Q=M_hidden)
        # Layer 2: hidden -> hidden
        out2 = layer2(out1, distances, P=M_hidden, Q=M_hidden)
        # Layer 3: hidden -> out
        out3 = layer3(out2, distances, P=M_hidden, Q=M_out)

    Args:
        reprs: Sequence of representations to compute matrices for.

    Example:
        >>> repr_in = Repr(lvals=[0, 1], mult=32)
        >>> repr_hidden = Repr(lvals=[0, 1, 2], mult=64)
        >>> repr_out = Repr(lvals=[0], mult=1)
        >>> basis = WignerDBasis([repr_in, repr_hidden, repr_out])
        >>>
        >>> directions = torch.randn(1000, 3)
        >>> M_in, M_hidden, M_out = basis(directions)
        >>> M_in.shape, M_hidden.shape, M_out.shape
        (torch.Size([1000, 4, 4]), torch.Size([1000, 9, 9]), torch.Size([1000, 1, 1]))
    """

    def __init__(self, reprs: Sequence[Repr]) -> None:
        super().__init__()

        if len(reprs) == 0:
            raise ValueError("reprs must contain at least one Repr")

        self.reprs = list(reprs)

        # Deduplicate by lvals (multiplicity doesn't affect the matrix)
        # Map each input index to its unique lvals index
        lvals_to_idx: dict[tuple[int, ...], int] = {}
        self._input_to_unique: list[int] = []
        unique_reprs: list[Repr] = []

        for repr in reprs:
            lvals_key = tuple(repr.lvals.tolist())
            if lvals_key not in lvals_to_idx:
                lvals_to_idx[lvals_key] = len(unique_reprs)
                unique_reprs.append(repr)
            self._input_to_unique.append(lvals_to_idx[lvals_key])

        self._num_unique = len(unique_reprs)

        # Create WignerD modules and permutation buffers for unique reprs only
        self._wigner_modules = nn.ModuleList([WignerD(r) for r in unique_reprs])
        for i, repr in enumerate(unique_reprs):
            self.register_buffer(f'_perm_{i}', _build_m_order_permutation(repr.lvals))

    def forward(self, directions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute Wigner-D basis matrices for given directions.

        Args:
            directions: (batch, 3) direction vectors in Cartesian coordinates
                        (need not be normalized)

        Returns:
            Tuple of matrices, one per input Repr. Each matrix has shape
            (batch, dim, dim) where dim is the representation dimension.

        Note:
            These are the Wigner-D matrices for the rotation g_x that takes
            e_z to the direction x. The equivariant layer computes:
                out = Q @ W @ P^T @ f
            where W is block-diagonal in the m-basis.
        """
        # Compute matrices for unique lvals only
        unique_matrices: list[torch.Tensor] = []
        for i in range(self._num_unique):
            # rot_to_ez returns D(g_x^{-1}) where g_x takes e_z to x
            # We need D(g_x), so we transpose: D(g_x) = D(g_x^{-1})^T
            M_std = self._wigner_modules[i].rot_to_ez(directions).mT
            perm = getattr(self, f'_perm_{i}')
            M = M_std[..., perm]
            unique_matrices.append(M)

        # Map back to input order
        return tuple(unique_matrices[idx] for idx in self._input_to_unique)

    def extra_repr(self) -> str:
        repr_strs = [str(r) for r in self.reprs]
        return f"reprs=[{', '.join(repr_strs)}], num_unique={self._num_unique}"
