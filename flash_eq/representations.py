"""SO(3) representation theory for equivariant neural networks.

This module implements the real irreducible representations of SO(3),
the 3D rotation group. These are the mathematical foundation for
building rotation-equivariant neural networks.

Key concepts:
- **Irreducible representation (irrep)**: An irrep of degree l has
  dimension 2l+1 and transforms vectors according to Wigner D-matrices.

- **Wigner D-matrix**: D^l(R) gives the action of rotation R on the
  degree-l irrep. Computed via matrix exponential of the Lie algebra.

Conventions:
- Basis ordering: m = -l, -l+1, ..., l-1, l
- Real spherical harmonics: Y_l^m with Wikipedia conventions

Classes:
    Irrep: Single irreducible representation
    ProductIrrep: Tensor product decomposition of two irreps
    Repr: Collection of irreps into a unified representation
    ProductRepr: Tensor product of two representations
"""
from __future__ import annotations

from typing import Generator, Any, List

import torch
import torch.nn as nn


class Irrep:
    """A real irreducible representation of SO(3).

    An irreducible representation (irrep) of degree l is a (2l+1)-dimensional
    vector space on which SO(3) acts via Wigner D-matrices.

    Special cases:
        - l=0: Scalars (1D, invariant under rotation)
        - l=1: Vectors (3D, rotate like coordinates)
        - l=2: Traceless symmetric matrices (5D)

    Args:
        l: The degree (non-negative integer).
        mult: Multiplicity (number of independent copies).
    """

    REAL_DTYPE = torch.float32
    COMPLEX_DTYPE = torch.complex128

    def __init__(self, l: int, mult: int = 1) -> None:
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"Degree l must be a non-negative integer, got {l}")
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be a positive integer, got {mult}")

        self.l = l
        self.mult = mult

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Irrep):
            return False
        return self.l == other.l and self.mult == other.mult

    def __hash__(self) -> int:
        return hash((self.l, self.mult))

    def dim(self) -> int:
        """Return the dimension of the representation (2l+1)."""
        return 2 * self.l + 1

    def mvals(self) -> List[int]:
        """Return the magnetic quantum numbers from -l to l."""
        return list(range(-self.l, self.l + 1))

    def raising(self) -> torch.Tensor:
        """Compute the raising operator J_+ for this irrep."""
        j = self.l
        m = torch.arange(-j, j)
        return torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    def lowering(self) -> torch.Tensor:
        """Compute the lowering operator J_- for this irrep."""
        j = self.l
        m = torch.arange(-j + 1, j + 1)
        return torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    def _generators(self) -> torch.Tensor:
        """Return the so(3) generators in the real spherical harmonic basis.

        Returns generators ordered as [Lz, Lx, Ly] so that with cartesian=False,
        axis=(0,0,1) produces rotation around the SH z-axis. This yields
        m-diagonal Wigner-D matrices where each |m| block rotates by mθ.

        Returns:
            Shape (3, 2l+1, 2l+1) tensor of generator matrices [Lz, Lx, Ly].
        """
        raising = self.raising().to(self.COMPLEX_DTYPE)
        lowering = self.lowering().to(self.COMPLEX_DTYPE)

        # Angular momentum operators in complex |l,m> basis
        Jx = 0.5 * (raising + lowering)
        Jy = -0.5j * (raising - lowering)
        Jz = 1j * torch.diag(torch.tensor(self.mvals(), dtype=self.COMPLEX_DTYPE))

        # Transform to real spherical harmonic basis.
        # The toreal() transformation swaps y<->z: Jx -> Lx, Jy -> Lz, Jz -> Ly
        Q = self.toreal()
        Lx = (Q.t().conj() @ Jx @ Q).real.to(self.REAL_DTYPE)
        Lz = (Q.t().conj() @ Jy @ Q).real.to(self.REAL_DTYPE)
        Ly = (Q.t().conj() @ Jz @ Q).real.to(self.REAL_DTYPE)

        # Order [Lz, Lx, Ly] so axis=(0,0,1) with cartesian=False gives Lz rotation
        return torch.stack([Lz, Lx, Ly], dim=0)

    def toreal(self) -> torch.Tensor:
        """
        Get the conversion matrix from complex to real spherical harmonics.
        """
        SQRT2 = 2 ** -0.5

        q = torch.zeros(self.dim(), self.dim(), dtype=torch.complex128)

        for m in range(-self.l, 0):
            q[self.l + m, self.l + abs(m)] = SQRT2
            q[self.l + m, self.l - abs(m)] = -1j * SQRT2

        for m in range(1, self.l + 1):
            q[self.l + m, self.l + abs(m)] = (-1)**m * SQRT2
            q[self.l + m, self.l - abs(m)] = 1j * (-1)**m * SQRT2

        q[self.l, self.l] = 1
        q = (-1j)**self.l * q

        return q

    def __str__(self) -> str:
        return f"Irrep(l={self.l}, mult={self.mult})"


class ProductIrrep:
    """Tensor product decomposition of two irreducible representations.

    When taking the tensor product of two irreps with degrees l1 and l2,
    the result decomposes into a direct sum of irreps with degrees
    |l1-l2|, |l1-l2|+1, ..., l1+l2.

    Args:
        rep1: First irreducible representation.
        rep2: Second irreducible representation.
    """

    def __init__(self, rep1: Irrep, rep2: Irrep) -> None:
        self.rep1 = rep1
        self.rep2 = rep2

        self.lmin = abs(rep1.l - rep2.l)
        self.lmax = rep1.l + rep2.l

        self.reps = [Irrep(l) for l in range(self.lmin, self.lmax + 1)]

    def dim(self) -> int:
        """Return total dimension of the tensor product decomposition."""
        return sum(rep.dim() for rep in self.reps)

    def maxdim(self) -> int:
        """Return the maximum dimension among irreps in the decomposition."""
        return max(rep.dim() for rep in self.reps)

    def nreps(self) -> int:
        """Return the number of irreps in the decomposition."""
        return len(self.reps)

    def __str__(self) -> str:
        return f"ProductIrrep({self.rep1.l} x {self.rep2.l})"


class Repr:
    """A collection of irreducible representations.

    Simple data class storing representation structure. Use WignerD for
    computing rotation matrices.

    Args:
        lvals: List of degrees to include. Defaults to [1] (vectors only).
        mult: Multiplicity for all irreps.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> repr.dim()  # 1 + 3 + 5 = 9
        9
    """

    def __init__(self, lvals: List[int] = None, mult: int = 1) -> None:
        if lvals is None:
            lvals = [1]

        if not isinstance(lvals, (list, tuple)):
            raise TypeError(f"lvals must be a list or tuple, got {type(lvals).__name__}")
        if len(lvals) == 0:
            raise ValueError("lvals must contain at least one degree")
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be a positive integer, got {mult}")

        self.irreps = [Irrep(l, mult) for l in lvals]
        self.lvals = [irrep.l for irrep in self.irreps]
        self.mult = mult

        self._cumdims = [
            sum(rep.dim() for rep in self.irreps[:i])
            for i in range(len(self.irreps) + 1)
        ]

    def nreps(self) -> int:
        """Return the number of irreducible representations."""
        return len(self.irreps)

    def __iter__(self) -> Generator[Irrep, None, None]:
        yield from self.irreps

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Repr):
            return False
        return self.lvals == other.lvals and self.mult == other.mult

    def __hash__(self) -> int:
        return hash((tuple(self.lvals), self.mult))

    def dim(self) -> int:
        """Get total dimension of the representation."""
        return sum(irrep.dim() for irrep in self)

    def lmax(self) -> int:
        """Get the largest degree among all irreps."""
        return max(irrep.l for irrep in self)

    def cumdims(self) -> List[int]:
        """Get cumulative dimensions for indexing into subspaces."""
        return self._cumdims

    def __str__(self) -> str:
        degrees = ', '.join(str(rep.l) for rep in self.irreps)
        return f"Repr(lvals=[{degrees}])"


class WignerD(nn.Module):
    """Wigner D-matrix computation for a representation.

    Computes rotation matrices (Wigner D-matrices) for a given Repr.

    Args:
        repr: The representation to compute rotations for.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> wigner = WignerD(repr)
        >>> axis = torch.tensor([[0., 0., 1.]])
        >>> angle = torch.tensor([3.14159 / 4])
        >>> D = wigner.rot(axis, angle)
    """

    def __init__(self, repr: Repr) -> None:
        super().__init__()
        self.repr = repr

        # Build generators eagerly
        NUM_GENS = 3
        dim = repr.dim()
        gens = torch.zeros(NUM_GENS, dim, dim)

        cumdim = 0
        for irrep in repr.irreps:
            gens[
                ...,
                cumdim: cumdim + irrep.dim(),
                cumdim: cumdim + irrep.dim(),
            ] = irrep._generators()
            cumdim += irrep.dim()

        self.register_buffer('generators', gens.view(3, -1))

    def rot(
        self,
        axis: torch.Tensor,
        angle: torch.Tensor,
        cartesian: bool = False,
    ) -> torch.Tensor:
        """Compute the Wigner D-matrix for a rotation.

        Args:
            axis: Rotation axis of shape (..., 3).
                  Should be normalized (or will be treated as direction).
            angle: Rotation angle in radians of shape (...).
            cartesian: If False (default), axis components index generators as
                  [Lz, Lx, Ly]. This means axis=(0,0,1) rotates around the SH
                  z-axis, producing m-diagonal Wigner-D matrices.
                  If True, axis is in Cartesian (x,y,z) coordinates and is
                  permuted to [z,x,y] to match the generator ordering.

        Returns:
            Wigner D-matrix of shape (..., dim, dim).
        """
        if cartesian:
            # Permute axis to match generator ordering in real SH basis
            # (ax, ay, az) -> (az, ax, ay)
            axis = axis[..., [2, 0, 1]]

        dim = self.repr.dim()
        *b, _ = axis.size()
        gens = (axis @ self.generators.to(dtype=axis.dtype, device=axis.device)).view(*b, dim, dim)

        rot = torch.linalg.matrix_exp(angle[..., None, None] * gens)
        rot = torch.nan_to_num(rot, 0.0)

        # Restore identity for degree-0 (scalar) irreps
        cdims = self.repr.cumdims()
        for i, irrep in enumerate(self.repr.irreps):
            if irrep.l == 0:
                rot[..., cdims[i], cdims[i]] = 1.0

        return rot

    def rot_to_ez(self, directions: torch.Tensor, cartesian: bool = False) -> torch.Tensor:
        """Compute Wigner D-matrix for rotation taking e_z to direction.

        This computes the rotation matrix D such that applying D to features
        is equivalent to rotating the coordinate frame so that e_z points
        along the given direction.

        Args:
            directions: (..., 3) direction vectors (need not be normalized)
            cartesian: If True, directions are in Cartesian coordinates.

        Returns:
            D: (..., dim, dim) Wigner D-matrix
        """
        # Normalize direction
        d = directions / (torch.linalg.norm(directions, dim=-1, keepdim=True) + 1e-8)

        # Rotation axis is perpendicular to both e_z and d
        # axis = e_z × d = (-d_y, d_x, 0)
        axis = torch.stack([
            -d[..., 1],
            d[..., 0],
            torch.zeros_like(d[..., 0])
        ], dim=-1)

        # Normalize axis (handle d ≈ ±e_z case)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)

        # Angle is arccos(e_z · d) = arccos(d_z)
        angle = torch.arccos(d[..., 2].clamp(-1 + 1e-7, 1 - 1e-7))

        return self.rot(axis, angle, cartesian).mT


class ProductRepr:
    """Tensor product of two representations.

    Computes the tensor product structure for the product of two Repr objects.
    Useful for determining weight dimensions and basis counts.

    Args:
        rep1: First representation.
        rep2: Second representation.
    """

    def __init__(self, rep1: Repr, rep2: Repr) -> None:
        self.rep1 = rep1
        self.rep2 = rep2

        self.reps = [
            ProductIrrep(irrep1, irrep2)
            for irrep1 in rep1
            for irrep2 in rep2
        ]

    def dim(self) -> int:
        """Get total dimension of the tensor product."""
        return sum(rep.dim() for rep in self.reps)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ProductRepr):
            return False
        return self.rep1 == other.rep1 and self.rep2 == other.rep2

    def __hash__(self) -> int:
        return hash((hash(self.rep1), hash(self.rep2)))

    def lmax(self) -> int:
        """Get the largest degree in the decomposition."""
        return max(rep.lmax for rep in self.reps)

    def nreps(self) -> int:
        """Return total number of irreps in the tensor product."""
        return sum(rep.nreps() for rep in self.reps)

    def _compute_block_structure(self) -> tuple:
        """Compute block structure for the m-basis parameterization.

        Returns:
            Tuple of (blocks, dim_in, dim_out, weight_dim) where blocks is a list
            of dicts with keys: m, n_in, n_out, in_off, out_off, w_off
        """
        lvals_in = torch.tensor(self.rep1.lvals)
        lvals_out = torch.tensor(self.rep2.lvals)
        lmax_in = int(lvals_in.max())
        lmax_out = int(lvals_out.max())
        lmax = max(lmax_in, lmax_out)

        # Vectorized count: for each m, count how many l >= m
        m_vals = torch.arange(lmax + 1)
        n_in_all = (lvals_in.unsqueeze(0) >= m_vals.unsqueeze(1)).sum(dim=1)
        n_out_all = (lvals_out.unsqueeze(0) >= m_vals.unsqueeze(1)).sum(dim=1)

        # Compute offsets for all m-values (needed even for skipped blocks)
        mult = torch.where(m_vals == 0, 1, 2)
        in_sizes = mult * n_in_all
        out_sizes = mult * n_out_all

        # Cumulative offsets (exclusive prefix sum)
        in_offsets = torch.cat([torch.tensor([0]), in_sizes[:-1].cumsum(0)])
        out_offsets = torch.cat([torch.tensor([0]), out_sizes[:-1].cumsum(0)])

        # Build blocks only where coupling exists
        blocks = []
        w_off = 0
        for m in range(lmax + 1):
            n_in = int(n_in_all[m])
            n_out = int(n_out_all[m])
            if n_in > 0 and n_out > 0:
                m_mult = 1 if m == 0 else 2
                blocks.append({
                    'm': m,
                    'n_in': n_in,
                    'n_out': n_out,
                    'in_off': int(in_offsets[m]),
                    'out_off': int(out_offsets[m]),
                    'w_off': w_off,
                })
                w_off += m_mult * n_out * n_in

        dim_in = int(in_sizes[n_in_all > 0].sum()) if (n_in_all > 0).any() else 0
        dim_out = int(out_sizes[n_out_all > 0].sum()) if (n_out_all > 0).any() else 0

        return blocks, dim_in, dim_out, w_off


    def build_block_metadata(self, device: torch.device) -> tuple:
        """Build metadata tensors for CUDA kernel.

        Args:
            device: Target device for metadata tensors.

        Returns:
            Tuple of (block_data, dim_out, max_in_size, max_out_size) where:
            - block_data: (num_blocks, 6) tensor with columns [m, n_in, n_out, in_off, out_off, w_off]
            - dim_out: Total output dimension
            - max_in_size: Maximum input block size (for shared memory)
            - max_out_size: Maximum output block size (for shared memory)
        """
        blocks, _, dim_out, _ = self._compute_block_structure()

        # Pack into tensor
        block_data = torch.tensor(
            [[b['m'], b['n_in'], b['n_out'], b['in_off'], b['out_off'], b['w_off']] for b in blocks],
            dtype=torch.int32, device=device
        )

        # Compute max sizes for shared memory allocation
        max_in_size = max(b['n_in'] if b['m'] == 0 else 2 * b['n_in'] for b in blocks)
        max_out_size = max(b['n_out'] if b['m'] == 0 else 2 * b['n_out'] for b in blocks)

        return (block_data, dim_out, max_in_size, max_out_size)

    def __str__(self) -> str:
        return f"ProductRepr({self.rep1} x {self.rep2})"
