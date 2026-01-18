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

from typing import Generator, Any, List, Sequence

import torch
import torch.nn as nn

from .utils import get_epsilon


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

    def mvals(self) -> torch.Tensor:
        """Return the magnetic quantum numbers from -l to l."""
        return torch.arange(-self.l, self.l + 1)

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
        Jz = 1j * torch.diag(self.mvals().to(self.COMPLEX_DTYPE))

        # Transform to real spherical harmonic basis.
        # The toreal() transformation swaps y<->z: Jx -> Lx, Jy -> Lz, Jz -> Ly
        Q = self.toreal()
        Lx = (Q.t().conj() @ Jx @ Q).real.to(self.REAL_DTYPE)
        Lz = (Q.t().conj() @ Jy @ Q).real.to(self.REAL_DTYPE)
        Ly = (Q.t().conj() @ Jz @ Q).real.to(self.REAL_DTYPE)

        # Order [Lz, Lx, Ly] so axis=(0,0,1) with cartesian=False gives Lz rotation
        return torch.stack([Lz, Lx, Ly], dim=0)

    def toreal(self) -> torch.Tensor:
        """Get the conversion matrix from complex to real spherical harmonics.

        Returns:
            Unitary matrix Q of shape (2l+1, 2l+1) such that Y_real = Q @ Y_complex.
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
        lvals: Sequence of degrees to include.
        mult: Multiplicity for all irreps.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> repr.dim()  # 1 + 3 + 5 = 9
        9
    """

    def __init__(self, lvals: Sequence[int], mult: int = 1) -> None:
        if not hasattr(lvals, '__len__'):
            raise TypeError(f"lvals must be a sequence, got {type(lvals).__name__}")
        if len(lvals) == 0:
            raise ValueError("lvals must contain at least one degree")
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be a positive integer, got {mult}")

        self.irreps = [Irrep(l, mult) for l in lvals]
        self.lvals = torch.tensor([irrep.l for irrep in self.irreps], dtype=torch.long)
        self.mult = mult

        self._cumdims = torch.tensor([
            sum(rep.dim() for rep in self.irreps[:i])
            for i in range(len(self.irreps) + 1)
        ])

    def nreps(self) -> int:
        """Return the number of irreducible representations."""
        return len(self.irreps)

    def __iter__(self) -> Generator[Irrep, None, None]:
        yield from self.irreps

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Repr):
            return False
        return torch.equal(self.lvals, other.lvals) and self.mult == other.mult

    def __hash__(self) -> int:
        return hash((tuple(self.lvals.tolist()), self.mult))

    def dim(self) -> int:
        """Get total dimension of the representation."""
        return sum(irrep.dim() for irrep in self)

    def lmax(self) -> int:
        """Get the largest degree among all irreps."""
        return max(irrep.l for irrep in self)

    def cumdims(self) -> torch.Tensor:
        """Get cumulative dimensions for indexing into subspaces."""
        return self._cumdims

    def mvals(self) -> torch.Tensor:
        """Return the magnetic quantum numbers for all components.

        Returns m-values in standard (l-first) basis ordering:
        [m values for l=0], [m values for l=1], ...

        Example:
            >>> Repr(lvals=[0, 1, 2]).mvals()
            tensor([0, -1, 0, 1, -2, -1, 0, 1, 2])
        """
        return torch.cat([irrep.mvals() for irrep in self.irreps])

    def indices(self) -> List[int]:
        """Return irrep index for each dimension.

        Maps each dimension of the representation to its corresponding
        irrep index. Useful for scatter/gather operations.

        Example:
            >>> Repr(lvals=[0, 1, 2]).indices()
            [0, 1, 1, 1, 2, 2, 2, 2, 2]
        """
        result = []
        for i, irrep in enumerate(self.irreps):
            result.extend([i] * irrep.dim())
        return result

    def find_scalar(self) -> tuple[int, List[int]]:
        """Find l=0 (scalar) components in the representation.

        Returns:
            Tuple of (count, locations) where count is the number of
            scalar components and locations is the list of dimension
            indices where scalars appear.

        Example:
            >>> Repr(lvals=[0, 1, 2]).find_scalar()
            (1, [0])
            >>> Repr(lvals=[1, 2]).find_scalar()
            (0, [])
        """
        count = 0
        locations = []
        offset = 0
        for irrep in self.irreps:
            if irrep.l == 0:
                locations.append(offset)
                count += 1
            offset += irrep.dim()
        return count, locations

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

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def rot(
        self,
        axis: torch.Tensor,
        angle: torch.Tensor,
        cartesian: bool = False,
    ) -> torch.Tensor:
        """Compute the Wigner D-matrix for a rotation.

        This method is forced to run in FP32 even under AMP, because the
        matrix exponential is numerically unstable in FP16.

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
        gens = (axis @ self.generators).view(*b, dim, dim)

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
        eps = get_epsilon(directions.dtype)

        # Normalize direction
        d = directions / (torch.linalg.norm(directions, dim=-1, keepdim=True) + eps)

        # Rotation axis is perpendicular to both e_z and d
        # axis = e_z × d = (-d_y, d_x, 0)
        axis = torch.stack([
            -d[..., 1],
            d[..., 0],
            torch.zeros_like(d[..., 0])
        ], dim=-1)

        # Normalize axis (handle d ≈ ±e_z case)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        axis = axis / (axis_norm + eps)

        # Angle is arccos(e_z · d) = arccos(d_z)
        angle = torch.arccos(d[..., 2].clamp(-1 + eps, 1 - eps))

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

    def __str__(self) -> str:
        return f"ProductRepr({self.rep1} x {self.rep2})"
