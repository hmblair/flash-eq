"""Wigner-D matrices for SO(3)-equivariant operations.

This module provides:
- WignerD: Computes Wigner D-matrices for a representation
- WignerDBasis: Computes basis matrices from direction vectors
- random_rotation: Generate random SO(3) rotations

The key insight is that Wigner-D matrices depend only on directions, not on
layer-specific parameters. By computing them once and sharing across layers,
we avoid redundant computation in multi-layer networks.

Reference: docs/theory.tex, Section 2

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

from typing import Sequence, cast

import torch
import torch.nn as nn

from ..utils import get_epsilon
from .types import Repr


def random_rotation(
    shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random SO(3) rotation(s) uniform w.r.t. Haar measure.

    Samples rotations uniformly on SO(3) by generating unit quaternions
    uniformly on S³, then converting to axis-angle representation.

    Args:
        shape: Batch shape for the output tensors. Default () returns unbatched.
        device: Device for output tensors.
        dtype: Data type for output tensors.

    Returns:
        axis: (*shape, 3) unit vectors representing rotation axes
        angle: (*shape,) rotation angles in radians [0, π]

    Example:
        >>> axis, angle = random_rotation()  # Single rotation
        >>> axis.shape, angle.shape
        (torch.Size([3]), torch.Size([]))

        >>> axis, angle = random_rotation((10,))  # Batch of 10
        >>> axis.shape, angle.shape
        (torch.Size([10, 3]), torch.Size([10]))

        >>> axis, angle = random_rotation((2, 3))  # 2x3 batch
        >>> axis.shape, angle.shape
        (torch.Size([2, 3, 3]), torch.Size([2, 3]))
    """
    # Sample unit quaternion uniformly on S³ (gives Haar measure on SO(3))
    # Quaternion: q = [w, x, y, z] = [cos(θ/2), sin(θ/2) * axis]
    q = torch.randn((*shape, 4), device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)

    # Ensure w >= 0 (choose canonical quaternion, since q and -q represent same rotation)
    sign = torch.sign(q[..., 0:1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign

    # Convert to axis-angle
    w = q[..., 0]
    xyz = q[..., 1:]

    # angle = 2 * arccos(w), but w is clamped for numerical stability
    half_angle = torch.acos(w.clamp(-1.0, 1.0))
    angle = 2 * half_angle

    # axis = xyz / sin(θ/2), with special handling for small angles
    sin_half = torch.sin(half_angle)
    # For small angles, sin(θ/2) ≈ θ/2, and axis direction doesn't matter much
    # Use a default axis [0, 0, 1] when sin_half is too small
    small_angle = sin_half < 1e-6
    safe_sin_half = torch.where(small_angle, torch.ones_like(sin_half), sin_half)
    axis = xyz / safe_sin_half[..., None]

    # For small angles, set axis to [0, 0, 1] (arbitrary but consistent)
    default_axis = torch.zeros_like(axis)
    default_axis[..., 2] = 1.0
    axis = torch.where(small_angle[..., None], default_axis, axis)

    # Normalize axis (should already be unit, but ensure numerical stability)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    return axis, angle


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

        # Build generators eagerly (always FP32 - see _apply)
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

    def _apply(self, fn):
        """Override to keep generators in FP32 when FP16 would be used.

        torch.linalg.matrix_exp is numerically unstable in FP16, so we promote
        FP16 generators back to FP32. FP64 is left alone for full precision.
        """
        super()._apply(fn)
        # Only promote to FP32 if generators became FP16 (matrix_exp is broken in FP16)
        if self.generators.dtype == torch.float16:
            self.generators = self.generators.float()
        return self

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
        input_dtype = axis.dtype

        # FP16 matrix_exp is numerically unstable (produces NaN/Inf).
        # Compute in FP32, then cast back. Generators are always stored as FP32.
        if input_dtype == torch.float16:
            axis = axis.float()
            angle = angle.float()

        gens = (axis @ self.generators).view(*b, dim, dim)
        K = angle[..., None, None] * gens

        rot = torch.linalg.matrix_exp(K)
        rot = torch.nan_to_num(rot, 0.0)

        # Cast back to input dtype
        if input_dtype == torch.float16:
            rot = rot.half()

        # Restore identity for degree-0 (scalar) irreps
        cdims = self.repr.cumdims()
        for i, irrep in enumerate(self.repr.irreps):
            if irrep.l == 0:
                rot[..., cdims[i], cdims[i]] = 1.0

        return rot

    def rot_to_ez(
        self,
        directions: torch.Tensor,
        cartesian: bool = False,
    ) -> torch.Tensor:
        """Compute Wigner D-matrix for rotation taking e_z to direction.

        This computes the rotation matrix D such that applying D to features
        is equivalent to rotating the coordinate frame so that e_z points
        along the given direction.

        For zero-length direction vectors (e.g., from self-loops), the result
        preserves l=0 (scalar) components and zeros out l>0 components, since
        there is no well-defined direction to align with.

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
            wigner = cast(WignerD, self._wigner_modules[i])
            M_std = wigner.rot_to_ez(directions).mT
            perm = getattr(self, f'_perm_{i}')
            M = M_std[..., perm]
            unique_matrices.append(M)

        # Map back to input order
        return tuple(unique_matrices[idx] for idx in self._input_to_unique)

    def extra_repr(self) -> str:
        repr_strs = [str(r) for r in self.reprs]
        return f"reprs=[{', '.join(repr_strs)}], num_unique={self._num_unique}"
