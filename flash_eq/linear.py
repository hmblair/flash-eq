"""
SO(3)-equivariant linear layer using Wigner-D diagonalization.

The key insight: equivariant weight matrices W(x) are diagonalized by
Wigner-D matrices, yielding a block-diagonal structure with O(L^2) parameters
instead of O(L^4) for dense matrices.

Reference: docs/theory.tex, Theorem 2.1
"""

import torch
import torch.nn as nn

from ciffy.nn.geometric.representations import Repr

# Import CUDA kernel (optional - falls back to Python if unavailable)
try:
    from .block_diagonal_cuda import build_block_metadata, block_diagonal_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def _direction_to_rotation(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute axis and angle to rotate e_z to direction."""
    d = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)
    axis = torch.stack([d[..., 0], torch.zeros_like(d[..., 0]), -d[..., 1]], dim=-1)
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.arccos(d[..., 2].clamp(-1 + 1e-7, 1 - 1e-7))
    return axis, angle


def _build_m_order_permutation(lvals: list[int]) -> torch.Tensor:
    """Build permutation from standard (ℓ,m) order to m-first order.

    Output layout: [m=0 components] [m=1 reals] [m=1 imags] [m=2 reals] [m=2 imags] ...
    """
    perm = []
    lmax = max(lvals) if lvals else 0

    def std_pos(l_idx, m):
        return sum(2 * lvals[i] + 1 for i in range(l_idx)) + lvals[l_idx] + m

    # m=0
    for l_idx, l in enumerate(lvals):
        perm.append(std_pos(l_idx, 0))

    # m>0: reals first, then imags
    for m in range(1, lmax + 1):
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, m))
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, -m))

    return torch.tensor(perm, dtype=torch.long)


class EquivariantLinear(nn.Module):
    """SO(3)-equivariant linear layer with O(L^2) memory.

    Uses Wigner-D diagonalization: output = Q @ Λ @ P^T @ features
    where Λ is block-diagonal with 1×1 real blocks (m=0) and 2×2 real blocks (m>0).

    Args:
        repr_in: Input representation
        repr_out: Output representation
        use_cuda: Whether to use CUDA kernel for block-diagonal multiplication.
            If None (default), uses CUDA when available and input is on GPU.
    """

    def __init__(self, repr_in: Repr, repr_out: Repr, use_cuda: bool = None):
        super().__init__()
        self.repr_in = repr_in
        self.repr_out = repr_out
        self.add_module('_repr_in', repr_in)
        self.add_module('_repr_out', repr_out)

        self.lmax = max(max(repr_in.lvals), max(repr_out.lvals))
        self.dim_in = repr_in.dim()
        self.dim_out = repr_out.dim()
        self._use_cuda = use_cuda
        self._cuda_metadata = None

        self.register_buffer('_perm_in', _build_m_order_permutation(repr_in.lvals))
        self.register_buffer('_perm_out', _build_m_order_permutation(repr_out.lvals))
        self._build_block_info()

    def _build_block_info(self):
        """Compute block sizes and weight offsets for each m value."""
        lvals_in, lvals_out = self.repr_in.lvals, self.repr_out.lvals
        count = lambda lvals, m: sum(1 for l in lvals if l >= m)

        blocks = []
        in_off = out_off = w_off = 0

        for m in range(self.lmax + 1):
            n_in, n_out = count(lvals_in, m), count(lvals_out, m)
            if n_in > 0 and n_out > 0:
                blocks.append((m, n_in, n_out, in_off, out_off, w_off))
                mult = 1 if m == 0 else 2
                in_off += mult * n_in
                out_off += mult * n_out
                w_off += mult * n_out * n_in

        self._blocks = blocks
        self._weight_dim = w_off

    @property
    def weight_dim(self) -> int:
        return self._weight_dim

    def _should_use_cuda(self, features: torch.Tensor) -> bool:
        """Determine whether to use CUDA kernel."""
        if self._use_cuda is not None:
            return self._use_cuda
        return CUDA_AVAILABLE and features.is_cuda

    def _get_cuda_metadata(self, device: torch.device):
        """Get or build CUDA metadata for the given device."""
        if self._cuda_metadata is None or self._cuda_metadata[0].device != device:
            self._cuda_metadata = build_block_metadata(
                self.repr_in.lvals, self.repr_out.lvals, device
            )
        return self._cuda_metadata

    def _block_diagonal_python(self, f_diag: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply block-diagonal weights using Python/einsum."""
        batch, channels_in, _ = f_diag.shape
        channels_out = weights.shape[1]
        out_diag = torch.zeros(batch, channels_out, self.dim_out, device=f_diag.device, dtype=f_diag.dtype)

        for m, n_in, n_out, in_s, out_s, w_off in self._blocks:
            if m == 0:
                f_m = f_diag[:, :, in_s:in_s + n_in]
                w_m = weights[:, :, :, w_off:w_off + n_out * n_in].view(batch, channels_out, channels_in, n_out, n_in)
                out_diag[:, :, out_s:out_s + n_out] = torch.einsum('bocji,bci->boj', w_m, f_m)
            else:
                # 2x2 blocks: [a, b; -b, a] @ [f_re; f_im]
                f_re = f_diag[:, :, in_s:in_s + n_in]
                f_im = f_diag[:, :, in_s + n_in:in_s + 2*n_in]

                w_m = weights[:, :, :, w_off:w_off + 2*n_out*n_in].view(batch, channels_out, channels_in, n_out, n_in, 2)
                a, b = w_m[..., 0], w_m[..., 1]

                out_diag[:, :, out_s:out_s + n_out] = torch.einsum('bocji,bci->boj', a, f_re).add_(torch.einsum('bocji,bci->boj', b, f_im))
                out_diag[:, :, out_s + n_out:out_s + 2*n_out] = torch.einsum('bocji,bci->boj', a, f_im).sub_(torch.einsum('bocji,bci->boj', b, f_re))

        return out_diag

    def forward(self, features: torch.Tensor, directions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            features: (batch, channels_in, dim_in)
            directions: (batch, 3)
            weights: (batch, channels_out, channels_in, weight_dim)
        """
        # Compute Wigner-D matrices and permute to m-first order
        axis, angle = _direction_to_rotation(directions)
        P = self.repr_in.rot(axis, angle, perm=False)[:, :, self._perm_in]
        Q = self.repr_out.rot(axis, angle, perm=False)[:, :, self._perm_out]

        # Transform to diagonal basis
        f_diag = torch.bmm(P.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)

        # Apply block-diagonal weights
        if self._should_use_cuda(features):
            metadata = self._get_cuda_metadata(features.device)
            out_diag = block_diagonal_cuda(f_diag, weights, metadata)
        else:
            out_diag = self._block_diagonal_python(f_diag, weights)

        # Transform back
        return torch.bmm(Q, out_diag.transpose(-1, -2)).transpose(-1, -2)
