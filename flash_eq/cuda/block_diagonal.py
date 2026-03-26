"""
CUDA-accelerated block-diagonal multiplication for SO(3)-equivariant layers.

This module provides the core computational kernel for the equivariant layer:
    out = Λ(r) @ f

where:
    f: input features in m-first diagonal basis (num_edges, channels_in, dim_in)
    Λ(r): block-diagonal weights interpolated from radial table
    out: output features in m-first diagonal basis (num_edges, channels_out, dim_out)

The kernel does NOT handle:
    - Gathering node features to edges (done in PyTorch)
    - P^T or Q basis transforms (done in PyTorch)

Optimizations:
    - Binned radial weights: O(bins) memory instead of O(edges)
    - Linear interpolation between bin edges

Supports FP32, FP64, and FP16.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..representations import ProductRepr

_cuda_module = None


class CUDANotAvailableError(RuntimeError):
    """Raised when CUDA is required but not available."""
    pass


def _get_cuda_module():
    """Load the pre-built CUDA extension."""
    global _cuda_module

    if _cuda_module is not None:
        return _cuda_module

    if not torch.cuda.is_available():
        raise CUDANotAvailableError(
            "flash-eq requires a CUDA-capable GPU. No CUDA device was detected.\n\n"
            "To use flash-eq, you need:\n"
            "  1. An NVIDIA GPU with CUDA support\n"
            "  2. PyTorch with CUDA support (torch.cuda.is_available() == True)\n\n"
            "If you have a GPU but see this error, check:\n"
            "  - nvidia-smi shows your GPU\n"
            "  - PyTorch was installed with CUDA support"
        )

    try:
        from .. import _block_diagonal_cuda
        _cuda_module = _block_diagonal_cuda
    except ImportError:
        raise ImportError(
            "flash-eq CUDA extension is not installed.\n\n"
            "Install from a pre-built wheel:\n"
            "  pip install flash-eq --find-links "
            "https://github.com/hmblair/flash-eq/releases/latest/download/\n\n"
            "Or build from source (requires CUDA toolkit):\n"
            "  CUDA_HOME=/usr/local/cuda pip install flash-eq"
        ) from None

    return _cuda_module


# =============================================================================
# Internal Autograd Function
# =============================================================================

class _BlockDiagonalFunction(Function):
    """Autograd function wrapping the CUDA kernel with distance-sorted edges."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        features: torch.Tensor,
        radial_table: torch.Tensor,
        distances: torch.Tensor,
        sh_scale: float,
        bin_param1: float,
        bin_param2: float,
        num_bins: int,
        log_bins: bool,
        channels_out: int,
        lvals_in: torch.Tensor,
        lvals_out: torch.Tensor,
        dim_out: int,
        max_in_size: int,
        max_out_size: int,
    ) -> torch.Tensor:
        cuda_module = _get_cuda_module()

        # Sort edges by distance for better memory access patterns
        # (since binning is monotonic, sorting by distance is equivalent to sorting by bin)
        sort_idx = torch.argsort(distances)
        unsort_idx = torch.argsort(sort_idx)

        features_sorted = features[sort_idx]
        distances_sorted = distances[sort_idx]

        output_sorted, = cuda_module.forward(
            features_sorted,
            radial_table,
            distances_sorted,
            sh_scale,
            bin_param1,
            bin_param2,
            num_bins,
            log_bins,
            lvals_in,
            lvals_out,
            channels_out,
            dim_out,
            max_in_size
        )

        # Unsort output back to original edge order
        output = output_sorted[unsort_idx]

        ctx.save_for_backward(features_sorted, radial_table, distances_sorted,
                              lvals_in, lvals_out, sort_idx, unsort_idx)
        ctx.dim_in = features.size(2)  # type: ignore[attr-defined]
        ctx.max_in_size = max_in_size  # type: ignore[attr-defined]
        ctx.max_out_size = max_out_size  # type: ignore[attr-defined]
        ctx.sh_scale = sh_scale  # type: ignore[attr-defined]
        ctx.bin_param1 = bin_param1  # type: ignore[attr-defined]
        ctx.bin_param2 = bin_param2  # type: ignore[attr-defined]
        ctx.num_bins = num_bins  # type: ignore[attr-defined]
        ctx.log_bins = log_bins  # type: ignore[attr-defined]

        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (features_sorted, radial_table, distances_sorted,
         lvals_in, lvals_out, sort_idx, unsort_idx) = ctx.saved_tensors  # type: ignore[attr-defined]
        cuda_module = _get_cuda_module()

        # Sort grad_output to match the sorted forward pass order
        grad_output_sorted = grad_output[sort_idx]

        grad_features_sorted, grad_radial_table, grad_distances_sorted = cuda_module.backward(
            grad_output_sorted,
            features_sorted,
            radial_table,
            distances_sorted,
            ctx.sh_scale,  # type: ignore[attr-defined]
            ctx.bin_param1,  # type: ignore[attr-defined]
            ctx.bin_param2,  # type: ignore[attr-defined]
            ctx.num_bins,  # type: ignore[attr-defined]
            ctx.log_bins,  # type: ignore[attr-defined]
            lvals_in,
            lvals_out,
            ctx.dim_in,  # type: ignore[attr-defined]
            ctx.max_in_size,  # type: ignore[attr-defined]
            ctx.max_out_size,  # type: ignore[attr-defined]
        )

        # Unsort gradients back to original edge order
        grad_features = grad_features_sorted[unsort_idx]
        grad_distances = grad_distances_sorted[unsort_idx]

        return (grad_features, grad_radial_table, grad_distances,
                None, None, None, None, None, None, None, None, None, None, None)


# =============================================================================
# Main Public API
# =============================================================================

def _compute_block_sizes(lvals_in: torch.Tensor, lvals_out: torch.Tensor) -> tuple[int, int]:
    """Compute max block sizes for shared memory allocation.

    Args:
        lvals_in: Angular momentum values for input representation.
        lvals_out: Angular momentum values for output representation.

    Returns:
        Tuple of (max_in_size, max_out_size) for shared memory allocation.
    """
    m_max = max(int(lvals_in.max()), int(lvals_out.max()))
    max_in_size, max_out_size = 0, 0

    for m in range(m_max + 1):
        n_in = int((lvals_in >= m).sum())
        n_out = int((lvals_out >= m).sum())

        if n_in > 0 and n_out > 0:
            m_mult = 1 if m == 0 else 2
            max_in_size = max(max_in_size, m_mult * n_in)
            max_out_size = max(max_out_size, m_mult * n_out)

    return max_in_size, max_out_size


def block_diagonal_cuda(
    features: torch.Tensor,
    radial_table: torch.Tensor,
    distances: torch.Tensor,
    product_repr: "ProductRepr",
    bin_param1: float,
    bin_param2: float,
    num_bins: int,
    log_bins: bool = False,
    sh_scale: float = 0.1,
) -> torch.Tensor:
    """Apply block-diagonal multiplication with binned radial weights.

    This is the core computational kernel for SO(3)-equivariant layers.
    It computes: out[e] = Λ(r[e]) @ f[e] for each edge e, where Λ(r) is
    a block-diagonal weight matrix interpolated from the radial table.

    Includes solid harmonic scaling: weights are multiplied by
    (r/(r+scale))^(l_in+l_out) to suppress higher angular momentum at
    short distances, following the structure of solid harmonics.

    Input/output are in the m-first diagonal basis (not standard SH basis).
    The caller is responsible for P^T and Q basis transforms.

    Args:
        features: (num_edges, channels_in, dim_in)
            Edge features in m-first diagonal basis.
        radial_table: (num_bins + 1, channels_out, channels_in, weight_dim)
            Block-diagonal weights at bin edges.
        distances: (num_edges,)
            Edge distances for radial binning and solid harmonic scaling.
        product_repr: ProductRepr describing input/output representation structure.
        bin_param1: First binning parameter (min_val for linear, log_min for log).
        bin_param2: Second binning parameter (inv_bin_width for linear, inv_log_range for log).
        num_bins: Number of bins.
        log_bins: Whether to use logarithmic bin spacing (default False).
        sh_scale: Length scale for solid harmonic scaling (default 0.1).

    Returns:
        output: (num_edges, channels_out, dim_out)
            Output features in m-first diagonal basis.
    """
    device = features.device
    num_edges = features.size(0)
    channels_out = radial_table.size(1)
    dim_out = product_repr.rep2.dim()

    # Handle empty input
    if num_edges == 0:
        return torch.zeros(0, channels_out, dim_out, device=device, dtype=features.dtype)

    # Ensure contiguous memory layout for CUDA kernel
    features = features.contiguous()
    radial_table = radial_table.contiguous()
    distances = distances.to(features.dtype).contiguous()

    # Extract lvals tensors from representations
    lvals_in = product_repr.rep1.lvals.to(device=device, dtype=torch.int32).contiguous()
    lvals_out = product_repr.rep2.lvals.to(device=device, dtype=torch.int32).contiguous()

    # Compute block sizes for shared memory allocation
    max_in_size, max_out_size = _compute_block_sizes(lvals_in, lvals_out)

    # Run kernel
    return _BlockDiagonalFunction.apply(
        features, radial_table, distances, sh_scale,
        bin_param1, bin_param2, num_bins, log_bins,
        channels_out, lvals_in, lvals_out, dim_out, max_in_size, max_out_size
    )
