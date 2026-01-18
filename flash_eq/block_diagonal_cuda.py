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

import os
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .representations import ProductRepr

# Set CUDA_HOME if available
if os.path.exists("/usr/local/cuda-12.6"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.6"

_cuda_module = None


def _get_cuda_module():
    """JIT compile and load the CUDA extension."""
    global _cuda_module
    if _cuda_module is None:
        csrc_dir = Path(__file__).parent / "csrc"
        _cuda_module = load(
            name="block_diagonal_cuda",
            sources=[str(csrc_dir / "block_diagonal.cu")],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _cuda_module


# =============================================================================
# Internal Autograd Function
# =============================================================================

class _BlockDiagonalFunction(Function):
    """Autograd function wrapping the CUDA kernel with bin-sorted edges."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        features: torch.Tensor,
        radial_table: torch.Tensor,
        bin_lo: torch.Tensor,
        interp_weight: torch.Tensor,
        channels_out: int,
        num_bins: int,
        lvals_in: torch.Tensor,
        lvals_out: torch.Tensor,
        dim_out: int,
        max_in_size: int,
        max_out_size: int,
    ) -> torch.Tensor:
        cuda_module = _get_cuda_module()

        # Sort edges by bin for better memory access patterns
        sort_idx = torch.argsort(bin_lo)
        unsort_idx = torch.argsort(sort_idx)

        features_sorted = features[sort_idx]
        bin_lo_sorted = bin_lo[sort_idx]
        interp_weight_sorted = interp_weight[sort_idx]

        output_sorted, = cuda_module.forward(
            features_sorted,
            radial_table,
            bin_lo_sorted,
            interp_weight_sorted,
            lvals_in,
            lvals_out,
            channels_out,
            dim_out,
            num_bins,
            max_in_size
        )

        # Unsort output back to original edge order
        output = output_sorted[unsort_idx]

        ctx.save_for_backward(features_sorted, radial_table, bin_lo_sorted,
                              interp_weight_sorted, lvals_in, lvals_out, sort_idx, unsort_idx)
        ctx.dim_in = features.size(2)
        ctx.max_in_size = max_in_size
        ctx.max_out_size = max_out_size

        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (features_sorted, radial_table, bin_lo_sorted, interp_weight_sorted,
         lvals_in, lvals_out, sort_idx, unsort_idx) = ctx.saved_tensors
        cuda_module = _get_cuda_module()

        # Sort grad_output to match the sorted forward pass order
        grad_output_sorted = grad_output[sort_idx]

        grad_features_sorted, grad_radial_table, grad_interp_weight_sorted = cuda_module.backward(
            grad_output_sorted,
            features_sorted,
            radial_table,
            bin_lo_sorted,
            interp_weight_sorted,
            lvals_in,
            lvals_out,
            ctx.dim_in,
            ctx.max_in_size,
            ctx.max_out_size
        )

        # Unsort gradients back to original edge order
        grad_features = grad_features_sorted[unsort_idx]
        grad_interp_weight = grad_interp_weight_sorted[unsort_idx]

        return (grad_features, grad_radial_table, None, grad_interp_weight,
                None, None, None, None, None, None, None)


# =============================================================================
# Main Public API
# =============================================================================

def _compute_block_sizes(lvals_in: torch.Tensor, lvals_out: torch.Tensor) -> tuple[int, int]:
    """Compute max block sizes for shared memory allocation.

    Returns:
        (max_in_size, max_out_size): Maximum block sizes across all m values.
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
    bin_lo: torch.Tensor,
    interp_weight: torch.Tensor,
    product_repr: "ProductRepr",
) -> torch.Tensor:
    """
    Apply block-diagonal multiplication with binned radial weights.

    This is the core computational kernel for SO(3)-equivariant layers.
    It computes: out[e] = Λ(r[e]) @ f[e] for each edge e, where Λ(r) is
    a block-diagonal weight matrix interpolated from the radial table.

    Input/output are in the m-first diagonal basis (not standard SH basis).
    The caller is responsible for P^T and Q basis transforms.

    Args:
        features: (num_edges, channels_in, dim_in)
            Edge features in m-first diagonal basis.
        radial_table: (num_bins + 1, channels_out, channels_in, weight_dim)
            Block-diagonal weights at bin edges.
        bin_lo: (num_edges,) int32
            Lower bin index for each edge (from BinnedModule.bin_indices).
        interp_weight: (num_edges,)
            Interpolation weight in [0, 1] for each edge.
        product_repr: ProductRepr describing input/output representation structure.

    Returns:
        output: (num_edges, channels_out, dim_out)
            Output features in m-first diagonal basis.
    """
    device = features.device
    num_bins = radial_table.size(0) - 1
    channels_out = radial_table.size(1)

    # Ensure contiguous memory layout for CUDA kernel
    features = features.contiguous()
    radial_table = radial_table.contiguous()
    bin_lo = bin_lo.contiguous()
    interp_weight = interp_weight.to(features.dtype).contiguous()

    # Extract lvals tensors from representations
    lvals_in = product_repr.rep1.lvals.to(device=device, dtype=torch.int32).contiguous()
    lvals_out = product_repr.rep2.lvals.to(device=device, dtype=torch.int32).contiguous()

    # Compute block sizes for shared memory allocation
    dim_out = product_repr.rep2.dim()
    max_in_size, max_out_size = _compute_block_sizes(lvals_in, lvals_out)

    # Run kernel
    return _BlockDiagonalFunction.apply(
        features, radial_table, bin_lo, interp_weight,
        channels_out, num_bins, lvals_in, lvals_out, dim_out, max_in_size, max_out_size
    )
