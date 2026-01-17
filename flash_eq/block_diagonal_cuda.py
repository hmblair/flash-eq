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
    - Bin-sorted edge ordering: L2 cache locality for weight table
    - Linear interpolation between bin edges

Supports FP32 and FP64. Also supports FP16.
"""

import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import List, Tuple, Union

from .representations import Repr, ProductRepr

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
# Metadata Construction
# =============================================================================

def build_block_metadata(
    in_repr: Union[Repr, List[int]],
    out_repr: Union[Repr, List[int]],
    device: torch.device
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Build metadata tensors describing the block-diagonal structure.

    The block structure groups components by |m| value:
        m=0: n_in × n_out block (real scalars)
        m>0: 2×n_in × 2×n_out block (complex, stored as real/imag pairs)

    Args:
        in_repr: Input representation (Repr or list of l-values).
        out_repr: Output representation (Repr or list of l-values).
        device: Target device for metadata tensors.

    Returns:
        Tuple of (block_data, dim_out, max_in_size, max_out_size):
            block_data: (num_blocks, 6) int32 tensor with columns
                        [m, n_in, n_out, in_offset, out_offset, weight_offset]
            dim_out: Total output dimension.
            max_in_size: Largest input block size (for shared memory).
            max_out_size: Largest output block size (for shared memory).
    """
    if isinstance(in_repr, list):
        in_repr = Repr(lvals=in_repr)
    if isinstance(out_repr, list):
        out_repr = Repr(lvals=out_repr)

    return ProductRepr(in_repr, out_repr).build_block_metadata(device)


def get_weight_dim(lvals_in: List[int], lvals_out: List[int]) -> int:
    """
    Compute the weight dimension for given input/output representations.

    This is the number of scalar weights per (channel_out, channel_in) pair
    in the block-diagonal parameterization.

    Args:
        lvals_in: List of input angular momentum values.
        lvals_out: List of output angular momentum values.

    Returns:
        Weight dimension (sum over blocks of block_size).
    """
    return ProductRepr(Repr(lvals=lvals_in), Repr(lvals=lvals_out)).nreps()


# =============================================================================
# Internal Autograd Function
# =============================================================================

class _BlockDiagonalFunction(Function):
    """Autograd function wrapping the CUDA kernel."""

    @staticmethod
    def forward(ctx, features, radial_table, bin_lo, interp_weight,
                channels_out, num_bins, block_data, dim_out, max_in_size, max_out_size):
        cuda_module = _get_cuda_module()

        output, = cuda_module.forward(
            features.contiguous(),
            radial_table.contiguous(),
            bin_lo.contiguous().int(),
            interp_weight.contiguous(),
            block_data,
            channels_out,
            dim_out,
            num_bins,
            max_in_size
        )

        ctx.save_for_backward(features, radial_table, bin_lo, interp_weight, block_data)
        ctx.dim_in = features.size(2)
        ctx.max_in_size = max_in_size
        ctx.max_out_size = max_out_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, radial_table, bin_lo, interp_weight, block_data = ctx.saved_tensors
        cuda_module = _get_cuda_module()

        grad_features, grad_radial_table, grad_interp_weight = cuda_module.backward(
            grad_output.contiguous(),
            features.contiguous(),
            radial_table.contiguous(),
            bin_lo.contiguous().int(),
            interp_weight.contiguous(),
            block_data.contiguous(),
            ctx.dim_in,
            ctx.max_in_size,
            ctx.max_out_size
        )

        return (grad_features, grad_radial_table, None, grad_interp_weight,
                None, None, None, None, None, None)


# =============================================================================
# Bin Sorting (Cache Optimization)
# =============================================================================

def _compute_bin_sorted_order(
    distances: torch.Tensor,
    num_bins: int,
    min_dist: float,
    max_dist: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort edges by distance bin for improved cache locality.

    Edges accessing similar bins are grouped together, improving L2 cache
    hit rate when reading the radial weight table.

    Args:
        distances: (num_edges,) edge distances.
        num_bins: Number of bins in the radial table.
        min_dist: Minimum distance (maps to bin 0).
        max_dist: Maximum distance (maps to bin num_bins).

    Returns:
        sorted_order: (num_edges,) indices to reorder edges by bin.
        unsort_indices: (num_edges,) indices to restore original order.
        bin_lo: (num_edges,) lower bin index for each sorted edge.
        interp_weight: (num_edges,) interpolation weight t in [0,1] for each sorted edge.
    """
    distances_f32 = distances.float()
    inv_bin_width = num_bins / (max_dist - min_dist)
    normalized = (distances_f32 - min_dist) * inv_bin_width
    normalized = normalized.clamp(0.0, num_bins)
    bin_lo = normalized.floor().int().clamp(max=num_bins - 1)

    # Stable sort preserves order within bins
    sorted_order = torch.argsort(bin_lo, stable=True)
    unsort_indices = torch.argsort(sorted_order)

    bin_lo_sorted = bin_lo[sorted_order]
    normalized_sorted = normalized[sorted_order]
    interp_weight = (normalized_sorted - bin_lo_sorted.float()).clamp(0.0, 1.0)

    return sorted_order, unsort_indices, bin_lo_sorted, interp_weight


# =============================================================================
# Main Public API
# =============================================================================

def block_diagonal_cuda(
    features: torch.Tensor,
    radial_table: torch.Tensor,
    distances: torch.Tensor,
    metadata: Tuple[torch.Tensor, int, int, int],
    min_dist: float = 0.0,
    max_dist: float = 10.0,
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
            Block-diagonal weights at bin edges. Interpolated linearly.
        distances: (num_edges,)
            Edge distances for weight interpolation.
        metadata: Tuple from build_block_metadata().
        min_dist: Minimum distance for binning (default 0.0).
        max_dist: Maximum distance for binning (default 10.0).

    Returns:
        output: (num_edges, channels_out, dim_out)
            Output features in m-first diagonal basis.
    """
    block_data, dim_out, max_in_size, max_out_size = metadata
    num_bins = radial_table.size(0) - 1
    channels_out = radial_table.size(1)

    # Sort edges by bin for cache locality
    sorted_order, unsort_indices, bin_lo, interp_weight = _compute_bin_sorted_order(
        distances, num_bins, min_dist, max_dist
    )
    interp_weight = interp_weight.to(features.dtype)

    # Reorder features to match sorted order
    features_sorted = features[sorted_order]

    # Run kernel
    output_sorted = _BlockDiagonalFunction.apply(
        features_sorted, radial_table, bin_lo, interp_weight,
        channels_out, num_bins, block_data, dim_out, max_in_size, max_out_size
    )

    # Restore original edge order
    return output_sorted[unsort_indices]
