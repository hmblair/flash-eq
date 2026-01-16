"""
CUDA-accelerated block-diagonal multiplication for SO(3)-equivariant layers.

This module provides two implementations:
  - Binned (production): Memory-efficient with 5-13x reduction, 1.2-1.8x faster
  - Reference: Standard per-edge weights for baseline comparisons

The binned approach precomputes weights at K bin edges and interpolates at
runtime, avoiding the O(B * Cout * Cin * Wdim) memory bottleneck.

Supports FP16, FP32, and FP64 with FP32 accumulation for numerical stability.
Uses int64 indexing for batch dimension to support >2^31 edges.
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

_binned_module = None
_reference_module = None


def _get_binned_module():
    """JIT compile and load the binned CUDA extension."""
    global _binned_module
    if _binned_module is None:
        csrc_dir = Path(__file__).parent / "csrc"
        _binned_module = load(
            name="block_diagonal_binned",
            sources=[str(csrc_dir / "block_diagonal_binned.cu")],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _binned_module


def _get_reference_module():
    """JIT compile and load the reference CUDA extension."""
    global _reference_module
    if _reference_module is None:
        csrc_dir = Path(__file__).parent / "csrc"
        _reference_module = load(
            name="block_diagonal_reference",
            sources=[str(csrc_dir / "block_diagonal_reference.cu")],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _reference_module


def build_block_metadata(
    in_repr: Union[Repr, List[int]],
    out_repr: Union[Repr, List[int]],
    device: torch.device
) -> Tuple[torch.Tensor, ...]:
    """
    Build metadata tensors for the CUDA kernel.

    Args:
        in_repr: Input representation (Repr object or list of l-values)
        out_repr: Output representation (Repr object or list of l-values)
        device: Target device for metadata tensors

    Returns:
        Tuple of (block_data, dim_out, max_in_size, max_out_size)

    Note:
        When in_repr != out_repr, the block-diagonal weight matrix is non-square.
        Output m-components with no corresponding input (n_in=0) remain zero.
        Input m-components with no corresponding output (n_out=0) are ignored.
    """
    # Convert lists to Repr objects for backward compatibility
    if isinstance(in_repr, list):
        in_repr = Repr(lvals=in_repr)
    if isinstance(out_repr, list):
        out_repr = Repr(lvals=out_repr)

    return ProductRepr(in_repr, out_repr).build_block_metadata(device)


def get_weight_dim(lvals_in: List[int], lvals_out: List[int]) -> int:
    """
    Compute the weight dimension for given input/output representations.

    Args:
        lvals_in: List of input angular momentum values
        lvals_out: List of output angular momentum values

    Returns:
        Total number of weight parameters per (channel_out, channel_in) pair
    """
    return ProductRepr(Repr(lvals=lvals_in), Repr(lvals=lvals_out)).weight_dim()


# =============================================================================
# Reference Implementation (per-edge weights)
# =============================================================================

class BlockDiagonalFunction(Function):
    """Autograd function for standard block-diagonal multiplication."""

    @staticmethod
    def forward(ctx, features, weights, block_data, dim_out, max_in_size, max_out_size):
        cuda_module = _get_reference_module()

        output, = cuda_module.forward_v2(
            features.contiguous(),
            weights.contiguous(),
            block_data,
            dim_out,
            max_in_size
        )

        ctx.save_for_backward(features, weights, block_data)
        ctx.dim_in = features.size(2)
        ctx.max_in_size = max_in_size
        ctx.max_out_size = max_out_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, block_data = ctx.saved_tensors

        cuda_module = _get_reference_module()

        grad_features, grad_weights = cuda_module.backward_v2(
            grad_output.contiguous(),
            features,
            weights,
            block_data,
            ctx.dim_in,
            ctx.max_in_size,
            ctx.max_out_size
        )

        return grad_features, grad_weights, None, None, None, None


def block_diagonal_cuda(
    features: torch.Tensor,
    weights: torch.Tensor,
    metadata: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """
    Apply block-diagonal multiplication with per-edge weights.

    This is the reference implementation. For production use with radial MLPs,
    prefer block_diagonal_binned_interp_cuda() which provides significant
    memory and speed improvements.

    Args:
        features: (batch, channels_in, dim_in) - features in diagonal basis
        weights: (batch, channels_out, channels_in, weight_dim) - per-edge weights
        metadata: Tuple from build_block_metadata()

    Returns:
        output: (batch, channels_out, dim_out)
    """
    block_data, dim_out, max_in_size, max_out_size = metadata

    return BlockDiagonalFunction.apply(
        features, weights, block_data, dim_out, max_in_size, max_out_size
    )


# =============================================================================
# Binned Implementation (production)
# =============================================================================

class BlockDiagonalBinnedInterpFunction(Function):
    """Autograd function for binned interpolated block-diagonal multiplication."""

    @staticmethod
    def forward(ctx, features, radial_table, bin_lo, interp_weight,
                channels_out, num_bins, block_data, dim_out, max_in_size, max_out_size):
        cuda_module = _get_binned_module()

        output, = cuda_module.forward_binned_interp(
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
        ctx.channels_out = channels_out
        ctx.dim_in = features.size(2)
        ctx.max_in_size = max_in_size
        ctx.max_out_size = max_out_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, radial_table, bin_lo, interp_weight, block_data = ctx.saved_tensors

        cuda_module = _get_binned_module()

        grad_features, grad_radial_table, grad_interp_weight = cuda_module.backward_binned_interp(
            grad_output.contiguous(),
            features,
            radial_table,
            bin_lo.int(),
            interp_weight,
            block_data,
            ctx.dim_in,
            ctx.max_in_size,
            ctx.max_out_size
        )

        return (grad_features, grad_radial_table, None, grad_interp_weight,
                None, None, None, None, None, None)


def block_diagonal_binned_interp_cuda(
    features: torch.Tensor,
    radial_table: torch.Tensor,
    distances: torch.Tensor,
    metadata: Tuple[torch.Tensor, ...],
    min_dist: float = 0.0,
    max_dist: float = 10.0,
) -> torch.Tensor:
    """
    Apply block-diagonal multiplication with binned interpolated weights.

    This is the production implementation providing:
      - 5-13x memory reduction vs standard approach
      - 1.2-1.8x speedup during training
      - ~0.1% interpolation error at 100 bins

    Args:
        features: (batch, channels_in, dim_in) - features in diagonal basis
        radial_table: (num_bins + 1, channels_out, channels_in, weight_dim) - weights at bin edges
        distances: (batch,) - edge distances for binning
        metadata: Tuple from build_block_metadata()
        min_dist: Minimum distance for binning (default 0.0)
        max_dist: Maximum distance for binning (default 10.0 Angstroms)

    Returns:
        output: (batch, channels_out, dim_out)

    Gradient support:
        - grad_features: backprop through feature pathway
        - grad_radial_table: backprop through MLP (weighted scatter-add to bins)
        - grad_distances: for force computation
    """
    block_data, dim_out, max_in_size, max_out_size = metadata
    num_bins = radial_table.size(0) - 1
    channels_out = radial_table.size(1)

    # Compute bin indices and interpolation weights (O(1) arithmetic)
    # Use float32 for binning arithmetic, then convert to feature dtype
    distances_f32 = distances.float()
    inv_bin_width = num_bins / (max_dist - min_dist)
    normalized = (distances_f32 - min_dist) * inv_bin_width
    normalized = normalized.clamp(0.0, num_bins)
    bin_lo = normalized.floor().int().clamp(max=num_bins - 1)
    interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0).to(features.dtype)

    return BlockDiagonalBinnedInterpFunction.apply(
        features, radial_table, bin_lo, interp_weight,
        channels_out, num_bins, block_data, dim_out, max_in_size, max_out_size
    )


# =============================================================================
# Gathered Implementation (fused node->edge gather with bin-sorted edges)
# =============================================================================

class BlockDiagonalGatheredFunction(Function):
    """Autograd function for gathered block-diagonal multiplication.

    This variant reads node features directly using src_indices, avoiding the
    need to pre-expand features to edges. Combined with bin-sorted edge ordering,
    this provides better L2 cache utilization for the radial weight table.
    """

    @staticmethod
    def forward(ctx, node_features, src_indices, radial_table, bin_lo, interp_weight,
                channels_out, num_bins, num_nodes, block_data, dim_out, max_in_size, max_out_size):
        cuda_module = _get_binned_module()

        output, = cuda_module.forward_gathered(
            node_features.contiguous(),
            src_indices.contiguous(),
            radial_table.contiguous(),
            bin_lo.contiguous().int(),
            interp_weight.contiguous(),
            block_data,
            channels_out,
            dim_out,
            num_bins,
            max_in_size
        )

        ctx.save_for_backward(node_features, src_indices, radial_table, bin_lo, interp_weight, block_data)
        ctx.num_nodes = num_nodes
        ctx.dim_in = node_features.size(2)
        ctx.max_in_size = max_in_size
        ctx.max_out_size = max_out_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        node_features, src_indices, radial_table, bin_lo, interp_weight, block_data = ctx.saved_tensors

        cuda_module = _get_binned_module()

        grad_node_features, grad_radial_table, grad_interp_weight = cuda_module.backward_gathered(
            grad_output.contiguous(),
            node_features,
            src_indices,
            radial_table,
            bin_lo.int(),
            interp_weight,
            block_data,
            ctx.num_nodes,
            ctx.dim_in,
            ctx.max_in_size,
            ctx.max_out_size
        )

        return (grad_node_features, None, grad_radial_table, None, grad_interp_weight,
                None, None, None, None, None, None, None)


def compute_bin_sorted_order(
    distances: torch.Tensor,
    num_bins: int = 100,
    min_dist: float = 0.0,
    max_dist: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute bin-sorted edge ordering for improved cache utilization.

    Sorting edges by their distance bin ensures that consecutive edges access
    the same region of the radial weight table, improving L2 cache hit rates.

    Args:
        distances: (num_edges,) - edge distances
        num_bins: Number of bins for the radial table
        min_dist: Minimum distance for binning
        max_dist: Maximum distance for binning

    Returns:
        sorted_order: (num_edges,) - indices to reorder edges by bin
        bin_lo: (num_edges,) - lower bin index for each sorted edge
        interp_weight: (num_edges,) - interpolation weight for each sorted edge
    """
    # Compute bin indices
    distances_f32 = distances.float()
    inv_bin_width = num_bins / (max_dist - min_dist)
    normalized = (distances_f32 - min_dist) * inv_bin_width
    normalized = normalized.clamp(0.0, num_bins)
    bin_lo = normalized.floor().int().clamp(max=num_bins - 1)

    # Sort edges by bin index (stable sort preserves order within bins)
    sorted_order = torch.argsort(bin_lo, stable=True)

    # Reorder bin indices and compute interpolation weights
    bin_lo_sorted = bin_lo[sorted_order]
    normalized_sorted = normalized[sorted_order]
    interp_weight = (normalized_sorted - bin_lo_sorted.float()).clamp(0.0, 1.0)

    return sorted_order, bin_lo_sorted, interp_weight


def block_diagonal_gathered_cuda(
    node_features: torch.Tensor,
    src_indices: torch.Tensor,
    radial_table: torch.Tensor,
    distances: torch.Tensor,
    metadata: Tuple[torch.Tensor, ...],
    min_dist: float = 0.0,
    max_dist: float = 10.0,
    sort_by_bin: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply block-diagonal multiplication with fused gather from node features.

    This implementation:
      - Reads features directly from nodes via src_indices (no pre-expansion)
      - Optionally sorts edges by bin for better L2 cache utilization
      - Returns output in sorted order (with unsort indices if sorted)

    Args:
        node_features: (num_nodes, channels_in, dim_in) - node features
        src_indices: (num_edges,) - source node index for each edge
        radial_table: (num_bins + 1, channels_out, channels_in, weight_dim)
        distances: (num_edges,) - edge distances for binning
        metadata: Tuple from build_block_metadata()
        min_dist: Minimum distance for binning (default 0.0)
        max_dist: Maximum distance for binning (default 10.0)
        sort_by_bin: Whether to sort edges by bin (default True, recommended)

    Returns:
        output: (num_edges, channels_out, dim_out) - in sorted order if sort_by_bin=True
        unsort_indices: (num_edges,) - indices to restore original order, or None if not sorted
    """
    block_data, dim_out, max_in_size, max_out_size = metadata
    num_bins = radial_table.size(0) - 1
    channels_out = radial_table.size(1)
    num_nodes = node_features.size(0)

    if sort_by_bin:
        # Sort edges by bin for better cache utilization
        sorted_order, bin_lo, interp_weight = compute_bin_sorted_order(
            distances, num_bins, min_dist, max_dist
        )
        interp_weight = interp_weight.to(node_features.dtype)

        # Reorder src_indices to match sorted order
        src_indices_sorted = src_indices[sorted_order]

        # Compute unsort indices (inverse permutation)
        unsort_indices = torch.argsort(sorted_order)

        output = BlockDiagonalGatheredFunction.apply(
            node_features, src_indices_sorted, radial_table, bin_lo, interp_weight,
            channels_out, num_bins, num_nodes, block_data, dim_out, max_in_size, max_out_size
        )

        return output, unsort_indices
    else:
        # No sorting - compute bin indices directly
        distances_f32 = distances.float()
        inv_bin_width = num_bins / (max_dist - min_dist)
        normalized = (distances_f32 - min_dist) * inv_bin_width
        normalized = normalized.clamp(0.0, num_bins)
        bin_lo = normalized.floor().int().clamp(max=num_bins - 1)
        interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0).to(node_features.dtype)

        output = BlockDiagonalGatheredFunction.apply(
            node_features, src_indices, radial_table, bin_lo, interp_weight,
            channels_out, num_bins, num_nodes, block_data, dim_out, max_in_size, max_out_size
        )

        return output, None
