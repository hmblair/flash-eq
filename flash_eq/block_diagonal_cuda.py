"""
CUDA-accelerated block-diagonal multiplication for SO(3)-equivariant layers.

This module provides two implementations:
  - Binned (production): Memory-efficient with 5-13x reduction, 1.2-1.8x faster
  - Reference: Standard per-edge weights for baseline comparisons

The binned approach precomputes weights at K bin edges and interpolates at
runtime, avoiding the O(B * Cout * Cin * Wdim) memory bottleneck.

Supports FP16, FP32, and FP64 with FP32 accumulation for numerical stability.
"""

import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import List, Tuple

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
    lvals_in: List[int],
    lvals_out: List[int],
    device: torch.device
) -> Tuple[torch.Tensor, ...]:
    """
    Build metadata tensors for the CUDA kernel.

    Args:
        lvals_in: List of input angular momentum values (e.g., [0, 1, 2])
        lvals_out: List of output angular momentum values
        device: Target device for metadata tensors

    Returns:
        Tuple of metadata tensors for block structure
    """
    lmax = max(max(lvals_in), max(lvals_out))

    def count(lvals, m):
        return sum(1 for l in lvals if l >= m)

    blocks = []
    in_off = out_off = w_off = 0

    for m in range(lmax + 1):
        n_in, n_out = count(lvals_in, m), count(lvals_out, m)
        if n_in > 0 and n_out > 0:
            blocks.append({
                'm': m,
                'n_in': n_in,
                'n_out': n_out,
                'in_off': in_off,
                'out_off': out_off,
                'w_off': w_off,
            })
            mult = 1 if m == 0 else 2
            in_off += mult * n_in
            out_off += mult * n_out
            w_off += mult * n_out * n_in

    dim_out = out_off

    # Pack block metadata into single (num_blocks, 6) tensor
    # Columns: [m, n_in, n_out, in_off, out_off, w_off]
    block_data = torch.tensor(
        [[b['m'], b['n_in'], b['n_out'], b['in_off'], b['out_off'], b['w_off']] for b in blocks],
        dtype=torch.int32, device=device
    )

    # Compute max sizes for shared memory allocation
    max_in_size = max(b['n_in'] if b['m'] == 0 else 2 * b['n_in'] for b in blocks)
    max_out_size = max(b['n_out'] if b['m'] == 0 else 2 * b['n_out'] for b in blocks)

    return (block_data, dim_out, max_in_size, max_out_size)


def get_weight_dim(lvals_in: List[int], lvals_out: List[int]) -> int:
    """
    Compute the weight dimension for given input/output representations.

    Args:
        lvals_in: List of input angular momentum values
        lvals_out: List of output angular momentum values

    Returns:
        Total number of weight parameters per (channel_out, channel_in) pair
    """
    lmax = max(max(lvals_in), max(lvals_out))

    def count(lvals, m):
        return sum(1 for l in lvals if l >= m)

    weight_dim = 0
    for m in range(lmax + 1):
        n_in, n_out = count(lvals_in, m), count(lvals_out, m)
        if n_in > 0 and n_out > 0:
            mult = 1 if m == 0 else 2
            weight_dim += mult * n_out * n_in

    return weight_dim


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
