"""
CUDA-accelerated block-diagonal multiplication for SO(3)-equivariant layers.

This module provides optimized CUDA kernels for the Λ (Lambda) block-diagonal
multiplication step, which handles both real (m=0) and complex (m>0) irreps.

The default kernel (V2) parallelizes by (batch, m-block) with shared memory
caching for 2-9x speedup over the original per-output parallelization (V1).

Supports FP16, FP32, and FP64 with FP32 accumulation for numerical stability.
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import List, Tuple

# Set CUDA_HOME to 12.6 if available
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
        Tuple of metadata tensors: (block_m, block_n_in, block_n_out, block_in_off,
        block_out_off, block_w_off, out_to_block, out_to_local, block_in_size, dim_out)
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

    # Build output-to-block mapping
    out_to_block = []
    out_to_local = []
    for blk_idx, blk in enumerate(blocks):
        m, n_out = blk['m'], blk['n_out']
        size = n_out if m == 0 else 2 * n_out
        for local_idx in range(size):
            out_to_block.append(blk_idx)
            out_to_local.append(local_idx)

    # Convert to tensors
    block_m = torch.tensor([b['m'] for b in blocks], dtype=torch.int32, device=device)
    block_n_in = torch.tensor([b['n_in'] for b in blocks], dtype=torch.int32, device=device)
    block_n_out = torch.tensor([b['n_out'] for b in blocks], dtype=torch.int32, device=device)
    block_in_off = torch.tensor([b['in_off'] for b in blocks], dtype=torch.int32, device=device)
    block_out_off = torch.tensor([b['out_off'] for b in blocks], dtype=torch.int32, device=device)
    block_w_off = torch.tensor([b['w_off'] for b in blocks], dtype=torch.int32, device=device)
    out_to_block = torch.tensor(out_to_block, dtype=torch.int32, device=device)
    out_to_local = torch.tensor(out_to_local, dtype=torch.int32, device=device)

    # block_in_size: n_in for m=0, 2*n_in for m>0 (for v2 kernel shared memory)
    block_in_size = torch.tensor(
        [b['n_in'] if b['m'] == 0 else 2 * b['n_in'] for b in blocks],
        dtype=torch.int32, device=device
    )

    return (block_m, block_n_in, block_n_out, block_in_off, block_out_off,
            block_w_off, out_to_block, out_to_local, block_in_size, dim_out)


class BlockDiagonalFunction(Function):
    """Autograd function for block-diagonal multiplication using V2 kernel."""

    @staticmethod
    def forward(ctx, features, weights, block_m, block_n_in, block_n_out,
                block_in_off, block_out_off, block_w_off, out_to_block,
                out_to_local, block_in_size, dim_out):
        cuda_module = _get_cuda_module()

        # Use V2 (m-block parallel) kernel for forward pass
        output, = cuda_module.forward_v2(
            features.contiguous(),
            weights.contiguous(),
            block_m, block_n_in, block_n_out,
            block_in_off, block_out_off, block_w_off,
            block_in_size,
            dim_out
        )

        ctx.save_for_backward(features, weights, block_m, block_n_in, block_n_out,
                              block_in_off, block_out_off, block_w_off)
        ctx.dim_in = features.size(2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, block_m, block_n_in, block_n_out, \
            block_in_off, block_out_off, block_w_off = ctx.saved_tensors

        cuda_module = _get_cuda_module()

        # Use V1 kernel for backward (V2 backward not yet implemented)
        grad_features, grad_weights = cuda_module.backward(
            grad_output.contiguous(),
            features,
            weights,
            block_m, block_n_in, block_n_out,
            block_in_off, block_out_off, block_w_off,
            ctx.dim_in
        )

        return grad_features, grad_weights, None, None, None, None, None, None, None, None, None, None


class BlockDiagonalFunctionV1(Function):
    """Autograd function using V1 (per-output) kernel. Kept for compatibility."""

    @staticmethod
    def forward(ctx, features, weights, block_m, block_n_in, block_n_out,
                block_in_off, block_out_off, block_w_off, out_to_block,
                out_to_local, block_in_size, dim_out):
        cuda_module = _get_cuda_module()

        output, = cuda_module.forward(
            features.contiguous(),
            weights.contiguous(),
            block_m, block_n_in, block_n_out,
            block_in_off, block_out_off, block_w_off,
            out_to_block, out_to_local,
            dim_out
        )

        ctx.save_for_backward(features, weights, block_m, block_n_in, block_n_out,
                              block_in_off, block_out_off, block_w_off)
        ctx.dim_in = features.size(2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, block_m, block_n_in, block_n_out, \
            block_in_off, block_out_off, block_w_off = ctx.saved_tensors

        cuda_module = _get_cuda_module()

        grad_features, grad_weights = cuda_module.backward(
            grad_output.contiguous(),
            features,
            weights,
            block_m, block_n_in, block_n_out,
            block_in_off, block_out_off, block_w_off,
            ctx.dim_in
        )

        return grad_features, grad_weights, None, None, None, None, None, None, None, None, None, None


def block_diagonal_cuda(
    features: torch.Tensor,
    weights: torch.Tensor,
    metadata: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """
    Apply block-diagonal multiplication using optimized CUDA kernel.

    Uses V2 kernel (m-block parallelization with shared memory) which is
    2-9x faster than V1 depending on configuration.

    This implements the Λ step in: output = Q @ Λ @ P^T @ features

    Args:
        features: (batch, channels_in, dim_in) - features in diagonal (m-ordered) basis
        weights: (batch, channels_out, channels_in, weight_dim) - block-diagonal weights
        metadata: Tuple from build_block_metadata()

    Returns:
        output: (batch, channels_out, dim_out)

    Supports FP16, FP32, and FP64. Internal accumulation is done in FP32.
    """
    (block_m, block_n_in, block_n_out, block_in_off, block_out_off,
     block_w_off, out_to_block, out_to_local, block_in_size, dim_out) = metadata

    return BlockDiagonalFunction.apply(
        features, weights, block_m, block_n_in, block_n_out,
        block_in_off, block_out_off, block_w_off, out_to_block,
        out_to_local, block_in_size, dim_out
    )


def block_diagonal_cuda_v1(
    features: torch.Tensor,
    weights: torch.Tensor,
    metadata: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """
    Apply block-diagonal multiplication using V1 CUDA kernel (per-output parallel).

    V1 parallelizes by (batch, cout, out_idx) with reduction over channels_in.
    Slower than V2 but kept for compatibility and comparison.

    Args:
        features: (batch, channels_in, dim_in) - features in diagonal (m-ordered) basis
        weights: (batch, channels_out, channels_in, weight_dim) - block-diagonal weights
        metadata: Tuple from build_block_metadata()

    Returns:
        output: (batch, channels_out, dim_out)
    """
    (block_m, block_n_in, block_n_out, block_in_off, block_out_off,
     block_w_off, out_to_block, out_to_local, block_in_size, dim_out) = metadata

    return BlockDiagonalFunctionV1.apply(
        features, weights, block_m, block_n_in, block_n_out,
        block_in_off, block_out_off, block_w_off, out_to_block,
        out_to_local, block_in_size, dim_out
    )


# Alias for explicit V2 usage
block_diagonal_cuda_v2 = block_diagonal_cuda


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
