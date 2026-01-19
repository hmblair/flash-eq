"""CUDA-accelerated kernels for SO(3)-equivariant operations.

This package provides the core CUDA kernels for flash-eq:
- block_diagonal_cuda: Block-diagonal multiplication with binned radial weights

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from .block_diagonal import block_diagonal_cuda, CUDANotAvailableError

__all__ = [
    "block_diagonal_cuda",
    "CUDANotAvailableError",
]
