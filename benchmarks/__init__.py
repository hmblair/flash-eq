"""Benchmarks for flash-eq CUDA kernels and pipelines."""

from .kernel import benchmark_kernel
from .pipeline import benchmark_pipeline

__all__ = ["benchmark_kernel", "benchmark_pipeline"]
