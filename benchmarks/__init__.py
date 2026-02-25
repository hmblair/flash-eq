"""Benchmarks for flash-eq components.

Modules:
    - benchmark_suite: Comprehensive benchmarks for all Flash-eq components
    - compare_all: 3-way comparison (SE(3)-T vs EquiformerV2 vs Flash-eq)
    - benchmark_patch: NVIDIA ConvSE3 vs PatchedConvSE3 comparison
    - se3t_baseline: SE(3)-Transformer reference implementation
    - equiformer_v2_baseline: EquiformerV2/eSCN reference implementation
    - nvidia_weight_conversion_prototype: Development prototype for patch module

Usage:
    python -m benchmarks.benchmark_suite       # Flash-eq component benchmarks
    python benchmarks/compare_all.py           # 3-way comparison
    python benchmarks/benchmark_patch.py       # Patch benchmarks
"""

from .benchmark_suite import (
    Scenario,
    SCENARIOS,
    QUICK_SCENARIOS,
    benchmark,
    run_component_benchmarks,
    run_transformer_benchmarks,
    run_scaling_analysis,
)

__all__ = [
    "Scenario",
    "SCENARIOS",
    "QUICK_SCENARIOS",
    "benchmark",
    "run_component_benchmarks",
    "run_transformer_benchmarks",
    "run_scaling_analysis",
]
