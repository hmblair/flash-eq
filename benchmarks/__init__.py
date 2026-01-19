"""Benchmarks for flash-eq components.

Provides benchmark utilities:
    - benchmark_suite: Comprehensive benchmarks for all Flash-eq components

Usage:
    python -m benchmarks.benchmark_suite  # Run full benchmark suite
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
