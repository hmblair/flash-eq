#!/usr/bin/env python
"""
CLI for running flash-eq benchmarks.

Usage:
    python -m benchmarks.run kernel     # Benchmark CUDA kernel vs Python
    python -m benchmarks.run pipeline   # Benchmark full pipeline vs dense
    python -m benchmarks.run all        # Run all benchmarks
"""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run flash-eq benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks.run kernel      # CUDA kernel vs Python/einsum
    python -m benchmarks.run pipeline    # Full pipeline (Wigner-D + kernel) vs dense
    python -m benchmarks.run all         # All benchmarks
    python -m benchmarks.run kernel --fp16-only  # Only FP16
    python -m benchmarks.run kernel --fp32-only  # Only FP32
        """,
    )

    parser.add_argument(
        "benchmark",
        choices=["kernel", "pipeline", "all"],
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--fp16-only",
        action="store_true",
        help="Only run FP16 benchmarks",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="Only run FP32 benchmarks",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for benchmarks")
        sys.exit(1)

    # Determine dtypes
    if args.fp16_only and args.fp32_only:
        print("ERROR: Cannot specify both --fp16-only and --fp32-only")
        sys.exit(1)

    dtypes = None
    if args.fp16_only:
        dtypes = [torch.float16]
    elif args.fp32_only:
        dtypes = [torch.float32]

    # Run benchmarks
    if args.benchmark in ["kernel", "all"]:
        from .kernel import run_kernel_benchmark
        run_kernel_benchmark(dtypes=dtypes)

    if args.benchmark in ["pipeline", "all"]:
        from .pipeline import run_pipeline_benchmark
        if args.benchmark == "all":
            print("\n\n")
        run_pipeline_benchmark(dtypes=dtypes)


if __name__ == "__main__":
    main()
