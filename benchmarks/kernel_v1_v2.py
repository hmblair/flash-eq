"""
Benchmark V1 vs V2 block-diagonal kernels.

V1: Parallelizes by (batch, cout, out_idx) - one output per thread block
V2: Parallelizes by (batch, m_block) - one m-block per thread block with shared memory
"""

import torch
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_cuda_v2,
    get_weight_dim,
)


def benchmark_kernel(fn, num_iters=100, warmup=10):
    """Benchmark a kernel function."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def check_correctness(lvals, batch_size, channels_in, channels_out, dtype=torch.float32):
    """Verify V2 produces same results as V1."""
    device = torch.device("cuda")
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    features = torch.randn(batch_size, channels_in, dim, device=device, dtype=dtype)
    weights = torch.randn(batch_size, channels_out, channels_in, weight_dim, device=device, dtype=dtype)
    metadata = build_block_metadata(lvals, lvals, device)

    out_v1 = block_diagonal_cuda(features, weights, metadata)
    out_v2 = block_diagonal_cuda_v2(features, weights, metadata)

    diff = (out_v1 - out_v2).abs().max().item()
    rel_diff = diff / out_v1.abs().mean().item()
    return rel_diff


def run_benchmark(lvals, batch_size, channels_in, channels_out, dtype=torch.float32):
    """Run benchmark comparing V1 and V2."""
    device = torch.device("cuda")
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    features = torch.randn(batch_size, channels_in, dim, device=device, dtype=dtype)
    weights = torch.randn(batch_size, channels_out, channels_in, weight_dim, device=device, dtype=dtype)
    metadata = build_block_metadata(lvals, lvals, device)

    t_v1 = benchmark_kernel(lambda: block_diagonal_cuda(features, weights, metadata))
    t_v2 = benchmark_kernel(lambda: block_diagonal_cuda_v2(features, weights, metadata))

    return t_v1, t_v2


def main():
    print("=" * 90)
    print("Block-Diagonal Kernel Benchmark: V1 (per-output) vs V2 (per-m-block)")
    print("=" * 90)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    # First check correctness
    print("\n" + "-" * 90)
    print("Correctness Check (V2 vs V1)")
    print("-" * 90)

    for lmax in [3, 6]:
        lvals = list(range(lmax + 1))
        for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            rel_diff = check_correctness(lvals, 32, 64, 64, dtype)
            status = "PASS" if rel_diff < 1e-3 else "FAIL"
            print(f"L={lmax}, {name}: rel_diff={rel_diff:.2e} [{status}]")

    # Benchmark configurations
    configs = [
        # (lmax, batch, cin, cout)
        (3, 1000, 64, 64),
        (4, 1000, 64, 64),
        (6, 1000, 64, 64),
        (6, 2000, 64, 64),
        (6, 5000, 64, 64),
        (6, 5000, 32, 32),
        (6, 10000, 32, 32),
    ]

    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n{'-' * 90}")
        print(f"{dtype_name} Performance")
        print(f"{'-' * 90}")
        print(f"{'Config':<40} {'V1':>12} {'V2':>12} {'Speedup':>12} {'V1 blocks':>14} {'V2 blocks':>14}")
        print(f"{'-' * 90}")

        for lmax, batch, cin, cout in configs:
            lvals = list(range(lmax + 1))
            dim = sum(2 * l + 1 for l in lvals)
            num_m_blocks = lmax + 1

            # Thread block counts
            v1_blocks = batch * cout * dim
            v2_blocks = batch * num_m_blocks

            try:
                t_v1, t_v2 = run_benchmark(lvals, batch, cin, cout, dtype)
                speedup = t_v1 / t_v2

                config_str = f"L={lmax}, B={batch}, C={cin}x{cout}, D={dim}"
                print(f"{config_str:<40} {t_v1:>11.3f}ms {t_v2:>11.3f}ms {speedup:>11.2f}x {v1_blocks:>14,} {v2_blocks:>14,}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"L={lmax}, B={batch}, C={cin}x{cout}: OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

    print("\n" + "=" * 90)
    print("V1: One thread block per (batch, cout, out_idx), 128 threads reduce over cin")
    print("V2: One thread block per (batch, m_block), 256 threads handle (cout, out_local) pairs")
    print("=" * 90)


if __name__ == "__main__":
    main()
