"""
Benchmark CUDA block-diagonal kernel vs Python implementation.

This measures the performance of the isolated block-diagonal multiplication step,
which is the Lambda (Λ) operation in: output = Q @ Λ @ P^T @ features
"""

import torch
import time
from typing import List, Tuple

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    get_weight_dim,
)


def _block_diagonal_python(features, weights, lvals_in, lvals_out):
    """Pure Python reference implementation for comparison."""
    lmax = max(max(lvals_in), max(lvals_out))
    count = lambda lvals, m: sum(1 for l in lvals if l >= m)

    batch, channels_in, _ = features.shape
    channels_out = weights.shape[1]

    blocks = []
    in_off = out_off = w_off = 0
    for m in range(lmax + 1):
        n_in, n_out = count(lvals_in, m), count(lvals_out, m)
        if n_in > 0 and n_out > 0:
            blocks.append({
                'm': m, 'n_in': n_in, 'n_out': n_out,
                'in_off': in_off, 'out_off': out_off, 'w_off': w_off
            })
            mult = 1 if m == 0 else 2
            in_off += mult * n_in
            out_off += mult * n_out
            w_off += mult * n_out * n_in

    dim_out = out_off
    out = torch.zeros(batch, channels_out, dim_out, device=features.device, dtype=features.dtype)

    for blk in blocks:
        m = blk['m']
        n_in, n_out = blk['n_in'], blk['n_out']
        in_s, out_s, w_off = blk['in_off'], blk['out_off'], blk['w_off']

        if m == 0:
            f_m = features[:, :, in_s:in_s + n_in]
            w_m = weights[:, :, :, w_off:w_off + n_out * n_in].view(
                batch, channels_out, channels_in, n_out, n_in)
            out[:, :, out_s:out_s + n_out] = torch.einsum('bocji,bci->boj', w_m, f_m)
        else:
            f_re = features[:, :, in_s:in_s + n_in]
            f_im = features[:, :, in_s + n_in:in_s + 2*n_in]
            w_m = weights[:, :, :, w_off:w_off + 2*n_out*n_in].view(
                batch, channels_out, channels_in, n_out, n_in, 2)
            a, b = w_m[..., 0], w_m[..., 1]
            out[:, :, out_s:out_s + n_out] = (
                torch.einsum('bocji,bci->boj', a, f_re) +
                torch.einsum('bocji,bci->boj', b, f_im)
            )
            out[:, :, out_s + n_out:out_s + 2*n_out] = (
                torch.einsum('bocji,bci->boj', a, f_im) -
                torch.einsum('bocji,bci->boj', b, f_re)
            )

    return out


def benchmark_kernel(
    lvals: List[int],
    batch: int,
    channels_in: int,
    channels_out: int,
    dtype: torch.dtype = torch.float32,
    n_warmup: int = 10,
    n_iter: int = 100,
) -> dict:
    """
    Benchmark CUDA kernel vs Python implementation.

    Args:
        lvals: List of angular momentum values (e.g., [0, 1, 2])
        batch: Batch size
        channels_in: Number of input channels
        channels_out: Number of output channels
        dtype: Data type (torch.float32, torch.float16)
        n_warmup: Number of warmup iterations
        n_iter: Number of benchmark iterations

    Returns:
        Dictionary with timing and memory results
    """
    device = torch.device("cuda")

    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    features = torch.randn(batch, channels_in, dim, device=device, dtype=dtype)
    weights = torch.randn(batch, channels_out, channels_in, weight_dim, device=device, dtype=dtype)

    # Warmup CUDA
    for _ in range(n_warmup):
        _ = block_diagonal_cuda(features, weights, metadata)
    torch.cuda.synchronize()

    # Benchmark CUDA
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = block_diagonal_cuda(features, weights, metadata)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / n_iter * 1000
    cuda_mem = torch.cuda.max_memory_allocated() / 1024**2

    # Warmup Python
    for _ in range(n_warmup):
        _ = _block_diagonal_python(features, weights, lvals, lvals)
    torch.cuda.synchronize()

    # Benchmark Python
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = _block_diagonal_python(features, weights, lvals, lvals)
    torch.cuda.synchronize()
    python_time = (time.perf_counter() - start) / n_iter * 1000
    python_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "cuda_time_ms": cuda_time,
        "python_time_ms": python_time,
        "speedup": python_time / cuda_time,
        "cuda_mem_mb": cuda_mem,
        "python_mem_mb": python_mem,
        "mem_ratio": python_mem / cuda_mem,
    }


def run_kernel_benchmark(
    configs: List[Tuple[List[int], int, int, int]] = None,
    dtypes: List[torch.dtype] = None,
):
    """
    Run kernel benchmarks across multiple configurations.

    Args:
        configs: List of (lvals, batch, channels_in, channels_out) tuples
        dtypes: List of dtypes to benchmark
    """
    if configs is None:
        configs = [
            ([0, 1, 2], 1000, 64, 64),
            ([0, 1, 2, 3], 1000, 64, 64),
            ([0, 1, 2, 3], 1000, 128, 128),
            ([0, 1, 2, 3, 4], 500, 128, 128),
        ]

    if dtypes is None:
        dtypes = [torch.float32, torch.float16]

    print("=" * 90)
    print("CUDA Kernel Benchmark: block_diagonal_cuda vs Python/einsum")
    print("=" * 90)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    for dtype in dtypes:
        dtype_name = "FP32" if dtype == torch.float32 else "FP16"
        print(f"\n{'-' * 90}")
        print(f"{dtype_name} Results")
        print(f"{'-' * 90}")
        print(f"{'Config':<35} {'CUDA':>10} {'Python':>10} {'Speedup':>10} {'CUDA Mem':>12} {'Python Mem':>12}")
        print("-" * 90)

        for lvals, batch, cin, cout in configs:
            try:
                r = benchmark_kernel(lvals, batch, cin, cout, dtype)
                config = f"L={lvals}, B={batch}, C={cin}x{cout}"
                print(f"{config:<35} {r['cuda_time_ms']:>9.3f}ms {r['python_time_ms']:>9.3f}ms "
                      f"{r['speedup']:>9.1f}x {r['cuda_mem_mb']:>10.1f}MB {r['python_mem_mb']:>10.1f}MB")
            except Exception as e:
                config = f"L={lvals}, B={batch}, C={cin}x{cout}"
                print(f"{config:<35} ERROR: {e}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_kernel_benchmark()
