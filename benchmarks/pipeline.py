"""
Benchmark full equivariant pipeline: Low-rank vs Dense approaches.

Low-rank pipeline:
    output = Q @ Λ @ P^T @ features
    Where P, Q are Wigner-D matrices and Λ is block-diagonal with O(L²) parameters.

Dense pipeline:
    output = W @ features
    Where W is a dense weight matrix with O(L⁴) parameters.

This benchmark measures the theoretical maximum benefit of the low-rank representation,
not a comparison against a specific CG-basis implementation.
"""

import torch
import time
from typing import List, Tuple

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    get_weight_dim,
)


def _build_random_orthogonal(batch: int, dim: int, device: torch.device, dtype: torch.dtype):
    """Build random orthogonal matrices (stand-in for Wigner-D)."""
    rand = torch.randn(batch, dim, dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand)
    return Q.to(dtype)


def _build_m_order_permutation(lvals: List[int], device: torch.device):
    """Build permutation from standard (l,m) order to m-first order."""
    perm = []
    lmax = max(lvals) if lvals else 0

    def std_pos(l_idx, m):
        return sum(2 * lvals[i] + 1 for i in range(l_idx)) + lvals[l_idx] + m

    for l_idx, l in enumerate(lvals):
        perm.append(std_pos(l_idx, 0))

    for m in range(1, lmax + 1):
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, m))
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, -m))

    return torch.tensor(perm, dtype=torch.long, device=device)


def _lowrank_forward(features, weights, P, Q, perm, metadata):
    """Low-rank pipeline: P^T @ features -> CUDA kernel -> Q @ result."""
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
    out_diag = block_diagonal_cuda(f_diag, weights, metadata)
    return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)


def _dense_forward(features, weights):
    """Dense pipeline: einsum contraction with O(L⁴) weights."""
    return torch.einsum("bocij,bcj->boi", weights, features)


def benchmark_pipeline(
    lvals: List[int],
    batch: int,
    channels_in: int,
    channels_out: int,
    dtype: torch.dtype = torch.float32,
    n_warmup: int = 10,
    n_iter: int = 100,
) -> dict:
    """
    Benchmark low-rank pipeline vs dense pipeline.

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
    perm = _build_m_order_permutation(lvals, device)

    # Low-rank tensors
    features_lr = torch.randn(batch, channels_in, dim, device=device, dtype=dtype)
    weights_lr = torch.randn(batch, channels_out, channels_in, weight_dim, device=device, dtype=dtype)
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)

    # Dense tensors
    features_dense = torch.randn(batch, channels_in, dim, device=device, dtype=dtype)
    weights_dense = torch.randn(batch, channels_out, channels_in, dim, dim, device=device, dtype=dtype)

    # Warmup low-rank
    for _ in range(n_warmup):
        _ = _lowrank_forward(features_lr, weights_lr, P, Q, perm, metadata)
    torch.cuda.synchronize()

    # Benchmark low-rank
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = _lowrank_forward(features_lr, weights_lr, P, Q, perm, metadata)
    torch.cuda.synchronize()
    lr_time = (time.perf_counter() - start) / n_iter * 1000
    lr_mem = torch.cuda.max_memory_allocated() / 1024**2

    # Warmup dense
    for _ in range(n_warmup):
        _ = _dense_forward(features_dense, weights_dense)
    torch.cuda.synchronize()

    # Benchmark dense
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = _dense_forward(features_dense, weights_dense)
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / n_iter * 1000
    dense_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "lowrank_time_ms": lr_time,
        "dense_time_ms": dense_time,
        "speedup": dense_time / lr_time,
        "lowrank_mem_mb": lr_mem,
        "dense_mem_mb": dense_mem,
        "mem_ratio": dense_mem / lr_mem,
        "lowrank_params": weight_dim,
        "dense_params": dim * dim,
    }


def run_pipeline_benchmark(
    configs: List[Tuple[List[int], int, int, int]] = None,
    dtypes: List[torch.dtype] = None,
):
    """
    Run pipeline benchmarks across multiple configurations.

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

    print("=" * 95)
    print("Full Pipeline Benchmark: Low-Rank (Wigner-D + CUDA) vs Dense")
    print("=" * 95)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print("\nLow-rank: P^T @ features -> CUDA block-diag kernel -> Q @ result")
    print("Dense: einsum contraction with O(dim²) weight parameters per channel pair")

    for dtype in dtypes:
        dtype_name = "FP32" if dtype == torch.float32 else "FP16"
        print(f"\n{'-' * 95}")
        print(f"{dtype_name} Results")
        print(f"{'-' * 95}")
        print(f"{'Config':<35} {'Low-Rank':>12} {'Dense':>12} {'Speedup':>10} {'LR Mem':>12} {'Dense Mem':>12}")
        print("-" * 95)

        for lvals, batch, cin, cout in configs:
            try:
                r = benchmark_pipeline(lvals, batch, cin, cout, dtype)
                config = f"L={lvals}, B={batch}, C={cin}x{cout}"
                print(f"{config:<35} {r['lowrank_time_ms']:>11.3f}ms {r['dense_time_ms']:>11.3f}ms "
                      f"{r['speedup']:>9.1f}x {r['lowrank_mem_mb']:>10.1f}MB {r['dense_mem_mb']:>10.1f}MB")
            except Exception as e:
                config = f"L={lvals}, B={batch}, C={cin}x{cout}"
                print(f"{config:<35} ERROR: {e}")

    print("\n" + "=" * 95)
    print("Note: Dense weights are O(dim²) per channel pair vs O(L²) for low-rank.")
    print("This measures theoretical maximum benefit of the low-rank representation.")
    print("=" * 95)


if __name__ == "__main__":
    run_pipeline_benchmark()
