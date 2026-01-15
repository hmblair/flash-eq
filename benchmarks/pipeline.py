"""
Benchmark full equivariant pipeline: Low-rank vs Dense (VersatileConvSE3-style).

Low-rank pipeline:
    output = Q @ Λ @ P^T @ features
    Where P, Q are Wigner-D matrices and Λ is block-diagonal with O(L²) parameters.

Dense pipeline (mimics VersatileConvSE3):
    tmp = features @ basis
    output = radial_weights @ tmp
    Two matmuls matching the SE3-Transformer's fused convolution.
"""

import torch
import time
from typing import List, Tuple

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_cuda_v2,
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


def _lowrank_forward(features, weights, P, Q, perm, metadata, use_v2=True):
    """Low-rank pipeline: P^T @ features -> CUDA kernel -> Q @ result."""
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
    if use_v2:
        out_diag = block_diagonal_cuda_v2(f_diag, weights, metadata)
    else:
        out_diag = block_diagonal_cuda(f_diag, weights, metadata)
    return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)


def _dense_forward(features, basis, radial_weights, out_dim):
    """
    Dense pipeline mimicking VersatileConvSE3.

    Two matmuls:
    1. features @ basis_view -> tmp
    2. radial_weights @ tmp -> output

    This matches the einsum: n i l, n o i f, n l f k -> n o k
    where basis has shape [batch, in_dim, freq, out_dim].
    """
    # features: [batch, channels_in, in_dim]
    # basis: [batch, in_dim, freq, out_dim] -> viewed as [batch, in_dim, freq * out_dim]
    # radial_weights: [batch, channels_out, channels_in * freq]
    batch, channels_in, in_dim = features.shape

    # View basis as [batch, in_dim, freq * out_dim]
    basis_view = basis.view(batch, in_dim, -1)

    # First matmul: features @ basis_view -> [batch, channels_in, freq * out_dim]
    tmp = features @ basis_view

    # Reshape for second matmul: [batch, channels_in * freq, out_dim]
    tmp = tmp.view(batch, -1, out_dim)

    # Second matmul: radial_weights @ tmp -> [batch, channels_out, out_dim]
    return radial_weights @ tmp


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

    # Compute freq_sum for dense baseline (matches VersatileConvSE3 fully fused)
    lmax = max(lvals)
    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    # Low-rank tensors
    features_lr = torch.randn(batch, channels_in, dim, device=device, dtype=dtype)
    weights_lr = torch.randn(batch, channels_out, channels_in, weight_dim, device=device, dtype=dtype)
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)

    # Dense tensors (mimicking VersatileConvSE3)
    # basis: [batch, in_dim, freq_sum, out_dim]
    # radial_weights: [batch, channels_out, channels_in * freq_sum]
    features_dense = torch.randn(batch, channels_in, dim, device=device, dtype=dtype)
    basis_dense = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
    radial_weights_dense = torch.randn(batch, channels_out, channels_in * freq_sum, device=device, dtype=dtype)

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
        _ = _dense_forward(features_dense, basis_dense, radial_weights_dense, dim)
    torch.cuda.synchronize()

    # Benchmark dense
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = _dense_forward(features_dense, basis_dense, radial_weights_dense, dim)
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
            # Original configs
            ([0, 1, 2, 3], 1000, 64, 64),
            ([0, 1, 2, 3, 4], 1000, 64, 64),
            # Higher L values
            ([0, 1, 2, 3, 4, 5, 6], 1000, 64, 64),
            ([0, 1, 2, 3, 4, 5, 6], 2000, 64, 64),
            ([0, 1, 2, 3, 4, 5, 6], 5000, 64, 64),
            # Higher edge counts with moderate channels
            ([0, 1, 2, 3, 4, 5, 6], 10000, 32, 32),
            ([0, 1, 2, 3, 4, 5, 6], 20000, 32, 32),
        ]

    if dtypes is None:
        dtypes = [torch.float32, torch.float16]

    print("=" * 95)
    print("Full Pipeline Benchmark: Low-Rank (Wigner-D + CUDA) vs Dense (VersatileConvSE3-style)")
    print("=" * 95)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print("\nLow-rank: P^T @ features -> CUDA block-diag kernel -> Q @ result")
    print("Dense: features @ basis -> radial_weights @ tmp (two matmuls, mimics SE3-Transformer)")

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
    print("Note: Dense baseline mimics VersatileConvSE3 (two matmuls with basis and radial weights).")
    print("=" * 95)


if __name__ == "__main__":
    run_pipeline_benchmark()
