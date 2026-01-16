"""
Benchmark full equivariant pipeline: Low-rank (binned) vs Low-rank (full) vs Dense.

Compares:
1. Low-rank + binned weights: P^T @ f -> binned_kernel -> Q @ out
2. Low-rank + full weights: P^T @ f -> full_kernel -> Q @ out
3. Dense baseline: f @ basis -> radial @ tmp (VersatileConvSE3-style)

Measures both runtime and peak memory usage.
"""

import torch
import torch.nn as nn
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import (
    create_bin_edges,
    compute_bin_interpolation,
)


def _build_random_orthogonal(batch, dim, device, dtype):
    """Build random orthogonal matrices (stand-in for Wigner-D)."""
    rand = torch.randn(batch, dim, dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand)
    return Q.to(dtype)


def _build_m_order_permutation(lvals, device):
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


def benchmark_pipeline(lmax, batch, cin, cout, num_bins=100, dtype=torch.float32,
                       n_warmup=10, n_iter=50):
    """Benchmark all three approaches."""
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    perm = _build_m_order_permutation(lvals, device)

    # Dense baseline parameters
    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    # Shared: Wigner-D matrices (P, Q)
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    # Generate random edge lengths
    edge_lengths = torch.rand(batch, device=device) * 10.0

    # Binned weights setup
    bin_edges = create_bin_edges(0.0, 10.0, num_bins, device)
    # Table shape: (num_bins + 1, cout, cin, weight_dim)
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)
    bin_lo, bin_hi, interp_weight = compute_bin_interpolation(edge_lengths, bin_edges)
    interp_weight = interp_weight.to(dtype)

    # Full weights (for comparison)
    full_weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)

    # Dense baseline tensors
    basis_dense = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
    radial_dense = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype)

    # Define pipeline functions
    def lr_binned_forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_binned_interp_cuda(
            f_diag, radial_table, bin_lo, bin_hi, interp_weight, cout, metadata
        )
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    def lr_full_forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_cuda(f_diag, full_weights, metadata)
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    def dense_forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        basis_view = basis_dense.view(batch, dim, -1)
        tmp = features @ basis_view
        tmp = tmp.view(batch, -1, dim)
        return radial_dense @ tmp

    # Warmup
    for _ in range(n_warmup):
        lr_binned_forward()
        lr_full_forward()
        dense_forward()
    torch.cuda.synchronize()

    results = {}

    # Benchmark LR + Binned
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        lr_binned_forward()
    end.record()
    torch.cuda.synchronize()
    results['lr_binned_ms'] = start.elapsed_time(end) / n_iter
    results['lr_binned_mem_mb'] = torch.cuda.max_memory_allocated() / 1024**2

    # Benchmark LR + Full weights
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        lr_full_forward()
    end.record()
    torch.cuda.synchronize()
    results['lr_full_ms'] = start.elapsed_time(end) / n_iter
    results['lr_full_mem_mb'] = torch.cuda.max_memory_allocated() / 1024**2

    # Benchmark Dense
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        dense_forward()
    end.record()
    torch.cuda.synchronize()
    results['dense_ms'] = start.elapsed_time(end) / n_iter
    results['dense_mem_mb'] = torch.cuda.max_memory_allocated() / 1024**2

    # Compute speedups
    results['speedup_vs_full'] = results['lr_full_ms'] / results['lr_binned_ms']
    results['speedup_vs_dense'] = results['dense_ms'] / results['lr_binned_ms']
    results['mem_reduction_vs_full'] = results['lr_full_mem_mb'] / results['lr_binned_mem_mb']
    results['mem_reduction_vs_dense'] = results['dense_mem_mb'] / results['lr_binned_mem_mb']

    return results


def measure_memory_only(lmax, batch, cin, cout, num_bins=100, dtype=torch.float32):
    """Measure theoretical memory usage for each approach."""
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    # Dense baseline
    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    elem_size = 2 if dtype == torch.float16 else 4

    # Memory for each approach (main tensors only, excluding features/output)

    # LR + Binned: P, Q, radial_table, bin_indices
    lr_binned_mem = (
        2 * batch * dim * dim +  # P, Q
        (num_bins + 1) * weight_dim +  # radial_table
        batch  # bin indices (int32, ~4 bytes)
    ) * elem_size

    # LR + Full: P, Q, full_weights
    lr_full_mem = (
        2 * batch * dim * dim +  # P, Q
        batch * cout * cin * weight_dim  # full_weights
    ) * elem_size

    # Dense: basis, radial_weights
    dense_mem = (
        batch * dim * freq_sum * dim +  # basis
        batch * cout * cin * freq_sum  # radial_weights
    ) * elem_size

    return {
        'lr_binned_mb': lr_binned_mem / 1024**2,
        'lr_full_mb': lr_full_mem / 1024**2,
        'dense_mb': dense_mem / 1024**2,
    }


def main():
    print("=" * 120)
    print("Full Pipeline Benchmark: Low-Rank (Binned) vs Low-Rank (Full) vs Dense")
    print("=" * 120)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print("\nPipelines:")
    print("  LR-Binned: P^T @ features -> binned_block_diag -> Q @ result")
    print("  LR-Full:   P^T @ features -> full_block_diag -> Q @ result")
    print("  Dense:     features @ basis -> radial_weights @ tmp (VersatileConvSE3)")

    configs = [
        # (lmax, batch, cin, cout)
        (4, 1000, 64, 64),
        (4, 5000, 64, 64),
        (6, 1000, 64, 64),
        (6, 5000, 64, 64),
        (6, 10000, 32, 32),
        (6, 20000, 32, 32),
    ]

    # Theoretical memory comparison
    print("\n" + "-" * 120)
    print("Theoretical Memory Usage (main tensors only)")
    print("-" * 120)
    print(f"{'Config':<30} {'LR-Binned':>14} {'LR-Full':>14} {'Dense':>14} {'Binned/Full':>14} {'Binned/Dense':>14}")
    print("-" * 120)

    for lmax, batch, cin, cout in configs:
        mem = measure_memory_only(lmax, batch, cin, cout, num_bins=100)
        config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
        print(f"{config_str:<30} {mem['lr_binned_mb']:>12.1f}MB {mem['lr_full_mb']:>12.1f}MB "
              f"{mem['dense_mb']:>12.1f}MB {mem['lr_full_mb']/mem['lr_binned_mb']:>13.1f}x "
              f"{mem['dense_mb']/mem['lr_binned_mb']:>13.1f}x")

    # Runtime benchmark
    for dtype, dtype_name in [(torch.float32, "FP32")]:
        print(f"\n{'-' * 120}")
        print(f"Runtime Benchmark ({dtype_name})")
        print(f"{'-' * 120}")
        print(f"{'Config':<30} {'LR-Binned':>12} {'LR-Full':>12} {'Dense':>12} {'vs Full':>10} {'vs Dense':>10}")
        print("-" * 120)

        for lmax, batch, cin, cout in configs:
            try:
                r = benchmark_pipeline(lmax, batch, cin, cout, num_bins=100, dtype=dtype)
                config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
                print(f"{config_str:<30} {r['lr_binned_ms']:>10.2f}ms {r['lr_full_ms']:>10.2f}ms "
                      f"{r['dense_ms']:>10.2f}ms {r['speedup_vs_full']:>9.2f}x {r['speedup_vs_dense']:>9.2f}x")
            except Exception as e:
                print(f"L={lmax}, B={batch}, C={cin}x{cout}: {e}")
                torch.cuda.empty_cache()

    # Peak memory during execution
    print(f"\n{'-' * 120}")
    print("Peak Memory During Execution (FP32)")
    print(f"{'-' * 120}")
    print(f"{'Config':<30} {'LR-Binned':>14} {'LR-Full':>14} {'Dense':>14} {'Savings/Full':>14} {'Savings/Dense':>14}")
    print("-" * 120)

    for lmax, batch, cin, cout in configs:
        try:
            r = benchmark_pipeline(lmax, batch, cin, cout, num_bins=100, dtype=torch.float32)
            config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
            print(f"{config_str:<30} {r['lr_binned_mem_mb']:>12.1f}MB {r['lr_full_mem_mb']:>12.1f}MB "
                  f"{r['dense_mem_mb']:>12.1f}MB {r['mem_reduction_vs_full']:>13.2f}x "
                  f"{r['mem_reduction_vs_dense']:>13.2f}x")
        except Exception as e:
            print(f"L={lmax}, B={batch}, C={cin}x{cout}: {e}")
            torch.cuda.empty_cache()

    print("\n" + "=" * 120)
    print("Summary:")
    print("  - LR-Binned combines low-rank factorization with binned weight lookup")
    print("  - Memory reduction comes from: (1) low-rank P,Q vs dense basis, (2) binned table vs per-edge weights")
    print("  - Speed improvement comes from: (1) smaller matmuls, (2) weight sharing in shared memory")
    print("=" * 120)


if __name__ == "__main__":
    main()
