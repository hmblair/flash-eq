"""
Benchmark full pipeline with ISOLATED memory measurement.

Each approach is measured in a separate function with full cleanup between them.
"""

import torch
import gc
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
    rand = torch.randn(batch, dim, dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand)
    return Q.to(dtype)


def _build_m_order_permutation(lvals, device):
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


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_lr_binned(lmax, batch, cin, cout, num_bins, dtype, n_warmup, n_iter):
    """Benchmark LR + Binned in isolation."""
    clear_memory()

    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    perm = _build_m_order_permutation(lvals, device)

    # Allocate only what binned approach needs
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    edge_lengths = torch.rand(batch, device=device) * 10.0
    bin_edges = create_bin_edges(0.0, 10.0, num_bins, device)
    radial_table = torch.randn(num_bins + 1, weight_dim, device=device, dtype=dtype)
    bin_lo, bin_hi, interp_weight = compute_bin_interpolation(edge_lengths, bin_edges)
    interp_weight = interp_weight.to(dtype)

    # Print tensor sizes
    print(f"    P,Q: 2 x {list(P.shape)} = {2 * P.numel() * P.element_size() / 1024**2:.1f} MB")
    print(f"    radial_table: {list(radial_table.shape)} = {radial_table.numel() * radial_table.element_size() / 1024**2:.3f} MB")
    print(f"    bin indices: 3 x {list(bin_lo.shape)} = {3 * bin_lo.numel() * 4 / 1024**2:.3f} MB")

    def forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_binned_interp_cuda(
            f_diag, radial_table, bin_lo, bin_hi, interp_weight, cout, metadata
        )
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    # Warmup
    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_lr_full(lmax, batch, cin, cout, dtype, n_warmup, n_iter):
    """Benchmark LR + Full weights in isolation."""
    clear_memory()

    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    perm = _build_m_order_permutation(lvals, device)

    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    # Full weights tensor - THIS IS THE BIG ONE
    full_weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)

    print(f"    P,Q: 2 x {list(P.shape)} = {2 * P.numel() * P.element_size() / 1024**2:.1f} MB")
    print(f"    full_weights: {list(full_weights.shape)} = {full_weights.numel() * full_weights.element_size() / 1024**2:.1f} MB")

    def forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_cuda(f_diag, full_weights, metadata)
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    # Warmup
    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_dense(lmax, batch, cin, cout, dtype, n_warmup, n_iter):
    """Benchmark Dense baseline in isolation."""
    clear_memory()

    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    basis = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
    radial = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype)

    print(f"    basis: {list(basis.shape)} = {basis.numel() * basis.element_size() / 1024**2:.1f} MB")
    print(f"    radial: {list(radial.shape)} = {radial.numel() * radial.element_size() / 1024**2:.1f} MB")

    def forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        basis_view = basis.view(batch, dim, -1)
        tmp = features @ basis_view
        tmp = tmp.view(batch, -1, dim)
        return radial @ tmp

    # Warmup
    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    print("=" * 100)
    print("Full Pipeline Benchmark (Isolated Memory Measurement)")
    print("=" * 100)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    configs = [
        # (lmax, batch, cin, cout)
        (6, 1000, 64, 64),
        (6, 5000, 64, 64),
        (6, 5000, 32, 32),
        (6, 10000, 32, 32),
    ]

    num_bins = 100
    dtype = torch.float32
    n_warmup = 5
    n_iter = 20

    for lmax, batch, cin, cout in configs:
        print(f"\n{'='*100}")
        print(f"Config: L={lmax}, B={batch}, C={cin}x{cout}")
        print("=" * 100)

        # LR + Binned
        print("\n[LR-Binned] Allocating tensors:")
        try:
            r_binned = benchmark_lr_binned(lmax, batch, cin, cout, num_bins, dtype, n_warmup, n_iter)
            print(f"  -> Time: {r_binned['time_ms']:.2f} ms, Peak Memory: {r_binned['peak_mem_mb']:.1f} MB")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            r_binned = None
        clear_memory()

        # LR + Full
        print("\n[LR-Full] Allocating tensors:")
        try:
            r_full = benchmark_lr_full(lmax, batch, cin, cout, dtype, n_warmup, n_iter)
            print(f"  -> Time: {r_full['time_ms']:.2f} ms, Peak Memory: {r_full['peak_mem_mb']:.1f} MB")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            r_full = None
        clear_memory()

        # Dense
        print("\n[Dense] Allocating tensors:")
        try:
            r_dense = benchmark_dense(lmax, batch, cin, cout, dtype, n_warmup, n_iter)
            print(f"  -> Time: {r_dense['time_ms']:.2f} ms, Peak Memory: {r_dense['peak_mem_mb']:.1f} MB")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            r_dense = None
        clear_memory()

        # Summary
        print(f"\n--- Summary for L={lmax}, B={batch}, C={cin}x{cout} ---")
        if r_binned and r_full:
            print(f"  Speedup vs LR-Full: {r_full['time_ms']/r_binned['time_ms']:.2f}x")
            print(f"  Memory reduction vs LR-Full: {r_full['peak_mem_mb']/r_binned['peak_mem_mb']:.2f}x")
        if r_binned and r_dense:
            print(f"  Speedup vs Dense: {r_dense['time_ms']/r_binned['time_ms']:.2f}x")
            print(f"  Memory reduction vs Dense: {r_dense['peak_mem_mb']/r_binned['peak_mem_mb']:.2f}x")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
