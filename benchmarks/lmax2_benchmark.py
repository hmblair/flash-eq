"""
Benchmark at various Lmax: LR-Binned vs LR-Full vs Dense.

Compares performance across different angular momentum values.
"""

import torch
import torch.nn as nn
import gc
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import RadialBinning


class RadialMLP(nn.Module):
    """Radial MLP that outputs (N, cout, cin, weight_dim) for N distances."""

    def __init__(self, cout, cin, weight_dim, hidden=128):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.weight_dim = weight_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cout * cin * weight_dim),
        )

    def forward(self, distances):
        out = self.net(distances.unsqueeze(-1))
        return out.view(-1, self.cout, self.cin, self.weight_dim)


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
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_lr_binned(lmax, batch, cin, cout, num_bins, dtype, n_warmup, n_iter, interp=False):
    """Benchmark LR + Binned (includes MLP forward)."""
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

    radial_mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    edge_lengths = torch.rand(batch, device=device) * 10.0

    if interp:
        def forward():
            radial_table = radial_mlp(binning.bin_edges)
            bin_data = binning.compute_bins(edge_lengths)
            features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
            f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
            out_diag = block_diagonal_binned_interp_cuda(
                f_diag, radial_table, bin_data.lo, bin_data.hi, bin_data.weight.to(dtype), cout, metadata
            )
            return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)
    else:
        def forward():
            radial_table = radial_mlp(binning.bin_edges)
            bin_indices = binning.compute_indices(edge_lengths)
            features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
            f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
            out_diag = block_diagonal_binned_cuda(
                f_diag, radial_table, bin_indices, cout, metadata
            )
            return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

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
    """Benchmark LR + Full weights (includes MLP forward)."""
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

    radial_mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    edge_lengths = torch.rand(batch, device=device) * 10.0

    def forward():
        full_weights = radial_mlp(edge_lengths)
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_cuda(f_diag, full_weights, metadata)
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

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
    """Benchmark Dense baseline."""
    clear_memory()

    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    basis = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
    radial = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype)

    def forward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        basis_view = basis.view(batch, dim, -1)
        tmp = features @ basis_view
        tmp = tmp.view(batch, -1, dim)
        return radial @ tmp

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

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
    print("Binned Weights Benchmark: LR-Binned vs LR-Full vs Dense")
    print("=" * 100)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    num_bins = 100
    dtype = torch.float32
    n_warmup = 5
    n_iter = 20

    for lmax in [2, 4, 6]:
        lvals = list(range(lmax + 1))
        dim = sum(2 * l + 1 for l in lvals)
        weight_dim = get_weight_dim(lvals, lvals)

        print(f"\n{'='*100}")
        print(f"Lmax={lmax}: dim={dim}, weight_dim={weight_dim}")
        print("=" * 100)

        configs = [
            (5000, 32, 32),
            (5000, 64, 64),
            (10000, 32, 32),
        ]

        print(f"\n{'Config':<25} {'Binned(NN)':>12} {'LR-Full':>12} {'Dense':>12} {'vs Full':>10} {'vs Dense':>10}")
        print("-" * 90)

        for batch, cin, cout in configs:
            config_str = f"B={batch}, C={cin}x{cout}"

            try:
                r_binned = benchmark_lr_binned(lmax, batch, cin, cout, num_bins, dtype, n_warmup, n_iter, interp=False)
            except Exception as e:
                print(f"{config_str:<25} BINNED FAILED: {e}")
                r_binned = None
            clear_memory()

            try:
                r_full = benchmark_lr_full(lmax, batch, cin, cout, dtype, n_warmup, n_iter)
            except Exception as e:
                print(f"{config_str:<25} FULL FAILED: {e}")
                r_full = None
            clear_memory()

            try:
                r_dense = benchmark_dense(lmax, batch, cin, cout, dtype, n_warmup, n_iter)
            except Exception as e:
                print(f"{config_str:<25} DENSE FAILED: {e}")
                r_dense = None
            clear_memory()

            if r_binned and r_full and r_dense:
                speedup_full = r_full['time_ms'] / r_binned['time_ms']
                speedup_dense = r_dense['time_ms'] / r_binned['time_ms']
                print(f"{config_str:<25} {r_binned['time_ms']:>10.2f}ms {r_full['time_ms']:>10.2f}ms "
                      f"{r_dense['time_ms']:>10.2f}ms {speedup_full:>9.2f}x {speedup_dense:>9.2f}x")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
