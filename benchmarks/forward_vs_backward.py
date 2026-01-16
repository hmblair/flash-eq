"""
Compare forward-only vs forward+backward performance for binned approach.
"""

import torch
import torch.nn as nn
import gc
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import RadialBinning


class RadialMLP(nn.Module):
    def __init__(self, cout, cin, weight_dim, hidden=128):
        super().__init__()
        self.cout, self.cin, self.weight_dim = cout, cin, weight_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, cout * cin * weight_dim),
        )

    def forward(self, distances):
        return self.net(distances.unsqueeze(-1)).view(-1, self.cout, self.cin, self.weight_dim)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_standard(lmax, batch, cin, cout, dtype, forward_only, n_warmup=3, n_iter=10):
    """Benchmark standard approach."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=not forward_only)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0
    target = torch.randn(batch, cout, dim, device=device, dtype=dtype)

    if forward_only:
        def run():
            with torch.no_grad():
                weights = mlp(distances)
                output = block_diagonal_cuda(features, weights, metadata)
            return output
    else:
        def run():
            mlp.zero_grad()
            weights = mlp(distances)
            output = block_diagonal_cuda(features, weights, metadata)
            loss = ((output - target) ** 2).mean()
            loss.backward()
            return loss

    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()

    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        run()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_binned(lmax, batch, cin, cout, num_bins, dtype, forward_only, n_warmup=3, n_iter=10):
    """Benchmark binned approach."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=not forward_only)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0
    target = torch.randn(batch, cout, dim, device=device, dtype=dtype)

    if forward_only:
        def run():
            with torch.no_grad():
                radial_table = mlp(binning.bin_edges)
                bin_data = binning.compute_bins(distances)
                output = block_diagonal_binned_interp_cuda(
                    features, radial_table,
                    bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
                    cout, metadata
                )
            return output
    else:
        def run():
            mlp.zero_grad()
            radial_table = mlp(binning.bin_edges)
            bin_data = binning.compute_bins(distances)
            output = block_diagonal_binned_interp_cuda(
                features, radial_table,
                bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
                cout, metadata
            )
            loss = ((output - target) ** 2).mean()
            loss.backward()
            return loss

    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()

    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        run()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    print("=" * 110)
    print("Forward vs Forward+Backward: Standard vs Binned")
    print("=" * 110)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    dtype = torch.float32
    num_bins = 100

    configs = [
        (6, 5000, 32, 32),
        (6, 5000, 64, 64),
        (6, 10000, 32, 32),
    ]

    print(f"\nSettings: num_bins={num_bins}, dtype=float32")

    for lmax, batch, cin, cout in configs:
        config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
        print(f"\n{'='*110}")
        print(f"Config: {config_str}")
        print("=" * 110)

        # Forward only
        std_fwd = benchmark_standard(lmax, batch, cin, cout, dtype, forward_only=True)
        clear_memory()
        bin_fwd = benchmark_binned(lmax, batch, cin, cout, num_bins, dtype, forward_only=True)
        clear_memory()

        # Forward + Backward
        std_bwd = benchmark_standard(lmax, batch, cin, cout, dtype, forward_only=False)
        clear_memory()
        bin_bwd = benchmark_binned(lmax, batch, cin, cout, num_bins, dtype, forward_only=False)
        clear_memory()

        print(f"\n{'':20} {'Standard':>25} {'Binned':>25} {'Ratio':>15}")
        print("-" * 90)

        # Forward only
        fwd_speedup = std_fwd['time_ms'] / bin_fwd['time_ms']
        fwd_mem_ratio = std_fwd['peak_mem_mb'] / bin_fwd['peak_mem_mb']
        print(f"{'Forward Only':<20} {std_fwd['time_ms']:>10.1f}ms / {std_fwd['peak_mem_mb']:>6.0f}MB "
              f"{bin_fwd['time_ms']:>10.1f}ms / {bin_fwd['peak_mem_mb']:>6.0f}MB "
              f"{fwd_speedup:>6.2f}x / {fwd_mem_ratio:>5.1f}x")

        # Forward + Backward
        bwd_speedup = std_bwd['time_ms'] / bin_bwd['time_ms']
        bwd_mem_ratio = std_bwd['peak_mem_mb'] / bin_bwd['peak_mem_mb']
        print(f"{'Forward+Backward':<20} {std_bwd['time_ms']:>10.1f}ms / {std_bwd['peak_mem_mb']:>6.0f}MB "
              f"{bin_bwd['time_ms']:>10.1f}ms / {bin_bwd['peak_mem_mb']:>6.0f}MB "
              f"{bwd_speedup:>6.2f}x / {bwd_mem_ratio:>5.1f}x")

        # Compute backward-only time (approximate)
        std_bwd_only = std_bwd['time_ms'] - std_fwd['time_ms']
        bin_bwd_only = bin_bwd['time_ms'] - bin_fwd['time_ms']
        bwd_only_speedup = std_bwd_only / bin_bwd_only if bin_bwd_only > 0 else 0
        print(f"{'Backward Only (est)':<20} {std_bwd_only:>10.1f}ms {'':>14} "
              f"{bin_bwd_only:>10.1f}ms {'':>14} "
              f"{bwd_only_speedup:>6.2f}x")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
