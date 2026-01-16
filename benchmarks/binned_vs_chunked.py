"""
Compare memory usage: Binned vs Chunked approaches.

Binned: Weights stored per distance bin (num_bins), not per edge (B)
  - Memory: O(num_bins * Cout * Cin * Wdim)
  - Reduction factor: B / num_bins

Chunked: Process output channels in chunks
  - Memory: O(B * chunk_size * Cin * Wdim)  
  - Reduction factor: Cout / chunk_size
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
from flash_eq.fused_radial import FusedRadialBlockDiagonal
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


def benchmark_standard(lmax, batch, cin, cout, dtype):
    """Standard: Full (B, Cout, Cin, Wdim) weights."""
    clear_memory()
    device = torch.device("cuda")
    
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        weights = mlp(distances)
        _ = block_diagonal_cuda(features, weights, metadata)
    torch.cuda.synchronize()
    
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_binned(lmax, batch, cin, cout, num_bins, dtype):
    """Binned: Weights per distance bin."""
    clear_memory()
    device = torch.device("cuda")
    
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # Only evaluate MLP at bin edges (num_bins+1 points)
        radial_table = mlp(binning.bin_edges)
        bin_data = binning.compute_bins(distances)
        _ = block_diagonal_binned_interp_cuda(
            features, radial_table, 
            bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
            cout, metadata
        )
    torch.cuda.synchronize()
    
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_chunked(lmax, batch, cin, cout, chunk_size, dtype):
    """Chunked: Process output channels in chunks."""
    clear_memory()
    device = torch.device("cuda")
    
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    
    layer = FusedRadialBlockDiagonal(
        cout, cin, weight_dim, hidden_dim=128, chunk_size=chunk_size
    ).to(device).to(dtype)
    layer.set_metadata(metadata)
    
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = layer(features, distances)
    torch.cuda.synchronize()
    
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_standard_runtime(lmax, batch, cin, cout, dtype, n_warmup=5, n_iter=20):
    """Benchmark standard approach runtime."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0

    def forward():
        with torch.no_grad():
            weights = mlp(distances)
            return block_diagonal_cuda(features, weights, metadata)

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iter


def benchmark_binned_runtime(lmax, batch, cin, cout, num_bins, dtype, n_warmup=5, n_iter=20):
    """Benchmark binned approach runtime."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0

    def forward():
        with torch.no_grad():
            radial_table = mlp(binning.bin_edges)
            bin_data = binning.compute_bins(distances)
            return block_diagonal_binned_interp_cuda(
                features, radial_table,
                bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
                cout, metadata
            )

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iter


def benchmark_chunked_runtime(lmax, batch, cin, cout, chunk_size, dtype, n_warmup=5, n_iter=20):
    """Benchmark chunked approach runtime."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    layer = FusedRadialBlockDiagonal(
        cout, cin, weight_dim, hidden_dim=128, chunk_size=chunk_size
    ).to(device).to(dtype)
    layer.set_metadata(metadata)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0

    def forward():
        with torch.no_grad():
            return layer(features, distances)

    for _ in range(n_warmup):
        forward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        forward()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iter


def main():
    print("=" * 90)
    print("Comparison: Standard vs Binned vs Chunked")
    print("=" * 90)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    dtype = torch.float32
    num_bins = 100
    chunk_size = 8

    configs = [
        (6, 5000, 32, 32),
        (6, 5000, 64, 64),
        (6, 10000, 32, 32),
        (6, 10000, 64, 64),
    ]

    print(f"\nSettings: num_bins={num_bins}, chunk_size={chunk_size}, dtype=float32")

    # Memory comparison
    print(f"\n{'='*90}")
    print("MEMORY USAGE")
    print(f"{'='*90}")
    print(f"\n{'Config':<25} {'Standard':>12} {'Binned':>12} {'Chunked':>12} {'B/S':>8} {'C/S':>8}")
    print("-" * 90)

    for lmax, batch, cin, cout in configs:
        config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"

        try:
            std_mem = benchmark_standard(lmax, batch, cin, cout, dtype)
        except Exception as e:
            std_mem = float('inf')
        clear_memory()

        try:
            bin_mem = benchmark_binned(lmax, batch, cin, cout, num_bins, dtype)
        except Exception as e:
            bin_mem = float('inf')
        clear_memory()

        try:
            chunk_mem = benchmark_chunked(lmax, batch, cin, cout, chunk_size, dtype)
        except Exception as e:
            chunk_mem = float('inf')
        clear_memory()

        bin_ratio = std_mem / bin_mem if bin_mem > 0 else 0
        chunk_ratio = std_mem / chunk_mem if chunk_mem > 0 else 0

        print(f"{config_str:<25} {std_mem:>10.1f}MB {bin_mem:>10.1f}MB {chunk_mem:>10.1f}MB "
              f"{bin_ratio:>7.1f}x {chunk_ratio:>7.1f}x")

    # Runtime comparison
    print(f"\n{'='*90}")
    print("RUNTIME")
    print(f"{'='*90}")
    print(f"\n{'Config':<25} {'Standard':>12} {'Binned':>12} {'Chunked':>12} {'B/S':>8} {'C/S':>8}")
    print("-" * 90)

    for lmax, batch, cin, cout in configs:
        config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"

        try:
            std_time = benchmark_standard_runtime(lmax, batch, cin, cout, dtype)
        except Exception as e:
            std_time = float('inf')
        clear_memory()

        try:
            bin_time = benchmark_binned_runtime(lmax, batch, cin, cout, num_bins, dtype)
        except Exception as e:
            bin_time = float('inf')
        clear_memory()

        try:
            chunk_time = benchmark_chunked_runtime(lmax, batch, cin, cout, chunk_size, dtype)
        except Exception as e:
            chunk_time = float('inf')
        clear_memory()

        bin_ratio = std_time / bin_time if bin_time > 0 else 0
        chunk_ratio = std_time / chunk_time if chunk_time > 0 else 0

        print(f"{config_str:<25} {std_time:>10.2f}ms {bin_time:>10.2f}ms {chunk_time:>10.2f}ms "
              f"{bin_ratio:>7.2f}x {chunk_ratio:>7.2f}x")

    print("\n" + "=" * 90)
    print("Analysis:")
    print("  Memory:")
    print("    - Binned reduction ≈ B / num_bins (e.g., 5000/100 = 50x theoretical)")
    print("    - Chunked reduction ≈ Cout / chunk_size (e.g., 64/8 = 8x theoretical)")
    print("  Runtime:")
    print("    - Binned: MLP evaluates at num_bins+1 points instead of B points")
    print("    - Chunked: Same MLP compute, but chunked block-diagonal (slight overhead)")
    print("  Trade-offs:")
    print("    - Binned: Best memory savings, faster when B >> num_bins")
    print("    - Chunked: Moderate memory savings, supports gradients")
    print("=" * 90)


if __name__ == "__main__":
    main()
