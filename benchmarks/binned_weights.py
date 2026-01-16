"""
Benchmark binned radial weights vs full weights.

Compares:
1. Memory usage: binned table vs full (batch, cout, cin, weight_dim) tensor
2. Speed: lookup + kernel vs full tensor kernel
"""

import torch
import torch.nn as nn
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import (
    create_bin_edges,
    create_radial_table,
    compute_bin_indices,
    compute_bin_interpolation,
    BinnedRadialWeights,
)


class SimpleRadialMLP(nn.Module):
    """Simple MLP that maps distance -> weight_dim outputs."""

    def __init__(self, weight_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, weight_dim),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: (N,) -> (N, weight_dim)
        return self.net(distances.unsqueeze(-1))


def benchmark_memory(lmax, batch, cin, cout, num_bins=100, dtype=torch.float32):
    """Compare memory usage of binned vs full weights."""
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    weight_dim = get_weight_dim(lvals, lvals)

    # Full weights memory
    full_elements = batch * cout * cin * weight_dim
    full_bytes = full_elements * (2 if dtype == torch.float16 else 4)

    # Binned table memory
    table_elements = num_bins * weight_dim
    table_bytes = table_elements * (2 if dtype == torch.float16 else 4)

    # Bin indices memory (int32)
    indices_bytes = batch * 4

    return {
        'full_elements': full_elements,
        'full_mb': full_bytes / 1024**2,
        'table_elements': table_elements,
        'table_mb': table_bytes / 1024**2,
        'indices_mb': indices_bytes / 1024**2,
        'binned_total_mb': (table_bytes + indices_bytes) / 1024**2,
        'reduction': full_bytes / (table_bytes + indices_bytes),
    }


def benchmark_speed(lmax, batch, cin, cout, num_bins=100, dtype=torch.float32,
                    n_warmup=10, n_iter=100):
    """Compare speed of binned vs full weights."""
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create radial MLP
    radial_mlp = SimpleRadialMLP(weight_dim).to(device).to(dtype)

    # Generate random edge lengths (0 to 10 Angstroms)
    edge_lengths = torch.rand(batch, device=device) * 10.0

    # Create binned lookup table
    bin_edges = create_bin_edges(0.0, 10.0, num_bins, device)
    radial_table = create_radial_table(
        lambda d: radial_mlp(d.to(dtype)),
        bin_edges,
        eval_at="edges"  # For interpolation
    )
    bin_lo, bin_hi, interp_weight = compute_bin_interpolation(edge_lengths, bin_edges)
    interp_weight = interp_weight.to(dtype)

    # Also create nearest-neighbor indices
    bin_indices = compute_bin_indices(edge_lengths, bin_edges)

    # Create full weights (simulating what you'd have without binning)
    # In practice, this would be: radial_mlp(edge_lengths) expanded to (batch, cout, cin, weight_dim)
    full_weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)

    # Features
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)

    # Define benchmark functions
    def full_forward():
        return block_diagonal_cuda(features, full_weights, metadata)

    def binned_forward():
        return block_diagonal_binned_cuda(
            features, radial_table, bin_indices, cout, metadata
        )

    def binned_interp_forward():
        return block_diagonal_binned_interp_cuda(
            features, radial_table, bin_lo, bin_hi, interp_weight, cout, metadata
        )

    # Warmup
    for _ in range(n_warmup):
        full_forward()
        binned_forward()
        binned_interp_forward()
    torch.cuda.synchronize()

    # Benchmark full weights
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        full_forward()
    end.record()
    torch.cuda.synchronize()
    full_time = start.elapsed_time(end) / n_iter

    # Benchmark binned (nearest)
    start.record()
    for _ in range(n_iter):
        binned_forward()
    end.record()
    torch.cuda.synchronize()
    binned_time = start.elapsed_time(end) / n_iter

    # Benchmark binned (interpolated)
    start.record()
    for _ in range(n_iter):
        binned_interp_forward()
    end.record()
    torch.cuda.synchronize()
    binned_interp_time = start.elapsed_time(end) / n_iter

    return {
        'full_ms': full_time,
        'binned_ms': binned_time,
        'binned_interp_ms': binned_interp_time,
        'speedup_nearest': full_time / binned_time,
        'speedup_interp': full_time / binned_interp_time,
    }


def main():
    print("=" * 100)
    print("Binned Radial Weights Benchmark")
    print("=" * 100)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    configs = [
        # (lmax, batch, cin, cout, num_bins)
        (4, 1000, 64, 64, 100),
        (4, 5000, 64, 64, 100),
        (6, 1000, 64, 64, 100),
        (6, 5000, 64, 64, 100),
        (6, 10000, 32, 32, 100),
        (6, 20000, 32, 32, 100),
    ]

    # Memory comparison
    print("\n" + "-" * 100)
    print("Memory Comparison")
    print("-" * 100)
    print(f"{'Config':<35} {'Full':>12} {'Binned':>12} {'Reduction':>12}")
    print("-" * 100)

    for lmax, batch, cin, cout, num_bins in configs:
        mem = benchmark_memory(lmax, batch, cin, cout, num_bins)
        config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
        print(f"{config_str:<35} {mem['full_mb']:>10.1f}MB {mem['binned_total_mb']:>10.3f}MB "
              f"{mem['reduction']:>11.0f}x")

    # Speed comparison
    print("\n" + "-" * 100)
    print("Speed Comparison (FP32)")
    print("-" * 100)
    print(f"{'Config':<35} {'Full':>10} {'Binned':>10} {'Interp':>10} {'Speedup':>10} {'Speedup(I)':>12}")
    print("-" * 100)

    for lmax, batch, cin, cout, num_bins in configs:
        try:
            speed = benchmark_speed(lmax, batch, cin, cout, num_bins, torch.float32)
            config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
            print(f"{config_str:<35} {speed['full_ms']:>9.2f}ms {speed['binned_ms']:>9.2f}ms "
                  f"{speed['binned_interp_ms']:>9.2f}ms {speed['speedup_nearest']:>9.2f}x "
                  f"{speed['speedup_interp']:>11.2f}x")
        except Exception as e:
            print(f"L={lmax}, B={batch}, C={cin}x{cout}: {e}")
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("Notes:")
    print("- Full: stores (batch, cout, cin, weight_dim) tensor")
    print("- Binned: stores (num_bins, weight_dim) table + (batch,) indices")
    print("- Interp: linear interpolation between adjacent bins")
    print("- Binned kernel also benefits from weight sharing across channels")
    print("=" * 100)


if __name__ == "__main__":
    main()
