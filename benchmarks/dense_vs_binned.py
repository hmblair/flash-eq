"""
Benchmark training loop: Dense vs Binned vs Gathered (bin-sorted).

Compares memory usage and runtime for forward + backward pass.
"""

import torch
import torch.nn as nn
import gc
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_interp_cuda,
    block_diagonal_gathered_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import RadialBinning


class RadialMLP(nn.Module):
    """Standard radial MLP that outputs per-edge weights."""

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


def benchmark_standard_training(lmax, batch, cin, cout, dtype, n_warmup=3, n_iter=10):
    """Benchmark standard approach: full per-edge weights."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Model and optimizer
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Fixed inputs for benchmarking
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0
    distances.requires_grad_(True)
    target = torch.randn(batch, cout, dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()
        weights = mlp(distances)
        output = block_diagonal_cuda(features, weights, metadata)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Warmup
    for _ in range(n_warmup):
        train_step()
    torch.cuda.synchronize()

    # Benchmark
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        train_step()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_binned_training(lmax, batch, cin, cout, num_bins, dtype, n_warmup=3, n_iter=10):
    """Benchmark binned approach with gradients."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Model and optimizer
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)

    # Fixed inputs for benchmarking
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0
    distances.requires_grad_(True)
    target = torch.randn(batch, cout, dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()

        # Binned approach: MLP at bin edges, kernel handles interpolation
        radial_table = mlp(binning.bin_edges)

        output = block_diagonal_binned_interp_cuda(
            features, radial_table, distances, metadata
        )

        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Warmup
    for _ in range(n_warmup):
        train_step()
    torch.cuda.synchronize()

    # Benchmark
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        train_step()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def benchmark_gathered_training(lmax, num_nodes, num_edges, cin, cout, num_bins, dtype, n_warmup=3, n_iter=10):
    """Benchmark gathered (bin-sorted) approach with gradients."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Model and optimizer
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)

    # Fixed inputs for benchmarking
    # Node features (smaller than edge features)
    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype, requires_grad=True)
    # Edge structure
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
    distances.requires_grad_(True)
    target = torch.randn(num_edges, cout, dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()

        # Gathered approach: MLP at bin edges, kernel handles gather + interpolation
        radial_table = mlp(binning.bin_edges)

        output, unsort_indices = block_diagonal_gathered_cuda(
            node_features, src_indices, radial_table, distances, metadata, sort_by_bin=True
        )
        # Unsort to match target order
        output = output[unsort_indices]

        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Warmup
    for _ in range(n_warmup):
        train_step()
    torch.cuda.synchronize()

    # Benchmark
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        train_step()
    end.record()
    torch.cuda.synchronize()

    return {
        'time_ms': start.elapsed_time(end) / n_iter,
        'peak_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    print("=" * 120)
    print("Training Benchmark: Dense vs Binned vs Gathered (bin-sorted)")
    print("=" * 120)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    dtype = torch.float32
    num_bins = 100

    # Configs: (lmax, num_nodes, num_edges, cin, cout)
    # For dense/binned, we use num_edges as "batch" (edge features pre-expanded)
    configs = [
        # Small scale
        (4, 1000, 5000, 32, 32),
        (6, 1000, 5000, 32, 32),
        # Medium scale
        (4, 2000, 20000, 32, 32),
        (6, 2000, 20000, 32, 32),
        # Large scale (typical GNN)
        (4, 5000, 128000, 32, 32),
        (6, 5000, 128000, 32, 32),
        # Very large
        (6, 10000, 256000, 32, 32),
    ]

    print(f"\nSettings: num_bins={num_bins}, dtype=float32")
    print(f"\n{'Config':<35} {'Dense':>22} {'Binned':>22} {'Gathered':>22}")
    print("-" * 120)

    results = []

    for lmax, num_nodes, num_edges, cin, cout in configs:
        config_str = f"L={lmax}, N={num_nodes}, E={num_edges}, C={cin}"

        # Dense approach (per-edge weights)
        try:
            r_dense = benchmark_standard_training(lmax, num_edges, cin, cout, dtype)
            dense_str = f"{r_dense['time_ms']:.1f}ms / {r_dense['peak_mem_mb']:.0f}MB"
        except torch.cuda.OutOfMemoryError:
            r_dense = None
            dense_str = "OOM"
        clear_memory()

        # Binned approach (pre-expanded edge features)
        try:
            r_binned = benchmark_binned_training(lmax, num_edges, cin, cout, num_bins, dtype)
            binned_str = f"{r_binned['time_ms']:.1f}ms / {r_binned['peak_mem_mb']:.0f}MB"
        except torch.cuda.OutOfMemoryError:
            r_binned = None
            binned_str = "OOM"
        clear_memory()

        # Gathered approach (bin-sorted, no pre-expansion)
        try:
            r_gathered = benchmark_gathered_training(lmax, num_nodes, num_edges, cin, cout, num_bins, dtype)
            gathered_str = f"{r_gathered['time_ms']:.1f}ms / {r_gathered['peak_mem_mb']:.0f}MB"
        except torch.cuda.OutOfMemoryError:
            r_gathered = None
            gathered_str = "OOM"
        clear_memory()

        print(f"{config_str:<35} {dense_str:>22} {binned_str:>22} {gathered_str:>22}")
        results.append((config_str, r_dense, r_binned, r_gathered))

    # Summary table with ratios
    print(f"\n{'='*120}")
    print("Summary: Memory and Speed Ratios (vs Dense baseline)")
    print("=" * 120)
    print(f"\n{'Config':<35} {'Binned Mem':>12} {'Binned Speed':>14} {'Gathered Mem':>14} {'Gathered Speed':>14}")
    print("-" * 120)

    for config_str, r_dense, r_binned, r_gathered in results:
        if r_dense and r_binned:
            binned_mem = f"{r_dense['peak_mem_mb'] / r_binned['peak_mem_mb']:.1f}x"
            binned_speed = f"{r_dense['time_ms'] / r_binned['time_ms']:.2f}x"
        else:
            binned_mem = "N/A"
            binned_speed = "N/A"

        if r_dense and r_gathered:
            gathered_mem = f"{r_dense['peak_mem_mb'] / r_gathered['peak_mem_mb']:.1f}x"
            gathered_speed = f"{r_dense['time_ms'] / r_gathered['time_ms']:.2f}x"
        else:
            gathered_mem = "N/A"
            gathered_speed = "N/A"

        print(f"{config_str:<35} {binned_mem:>12} {binned_speed:>14} {gathered_mem:>14} {gathered_speed:>14}")

    # Detailed breakdown
    print(f"\n{'='*120}")
    print("Detailed Analysis: L=6, N=5000, E=128000, C=32x32")
    print("=" * 120)

    lmax, num_nodes, num_edges, cin, cout = 6, 5000, 128000, 32, 32
    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    # Memory breakdown
    print(f"\nTheoretical memory usage:")
    print(f"  Weight dimension: {weight_dim}")
    print(f"  Feature dimension: {dim}")

    # Dense weights
    dense_weights_mb = num_edges * cout * cin * weight_dim * 4 / 1024**2
    print(f"\n  Dense weights: ({num_edges}, {cout}, {cin}, {weight_dim}) = {dense_weights_mb:.1f} MB")

    # Binned table
    binned_table_mb = (num_bins + 1) * cout * cin * weight_dim * 4 / 1024**2
    print(f"  Binned table:  ({num_bins+1}, {cout}, {cin}, {weight_dim}) = {binned_table_mb:.1f} MB")

    # Edge features (binned needs this, gathered doesn't)
    edge_features_mb = num_edges * cin * dim * 4 / 1024**2
    print(f"\n  Edge features (binned): ({num_edges}, {cin}, {dim}) = {edge_features_mb:.1f} MB")

    # Node features (gathered uses this instead)
    node_features_mb = num_nodes * cin * dim * 4 / 1024**2
    print(f"  Node features (gathered): ({num_nodes}, {cin}, {dim}) = {node_features_mb:.1f} MB")
    print(f"  Feature memory savings: {edge_features_mb / node_features_mb:.1f}x")

    print("\n" + "=" * 120)
    print("Summary:")
    print("  - Dense: O(E * Cout * Cin * Wdim) memory for weights - scales poorly")
    print("  - Binned: O(bins * Cout * Cin * Wdim) weights + O(E * Cin * dim) features")
    print("  - Gathered: O(bins * Cout * Cin * Wdim) weights + O(N * Cin * dim) features")
    print("  - Gathered wins when E >> N (typical in molecular graphs)")
    print("=" * 120)


if __name__ == "__main__":
    main()
