"""
Benchmark flash-eq vs ciffy EquivariantConvolution.
Measures runtime and memory usage on GPU.
"""

import torch
import torch.nn as nn
import time
import gc

from ciffy.nn.geometric.representations import Repr, ProductRepr
from ciffy.nn.geometric.layers import EquivariantConvolution, RadialWeight
from ciffy.nn.geometric.equivariant import EquivariantBasis

from flash_eq import EquivariantLinear


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e6
    return 0


class CiffyWrapper(nn.Module):
    """Wrapper to make ciffy API match flash-eq for benchmarking."""

    def __init__(self, repr_in: Repr, repr_out: Repr, edge_dim: int, hidden_dim: int, rank: int | None = 0):
        super().__init__()
        self.repr_in = repr_in
        self.repr_out = repr_out

        # Create ProductRepr for ciffy
        self.prepr = ProductRepr(repr_in, repr_out)

        # Basis computation
        self.basis_module = EquivariantBasis(self.prepr, rank=rank)
        num_basis = self.basis_module.num_basis

        # Convolution layer
        self.conv = EquivariantConvolution(
            self.prepr,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
        )

        self.edge_dim = edge_dim

    def forward(self, features, directions, edge_feats):
        """
        Args:
            features: (E, channels_in, dim_in)
            directions: (E, 3)
            edge_feats: (E, edge_dim)
        """
        E = features.shape[0]

        # Compute basis matrices
        basis = self.basis_module(directions)  # (E, num_basis, dim_in, dim_out)

        # For ciffy, we need src_idx - use identity (each edge is its own source)
        src_idx = torch.arange(E, device=features.device)

        # Apply convolution
        output = self.conv(basis, edge_feats, features, src_idx)

        return output


class FlashEqWrapper(nn.Module):
    """Wrapper for flash-eq with radial weight network."""

    def __init__(self, repr_in: Repr, repr_out: Repr, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.layer = EquivariantLinear(repr_in, repr_out)

        # Radial weight network (similar to ciffy's RadialWeight)
        self.weight_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_out.mult * repr_in.mult * self.layer.weight_dim),
        )

        self.repr_in = repr_in
        self.repr_out = repr_out

    def forward(self, features, directions, edge_feats):
        """
        Args:
            features: (E, channels_in, dim_in)
            directions: (E, 3)
            edge_feats: (E, edge_dim)
        """
        E = features.shape[0]
        channels_in = self.repr_in.mult
        channels_out = self.repr_out.mult

        # Compute weights from edge features
        weights = self.weight_net(edge_feats)
        weights = weights.view(E, channels_out, channels_in, self.layer.weight_dim)

        # Apply equivariant linear
        output = self.layer(features, directions, weights)

        return output


def benchmark_forward(model, features, directions, edge_feats, n_warmup=10, n_trials=50):
    """Benchmark forward pass."""
    # Warmup
    for _ in range(n_warmup):
        _ = model(features, directions, edge_feats)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_trials):
        _ = model(features, directions, edge_feats)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_trials * 1000  # ms


def benchmark_memory(model, features, directions, edge_feats):
    """Benchmark peak memory during forward pass."""
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    # Run forward pass
    _ = model(features, directions, edge_feats)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return peak_mb


def run_benchmark(edges, channels, lvals, edge_dim, hidden_dim, device, rank=None):
    """Run a single benchmark comparison."""
    repr_in = Repr(lvals=lvals, mult=channels)
    repr_out = Repr(lvals=lvals, mult=channels)
    dim = repr_in.dim()

    # Create models
    ciffy_model = CiffyWrapper(repr_in, repr_out, edge_dim, hidden_dim, rank=rank).to(device)
    flash_model = FlashEqWrapper(repr_in, repr_out, edge_dim, hidden_dim).to(device)

    # Create inputs
    features = torch.randn(edges, channels, dim, device=device)
    directions = torch.randn(edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    edge_feats = torch.randn(edges, edge_dim, device=device)

    # Benchmark runtime
    ciffy_time = benchmark_forward(ciffy_model, features, directions, edge_feats)
    flash_time = benchmark_forward(flash_model, features, directions, edge_feats)

    # Benchmark memory
    del ciffy_model, flash_model
    clear_memory()

    ciffy_model = CiffyWrapper(repr_in, repr_out, edge_dim, hidden_dim, rank=rank).to(device)
    features = torch.randn(edges, channels, dim, device=device)
    directions = torch.randn(edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    edge_feats = torch.randn(edges, edge_dim, device=device)

    ciffy_mem = benchmark_memory(ciffy_model, features, directions, edge_feats)

    del ciffy_model
    clear_memory()

    flash_model = FlashEqWrapper(repr_in, repr_out, edge_dim, hidden_dim).to(device)
    features = torch.randn(edges, channels, dim, device=device)
    directions = torch.randn(edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    edge_feats = torch.randn(edges, edge_dim, device=device)

    flash_mem = benchmark_memory(flash_model, features, directions, edge_feats)

    del flash_model
    clear_memory()

    return {
        'ciffy_time_ms': ciffy_time,
        'flash_time_ms': flash_time,
        'ciffy_mem_mb': ciffy_mem,
        'flash_mem_mb': flash_mem,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    edge_dim = 16
    hidden_dim = 64

    # Benchmark 1: Varying edges
    print("=" * 80)
    print("Benchmark 1: Varying number of edges (channels=32, lmax=2)")
    print("=" * 80)
    print(f"{'Edges':<10} {'ciffy (ms)':<12} {'flash (ms)':<12} {'Speedup':<10} {'ciffy (MB)':<12} {'flash (MB)':<12} {'Mem Ratio':<10}")
    print("-" * 80)

    for edges in [1000, 2000, 5000, 10000]:
        try:
            res = run_benchmark(edges, channels=32, lvals=[0, 1, 2], edge_dim=edge_dim, hidden_dim=hidden_dim, device=device)
            speedup = res['ciffy_time_ms'] / res['flash_time_ms']
            mem_ratio = res['ciffy_mem_mb'] / res['flash_mem_mb']
            print(f"{edges:<10} {res['ciffy_time_ms']:<12.2f} {res['flash_time_ms']:<12.2f} {speedup:<10.2f}x {res['ciffy_mem_mb']:<12.1f} {res['flash_mem_mb']:<12.1f} {mem_ratio:<10.2f}x")
        except Exception as e:
            print(f"{edges:<10} Error: {e}")
        clear_memory()

    # Benchmark 2: Varying lmax
    print()
    print("=" * 80)
    print("Benchmark 2: Varying lmax (edges=5000, channels=32)")
    print("=" * 80)
    print(f"{'lmax':<10} {'dim':<6} {'ciffy (ms)':<12} {'flash (ms)':<12} {'Speedup':<10} {'ciffy (MB)':<12} {'flash (MB)':<12} {'Mem Ratio':<10}")
    print("-" * 80)

    for lmax in [1, 2, 3]:
        lvals = list(range(lmax + 1))
        dim = sum(2*l+1 for l in lvals)
        try:
            res = run_benchmark(5000, channels=32, lvals=lvals, edge_dim=edge_dim, hidden_dim=hidden_dim, device=device)
            speedup = res['ciffy_time_ms'] / res['flash_time_ms']
            mem_ratio = res['ciffy_mem_mb'] / res['flash_mem_mb']
            print(f"{lmax:<10} {dim:<6} {res['ciffy_time_ms']:<12.2f} {res['flash_time_ms']:<12.2f} {speedup:<10.2f}x {res['ciffy_mem_mb']:<12.1f} {res['flash_mem_mb']:<12.1f} {mem_ratio:<10.2f}x")
        except Exception as e:
            print(f"{lmax:<10} {dim:<6} Error: {e}")
        clear_memory()

    # Benchmark 3: Varying channels
    print()
    print("=" * 80)
    print("Benchmark 3: Varying channels (edges=5000, lmax=2)")
    print("=" * 80)
    print(f"{'Channels':<10} {'ciffy (ms)':<12} {'flash (ms)':<12} {'Speedup':<10} {'ciffy (MB)':<12} {'flash (MB)':<12} {'Mem Ratio':<10}")
    print("-" * 80)

    for channels in [16, 32, 64]:
        try:
            res = run_benchmark(5000, channels=channels, lvals=[0, 1, 2], edge_dim=edge_dim, hidden_dim=hidden_dim, device=device)
            speedup = res['ciffy_time_ms'] / res['flash_time_ms']
            mem_ratio = res['ciffy_mem_mb'] / res['flash_mem_mb']
            print(f"{channels:<10} {res['ciffy_time_ms']:<12.2f} {res['flash_time_ms']:<12.2f} {speedup:<10.2f}x {res['ciffy_mem_mb']:<12.1f} {res['flash_mem_mb']:<12.1f} {mem_ratio:<10.2f}x")
        except Exception as e:
            print(f"{channels:<10} Error: {e}")
        clear_memory()


if __name__ == "__main__":
    main()
