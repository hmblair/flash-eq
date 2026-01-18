"""
Benchmark: SE3-Transformer (dense matmuls) vs Flash-eq (public API).

Compares memory usage and runtime for forward + backward pass.

SE3-Transformer approach (NVIDIA):
  - Per-edge radial weights from MLP: O(E * cout * cin * freq_sum)
  - Two dense matmuls: features @ basis, radial_weights @ tmp

Flash-eq approach:
  - EquivariantEdgewiseLinear with binned radial weights
  - Block-diagonal kernel with Wigner-D diagonalization
"""

import torch
import torch.nn as nn
import gc
from flash_eq import EquivariantEdgewiseLinear, WignerDBasis, Repr


class RadialMLP(nn.Module):
    """Radial MLP that outputs weights (for SE3-T baseline)."""

    def __init__(self, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, distances):
        return self.net(distances.unsqueeze(-1))


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_se3_transformer(lmax, num_edges, cin, cout, dtype, use_amp=False, n_warmup=3, n_iter=10):
    """
    Benchmark SE3-Transformer style: two dense matmuls with per-edge weights.

    Simulates VersatileConvSE3:
        basis_view = basis.view(num_edges, in_dim, -1)
        tmp = (features @ basis_view).view(num_edges, -1, out_dim)
        output = radial_weights @ tmp
    """
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    in_dim = sum(2 * l + 1 for l in lvals)
    out_dim = in_dim

    # freq_sum for SE3-T: sum of min(l_in, l_out) + 1 for all pairs
    freq_sum = sum(min(l1, l2) + 1 for l1 in lvals for l2 in lvals)

    # Radial MLP outputs per-edge weights
    mlp = RadialMLP(cout * cin * freq_sum).to(device).to(dtype)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Edge-level features
    features = torch.randn(num_edges, cin, in_dim, device=device, dtype=dtype, requires_grad=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0

    # Basis matrix (random, simulating CG coefficients)
    basis = torch.randn(num_edges, in_dim, freq_sum * out_dim, device=device, dtype=dtype)

    target = torch.randn(num_edges, cout, out_dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Per-edge radial weights from MLP
            radial_weights = mlp(distances).view(num_edges, cout, cin * freq_sum)

            # Two matmuls (SE3-Transformer style)
            tmp = (features @ basis).view(num_edges, cin * freq_sum, out_dim)
            output = radial_weights @ tmp

            loss = ((output - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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


def benchmark_flash_eq(lmax, num_nodes, num_edges, cin, cout, num_bins, dtype, use_amp=False, n_warmup=3, n_iter=10):
    """
    Benchmark Flash-eq using the public API: EquivariantEdgewiseLinear + WignerDBasis.
    """
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    # Create representations
    in_repr = Repr(lvals=lvals, mult=cin)
    out_repr = Repr(lvals=lvals, mult=cout)

    # Create layer and basis
    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr,
        num_bins=num_bins,
        min_dist=0.0,
        max_dist=10.0,
    ).to(device).to(dtype)

    basis = WignerDBasis(in_repr, out_repr).to(device)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Node features
    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype, requires_grad=True)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)

    # Edge directions (unit vectors) and distances
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0

    target = torch.randn(num_edges, cout, dim, device=device, dtype=dtype)

    # Precompute basis matrices
    P, Q = basis(directions)

    def train_step():
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass using public API
            output = layer(P, Q, node_features, distances, src_indices)

            loss = ((output - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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


def run_benchmark(dtype, use_amp, num_bins=100):
    """Run benchmark suite for a given precision setting."""
    precision_name = "float16 (AMP)" if use_amp else ("float16" if dtype == torch.float16 else "float32")

    print(f"\n{'='*100}")
    print(f"Precision: {precision_name}")
    print("=" * 100)

    # Configs: (lmax, num_nodes, num_edges, cin, cout)
    configs = [
        # Low lmax at 32K edges
        (1, 3000, 32000, 32, 32),
        (2, 3000, 32000, 32, 32),
        # Small scale
        (4, 1000, 5000, 32, 32),
        (6, 1000, 5000, 32, 32),
        # Medium scale
        (4, 2000, 20000, 32, 32),
        (6, 2000, 20000, 32, 32),
        # Large scale (typical GNN)
        (4, 5000, 50000, 32, 32),
        (6, 5000, 50000, 32, 32),
        # Very large
        (4, 5000, 128000, 32, 32),
        (6, 5000, 128000, 32, 32),
    ]

    print(f"\n{'Config':<40} {'SE3-Transformer':>25} {'Flash-eq':>25}")
    print("-" * 100)

    results = []

    for lmax, num_nodes, num_edges, cin, cout in configs:
        config_str = f"L={lmax}, N={num_nodes}, E={num_edges}, C={cin}"

        # SE3-Transformer approach
        try:
            r_se3 = benchmark_se3_transformer(lmax, num_edges, cin, cout, dtype, use_amp)
            se3_str = f"{r_se3['time_ms']:.1f}ms / {r_se3['peak_mem_mb']:.0f}MB"
        except torch.cuda.OutOfMemoryError:
            r_se3 = None
            se3_str = "OOM"
        clear_memory()

        # Flash-eq approach
        try:
            r_flash = benchmark_flash_eq(lmax, num_nodes, num_edges, cin, cout, num_bins, dtype, use_amp)
            flash_str = f"{r_flash['time_ms']:.1f}ms / {r_flash['peak_mem_mb']:.0f}MB"
        except torch.cuda.OutOfMemoryError:
            r_flash = None
            flash_str = "OOM"
        clear_memory()

        print(f"{config_str:<40} {se3_str:>25} {flash_str:>25}")
        results.append((config_str, r_se3, r_flash))

    # Summary table with ratios
    print(f"\nSummary ({precision_name}):")
    print(f"{'Config':<40} {'Memory Savings':>15} {'Speedup':>15}")
    print("-" * 70)

    for config_str, r_se3, r_flash in results:
        if r_se3 and r_flash:
            mem_savings = f"{r_se3['peak_mem_mb'] / r_flash['peak_mem_mb']:.1f}x"
            speedup = f"{r_se3['time_ms'] / r_flash['time_ms']:.2f}x"
        elif r_flash and not r_se3:
            mem_savings = "SE3 OOM"
            speedup = "SE3 OOM"
        else:
            mem_savings = "N/A"
            speedup = "N/A"

        print(f"{config_str:<40} {mem_savings:>15} {speedup:>15}")

    return results


def main():
    print("=" * 100)
    print("Benchmark: SE3-Transformer (dense matmuls) vs Flash-eq (public API)")
    print("=" * 100)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Run float32 benchmark
    results_fp32 = run_benchmark(torch.float32, use_amp=False)

    # Run float16 with AMP benchmark
    results_fp16 = run_benchmark(torch.float32, use_amp=True)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
