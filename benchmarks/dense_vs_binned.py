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
from flash_eq import EquivariantEdgewiseLinear, WignerDBasis, Repr, GraphPooling


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


def benchmark_basis_computation(lmax, num_edges, dtype, n_warmup=3, n_iter=10):
    """
    Benchmark WignerDBasis P/Q matrix computation.

    Measures:
    - Runtime to compute P and Q matrices from directions
    - Memory footprint of the resulting P and Q tensors
    """
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    # Create representations and basis
    repr_in = Repr(lvals=lvals, mult=1)
    repr_out = Repr(lvals=lvals, mult=1)
    basis = WignerDBasis(repr_in, repr_out).to(device)

    # Random directions (unit vectors)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Warmup
    for _ in range(n_warmup):
        P, Q = basis(directions)
        del P, Q
    torch.cuda.synchronize()
    clear_memory()

    # Benchmark runtime
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        P, Q = basis(directions)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end) / n_iter

    # Measure memory of P and Q tensors
    P_mem_mb = P.numel() * P.element_size() / 1024**2
    Q_mem_mb = Q.numel() * Q.element_size() / 1024**2

    return {
        'time_ms': time_ms,
        'P_mem_mb': P_mem_mb,
        'Q_mem_mb': Q_mem_mb,
        'total_mem_mb': P_mem_mb + Q_mem_mb,
        'P_shape': tuple(P.shape),
        'Q_shape': tuple(Q.shape),
        'dim': dim,
    }


def run_basis_benchmark(dtype):
    """Run benchmark suite for P/Q basis matrix computation."""
    dtype_name = "float32" if dtype == torch.float32 else "float16"

    print(f"\n{'='*100}")
    print(f"Benchmark: WignerDBasis P/Q Computation ({dtype_name})")
    print("=" * 100)

    # Configs: (lmax, num_edges)
    configs = [
        # Varying edges at fixed lmax
        (2, 1000),
        (2, 5000),
        (2, 10000),
        (2, 50000),
        (2, 100000),
        # Varying lmax at fixed edges
        (1, 50000),
        (2, 50000),
        (3, 50000),
        (4, 50000),
        (6, 50000),
        # Large scale
        (4, 100000),
        (6, 100000),
        (4, 200000),
        (6, 200000),
    ]

    print(f"\n{'Config':<25} {'P/Q Shape':<25} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 80)

    results = []

    for lmax, num_edges in configs:
        config_str = f"L={lmax}, E={num_edges}"

        try:
            r = benchmark_basis_computation(lmax, num_edges, dtype)
            shape_str = f"({num_edges}, {r['dim']}, {r['dim']})"
            time_str = f"{r['time_ms']:.2f}"
            mem_str = f"{r['total_mem_mb']:.2f}"
            results.append((config_str, r))
        except torch.cuda.OutOfMemoryError:
            shape_str = "OOM"
            time_str = "OOM"
            mem_str = "OOM"
            results.append((config_str, None))
        clear_memory()

        print(f"{config_str:<25} {shape_str:<25} {time_str:<15} {mem_str:<15}")

    # Summary: memory scaling analysis
    print(f"\nMemory Scaling Analysis:")
    print(f"{'Config':<25} {'Bytes/Edge':<15} {'ms/1K Edges':<15}")
    print("-" * 55)

    for config_str, r in results:
        if r:
            num_edges = int(config_str.split("E=")[1])
            bytes_per_edge = r['total_mem_mb'] * 1024**2 / num_edges
            ms_per_1k = r['time_ms'] / (num_edges / 1000)
            print(f"{config_str:<25} {bytes_per_edge:<15.0f} {ms_per_1k:<15.3f}")

    return results


def benchmark_graph_pooling(num_nodes, num_edges, channels, dim, dtype, n_warmup=3, n_iter=10):
    """
    Benchmark GraphPooling operations (sum, mean, max).

    Compares against a baseline using index_add for reference.
    """
    clear_memory()
    device = torch.device("cuda")

    # Create pooling modules
    pool_sum = GraphPooling(reduce='sum')
    pool_mean = GraphPooling(reduce='mean')
    pool_max = GraphPooling(reduce='max')

    # Random edge features and destination indices
    edge_features = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

    results = {}

    for name, pool in [('sum', pool_sum), ('mean', pool_mean), ('max', pool_max)]:
        # Warmup
        for _ in range(n_warmup):
            out = pool(edge_features, dst_indices, num_nodes)
        torch.cuda.synchronize()
        clear_memory()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iter):
            out = pool(edge_features, dst_indices, num_nodes)
        end.record()
        torch.cuda.synchronize()

        results[name] = {
            'time_ms': start.elapsed_time(end) / n_iter,
            'output_shape': tuple(out.shape),
        }
        clear_memory()

    return results


def run_pooling_benchmark(dtype):
    """Run benchmark suite for GraphPooling operations."""
    dtype_name = "float32" if dtype == torch.float32 else "float16"

    print(f"\n{'='*100}")
    print(f"Benchmark: GraphPooling ({dtype_name})")
    print("=" * 100)

    # Configs: (num_nodes, num_edges, channels, dim)
    configs = [
        # Varying edges
        (1000, 5000, 32, 9),
        (1000, 10000, 32, 9),
        (1000, 50000, 32, 9),
        (1000, 100000, 32, 9),
        # Varying nodes (affects sparsity)
        (100, 50000, 32, 9),
        (1000, 50000, 32, 9),
        (10000, 50000, 32, 9),
        # Varying channels
        (1000, 50000, 16, 9),
        (1000, 50000, 32, 9),
        (1000, 50000, 64, 9),
        # Varying dim (lmax)
        (1000, 50000, 32, 4),   # lmax=1
        (1000, 50000, 32, 9),   # lmax=2
        (1000, 50000, 32, 16),  # lmax=3
        (1000, 50000, 32, 49),  # lmax=6
        # Large scale
        (5000, 200000, 32, 9),
        (10000, 500000, 32, 9),
    ]

    print(f"\n{'Config':<35} {'Sum (ms)':<12} {'Mean (ms)':<12} {'Max (ms)':<12}")
    print("-" * 75)

    all_results = []

    for num_nodes, num_edges, channels, dim in configs:
        config_str = f"N={num_nodes}, E={num_edges}, C={channels}, D={dim}"

        try:
            r = benchmark_graph_pooling(num_nodes, num_edges, channels, dim, dtype)
            sum_str = f"{r['sum']['time_ms']:.3f}"
            mean_str = f"{r['mean']['time_ms']:.3f}"
            max_str = f"{r['max']['time_ms']:.3f}"
            all_results.append((config_str, r))
        except torch.cuda.OutOfMemoryError:
            sum_str = mean_str = max_str = "OOM"
            all_results.append((config_str, None))
        clear_memory()

        print(f"{config_str:<35} {sum_str:<12} {mean_str:<12} {max_str:<12}")

    # Throughput analysis
    print(f"\nThroughput Analysis (sum pooling):")
    print(f"{'Config':<35} {'Edges/ms':<15} {'GB/s':<15}")
    print("-" * 65)

    for config_str, r in all_results:
        if r:
            # Parse config
            parts = config_str.split(', ')
            num_edges = int(parts[1].split('=')[1])
            channels = int(parts[2].split('=')[1])
            dim = int(parts[3].split('=')[1])

            edges_per_ms = num_edges / r['sum']['time_ms']
            bytes_per_edge = channels * dim * (4 if dtype == torch.float32 else 2)
            gb_per_s = (num_edges * bytes_per_edge / 1e9) / (r['sum']['time_ms'] / 1000)

            print(f"{config_str:<35} {edges_per_ms:<15.0f} {gb_per_s:<15.1f}")

    return all_results


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

    # Run P/Q basis computation benchmark
    basis_results_fp32 = run_basis_benchmark(torch.float32)

    print("\n" + "=" * 100)

    # Run graph pooling benchmark
    pooling_results_fp32 = run_pooling_benchmark(torch.float32)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
