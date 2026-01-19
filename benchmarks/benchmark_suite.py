"""
Benchmark suite for Flash-eq components.

Compares performance of:
1. SE3-Transformer (dense matmuls) vs Flash-eq (binned radial weights)
2. WignerDBasis P/Q matrix computation
3. GraphPooling operations (sum, mean, max)
4. EquivariantEdgeAttention

Each benchmark measures runtime (ms) and peak memory (MB) for forward + backward.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from flash_eq import (
    EquivariantEdgeAttention,
    EquivariantEdgewiseLinear,
    EquivariantTransformerBlock,
    EquivariantTransformer,
    GraphPooling,
    Repr,
    WignerDBasis,
    S2Activation,
)


# =============================================================================
# Shared Utilities
# =============================================================================


@dataclass
class BenchmarkResult:
    """Stores benchmark results."""

    time_ms: float
    peak_mem_mb: float
    extra: Optional[dict] = None


def clear_memory():
    """Clear CUDA memory and reset peak stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@contextmanager
def cuda_timer():
    """Context manager for CUDA timing. Yields a callable that returns elapsed ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    elapsed = []
    yield lambda: elapsed[0] if elapsed else None
    end.record()
    torch.cuda.synchronize()
    elapsed.append(start.elapsed_time(end))


def run_timed_iterations(
    fn: Callable, n_warmup: int = 3, n_iter: int = 10
) -> BenchmarkResult:
    """Run warmup iterations, then benchmark and return results."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    with cuda_timer() as get_elapsed:
        for _ in range(n_iter):
            fn()

    return BenchmarkResult(
        time_ms=get_elapsed() / n_iter,
        peak_mem_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )


def format_result(result: Optional[BenchmarkResult], show_mem: bool = True) -> str:
    """Format a benchmark result for display."""
    if result is None:
        return "OOM"
    if show_mem:
        return f"{result.time_ms:.1f}ms / {result.peak_mem_mb:.0f}MB"
    return f"{result.time_ms:.2f}ms"


def print_header(title: str, width: int = 100):
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)


def print_table_header(columns: list[tuple[str, int]]):
    """Print table header with column names and widths."""
    header = "".join(f"{name:<{width}}" for name, width in columns)
    print(f"\n{header}")
    print("-" * sum(w for _, w in columns))


def print_table_row(values: list[tuple[str, int]]):
    """Print a table row with values and widths."""
    print("".join(f"{val:<{width}}" for val, width in values))


# =============================================================================
# Baseline: SE3-Transformer (Dense Matmuls)
# =============================================================================


class RadialMLP(nn.Module):
    """Radial MLP that outputs per-edge weights."""

    def __init__(self, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.net(distances.unsqueeze(-1))


def benchmark_se3_transformer(
    lmax: int,
    num_edges: int,
    cin: int,
    cout: int,
    dtype: torch.dtype,
    use_amp: bool = False,
) -> BenchmarkResult:
    """
    Benchmark SE3-Transformer: two dense matmuls with per-edge radial weights.

    Simulates the VersatileConvSE3 approach:
        tmp = features @ basis
        output = radial_weights @ tmp
    """
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    in_dim = sum(2 * l + 1 for l in lvals)
    out_dim = in_dim
    freq_sum = sum(min(l1, l2) + 1 for l1 in lvals for l2 in lvals)

    mlp = RadialMLP(cout * cin * freq_sum).to(device).to(dtype)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    features = torch.randn(
        num_edges, cin, in_dim, device=device, dtype=dtype, requires_grad=True
    )
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
    basis = torch.randn(num_edges, in_dim, freq_sum * out_dim, device=device, dtype=dtype)
    target = torch.randn(num_edges, cout, out_dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            radial_weights = mlp(distances).view(num_edges, cout, cin * freq_sum)
            tmp = (features @ basis).view(num_edges, cin * freq_sum, out_dim)
            output = radial_weights @ tmp
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return run_timed_iterations(train_step)


# =============================================================================
# Flash-eq: EquivariantEdgewiseLinear
# =============================================================================


def benchmark_flash_eq(
    lmax: int,
    num_nodes: int,
    num_edges: int,
    cin: int,
    cout: int,
    num_bins: int,
    dtype: torch.dtype,
    use_amp: bool = False,
) -> BenchmarkResult:
    """Benchmark Flash-eq EquivariantEdgewiseLinear + WignerDBasis."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    in_repr = Repr(lvals=lvals, mult=cin)
    out_repr = Repr(lvals=lvals, mult=cout)

    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr, num_bins=num_bins, min_dist=0.0, max_dist=10.0
    ).to(device).to(dtype)
    basis = WignerDBasis([in_repr, out_repr]).to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    node_features = torch.randn(
        num_nodes, cin, dim, device=device, dtype=dtype, requires_grad=True
    )
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
    target = torch.randn(num_edges, cout, dim, device=device, dtype=dtype)

    P, Q = basis(directions)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = layer(P, Q, node_features, distances, src_indices)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return run_timed_iterations(train_step)


# =============================================================================
# WignerDBasis P/Q Computation
# =============================================================================


def benchmark_basis_computation(
    lmax: int, num_edges: int, dtype: torch.dtype
) -> BenchmarkResult:
    """Benchmark WignerDBasis P/Q matrix computation."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    repr_in = Repr(lvals=lvals, mult=1)
    repr_out = Repr(lvals=lvals, mult=1)
    basis = WignerDBasis([repr_in, repr_out]).to(device)

    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    def compute_basis():
        return basis(directions)

    result = run_timed_iterations(compute_basis)
    P, Q = compute_basis()

    result.extra = {
        "P_mem_mb": P.numel() * P.element_size() / 1024**2,
        "Q_mem_mb": Q.numel() * Q.element_size() / 1024**2,
        "dim": dim,
    }
    return result


# =============================================================================
# GraphPooling
# =============================================================================


def benchmark_graph_pooling(
    num_nodes: int, num_edges: int, channels: int, dim: int, dtype: torch.dtype
) -> dict[str, BenchmarkResult]:
    """Benchmark GraphPooling operations (sum, mean, max)."""
    clear_memory()
    device = torch.device("cuda")

    edge_features = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

    results = {}
    for name in ["sum", "mean", "max"]:
        clear_memory()
        pool = GraphPooling(reduce=name)

        def pool_step():
            return pool(edge_features, dst_indices, num_nodes)

        results[name] = run_timed_iterations(pool_step)

    return results


# =============================================================================
# EquivariantEdgeAttention
# =============================================================================


def benchmark_edge_attention(
    num_nodes: int,
    num_edges: int,
    mult: int,
    lmax: int,
    num_heads: int,
    dtype: torch.dtype,
    use_amp: bool = False,
) -> BenchmarkResult:
    """Benchmark EquivariantEdgeAttention forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    repr = Repr(lvals=lvals, mult=mult)
    dim = repr.dim()

    attn = EquivariantEdgeAttention(repr, num_heads=num_heads, dropout=0.0).to(device).to(dtype)
    optimizer = torch.optim.Adam(attn.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    edge_features = torch.randn(
        num_edges, mult, dim, device=device, dtype=dtype, requires_grad=True
    )
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    target = torch.randn(num_edges, mult, dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = attn(edge_features, dst_indices, num_nodes)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return run_timed_iterations(train_step)


# =============================================================================
# S2Activation
# =============================================================================


def benchmark_s2_activation(
    lmax: int,
    mult: int,
    batch_size: int,
    dtype: torch.dtype,
    use_amp: bool = False,
    precision: int = 47,
) -> BenchmarkResult:
    """Benchmark S2Activation forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    repr = Repr(lvals=lvals, mult=mult)
    dim = repr.dim()

    act = S2Activation(repr, precision=precision).to(device).to(dtype)

    optimizer = torch.optim.Adam(act.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    features = torch.randn(
        batch_size, mult, dim, device=device, dtype=dtype, requires_grad=True
    )
    target = torch.randn(batch_size, mult, dim, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = act(features)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    result = run_timed_iterations(train_step)
    result.extra = {
        "n_points": act.n_points,
        "precision": act.precision,
    }
    return result


# =============================================================================
# EquivariantTransformerBlock
# =============================================================================


def benchmark_transformer_block(
    in_lvals: list[int],
    out_lvals: list[int],
    in_mult: int,
    out_mult: int,
    num_nodes: int,
    num_edges: int,
    num_heads: int,
    dtype: torch.dtype,
    use_amp: bool = False,
) -> BenchmarkResult:
    """Benchmark EquivariantTransformerBlock forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    in_repr = Repr(lvals=in_lvals, mult=in_mult)
    out_repr = Repr(lvals=out_lvals, mult=out_mult)

    block = EquivariantTransformerBlock(
        in_repr, out_repr, num_heads=num_heads, dropout=0.0
    ).to(device).to(dtype)
    basis = WignerDBasis([in_repr, out_repr]).to(device)

    optimizer = torch.optim.Adam(block.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    node_features = torch.randn(
        num_nodes, in_mult, in_repr.dim(), device=device, dtype=dtype, requires_grad=True
    )
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
    target = torch.randn(num_nodes, out_mult, out_repr.dim(), device=device, dtype=dtype)

    P, Q = basis(directions)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = block(P, Q, node_features, distances, src_indices, dst_indices, num_nodes)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return run_timed_iterations(train_step)


# =============================================================================
# EquivariantTransformer (Full Stack)
# =============================================================================


def benchmark_transformer(
    in_lvals: list[int],
    hidden_lvals: list[int],
    out_lvals: list[int],
    in_mult: int,
    hidden_mult: int,
    out_mult: int,
    num_layers: int,
    num_nodes: int,
    num_edges: int,
    num_heads: int,
    dtype: torch.dtype,
    use_amp: bool = False,
) -> BenchmarkResult:
    """Benchmark EquivariantTransformer forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    in_repr = Repr(lvals=in_lvals, mult=in_mult)
    hidden_repr = Repr(lvals=hidden_lvals, mult=hidden_mult)
    out_repr = Repr(lvals=out_lvals, mult=out_mult)

    model = EquivariantTransformer(
        in_repr, hidden_repr, out_repr,
        num_layers=num_layers, num_heads=num_heads, dropout=0.0
    ).to(device).to(dtype)
    basis = WignerDBasis(model.get_basis_reprs()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    node_features = torch.randn(
        num_nodes, in_mult, in_repr.dim(), device=device, dtype=dtype, requires_grad=True
    )
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
    target = torch.randn(num_nodes, out_mult, out_repr.dim(), device=device, dtype=dtype)

    matrices = basis(directions)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(matrices, node_features, distances, src_indices, dst_indices, num_nodes)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    result = run_timed_iterations(train_step)
    result.extra = {
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_layers": num_layers,
    }
    return result


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_layer_comparison(dtype: torch.dtype, use_amp: bool, num_bins: int = 100):
    """Compare SE3-Transformer vs Flash-eq across configurations."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"SE3-Transformer vs Flash-eq ({precision})")

    configs = [
        # (lmax, num_nodes, num_edges, cin, cout)
        (1, 3000, 32000, 32, 32),
        (2, 3000, 32000, 32, 32),
        (4, 1000, 5000, 32, 32),
        (6, 1000, 5000, 32, 32),
        (4, 2000, 20000, 32, 32),
        (6, 2000, 20000, 32, 32),
        (4, 5000, 50000, 32, 32),
        (6, 5000, 50000, 32, 32),
        (4, 5000, 128000, 32, 32),
        (6, 5000, 128000, 32, 32),
    ]

    cols = [("Config", 40), ("SE3-Transformer", 25), ("Flash-eq", 25)]
    print_table_header(cols)

    results = []
    for lmax, num_nodes, num_edges, cin, cout in configs:
        config_str = f"L={lmax}, N={num_nodes}, E={num_edges}, C={cin}"

        try:
            r_se3 = benchmark_se3_transformer(lmax, num_edges, cin, cout, dtype, use_amp)
        except torch.cuda.OutOfMemoryError:
            r_se3 = None
        clear_memory()

        try:
            r_flash = benchmark_flash_eq(
                lmax, num_nodes, num_edges, cin, cout, num_bins, dtype, use_amp
            )
        except torch.cuda.OutOfMemoryError:
            r_flash = None
        clear_memory()

        print_table_row([
            (config_str, 40),
            (format_result(r_se3), 25),
            (format_result(r_flash), 25),
        ])
        results.append((config_str, r_se3, r_flash))

    # Summary with ratios
    print(f"\nSpeedup Summary ({precision}):")
    cols = [("Config", 40), ("Memory Savings", 18), ("Speedup", 18)]
    print_table_header(cols)

    for config_str, r_se3, r_flash in results:
        if r_se3 and r_flash:
            mem = f"{r_se3.peak_mem_mb / r_flash.peak_mem_mb:.1f}x"
            speed = f"{r_se3.time_ms / r_flash.time_ms:.2f}x"
        elif r_flash and not r_se3:
            mem = speed = "SE3 OOM"
        else:
            mem = speed = "N/A"
        print_table_row([(config_str, 40), (mem, 18), (speed, 18)])

    return results


def run_basis_benchmark(dtype: torch.dtype):
    """Benchmark WignerDBasis P/Q computation."""
    dtype_name = "FP32" if dtype == torch.float32 else "FP16"
    print_header(f"WignerDBasis P/Q Computation ({dtype_name})")

    configs = [
        (2, 1000), (2, 5000), (2, 10000), (2, 50000), (2, 100000),
        (1, 50000), (2, 50000), (3, 50000), (4, 50000), (6, 50000),
        (4, 100000), (6, 100000), (4, 200000), (6, 200000),
    ]

    cols = [("Config", 25), ("P/Q Shape", 25), ("Time (ms)", 15), ("Memory (MB)", 15)]
    print_table_header(cols)

    results = []
    for lmax, num_edges in configs:
        config_str = f"L={lmax}, E={num_edges}"

        try:
            r = benchmark_basis_computation(lmax, num_edges, dtype)
            dim = r.extra["dim"]
            shape_str = f"({num_edges}, {dim}, {dim})"
            time_str = f"{r.time_ms:.2f}"
            mem_str = f"{r.extra['P_mem_mb'] + r.extra['Q_mem_mb']:.1f}"
            results.append((config_str, r, num_edges))
        except torch.cuda.OutOfMemoryError:
            shape_str = time_str = mem_str = "OOM"
            results.append((config_str, None, num_edges))
        clear_memory()

        print_table_row([
            (config_str, 25), (shape_str, 25), (time_str, 15), (mem_str, 15)
        ])

    # Scaling analysis
    print("\nScaling Analysis:")
    cols = [("Config", 25), ("Bytes/Edge", 15), ("ms/1K Edges", 15)]
    print_table_header(cols)

    for config_str, r, num_edges in results:
        if r:
            total_mem = r.extra["P_mem_mb"] + r.extra["Q_mem_mb"]
            bytes_per_edge = total_mem * 1024**2 / num_edges
            ms_per_1k = r.time_ms / (num_edges / 1000)
            print_table_row([
                (config_str, 25), (f"{bytes_per_edge:.0f}", 15), (f"{ms_per_1k:.3f}", 15)
            ])

    return results


def run_pooling_benchmark(dtype: torch.dtype):
    """Benchmark GraphPooling operations."""
    dtype_name = "FP32" if dtype == torch.float32 else "FP16"
    print_header(f"GraphPooling ({dtype_name})")

    configs = [
        (1000, 5000, 32, 9), (1000, 10000, 32, 9),
        (1000, 50000, 32, 9), (1000, 100000, 32, 9),
        (100, 50000, 32, 9), (1000, 50000, 32, 9), (10000, 50000, 32, 9),
        (1000, 50000, 16, 9), (1000, 50000, 32, 9), (1000, 50000, 64, 9),
        (1000, 50000, 32, 4), (1000, 50000, 32, 9),
        (1000, 50000, 32, 16), (1000, 50000, 32, 49),
        (5000, 200000, 32, 9), (10000, 500000, 32, 9),
    ]

    cols = [("Config", 38), ("Sum (ms)", 12), ("Mean (ms)", 12), ("Max (ms)", 12)]
    print_table_header(cols)

    results = []
    for num_nodes, num_edges, channels, dim in configs:
        config_str = f"N={num_nodes}, E={num_edges}, C={channels}, D={dim}"

        try:
            r = benchmark_graph_pooling(num_nodes, num_edges, channels, dim, dtype)
            sum_str = f"{r['sum'].time_ms:.3f}"
            mean_str = f"{r['mean'].time_ms:.3f}"
            max_str = f"{r['max'].time_ms:.3f}"
            results.append((config_str, r, num_edges, channels, dim))
        except torch.cuda.OutOfMemoryError:
            sum_str = mean_str = max_str = "OOM"
            results.append((config_str, None, num_edges, channels, dim))
        clear_memory()

        print_table_row([
            (config_str, 38), (sum_str, 12), (mean_str, 12), (max_str, 12)
        ])

    # Throughput analysis
    print("\nThroughput Analysis (sum):")
    cols = [("Config", 38), ("Edges/ms", 15), ("GB/s", 12)]
    print_table_header(cols)

    for config_str, r, num_edges, channels, dim in results:
        if r:
            edges_per_ms = num_edges / r["sum"].time_ms
            bytes_per_edge = channels * dim * (4 if dtype == torch.float32 else 2)
            gb_s = (num_edges * bytes_per_edge / 1e9) / (r["sum"].time_ms / 1000)
            print_table_row([
                (config_str, 38), (f"{edges_per_ms:.0f}", 15), (f"{gb_s:.1f}", 12)
            ])

    return results


def run_attention_benchmark(dtype: torch.dtype, use_amp: bool = False):
    """Benchmark EquivariantEdgeAttention."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"EquivariantEdgeAttention ({precision})")

    configs = [
        (1000, 5000, 32, 2, 4), (1000, 10000, 32, 2, 4),
        (1000, 50000, 32, 2, 4), (1000, 100000, 32, 2, 4),
        (1000, 50000, 32, 2, 1), (1000, 50000, 32, 2, 4),
        (1000, 50000, 32, 2, 8), (1000, 50000, 32, 2, 16),
        (1000, 50000, 16, 2, 4), (1000, 50000, 32, 2, 4),
        (1000, 50000, 64, 2, 8), (1000, 50000, 128, 2, 8),
        (1000, 50000, 32, 1, 4), (1000, 50000, 32, 2, 4),
        (1000, 50000, 32, 4, 4), (1000, 50000, 32, 6, 4),
        (5000, 200000, 32, 2, 4), (5000, 200000, 64, 4, 8),
        (10000, 500000, 32, 2, 4),
    ]

    cols = [("Config", 52), ("Time (ms)", 15), ("Memory (MB)", 15)]
    print_table_header(cols)

    results = []
    for num_nodes, num_edges, mult, lmax, num_heads in configs:
        config_str = f"N={num_nodes}, E={num_edges}, M={mult}, L={lmax}, H={num_heads}"

        try:
            r = benchmark_edge_attention(
                num_nodes, num_edges, mult, lmax, num_heads, dtype, use_amp
            )
            time_str = f"{r.time_ms:.2f}"
            mem_str = f"{r.peak_mem_mb:.0f}"
            results.append((config_str, r, num_edges, mult, lmax))
        except torch.cuda.OutOfMemoryError:
            time_str = mem_str = "OOM"
            results.append((config_str, None, num_edges, mult, lmax))
        clear_memory()

        print_table_row([(config_str, 52), (time_str, 15), (mem_str, 15)])

    # Throughput analysis
    print("\nThroughput Analysis:")
    cols = [("Config", 52), ("Edges/ms", 15), ("GB/s", 12)]
    print_table_header(cols)

    for config_str, r, num_edges, mult, lmax in results:
        if r:
            dim = sum(2 * l + 1 for l in range(lmax + 1))
            edges_per_ms = num_edges / r.time_ms
            bytes_per_edge = mult * dim * (4 if dtype == torch.float32 else 2)
            gb_s = (num_edges * bytes_per_edge / 1e9) / (r.time_ms / 1000)
            print_table_row([
                (config_str, 52), (f"{edges_per_ms:.0f}", 15), (f"{gb_s:.1f}", 12)
            ])

    return results


def run_s2_activation_benchmark(dtype: torch.dtype, use_amp: bool = False):
    """Benchmark S2Activation with Lebedev quadrature."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"S2Activation ({precision})")

    # Configurations: (lmax, mult, batch_size)
    configs = [
        # Vary batch size
        (2, 32, 100), (2, 32, 1000), (2, 32, 10000), (2, 32, 50000),
        # Vary lmax
        (2, 64, 10000), (4, 64, 10000), (6, 64, 10000),
        # Vary mult (channels)
        (4, 32, 10000), (4, 64, 10000), (4, 128, 10000),
        # EquiformerV2-like configs
        (6, 64, 5000), (6, 128, 5000), (6, 64, 20000),
    ]

    cols = [("Config", 30), ("Time", 18), ("Points", 12), ("Samples/ms", 15)]
    print_table_header(cols)

    results = []
    for lmax, mult, batch_size in configs:
        config_str = f"L={lmax}, M={mult}, B={batch_size}"

        try:
            r = benchmark_s2_activation(lmax, mult, batch_size, dtype, use_amp=use_amp)
            time_str = f"{r.time_ms:.2f}ms"
            n_points = r.extra['n_points']
            samples_per_ms = batch_size / r.time_ms
        except torch.cuda.OutOfMemoryError:
            r = None
            time_str = "OOM"
            n_points = "N/A"
            samples_per_ms = 0
        clear_memory()

        print_table_row([
            (config_str, 30), (time_str, 18), (str(n_points), 12),
            (f"{samples_per_ms:.0f}" if r else "N/A", 15)
        ])
        results.append((config_str, r))

    # Throughput analysis
    print("\nThroughput Analysis:")
    cols = [("Config", 30), ("GFLOP/s", 12)]
    print_table_header(cols)

    for config_str, r in results:
        if r:
            # Parse config
            parts = config_str.split(", ")
            lmax = int(parts[0].split("=")[1])
            mult = int(parts[1].split("=")[1])
            batch = int(parts[2].split("=")[1])
            dim = (lmax + 1) ** 2
            n_points = r.extra["n_points"]

            # FLOPs: 2 matmuls (batch, mult, dim) @ (dim, n_points) + MLP
            flops_matmul = 2 * batch * mult * dim * n_points * 2  # 2 matmuls, 2 ops per multiply-add
            flops_mlp = batch * n_points * mult * mult * 4 * 2  # hidden=2*mult, 2 layers
            gflops = (flops_matmul + flops_mlp) / 1e9 / (r.time_ms / 1000)
            print_table_row([
                (config_str, 30), (f"{gflops:.1f}", 12)
            ])

    return results


def run_transformer_block_benchmark(dtype: torch.dtype, use_amp: bool = False):
    """Benchmark EquivariantTransformerBlock."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"EquivariantTransformerBlock ({precision})")

    # Configurations: (in_lvals, out_lvals, in_mult, out_mult, num_nodes, num_edges, num_heads)
    configs = [
        # Same repr (with residual)
        ([0, 1, 2], [0, 1, 2], 32, 32, 1000, 10000, 4),
        ([0, 1, 2], [0, 1, 2], 64, 64, 1000, 10000, 8),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 32, 32, 1000, 10000, 4),
        # Different repr (no residual)
        ([0, 1], [0, 1, 2], 32, 64, 1000, 10000, 8),
        ([0, 1, 2], [0], 64, 32, 1000, 10000, 4),
        # Scaling with edges
        ([0, 1, 2], [0, 1, 2], 32, 32, 1000, 5000, 4),
        ([0, 1, 2], [0, 1, 2], 32, 32, 1000, 20000, 4),
        ([0, 1, 2], [0, 1, 2], 32, 32, 1000, 50000, 4),
        # Scaling with nodes
        ([0, 1, 2], [0, 1, 2], 32, 32, 500, 10000, 4),
        ([0, 1, 2], [0, 1, 2], 32, 32, 2000, 10000, 4),
        ([0, 1, 2], [0, 1, 2], 32, 32, 5000, 10000, 4),
    ]

    cols = [("Config", 55), ("Time (ms)", 15), ("Memory (MB)", 15)]
    print_table_header(cols)

    results = []
    for in_lvals, out_lvals, in_mult, out_mult, num_nodes, num_edges, num_heads in configs:
        in_str = "".join(str(l) for l in in_lvals)
        out_str = "".join(str(l) for l in out_lvals)
        config_str = f"L:{in_str}->{out_str}, M:{in_mult}->{out_mult}, N={num_nodes}, E={num_edges}"

        try:
            r = benchmark_transformer_block(
                in_lvals, out_lvals, in_mult, out_mult,
                num_nodes, num_edges, num_heads, dtype, use_amp
            )
            time_str = f"{r.time_ms:.2f}"
            mem_str = f"{r.peak_mem_mb:.0f}"
            results.append((config_str, r))
        except torch.cuda.OutOfMemoryError:
            time_str = mem_str = "OOM"
            results.append((config_str, None))
        clear_memory()

        print_table_row([(config_str, 55), (time_str, 15), (mem_str, 15)])

    return results


def run_transformer_benchmark(dtype: torch.dtype, use_amp: bool = False):
    """Benchmark EquivariantTransformer (full stack)."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"EquivariantTransformer ({precision})")

    # Configurations: (in_lvals, hidden_lvals, out_lvals, in_mult, hidden_mult, out_mult,
    #                  num_layers, num_nodes, num_edges, num_heads)
    configs = [
        # Small model
        ([0, 1], [0, 1, 2], [0], 16, 32, 8, 2, 500, 5000, 4),
        ([0, 1], [0, 1, 2], [0], 16, 32, 8, 4, 500, 5000, 4),
        # Medium model
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 4, 1000, 10000, 8),
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 6, 1000, 10000, 8),
        # Larger hidden lmax
        ([0, 1], [0, 1, 2, 3, 4], [0], 32, 64, 16, 4, 1000, 10000, 8),
        # Scaling with layers
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 2, 1000, 10000, 8),
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 4, 1000, 10000, 8),
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 8, 1000, 10000, 8),
        # Scaling with graph size
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 4, 500, 5000, 8),
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 4, 1000, 10000, 8),
        ([0, 1], [0, 1, 2], [0], 32, 64, 16, 4, 2000, 20000, 8),
    ]

    cols = [("Config", 70), ("Time (ms)", 12), ("Mem (MB)", 12), ("Params", 12)]
    print_table_header(cols)

    results = []
    for (in_lvals, hidden_lvals, out_lvals, in_mult, hidden_mult, out_mult,
         num_layers, num_nodes, num_edges, num_heads) in configs:

        in_str = "".join(str(l) for l in in_lvals)
        hid_str = "".join(str(l) for l in hidden_lvals)
        out_str = "".join(str(l) for l in out_lvals)
        config_str = (f"L:{in_str}->{hid_str}->{out_str}, M:{in_mult}/{hidden_mult}/{out_mult}, "
                      f"D={num_layers}, N={num_nodes}, E={num_edges}")

        try:
            r = benchmark_transformer(
                in_lvals, hidden_lvals, out_lvals,
                in_mult, hidden_mult, out_mult,
                num_layers, num_nodes, num_edges, num_heads, dtype, use_amp
            )
            time_str = f"{r.time_ms:.1f}"
            mem_str = f"{r.peak_mem_mb:.0f}"
            params_str = f"{r.extra['num_params'] / 1e6:.2f}M"
            results.append((config_str, r))
        except torch.cuda.OutOfMemoryError:
            time_str = mem_str = params_str = "OOM"
            results.append((config_str, None))
        clear_memory()

        print_table_row([(config_str, 70), (time_str, 12), (mem_str, 12), (params_str, 12)])

    # Scaling analysis
    print("\nScaling Analysis:")
    cols = [("Config", 70), ("ms/layer", 12), ("ms/1K edges", 12)]
    print_table_header(cols)

    for config_str, r in results:
        if r:
            ms_per_layer = r.time_ms / r.extra["num_layers"]
            # Parse edges from config string
            e_part = [p for p in config_str.split(", ") if p.startswith("E=")][0]
            num_edges = int(e_part.split("=")[1])
            ms_per_1k = r.time_ms / (num_edges / 1000)
            print_table_row([
                (config_str, 70), (f"{ms_per_layer:.2f}", 12), (f"{ms_per_1k:.3f}", 12)
            ])

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    print_header("Flash-eq Benchmark Suite")
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Layer comparison
    run_layer_comparison(torch.float32, use_amp=False)
    run_layer_comparison(torch.float32, use_amp=True)

    # Component benchmarks
    run_basis_benchmark(torch.float32)
    run_pooling_benchmark(torch.float32)
    run_attention_benchmark(torch.float32)
    run_attention_benchmark(torch.float32, use_amp=True)

    # S2Activation benchmarks
    run_s2_activation_benchmark(torch.float32)
    run_s2_activation_benchmark(torch.float32, use_amp=True)

    # Transformer benchmarks
    run_transformer_block_benchmark(torch.float32)
    run_transformer_block_benchmark(torch.float32, use_amp=True)
    run_transformer_benchmark(torch.float32)
    run_transformer_benchmark(torch.float32, use_amp=True)


if __name__ == "__main__":
    main()
