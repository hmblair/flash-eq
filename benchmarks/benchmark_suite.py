"""Benchmark suite for Flash-eq components.

Systematically profiles all components using standardized scenarios for
consistent comparison. Measures runtime (ms) and peak memory (MB) for
forward + backward passes.

Components benchmarked:
1. WignerDBasis - P/Q matrix computation
2. EquivariantEdgewiseLinear - edge-wise linear transformation
3. GraphPooling - edge to node aggregation
4. EquivariantEdgeAttention - attention mechanism
5. S2Activation - spherical activation function
6. EquivariantTransformerBlock - single transformer block
7. EquivariantTransformer - full model with component breakdown

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from flash_eq import (
    EquivariantEdgeAttention,
    EquivariantEdgewiseLinear,
    EquivariantTransformer,
    EquivariantTransformerBlock,
    GraphPooling,
    Repr,
    S2Activation,
    WignerDBasis,
)

# =============================================================================
# Standardized Test Scenarios
# =============================================================================


@dataclass
class Scenario:
    """A standardized test scenario for benchmarking."""

    name: str
    num_nodes: int
    num_edges: int
    lvals: list[int]
    mult: int
    num_heads: int = 4

    @property
    def repr(self) -> Repr:
        return Repr(lvals=self.lvals, mult=self.mult)

    @property
    def dim(self) -> int:
        return sum(2 * l + 1 for l in self.lvals)

    @property
    def lmax(self) -> int:
        return max(self.lvals)

    def __str__(self) -> str:
        l_str = "".join(str(l) for l in self.lvals)
        return f"N={self.num_nodes}, E={self.num_edges}, L={l_str}, M={self.mult}"


# Standard scenarios covering different scales and representations
SCENARIOS = [
    # Small graphs
    Scenario("small-low", 500, 5_000, [0, 1], 32),
    Scenario("small-mid", 500, 5_000, [0, 1, 2], 32),
    Scenario("small-high", 500, 5_000, [0, 1, 2, 3, 4], 32),
    # Medium graphs
    Scenario("med-low", 1000, 20_000, [0, 1], 32),
    Scenario("med-mid", 1000, 20_000, [0, 1, 2], 32),
    Scenario("med-high", 1000, 20_000, [0, 1, 2, 3, 4], 32),
    # Large graphs
    Scenario("large-low", 2000, 50_000, [0, 1], 32),
    Scenario("large-mid", 2000, 50_000, [0, 1, 2], 32),
    Scenario("large-high", 2000, 50_000, [0, 1, 2, 3, 4], 32),
]

# Subset for quick benchmarks
QUICK_SCENARIOS = [s for s in SCENARIOS if s.name.startswith("med")]


# =============================================================================
# Benchmark Infrastructure
# =============================================================================


@dataclass
class BenchmarkResult:
    """Stores benchmark results."""

    time_ms: float
    peak_mem_mb: float
    extra: dict = field(default_factory=dict)


def clear_memory():
    """Clear CUDA memory and reset peak stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@contextmanager
def cuda_timer():
    """Context manager for CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    elapsed = []
    yield lambda: elapsed[0] if elapsed else None
    end.record()
    torch.cuda.synchronize()
    elapsed.append(start.elapsed_time(end))


def benchmark(fn: Callable, n_warmup: int = 3, n_iter: int = 10) -> BenchmarkResult:
    """Run warmup iterations, then benchmark and return results."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    clear_memory()

    with cuda_timer() as get_elapsed:
        for _ in range(n_iter):
            fn()

    return BenchmarkResult(
        time_ms=get_elapsed() / n_iter,
        peak_mem_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )


# =============================================================================
# Formatting Utilities
# =============================================================================


def print_header(title: str, width: int = 90):
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 90):
    """Print a subsection header."""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print("-" * width)


def fmt_time(ms: float) -> str:
    """Format time in milliseconds."""
    if ms < 1:
        return f"{ms * 1000:.1f}us"
    return f"{ms:.2f}ms"


def fmt_mem(mb: float) -> str:
    """Format memory in megabytes."""
    if mb >= 1000:
        return f"{mb / 1024:.2f}GB"
    return f"{mb:.0f}MB"


def fmt_throughput(count: int, ms: float, unit: str = "edges") -> str:
    """Format throughput."""
    per_ms = count / ms
    if per_ms >= 1e6:
        return f"{per_ms / 1e6:.2f}M {unit}/ms"
    if per_ms >= 1e3:
        return f"{per_ms / 1e3:.1f}K {unit}/ms"
    return f"{per_ms:.0f} {unit}/ms"


# =============================================================================
# Component Benchmarks
# =============================================================================


def benchmark_wigner_basis(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult | None:
    """Benchmark WignerDBasis P/Q matrix computation."""
    clear_memory()
    device = torch.device("cuda")

    repr_obj = Repr(lvals=scenario.lvals, mult=1)
    basis = WignerDBasis([repr_obj, repr_obj]).to(device)

    directions = torch.randn(scenario.num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    try:
        result = benchmark(lambda: basis(directions))
        P, Q = basis(directions)
        result.extra = {
            "P_shape": tuple(P.shape),
            "Q_shape": tuple(Q.shape),
            "tensor_mb": (P.numel() + Q.numel()) * P.element_size() / 1024**2,
        }
        return result
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_edgewise_linear(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
    num_bins: int = 64,
) -> BenchmarkResult | None:
    """Benchmark EquivariantEdgewiseLinear forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    in_repr = scenario.repr
    out_repr = scenario.repr

    layer = (
        EquivariantEdgewiseLinear(
            in_repr, out_repr, num_bins=num_bins, min_dist=0.0, max_dist=10.0
        )
        .to(device)
        .to(dtype)
    )
    basis = WignerDBasis([in_repr, out_repr]).to(device)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    edge_features = torch.randn(
        scenario.num_edges,
        scenario.mult,
        scenario.dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    directions = torch.randn(scenario.num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(scenario.num_edges, device=device, dtype=dtype) * 10.0
    target = torch.randn(
        scenario.num_edges, scenario.mult, scenario.dim, device=device, dtype=dtype
    )

    P, Q = basis(directions)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = layer(P, Q, edge_features, distances)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    try:
        result = benchmark(train_step)
        result.extra = {"num_params": sum(p.numel() for p in layer.parameters())}
        return result
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_graph_pooling(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    reduce: str = "sum",
) -> BenchmarkResult | None:
    """Benchmark GraphPooling operation."""
    clear_memory()
    device = torch.device("cuda")

    pool = GraphPooling(reduce=reduce)

    edge_features = torch.randn(
        scenario.num_edges, scenario.mult, scenario.dim, device=device, dtype=dtype
    )
    dst_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)

    try:
        return benchmark(lambda: pool(edge_features, dst_indices, scenario.num_nodes))
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_edge_attention(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
) -> BenchmarkResult | None:
    """Benchmark EquivariantEdgeAttention forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    attn = (
        EquivariantEdgeAttention(scenario.repr, num_heads=scenario.num_heads, dropout=0.0)
        .to(device)
        .to(dtype)
    )
    optimizer = torch.optim.Adam(attn.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    edge_features = torch.randn(
        scenario.num_edges,
        scenario.mult,
        scenario.dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dst_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)
    target = torch.randn(
        scenario.num_edges, scenario.mult, scenario.dim, device=device, dtype=dtype
    )

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = attn(edge_features, dst_indices, scenario.num_nodes)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    try:
        result = benchmark(train_step)
        result.extra = {"num_params": sum(p.numel() for p in attn.parameters())}
        return result
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_s2_activation(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
    precision: int = 47,
) -> BenchmarkResult | None:
    """Benchmark S2Activation forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    act = S2Activation(scenario.repr, precision=precision).to(device).to(dtype)
    optimizer = torch.optim.Adam(act.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # S2Activation operates on node features, not edge features
    features = torch.randn(
        scenario.num_nodes,
        scenario.mult,
        scenario.dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.randn(
        scenario.num_nodes, scenario.mult, scenario.dim, device=device, dtype=dtype
    )

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = act(features)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    try:
        result = benchmark(train_step)
        result.extra = {
            "n_points": act.n_points,
            "num_params": sum(p.numel() for p in act.parameters()),
        }
        return result
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_transformer_block(
    scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
) -> BenchmarkResult | None:
    """Benchmark EquivariantTransformerBlock forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    in_repr = scenario.repr
    out_repr = scenario.repr

    block = (
        EquivariantTransformerBlock(
            in_repr, out_repr, num_heads=scenario.num_heads, dropout=0.0
        )
        .to(device)
        .to(dtype)
    )
    basis = WignerDBasis([in_repr, out_repr]).to(device)

    optimizer = torch.optim.Adam(block.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    node_features = torch.randn(
        scenario.num_nodes,
        scenario.mult,
        scenario.dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    src_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)
    dst_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)
    directions = torch.randn(scenario.num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(scenario.num_edges, device=device, dtype=dtype) * 10.0
    target = torch.randn(
        scenario.num_nodes, scenario.mult, scenario.dim, device=device, dtype=dtype
    )

    P, Q = basis(directions)

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = block(
                P, Q, node_features, distances, src_indices, dst_indices, scenario.num_nodes
            )
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    try:
        result = benchmark(train_step)
        result.extra = {"num_params": sum(p.numel() for p in block.parameters())}
        return result
    except torch.cuda.OutOfMemoryError:
        return None


def benchmark_transformer(
    scenario: Scenario,
    num_layers: int = 4,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
) -> BenchmarkResult | None:
    """Benchmark EquivariantTransformer forward + backward."""
    clear_memory()
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1], mult=scenario.mult)
    hidden_repr = scenario.repr
    out_repr = Repr(lvals=[0], mult=scenario.mult)

    model = (
        EquivariantTransformer(
            in_repr,
            hidden_repr,
            out_repr,
            num_layers=num_layers,
            num_heads=scenario.num_heads,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Generate random coordinates for nodes
    coordinates = torch.randn(scenario.num_nodes, 3, device=device, dtype=dtype) * 5.0
    node_features = torch.randn(
        scenario.num_nodes,
        scenario.mult,
        in_repr.dim(),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    src_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)
    dst_indices = torch.randint(0, scenario.num_nodes, (scenario.num_edges,), device=device)
    target = torch.randn(
        scenario.num_nodes, scenario.mult, out_repr.dim(), device=device, dtype=dtype
    )

    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(coordinates, node_features, src_indices, dst_indices)
            loss = ((output - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    try:
        result = benchmark(train_step)
        result.extra = {
            "num_params": sum(p.numel() for p in model.parameters()),
            "num_layers": num_layers,
        }
        return result
    except torch.cuda.OutOfMemoryError:
        return None


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_component_benchmarks(
    scenarios: list[Scenario],
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
):
    """Run benchmarks for all components across scenarios."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"Component Benchmarks ({precision})")

    # Column widths
    _name_w, scenario_w, time_w, mem_w, tput_w = 22, 32, 12, 10, 18

    # Results storage for summary
    all_results: dict[str, dict[str, BenchmarkResult | None]] = {}

    # 1. WignerDBasis
    print_subheader("WignerDBasis (P/Q computation)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["WignerDBasis"] = {}
    for s in scenarios:
        r = benchmark_wigner_basis(s, dtype)
        all_results["WignerDBasis"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {fmt_throughput(s.num_edges, r.time_ms):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    # 2. EquivariantEdgewiseLinear
    print_subheader("EquivariantEdgewiseLinear (fwd + bwd)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["EdgewiseLinear"] = {}
    for s in scenarios:
        r = benchmark_edgewise_linear(s, dtype, use_amp)
        all_results["EdgewiseLinear"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {fmt_throughput(s.num_edges, r.time_ms):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    # 3. GraphPooling
    print_subheader("GraphPooling (sum)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["GraphPooling"] = {}
    for s in scenarios:
        r = benchmark_graph_pooling(s, dtype, "sum")
        all_results["GraphPooling"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {fmt_throughput(s.num_edges, r.time_ms):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    # 4. EquivariantEdgeAttention
    print_subheader("EquivariantEdgeAttention (fwd + bwd)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["EdgeAttention"] = {}
    for s in scenarios:
        r = benchmark_edge_attention(s, dtype, use_amp)
        all_results["EdgeAttention"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {fmt_throughput(s.num_edges, r.time_ms):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    # 5. S2Activation
    print_subheader("S2Activation (fwd + bwd)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["S2Activation"] = {}
    for s in scenarios:
        r = benchmark_s2_activation(s, dtype, use_amp)
        all_results["S2Activation"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} "
                f"{fmt_throughput(s.num_nodes, r.time_ms, 'nodes'):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    # 6. TransformerBlock
    print_subheader("EquivariantTransformerBlock (fwd + bwd)")
    print(
        f"{'Scenario':<{scenario_w}} {'Time':<{time_w}} {'Memory':<{mem_w}} {'Throughput':<{tput_w}}"
    )
    all_results["TransformerBlock"] = {}
    for s in scenarios:
        r = benchmark_transformer_block(s, dtype, use_amp)
        all_results["TransformerBlock"][s.name] = r
        if r:
            print(
                f"{str(s):<{scenario_w}} {fmt_time(r.time_ms):<{time_w}} "
                f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {fmt_throughput(s.num_edges, r.time_ms):<{tput_w}}"
            )
        else:
            print(f"{str(s):<{scenario_w}} {'OOM':<{time_w}}")
        clear_memory()

    return all_results


def run_transformer_benchmarks(
    scenarios: list[Scenario],
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
):
    """Benchmark full EquivariantTransformer with varying layers."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"EquivariantTransformer Benchmarks ({precision})")

    scenario_w, layers_w, time_w, mem_w, params_w, tput_w = 32, 8, 12, 10, 12, 15

    print(
        f"{'Scenario':<{scenario_w}} {'Layers':<{layers_w}} {'Time':<{time_w}} "
        f"{'Memory':<{mem_w}} {'Params':<{params_w}} {'ms/layer':<{tput_w}}"
    )
    print("-" * 90)

    results = []
    for s in scenarios:
        for num_layers in [2, 4, 6]:
            r = benchmark_transformer(s, num_layers, dtype, use_amp)
            results.append((s, num_layers, r))
            if r:
                ms_per_layer = r.time_ms / num_layers
                params_str = f"{r.extra['num_params'] / 1e6:.2f}M"
                print(
                    f"{str(s):<{scenario_w}} {num_layers:<{layers_w}} {fmt_time(r.time_ms):<{time_w}} "
                    f"{fmt_mem(r.peak_mem_mb):<{mem_w}} {params_str:<{params_w}} {fmt_time(ms_per_layer):<{tput_w}}"
                )
            else:
                print(f"{str(s):<{scenario_w}} {num_layers:<{layers_w}} {'OOM':<{time_w}}")
            clear_memory()

    return results


def run_scaling_analysis(
    base_scenario: Scenario,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
):
    """Analyze scaling behavior with varying graph sizes."""
    precision = "AMP" if use_amp else ("FP16" if dtype == torch.float16 else "FP32")
    print_header(f"Scaling Analysis ({precision})")

    # Edge scaling (fixed nodes)
    print_subheader(f"Edge Scaling (N={base_scenario.num_nodes}, L={''.join(str(l) for l in base_scenario.lvals)})")
    edge_counts = [5_000, 10_000, 20_000, 50_000, 100_000]

    col_w = 15
    print(
        f"{'Edges':<{col_w}} {'Basis':<{col_w}} {'Linear':<{col_w}} "
        f"{'Attention':<{col_w}} {'Block':<{col_w}}"
    )
    print("-" * (col_w * 5))

    for num_edges in edge_counts:
        s = Scenario(
            f"scale-{num_edges}",
            base_scenario.num_nodes,
            num_edges,
            base_scenario.lvals,
            base_scenario.mult,
        )

        r_basis = benchmark_wigner_basis(s, dtype)
        clear_memory()
        r_linear = benchmark_edgewise_linear(s, dtype, use_amp)
        clear_memory()
        r_attn = benchmark_edge_attention(s, dtype, use_amp)
        clear_memory()
        r_block = benchmark_transformer_block(s, dtype, use_amp)
        clear_memory()

        print(
            f"{num_edges:<{col_w}} "
            f"{fmt_time(r_basis.time_ms) if r_basis else 'OOM':<{col_w}} "
            f"{fmt_time(r_linear.time_ms) if r_linear else 'OOM':<{col_w}} "
            f"{fmt_time(r_attn.time_ms) if r_attn else 'OOM':<{col_w}} "
            f"{fmt_time(r_block.time_ms) if r_block else 'OOM':<{col_w}}"
        )

    # L-max scaling (fixed graph size)
    print_subheader(f"L-max Scaling (N={base_scenario.num_nodes}, E={base_scenario.num_edges})")
    lmax_configs = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6],
    ]

    print(
        f"{'Lmax':<{col_w}} {'Basis':<{col_w}} {'Linear':<{col_w}} "
        f"{'Attention':<{col_w}} {'Block':<{col_w}}"
    )
    print("-" * (col_w * 5))

    for lvals in lmax_configs:
        s = Scenario(
            f"lmax-{max(lvals)}",
            base_scenario.num_nodes,
            base_scenario.num_edges,
            lvals,
            base_scenario.mult,
        )

        r_basis = benchmark_wigner_basis(s, dtype)
        clear_memory()
        r_linear = benchmark_edgewise_linear(s, dtype, use_amp)
        clear_memory()
        r_attn = benchmark_edge_attention(s, dtype, use_amp)
        clear_memory()
        r_block = benchmark_transformer_block(s, dtype, use_amp)
        clear_memory()

        lmax_str = f"L={max(lvals)} (dim={s.dim})"
        print(
            f"{lmax_str:<{col_w}} "
            f"{fmt_time(r_basis.time_ms) if r_basis else 'OOM':<{col_w}} "
            f"{fmt_time(r_linear.time_ms) if r_linear else 'OOM':<{col_w}} "
            f"{fmt_time(r_attn.time_ms) if r_attn else 'OOM':<{col_w}} "
            f"{fmt_time(r_block.time_ms) if r_block else 'OOM':<{col_w}}"
        )


def print_summary(all_results: dict[str, dict[str, BenchmarkResult | None]]):
    """Print a summary table comparing all components."""
    print_header("Summary: Component Comparison")

    # Get scenario names from first component
    first_component = next(iter(all_results.values()))
    scenario_names = list(first_component.keys())
    components = list(all_results.keys())

    # Time comparison
    print_subheader("Time (ms) - Forward + Backward")
    col_w = 12
    header = f"{'Scenario':<20}" + "".join(f"{c:<{col_w}}" for c in components)
    print(header)
    print("-" * (20 + col_w * len(components)))

    for scenario_name in scenario_names:
        row = f"{scenario_name:<20}"
        for component in components:
            r = all_results[component].get(scenario_name)
            if r:
                row += f"{r.time_ms:<{col_w}.2f}"
            else:
                row += f"{'OOM':<{col_w}}"
        print(row)

    # Memory comparison
    print_subheader("Peak Memory (MB)")
    print(header)
    print("-" * (20 + col_w * len(components)))

    for scenario_name in scenario_names:
        row = f"{scenario_name:<20}"
        for component in components:
            r = all_results[component].get(scenario_name)
            if r:
                row += f"{r.peak_mem_mb:<{col_w}.0f}"
            else:
                row += f"{'OOM':<{col_w}}"
        print(row)

    # Time breakdown as percentage of TransformerBlock
    if "TransformerBlock" in all_results:
        print_subheader("Time as % of TransformerBlock")
        print(header)
        print("-" * (20 + col_w * len(components)))

        for scenario_name in scenario_names:
            block_r = all_results["TransformerBlock"].get(scenario_name)
            if not block_r:
                continue

            row = f"{scenario_name:<20}"
            for component in components:
                r = all_results[component].get(scenario_name)
                if r:
                    pct = 100 * r.time_ms / block_r.time_ms
                    row += f"{pct:<{col_w}.1f}%"
                else:
                    row += f"{'OOM':<{col_w}}"
            print(row)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the complete benchmark suite."""
    print_header("Flash-eq Benchmark Suite", width=90)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Run component benchmarks
    results_fp32 = run_component_benchmarks(SCENARIOS, torch.float32, use_amp=False)
    print_summary(results_fp32)

    results_amp = run_component_benchmarks(SCENARIOS, torch.float32, use_amp=True)
    print_summary(results_amp)

    # Full transformer benchmarks
    run_transformer_benchmarks(QUICK_SCENARIOS, torch.float32, use_amp=False)
    run_transformer_benchmarks(QUICK_SCENARIOS, torch.float32, use_amp=True)

    # Scaling analysis
    base_scenario = Scenario("base", 1000, 20_000, [0, 1, 2], 32)
    run_scaling_analysis(base_scenario, torch.float32, use_amp=True)


if __name__ == "__main__":
    main()
