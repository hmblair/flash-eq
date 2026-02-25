"""
Benchmark comparison: SE(3)-Transformer vs EquiformerV2 vs Flash-eq.

Measures forward + backward pass runtime (ms) and peak GPU memory (MB)
across varying lmax and edge counts.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

from __future__ import annotations

import gc
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from pathlib import Path

import torch
import torch.nn as nn

# Ensure benchmarks/ and project root are on sys.path.
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir, _this_dir.parent]:
    _benchmarks = _candidate / "benchmarks"
    if _benchmarks.is_dir():
        sys.path.insert(0, str(_benchmarks))
        sys.path.insert(0, str(_candidate))
        break

from se3t_baseline import SE3TBaseline, compute_dim, create_random_basis
from equiformer_v2_baseline import (
    EquiformerV2Baseline,
    create_random_wigner,
    compute_dim as ev2_compute_dim,
)

# ---------------------------------------------------------------------------
# Flash-eq imports (project root already on sys.path above)
# ---------------------------------------------------------------------------
from flash_eq import (
    EquivariantEdgewiseLinear,
    Repr,
    WignerDBasis,
)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@contextmanager
def cuda_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    elapsed = []
    yield lambda: elapsed[0] if elapsed else None
    end.record()
    torch.cuda.synchronize()
    elapsed.append(start.elapsed_time(end))


def benchmark(fn, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    clear()

    with cuda_timer() as get_elapsed:
        for _ in range(n_iter):
            fn()

    return get_elapsed() / n_iter, torch.cuda.max_memory_allocated() / 1024**3


# ---------------------------------------------------------------------------
# Per-method benchmark functions
# ---------------------------------------------------------------------------

def bench_se3t(lmax, channels, num_edges, dtype, use_amp):
    clear()
    device = torch.device("cuda")
    try:
        model = SE3TBaseline(lmax, channels, channels).to(device).to(dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        dim = compute_dim(lmax)
        features = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)
        basis = create_random_basis(num_edges, lmax, device, dtype)
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
        target = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)

        def step():
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(features, basis, distances)
                loss = ((out - target) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return benchmark(step)
    except torch.cuda.OutOfMemoryError:
        clear()
        return None, None


def bench_ev2(lmax, channels, num_edges, dtype, use_amp):
    clear()
    device = torch.device("cuda")
    try:
        model = EquiformerV2Baseline(lmax, channels, channels).to(device).to(dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        dim = ev2_compute_dim(lmax)
        features = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)
        wigner = create_random_wigner(num_edges, lmax, device, dtype)
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
        target = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)

        def step():
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(features, wigner, distances)
                loss = ((out - target) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return benchmark(step)
    except torch.cuda.OutOfMemoryError:
        clear()
        return None, None


def bench_flasheq(lmax, channels, num_edges, dtype, use_amp):
    clear()
    device = torch.device("cuda")
    try:
        lvals = list(range(lmax + 1))
        repr_obj = Repr(lvals=lvals, mult=channels)
        dim = sum(2 * l + 1 for l in lvals)

        layer = (
            EquivariantEdgewiseLinear(repr_obj, repr_obj, num_bins=64, min_dist=0.0, max_dist=10.0)
            .to(device).to(dtype)
        )
        basis = WignerDBasis([repr_obj, repr_obj]).to(device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        features = torch.randn(num_edges, channels, dim, device=device, dtype=dtype, requires_grad=True)
        directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0
        target = torch.randn(num_edges, channels, dim, device=device, dtype=dtype)

        P, Q = basis(directions)

        def step():
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = layer(P, Q, features, distances)
                loss = ((out - target) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return benchmark(step)
    except torch.cuda.OutOfMemoryError:
        clear()
        return None, None


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

@dataclass
class Config:
    lmax: int
    num_edges: int

    @property
    def label(self):
        if self.num_edges >= 1000:
            return f"L={self.lmax}, E={self.num_edges // 1000}k"
        return f"L={self.lmax}, E={self.num_edges}"


CONFIGS = [
    Config(1, 32_000),
    Config(2, 32_000),
    Config(4, 5_000),
    Config(6, 5_000),
    Config(4, 20_000),
    Config(6, 20_000),
    Config(4, 50_000),
    Config(6, 50_000),
    Config(4, 128_000),
    Config(6, 128_000),
]

CHANNELS = 32


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(time_ms, mem_gb):
    if time_ms is None:
        return "OOM"
    return f"{time_ms:.1f}ms / {mem_gb:.1f}GB"


def run_suite(use_amp: bool):
    dtype = torch.float32
    precision = "FP16 (AMP)" if use_amp else "FP32"

    print(f"\n{'=' * 100}")
    print(f" {precision} Comparison — C={CHANNELS}, fwd+bwd, {torch.cuda.get_device_name()}")
    print(f"{'=' * 100}")
    print(f"{'Config':<16} {'SE(3)-Transformer':<24} {'EquiformerV2':<24} {'Flash-eq':<24}")
    print("-" * 100)

    results = []

    for cfg in CONFIGS:
        se3t_time, se3t_mem = bench_se3t(cfg.lmax, CHANNELS, cfg.num_edges, dtype, use_amp)
        ev2_time, ev2_mem = bench_ev2(cfg.lmax, CHANNELS, cfg.num_edges, dtype, use_amp)
        feq_time, feq_mem = bench_flasheq(cfg.lmax, CHANNELS, cfg.num_edges, dtype, use_amp)

        print(
            f"{cfg.label:<16} "
            f"{fmt(se3t_time, se3t_mem):<24} "
            f"{fmt(ev2_time, ev2_mem):<24} "
            f"{fmt(feq_time, feq_mem):<24}"
        )

        results.append({
            "config": cfg.label,
            "lmax": cfg.lmax,
            "num_edges": cfg.num_edges,
            "precision": precision,
            "se3t_time_ms": se3t_time,
            "se3t_mem_gb": se3t_mem,
            "ev2_time_ms": ev2_time,
            "ev2_mem_gb": ev2_mem,
            "feq_time_ms": feq_time,
            "feq_mem_gb": feq_mem,
        })

    return results


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    all_results = []
    all_results.extend(run_suite(use_amp=False))
    all_results.extend(run_suite(use_amp=True))

    # Save results as JSON for plotting
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
