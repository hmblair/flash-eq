"""
Benchmark: NVIDIA SE(3)-Transformer vs flash-eq patched SE(3)-Transformer.

Builds a full SE(3)-Transformer (attention blocks + convolutions + norms),
patches it with flash-eq, and compares the full forward pass in terms of
timing and peak GPU memory.

Usage:
    python benchmarks/benchmark_patch.py

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock DGL (avoids GLIBC issues on cluster)
# ---------------------------------------------------------------------------

_dgl_mock = types.ModuleType("dgl")
_dgl_mock.DGLGraph = type("DGLGraph", (), {})
_dgl_ops = types.ModuleType("dgl.ops")


def _mock_copy_e_sum(graph, edge_features):
    src, dst = graph.edges()
    num_nodes = graph.num_nodes()
    out = torch.zeros(
        num_nodes, *edge_features.shape[1:],
        device=edge_features.device, dtype=edge_features.dtype,
    )
    out.index_add_(0, dst, edge_features)
    return out


def _mock_e_dot_v(graph, edge_features, node_features):
    """Mock dgl.ops.e_dot_v: dot product of edge features with dst node features."""
    _, dst = graph.edges()
    dst_feats = node_features[dst]
    return (edge_features * dst_feats).sum(dim=-1, keepdim=True)


def _mock_edge_softmax(graph, edge_scores):
    """Mock dgl.ops.edge_softmax: per-destination-node softmax over incoming edges."""
    _, dst = graph.edges()
    num_nodes = graph.num_nodes()
    # Compute max per destination for numerical stability
    max_scores = torch.full(
        (num_nodes, *edge_scores.shape[1:]),
        float("-inf"), device=edge_scores.device, dtype=edge_scores.dtype,
    )
    max_scores.scatter_reduce_(0, dst.view(-1, *([1] * (edge_scores.dim() - 1))).expand_as(edge_scores),
                                edge_scores, reduce="amax", include_self=False)
    shifted = edge_scores - max_scores[dst]
    exp_scores = shifted.exp()
    sum_exp = torch.zeros(
        num_nodes, *edge_scores.shape[1:],
        device=edge_scores.device, dtype=edge_scores.dtype,
    )
    sum_exp.index_add_(0, dst, exp_scores)
    return exp_scores / sum_exp[dst].clamp(min=1e-12)


_dgl_ops.copy_e_sum = _mock_copy_e_sum
_dgl_ops.e_dot_v = _mock_e_dot_v
_dgl_ops.edge_softmax = _mock_edge_softmax
_dgl_mock.ops = _dgl_ops
_dgl_mock.graph = lambda src_dst: None
sys.modules["dgl"] = _dgl_mock
sys.modules["dgl.ops"] = _dgl_ops

# Mock dgl.nn.pytorch for pooling imports
_dgl_nn = types.ModuleType("dgl.nn")
_dgl_nn_pytorch = types.ModuleType("dgl.nn.pytorch")
_dgl_nn_pytorch.AvgPooling = type("AvgPooling", (), {})
_dgl_nn_pytorch.MaxPooling = type("MaxPooling", (), {})
_dgl_mock.nn = _dgl_nn
sys.modules["dgl.nn"] = _dgl_nn
sys.modules["dgl.nn.pytorch"] = _dgl_nn_pytorch


class MockGraph:
    def __init__(self, src, dst, num_nodes, rel_pos):
        self._src = src
        self._dst = dst
        self._num_nodes = num_nodes
        self.edata = {"rel_pos": rel_pos}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes


# ---------------------------------------------------------------------------
# Load NVIDIA SE(3)-Transformer
# ---------------------------------------------------------------------------

_nvidia_dir = Path("/Users/hmblair/Utils")
if not _nvidia_dir.exists():
    _nvidia_dir = Path("/home/groups/rhiju/hmblair/Utils")


def _load_module(name: str, filepath: Path):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            sys.modules[parent] = pkg
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_se3t = _nvidia_dir / "se3_transformer"
_runtime_utils = _load_module(
    "se3_transformer.runtime.utils", _se3t / "runtime" / "utils.py"
)
_fiber_mod = _load_module(
    "se3_transformer.model.fiber", _se3t / "model" / "fiber.py"
)
_basis_mod = _load_module(
    "se3_transformer.model.basis", _se3t / "model" / "basis.py"
)
_conv_mod = _load_module(
    "se3_transformer.model.layers.convolution",
    _se3t / "model" / "layers" / "convolution.py",
)
_linear_mod = _load_module(
    "se3_transformer.model.layers.linear",
    _se3t / "model" / "layers" / "linear.py",
)
_norm_mod = _load_module(
    "se3_transformer.model.layers.norm",
    _se3t / "model" / "layers" / "norm.py",
)
_attention_mod = _load_module(
    "se3_transformer.model.layers.attention",
    _se3t / "model" / "layers" / "attention.py",
)
_pooling_mod = _load_module(
    "se3_transformer.model.layers.pooling",
    _se3t / "model" / "layers" / "pooling.py",
)
_transformer_mod = _load_module(
    "se3_transformer.model.transformer",
    _se3t / "model" / "transformer.py",
)

SE3Transformer = _transformer_mod.SE3Transformer
Fiber = _fiber_mod.Fiber
ConvSE3FuseLevel = _conv_mod.ConvSE3FuseLevel

import copy

from flash_eq.patch import patch
from flash_eq.patch._convention import apply_convention_patches

# Apply convention patches once
apply_convention_patches(_conv_mod.ConvSE3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_timer(fn, warmup=3, repeats=10):
    """Time a GPU function with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return torch.tensor(times)


def peak_memory_mb(fn, repeats=3):
    """Measure peak GPU memory of a function (MB)."""
    fn()
    torch.cuda.synchronize()
    peaks = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() / 1024**2)
    return min(peaks)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_config(lmax, C, num_edges, num_nodes, num_layers=1,
                     num_heads=4, channels_div=2, num_bins=500, max_dist=10.0):
    """Run one benchmark configuration with a full SE(3)-Transformer."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(42)

    fiber = Fiber.create(lmax + 1, C)

    # Build full SE(3)-Transformer
    model = SE3Transformer(
        num_layers=num_layers,
        fiber_in=fiber,
        fiber_hidden=fiber,
        fiber_out=fiber,
        fiber_edge=Fiber({}),
        num_heads=num_heads,
        channels_div=channels_div,
        norm=True,
        use_layer_norm=False,
        tensor_cores=True,  # FULL fuse level
    ).to(device).to(dtype)

    # Data
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * (max_dist * 0.9) + 0.5
    rel_pos = directions * distances.unsqueeze(-1)

    node_feats = {}
    for d, c in fiber:
        node_feats[str(d)] = torch.randn(
            num_nodes, c, 2 * d + 1, device=device, dtype=dtype
        )

    g = MockGraph(src, dst, num_nodes, rel_pos)

    # ---- NVIDIA forward ----
    def run_nvidia():
        with torch.no_grad():
            return model(g, node_feats)

    # Capture output
    with torch.no_grad():
        nvidia_out = model(g, node_feats)

    nvidia_times = gpu_timer(run_nvidia)
    nvidia_peak = peak_memory_mb(run_nvidia)

    # ---- Patched forward ----
    patched_model = copy.deepcopy(model)
    patch(patched_model, num_bins=num_bins, max_dist=max_dist)

    def run_patched():
        with torch.no_grad():
            return patched_model(g, node_feats)

    # Capture output
    with torch.no_grad():
        patched_out = patched_model(g, node_feats)

    patched_times = gpu_timer(run_patched)
    patched_peak = peak_memory_mb(run_patched)

    # ---- Compare outputs ----
    lvals = list(range(lmax + 1))
    nvidia_cat = torch.cat([nvidia_out[str(d)] for d in lvals], dim=-1)
    patched_cat = torch.cat([patched_out[str(d)] for d in lvals], dim=-1)

    max_err = (nvidia_cat - patched_cat).abs().max().item()
    mean_err = (nvidia_cat - patched_cat).abs().mean().item()
    rel_err = max_err / nvidia_cat.abs().max().item() if nvidia_cat.abs().max().item() > 0 else 0.0

    degree_errors = {}
    for d in lvals:
        e = (nvidia_out[str(d)] - patched_out[str(d)]).abs().max().item()
        degree_errors[d] = e

    return {
        "lmax": lmax,
        "C": C,
        "num_edges": num_edges,
        "num_layers": num_layers,
        "nvidia_time_ms": nvidia_times.median().item(),
        "patched_time_ms": patched_times.median().item(),
        "speedup": nvidia_times.median().item() / patched_times.median().item(),
        "nvidia_peak_mb": nvidia_peak,
        "patched_peak_mb": patched_peak,
        "mem_ratio": nvidia_peak / patched_peak if patched_peak > 0 else float("inf"),
        "max_err": max_err,
        "mean_err": mean_err,
        "rel_err": rel_err,
        "degree_errors": degree_errors,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    configs = [
        # (lmax, C, num_edges, num_nodes)
        (2, 32, 4096, 256),
        (2, 32, 16384, 512),
        (2, 32, 65536, 1024),
        (3, 32, 4096, 256),
        (3, 32, 16384, 512),
        (3, 32, 65536, 1024),
    ]

    results = []
    for lmax, C, E, N in configs:
        label = f"lmax={lmax}, C={C}, E={E:,}"
        print(f"{'=' * 70}")
        print(f" {label}")
        print(f"{'=' * 70}")

        r = benchmark_config(lmax, C, E, N)
        results.append(r)

        # Output comparison
        print(f"\n  Output comparison:")
        print(f"    Max absolute error:  {r['max_err']:.2e}")
        print(f"    Mean absolute error: {r['mean_err']:.2e}")
        print(f"    Relative error:      {r['rel_err']:.2e}")
        for d, e in r["degree_errors"].items():
            print(f"    Degree {d}: {e:.2e}")

        # Timing
        print(f"\n  Timing (median of 10):")
        print(f"    NVIDIA:  {r['nvidia_time_ms']:8.2f} ms")
        print(f"    Patched: {r['patched_time_ms']:8.2f} ms")
        print(f"    Speedup: {r['speedup']:.2f}x")

        # Memory
        print(f"\n  Peak GPU memory (forward pass):")
        print(f"    NVIDIA:  {r['nvidia_peak_mb']:8.1f} MB")
        print(f"    Patched: {r['patched_peak_mb']:8.1f} MB")
        print(f"    Ratio:   {r['mem_ratio']:.2f}x")
        print()

    # Summary table
    print(f"\n{'=' * 70}")
    print(f" SUMMARY")
    print(f"{'=' * 70}")
    hdr = f"{'Config':<28} {'Max Err':>9} {'Time NVIDIA':>12} {'Time Patched':>13} {'Speedup':>8} {'Mem NVIDIA':>11} {'Mem Patched':>12} {'Mem Ratio':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        label = f"L={r['lmax']} C={r['C']} E={r['num_edges']:,}"
        print(
            f"{label:<28} {r['max_err']:>9.2e} "
            f"{r['nvidia_time_ms']:>10.2f}ms "
            f"{r['patched_time_ms']:>11.2f}ms "
            f"{r['speedup']:>7.2f}x "
            f"{r['nvidia_peak_mb']:>9.1f}MB "
            f"{r['patched_peak_mb']:>10.1f}MB "
            f"{r['mem_ratio']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
