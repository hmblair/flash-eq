"""
Benchmark: NVIDIA ConvSE3 vs flash-eq PatchedConvSE3.

Runs the NVIDIA layer, patches it, re-runs, then compares outputs,
timing, and peak GPU memory. Demonstrates that the patch produces
identical outputs while reducing memory.

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


_dgl_ops.copy_e_sum = _mock_copy_e_sum
_dgl_mock.ops = _dgl_ops
_dgl_mock.graph = lambda src_dst: None
sys.modules["dgl"] = _dgl_mock
sys.modules["dgl.ops"] = _dgl_ops


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

ConvSE3 = _conv_mod.ConvSE3
ConvSE3FuseLevel = _conv_mod.ConvSE3FuseLevel
Fiber = _fiber_mod.Fiber
degree_to_dim = _runtime_utils.degree_to_dim

import copy

from flash_eq import Repr, WignerDBasis
from flash_eq.patch import patch
from flash_eq.patch._convention import apply_convention_patches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_nvidia_basis(directions, max_degree, dtype, fully_fused=True):
    """Compute NVIDIA CG basis with optional full fusing."""
    basis = _basis_mod.get_basis(
        directions.float(), max_degree=max_degree,
        use_pad_trick=False, amp=False,
    )
    basis = {k: v.to(dtype) for k, v in basis.items()}

    if not fully_fused:
        return basis

    # Fuse basis for FULL fuse level (replicate update_basis_with_fused)
    num_edges = basis["0,0"].shape[0]
    device = basis["0,0"].device
    sum_dim = sum(degree_to_dim(d) for d in range(max_degree + 1))

    # Per-output-degree fused basis (needed as intermediate)
    for d_out in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_out)) for d in range(max_degree + 1))
        basis_fused = torch.zeros(
            num_edges, sum_dim, sum_freq, degree_to_dim(d_out),
            device=device, dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            dim_in = degree_to_dim(d_in)
            dim_out = degree_to_dim(d_out)
            dim_freq = degree_to_dim(min(d_out, d_in))
            basis_fused[
                :, acc_d:acc_d + dim_in, acc_f:acc_f + dim_freq, :dim_out
            ] = basis[f"{d_in},{d_out}"][:, :, :, :dim_out]
            acc_d += dim_in
            acc_f += dim_freq
        basis[f"out{d_out}_fused"] = basis_fused

    # Fully fused basis
    sum_freq = sum(
        sum(degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1))
        for d_out in range(max_degree + 1)
    )
    basis_fused = torch.zeros(
        num_edges, sum_dim, sum_freq, sum_dim, device=device, dtype=dtype,
    )
    acc_d, acc_f = 0, 0
    for d_out in range(max_degree + 1):
        b = basis[f"out{d_out}_fused"]
        dim_out = degree_to_dim(d_out)
        basis_fused[
            :, :, acc_f:acc_f + b.shape[2], acc_d:acc_d + dim_out
        ] = b[:, :, :, :dim_out]
        acc_f += b.shape[2]
        acc_d += dim_out
    basis["fully_fused"] = basis_fused

    # Clean up intermediate keys
    del basis["0,0"]

    return basis


unfuse_features = _runtime_utils.unfuse_features


def nvidia_forward(conv, features_dict, distances, basis, lvals,
                   node_feats=None, dst=None):
    """NVIDIA ConvSE3 forward (FULL or NONE fuse, no DGL), including self-interaction."""
    edge_feats = distances.unsqueeze(-1)

    if conv.used_fuse_level == ConvSE3FuseLevel.FULL:
        # FULL fuse: single fused conv
        in_fused = torch.cat([features_dict[str(d)] for d in lvals], dim=-1)
        out_fused = conv.conv(in_fused, edge_feats, basis["fully_fused"])
        out = unfuse_features(out_fused, lvals)
    else:
        # NONE fuse: per-pair loop
        out = {}
        for d_out in lvals:
            acc = 0
            for d_in in conv.fiber_in.degrees:
                key = f"{d_in},{d_out}"
                basis_used = basis.get(key)
                result = conv.conv[key](features_dict[str(d_in)], edge_feats, basis_used)
                if basis_used is not None:
                    result = result[..., :degree_to_dim(d_out)]
                acc = acc + result
            out[str(d_out)] = acc

    if conv.self_interaction and hasattr(conv, "to_kernel_self"):
        for d in lvals:
            if str(d) in conv.to_kernel_self:
                out[str(d)] = out[str(d)] + conv.to_kernel_self[str(d)] @ node_feats[str(d)][dst]

    return out


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
    # Warmup
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

def benchmark_config(lmax, C, num_edges, num_nodes, num_bins=500, max_dist=10.0):
    """Run one benchmark configuration."""
    device = torch.device("cuda")
    dtype = torch.float32

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    apply_convention_patches(ConvSE3)
    torch.manual_seed(42)

    fiber = Fiber.create(lmax + 1, C)
    conv = ConvSE3(
        fiber_in=fiber, fiber_out=fiber, fiber_edge=Fiber({}),
        pool=False, self_interaction=True, max_degree=lmax,
        fuse_level=ConvSE3FuseLevel.FULL,
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

    features_dict = {str(d): node_feats[str(d)][src] for d in lvals}

    # ---- NVIDIA forward ----
    basis = compute_nvidia_basis(directions, lmax, dtype, fully_fused=True)

    def run_nvidia():
        with torch.no_grad():
            return nvidia_forward(conv, features_dict, distances, basis, lvals,
                                  node_feats, dst)

    # Capture output
    with torch.no_grad():
        nvidia_out = nvidia_forward(conv, features_dict, distances, basis, lvals,
                                    node_feats, dst)
    nvidia_cat = torch.cat([nvidia_out[str(d)] for d in lvals], dim=-1)

    # Measure NVIDIA
    nvidia_times = gpu_timer(run_nvidia)

    # Memory: include basis storage
    basis_mem = sum(v.numel() * v.element_size() for v in basis.values()) / 1024**2
    nvidia_peak = peak_memory_mb(run_nvidia)

    # ---- Patch ----
    wrapper = nn.Module()
    wrapper.conv = copy.deepcopy(conv)
    patch(wrapper, num_bins=num_bins, max_dist=max_dist)
    patched = wrapper.conv

    # Basis for patched forward
    repr_obj = Repr(lvals=lvals, mult=1)
    wigner = WignerDBasis([repr_obj]).to(device)
    (M,) = wigner(directions)

    g = MockGraph(src, dst, num_nodes, rel_pos)
    patched_basis = {"_P": M, "_Q": M, "_distances": distances}
    edge_feats = {"0": distances.unsqueeze(-1)[..., None]}

    def run_patched():
        with torch.no_grad():
            return patched(node_feats, edge_feats, g, patched_basis)

    # Capture output
    with torch.no_grad():
        patched_out = patched(node_feats, edge_feats, g, patched_basis)
    patched_cat = torch.cat([patched_out[str(d)] for d in lvals], dim=-1)

    # Measure patched
    patched_times = gpu_timer(run_patched)
    wigner_mem = M.numel() * M.element_size() / 1024**2
    patched_peak = peak_memory_mb(run_patched)

    # ---- Compare ----
    max_err = (nvidia_cat - patched_cat).abs().max().item()
    mean_err = (nvidia_cat - patched_cat).abs().mean().item()
    rel_err = max_err / nvidia_cat.abs().max().item()

    # Per-degree errors
    degree_errors = {}
    for d in lvals:
        e = (nvidia_out[str(d)] - patched_out[str(d)]).abs().max().item()
        degree_errors[d] = e

    return {
        "lmax": lmax,
        "C": C,
        "num_edges": num_edges,
        "dim": dim,
        "nvidia_time_ms": nvidia_times.median().item(),
        "patched_time_ms": patched_times.median().item(),
        "speedup": nvidia_times.median().item() / patched_times.median().item(),
        "basis_mem_mb": basis_mem,
        "wigner_mem_mb": wigner_mem,
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
        print(f"\n  Basis memory:")
        print(f"    NVIDIA CG basis:     {r['basis_mem_mb']:8.2f} MB")
        print(f"    Flash-eq Wigner-D:   {r['wigner_mem_mb']:8.2f} MB")
        print(f"    Ratio:               {r['basis_mem_mb'] / r['wigner_mem_mb']:.1f}x")
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
