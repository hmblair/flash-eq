"""
Tests for flash_eq.patch — NVIDIA SE(3)-Transformer patching.

Tests the weight conversion and PatchedConvSE3 against NVIDIA ConvSE3
outputs for all fuse levels. Uses a mock DGL to avoid dependency issues.

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
    """Mock pooling: sum edge features to destination nodes."""
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
    """Minimal DGL-like graph for testing."""
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
# Load NVIDIA SE(3)-Transformer modules via importlib
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
unfuse_features = _runtime_utils.unfuse_features

# Flash-eq
from flash_eq import Repr, WignerDBasis  # noqa: E402
from flash_eq.patch import patch  # noqa: E402
from flash_eq.patch._convention import apply_convention_patches  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_data(num_edges, lvals, C, device, dtype, max_dist=10.0):
    """Create test edge directions and distances within the binning range."""
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * (max_dist * 0.9) + 0.5

    features_dict = {}
    for d in lvals:
        features_dict[str(d)] = torch.randn(
            num_edges, C, 2 * d + 1, device=device, dtype=dtype
        )
    features_cat = torch.cat([features_dict[str(d)] for d in lvals], dim=-1)
    return directions, distances, features_dict, features_cat


def nvidia_forward_no_dgl(conv, features_dict, distances, basis, lvals):
    """NVIDIA ConvSE3 forward without DGL."""
    edge_feats = distances.unsqueeze(-1)

    if conv.used_fuse_level == ConvSE3FuseLevel.FULL:
        in_fused = torch.cat([features_dict[str(d)] for d in conv.fiber_in.degrees], dim=-1)
        out_fused = conv.conv(in_fused, edge_feats, basis["fully_fused"])
        return unfuse_features(out_fused, lvals)

    elif conv.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(conv, "conv_out"):
        in_fused = torch.cat([features_dict[str(d)] for d in conv.fiber_in.degrees], dim=-1)
        out = {}
        for d_out in lvals:
            result = conv.conv_out[str(d_out)](in_fused, edge_feats, basis[f"out{d_out}_fused"])
            out[str(d_out)] = result[..., :degree_to_dim(d_out)]
        return out

    else:
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
        return out


def compute_nvidia_basis(directions, max_degree, fuse_level, dtype):
    """Compute NVIDIA CG basis with optional fusing."""
    basis = _basis_mod.get_basis(
        directions.float(), max_degree=max_degree, use_pad_trick=False, amp=False,
    )
    basis = {k: v.to(dtype) for k, v in basis.items()}

    if fuse_level in (ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL):
        num_edges = basis["0,0"].shape[0]
        device = basis["0,0"].device
        sum_dim = sum(degree_to_dim(d) for d in range(max_degree + 1))
        fully_fused = fuse_level == ConvSE3FuseLevel.FULL

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

        for d_in in range(max_degree + 1):
            sum_freq = sum(degree_to_dim(min(d, d_in)) for d in range(max_degree + 1))
            dim_in = degree_to_dim(d_in)
            basis_fused = torch.zeros(
                num_edges, dim_in, sum_freq, sum_dim, device=device, dtype=dtype,
            )
            acc_d, acc_f = 0, 0
            for d_out in range(max_degree + 1):
                dim_out = degree_to_dim(d_out)
                dim_freq = degree_to_dim(min(d_out, d_in))
                basis_fused[
                    :, :, acc_f:acc_f + dim_freq, acc_d:acc_d + dim_out
                ] = basis[f"{d_in},{d_out}"][:, :, :, :dim_out]
                acc_d += dim_out
                acc_f += dim_freq
            basis[f"in{d_in}_fused"] = basis_fused

        if fully_fused:
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

        del basis["0,0"]

    return basis


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_conv_conversion():
    """Test weight conversion for all fuse levels (float64)."""
    print("\n" + "=" * 70)
    print(" TEST: Weight conversion (all fuse levels, float64)")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.float64

    lmax = 2
    C = 4
    num_edges = 128
    num_bins = 500
    lvals = list(range(lmax + 1))

    apply_convention_patches(ConvSE3)
    torch.manual_seed(42)

    directions, distances, features_dict, features_cat = make_test_data(
        num_edges, lvals, C, device, dtype
    )

    from flash_eq.cuda.block_diagonal import block_diagonal_cuda
    from flash_eq.patch._conversion import convert_conv_to_table

    repr_obj = Repr(lvals=lvals, mult=1)
    wigner = WignerDBasis([repr_obj]).to(device).to(dtype)
    (P,) = wigner(directions)

    all_pass = True
    for fuse_level in [ConvSE3FuseLevel.NONE, ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL]:
        print(f"\n  --- {fuse_level.name} ---")

        fiber = Fiber.create(lmax + 1, C)
        conv = ConvSE3(
            fiber_in=fiber, fiber_out=fiber, fiber_edge=Fiber({}),
            pool=False, self_interaction=False, max_degree=lmax,
            fuse_level=fuse_level,
        ).to(device).to(dtype)

        basis = compute_nvidia_basis(directions, lmax, conv.used_fuse_level, dtype)
        with torch.no_grad():
            nvidia_out = nvidia_forward_no_dgl(conv, features_dict, distances, basis, lvals)
        nvidia_cat = torch.cat([nvidia_out[str(d)] for d in lvals], dim=-1)

        table, product_repr, _, _ = convert_conv_to_table(
            conv, _basis_mod, num_bins, 0.0, 10.0, device, dtype
        )

        with torch.no_grad():
            f_diag = torch.bmm(features_cat, P)
            out_diag = block_diagonal_cuda(
                f_diag, table, distances, product_repr,
                bin_param1=0.0, bin_param2=num_bins / 10.0,
                num_bins=num_bins, sh_scale=0.0,
            )
            feq_out = torch.bmm(out_diag, P.mT)

        max_err = (nvidia_cat - feq_out).abs().max().item()
        tol = 2e-4
        status = "PASS" if max_err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Max error: {max_err:.2e} [{status}]")

    assert all_pass, "Weight conversion failed for at least one fuse level"


def test_patched_conv():
    """Test PatchedConvSE3 forward with MockGraph (float32)."""
    print("\n" + "=" * 70)
    print(" TEST: PatchedConvSE3 forward (all fuse levels, float32)")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 2
    C = 8
    num_nodes = 32
    num_edges = 128
    num_bins = 500
    lvals = list(range(lmax + 1))

    apply_convention_patches(ConvSE3)
    torch.manual_seed(42)

    all_pass = True
    for fuse_level in [ConvSE3FuseLevel.NONE, ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL]:
        print(f"\n  --- {fuse_level.name} ---")

        fiber = Fiber.create(lmax + 1, C)
        conv = ConvSE3(
            fiber_in=fiber, fiber_out=fiber, fiber_edge=Fiber({}),
            pool=False, self_interaction=False, max_degree=lmax,
            fuse_level=fuse_level,
        ).to(device).to(dtype)

        # Create test data with distances in binning range
        src = torch.randint(0, num_nodes, (num_edges,), device=device)
        dst = torch.randint(0, num_nodes, (num_edges,), device=device)

        # Generate directions and distances separately to control range
        directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 9.0 + 0.5
        rel_pos = directions * distances.unsqueeze(-1)

        node_feats = {}
        for d, c in fiber:
            node_feats[str(d)] = torch.randn(
                num_nodes, c, 2 * d + 1, device=device, dtype=dtype
            )

        # Edge-level features for NVIDIA forward
        features_dict = {str(d): node_feats[str(d)][src] for d in lvals}
        edge_feats = {"0": distances.unsqueeze(-1)[..., None]}

        # NVIDIA forward
        basis = compute_nvidia_basis(directions, lmax, conv.used_fuse_level, dtype)
        with torch.no_grad():
            nvidia_out = nvidia_forward_no_dgl(conv, features_dict, distances, basis, lvals)

        # Patch
        import copy
        conv_copy = copy.deepcopy(conv)
        wrapper = nn.Module()
        wrapper.conv = conv_copy
        patch(wrapper, num_bins=num_bins, max_dist=10.0)

        # Create basis for patched forward
        repr_obj = Repr(lvals=lvals, mult=1)
        wigner = WignerDBasis([repr_obj]).to(device)
        (M,) = wigner(directions)

        g = MockGraph(src, dst, num_nodes, rel_pos)
        patched_basis = {"_P": M, "_Q": M, "_distances": distances}

        with torch.no_grad():
            patched_out = wrapper.conv(node_feats, edge_feats, g, patched_basis)

        max_err = 0.0
        for d in lvals:
            err = (nvidia_out[str(d)] - patched_out[str(d)]).abs().max().item()
            max_err = max(max_err, err)
            print(f"    degree {d}: {err:.2e}")

        tol = 5e-3
        status = "PASS" if max_err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Overall: max error = {max_err:.2e} [{status}]")

    assert all_pass, "PatchedConvSE3 forward failed for at least one fuse level"


def test_self_interaction():
    """Test PatchedConvSE3 with self_interaction=True."""
    print("\n" + "=" * 70)
    print(" TEST: PatchedConvSE3 self-interaction")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 2
    C = 8
    num_nodes = 32
    num_edges = 128
    num_bins = 500
    lvals = list(range(lmax + 1))

    apply_convention_patches(ConvSE3)
    torch.manual_seed(42)

    fiber = Fiber.create(lmax + 1, C)
    conv = ConvSE3(
        fiber_in=fiber, fiber_out=fiber, fiber_edge=Fiber({}),
        pool=False, self_interaction=True, max_degree=lmax,
        fuse_level=ConvSE3FuseLevel.NONE,
    ).to(device).to(dtype)

    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 9.0 + 0.5
    rel_pos = directions * distances.unsqueeze(-1)

    node_feats = {}
    for d, c in fiber:
        node_feats[str(d)] = torch.randn(
            num_nodes, c, 2 * d + 1, device=device, dtype=dtype
        )

    features_dict = {str(d): node_feats[str(d)][src] for d in lvals}
    edge_feats = {"0": distances.unsqueeze(-1)[..., None]}

    # NVIDIA forward + manual self-interaction
    basis = compute_nvidia_basis(directions, lmax, ConvSE3FuseLevel.NONE, dtype)
    with torch.no_grad():
        nvidia_edge_out = nvidia_forward_no_dgl(conv, features_dict, distances, basis, lvals)
        nvidia_out = {}
        for d in lvals:
            nvidia_out[str(d)] = nvidia_edge_out[str(d)]
            if str(d) in conv.to_kernel_self:
                nvidia_out[str(d)] = nvidia_out[str(d)] + conv.to_kernel_self[str(d)] @ node_feats[str(d)][dst]

    # Patch
    import copy
    wrapper = nn.Module()
    wrapper.conv = copy.deepcopy(conv)
    patch(wrapper, num_bins=num_bins, max_dist=10.0)

    repr_obj = Repr(lvals=lvals, mult=1)
    wigner = WignerDBasis([repr_obj]).to(device)
    (M,) = wigner(directions)

    g = MockGraph(src, dst, num_nodes, rel_pos)
    patched_basis = {"_P": M, "_Q": M, "_distances": distances}

    with torch.no_grad():
        patched_out = wrapper.conv(node_feats, edge_feats, g, patched_basis)

    max_err = 0.0
    for d in lvals:
        err = (nvidia_out[str(d)] - patched_out[str(d)]).abs().max().item()
        max_err = max(max_err, err)
        print(f"    degree {d}: {err:.2e}")

    tol = 5e-3
    status = "PASS" if max_err < tol else "FAIL"
    print(f"  Overall: max error = {max_err:.2e} [{status}]")
    assert max_err < tol, f"Self-interaction test failed: max_err={max_err:.2e}"


def test_pooling():
    """Test PatchedConvSE3 with pool=True."""
    print("\n" + "=" * 70)
    print(" TEST: PatchedConvSE3 pooling")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 2
    C = 8
    num_nodes = 32
    num_edges = 128
    num_bins = 500
    lvals = list(range(lmax + 1))

    apply_convention_patches(ConvSE3)
    torch.manual_seed(42)

    fiber = Fiber.create(lmax + 1, C)
    conv = ConvSE3(
        fiber_in=fiber, fiber_out=fiber, fiber_edge=Fiber({}),
        pool=True, self_interaction=True, max_degree=lmax,
        fuse_level=ConvSE3FuseLevel.NONE,
    ).to(device).to(dtype)

    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 9.0 + 0.5
    rel_pos = directions * distances.unsqueeze(-1)

    node_feats = {}
    for d, c in fiber:
        node_feats[str(d)] = torch.randn(
            num_nodes, c, 2 * d + 1, device=device, dtype=dtype
        )

    features_dict = {str(d): node_feats[str(d)][src] for d in lvals}
    edge_feats = {"0": distances.unsqueeze(-1)[..., None]}

    # NVIDIA forward + manual self-interaction + manual pooling
    basis = compute_nvidia_basis(directions, lmax, ConvSE3FuseLevel.NONE, dtype)
    with torch.no_grad():
        nvidia_edge_out = nvidia_forward_no_dgl(conv, features_dict, distances, basis, lvals)
        nvidia_out = {}
        for d in lvals:
            edge_result = nvidia_edge_out[str(d)]
            if str(d) in conv.to_kernel_self:
                edge_result = edge_result + conv.to_kernel_self[str(d)] @ node_feats[str(d)][dst]
            # Manual pooling (sum over edges to destination nodes)
            pooled = torch.zeros(num_nodes, *edge_result.shape[1:], device=device, dtype=dtype)
            pooled.index_add_(0, dst, edge_result)
            nvidia_out[str(d)] = pooled

    # Patch
    import copy
    wrapper = nn.Module()
    wrapper.conv = copy.deepcopy(conv)
    patch(wrapper, num_bins=num_bins, max_dist=10.0)

    repr_obj = Repr(lvals=lvals, mult=1)
    wigner = WignerDBasis([repr_obj]).to(device)
    (M,) = wigner(directions)

    g = MockGraph(src, dst, num_nodes, rel_pos)
    patched_basis = {"_P": M, "_Q": M, "_distances": distances}

    with torch.no_grad():
        patched_out = wrapper.conv(node_feats, edge_feats, g, patched_basis)

    max_err = 0.0
    for d in lvals:
        err = (nvidia_out[str(d)] - patched_out[str(d)]).abs().max().item()
        max_err = max(max_err, err)
        print(f"    degree {d}: {err:.2e}")

    tol = 5e-3
    status = "PASS" if max_err < tol else "FAIL"
    print(f"  Overall: max error = {max_err:.2e} [{status}]")
    assert max_err < tol, f"Pooling test failed: max_err={max_err:.2e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    test_conv_conversion()
    test_patched_conv()
    test_self_interaction()
    test_pooling()
    print("\nAll tests passed.")
