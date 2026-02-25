"""
Prototype: NVIDIA SE(3)-T → Flash-eq weight conversion.

NOTE: This is a development prototype. The productionized version lives in
flash_eq/patch/. This file is kept as reference for the conversion algorithm
and for standalone correctness/benchmark testing.

Demonstrates that an NVIDIA SE(3)-Transformer's (VersatileConvSE3) per-edge
equivariant kernel can be exactly converted to flash-eq format, producing
identical outputs. Supports all three ConvSE3FuseLevel modes (NONE, PARTIAL,
FULL) and benchmarks GPU memory/runtime vs flash-eq.

The NVIDIA SE(3)-T parameterizes:
    K(r, d) = sum_J w_J(r) * B_J(d)     [CG frequency basis]

Flash-eq parameterizes:
    K(r, d) = P(d) @ Lambda(r)^T @ Q(d)^T  [m-diagonal basis]

The conversion evaluates the NVIDIA kernel at e_z and projects into the
m-diagonal basis via flash-eq's Wigner-D matrices.

We monkey-patch the NVIDIA basis computation to use sphericart SH and
sphericart-convention CG coefficients (computed via Lebedev quadrature),
so that both the NVIDIA basis and flash-eq's P/Q share the same SH
convention. This eliminates any convention mismatch and makes the
conversion a direct projection with no correction needed.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import types
from functools import lru_cache
from itertools import product as iterproduct
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir, _this_dir.parent]:
    if (_candidate / "flash_eq").is_dir():
        sys.path.insert(0, str(_candidate))
        break

# Flash-eq
from flash_eq import (
    EquivariantEdgewiseLinear,
    Repr,
    ProductRepr,
    WignerDBasis,
    WignerD,
    real_spherical_harmonics,
    lebedev_grid,
)
from flash_eq.cuda.block_diagonal import block_diagonal_cuda

# ---------------------------------------------------------------------------
# NVIDIA SE(3)-Transformer imports via importlib (bypasses __init__.py → DGL)
# ---------------------------------------------------------------------------
import os
_nvidia_dir = Path(os.environ.get("SE3T_DIR", ""))
if not _nvidia_dir.exists():
    raise RuntimeError(
        "Set the SE3T_DIR environment variable to the directory containing "
        "se3_transformer/ (e.g., export SE3T_DIR=/path/to/Utils)"
    )


def _load_module(name: str, filepath: Path):
    """Load a single Python module by file path, registering parent packages."""
    parts = name.split('.')
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
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
    'se3_transformer.runtime.utils', _se3t / 'runtime' / 'utils.py')
_fiber_mod = _load_module(
    'se3_transformer.model.fiber', _se3t / 'model' / 'fiber.py')
_basis_mod = _load_module(
    'se3_transformer.model.basis', _se3t / 'model' / 'basis.py')

# convolution.py has `from dgl import DGLGraph` — provide a minimal mock
_dgl_mock = types.ModuleType('dgl')
_dgl_mock.DGLGraph = type('DGLGraph', (), {})
_dgl_mock.ops = types.ModuleType('dgl.ops')
_dgl_mock.ops.copy_e_sum = None
sys.modules.setdefault('dgl', _dgl_mock)
sys.modules.setdefault('dgl.ops', _dgl_mock.ops)

_conv_mod = _load_module(
    'se3_transformer.model.layers.convolution',
    _se3t / 'model' / 'layers' / 'convolution.py')

ConvSE3 = _conv_mod.ConvSE3
ConvSE3FuseLevel = _conv_mod.ConvSE3FuseLevel
VersatileConvSE3 = _conv_mod.VersatileConvSE3
Fiber = _fiber_mod.Fiber
degree_to_dim = _runtime_utils.degree_to_dim
unfuse_features = _runtime_utils.unfuse_features


# ============================================================================
# Monkey-patch: replace e3nn SH/CG with sphericart-convention equivalents
# ============================================================================

def _real_clebsch_gordan(l1: int, l2: int, l3: int) -> Tensor:
    """Compute real CG coefficients via Lebedev quadrature (sphericart convention).

    CG[m1, m2, m3] = integral Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} dΩ
    """
    points, weights = lebedev_grid(precision=131)
    points = points.to(torch.float64)
    weights = weights.to(torch.float64)

    lmax = max(l1, l2, l3)
    Y = real_spherical_harmonics(lmax, points).to(torch.float64)

    def get_Y(l):
        return Y[:, l * l:(l + 1) ** 2]

    return torch.einsum('n,na,nb,nc->abc', weights, get_Y(l1), get_Y(l2), get_Y(l3))


_cg_cache: dict[tuple[int, int, int, str], Tensor] = {}


@lru_cache(maxsize=None)
def _get_clebsch_gordon_sc(J: int, d_in: int, d_out: int, device) -> Tensor:
    """Replacement for NVIDIA's get_clebsch_gordon using sphericart convention.

    Original returns o3.wigner_3j(J, d_in, d_out).permute(2, 1, 0)
    with shape (2*d_out+1, 2*d_in+1, 2*J+1).

    We compute CG[m_in, m_J, m_out] via quadrature with sphericart SH, then
    permute to match the (d_out, d_in, J) index order the NVIDIA code expects.
    """
    cg = _real_clebsch_gordan(J, d_in, d_out)  # (2J+1, 2d_in+1, 2d_out+1)
    return cg.permute(2, 1, 0).to(device=device)  # (2d_out+1, 2d_in+1, 2J+1)


@lru_cache(maxsize=None)
def _get_all_clebsch_gordon_sc(max_degree: int, device) -> List[List[Tensor]]:
    """Replacement for NVIDIA's get_all_clebsch_gordon."""
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(_get_clebsch_gordon_sc(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def _get_spherical_harmonics_sc(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    """Replacement for NVIDIA's get_spherical_harmonics using sphericart."""
    all_degrees = list(range(2 * max_degree + 1))
    sh = real_spherical_harmonics(max(all_degrees), relative_pos.to(torch.float64))
    sh = sh.to(relative_pos.dtype)
    return [sh[:, d * d:(d + 1) ** 2] for d in all_degrees]


# Apply monkey-patches to the loaded basis module
_basis_mod.get_clebsch_gordon = _get_clebsch_gordon_sc
_basis_mod.get_all_clebsch_gordon = _get_all_clebsch_gordon_sc
_basis_mod.get_spherical_harmonics = _get_spherical_harmonics_sc

# get_basis_script is @torch.jit.script — it takes SH and CG as arguments,
# so the monkey-patches above are sufficient; no need to patch get_basis_script.
# But we do need a non-JIT version since the JIT-compiled one captured
# the original function signatures.
def _get_basis_no_jit(
    max_degree: int,
    use_pad_trick: bool,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
    amp: bool,
) -> Dict[str, Tensor]:
    """Non-JIT replacement for get_basis_script."""
    import torch.nn.functional as F
    basis = {}
    idx = 0
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f'{d_in},{d_out}'
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(torch.einsum(
                    'n f, k l f -> n l k',
                    spherical_harmonics[J].float(),
                    Q_J.float(),
                ))
            basis[key] = torch.stack(K_Js, 2)
            if amp:
                basis[key] = basis[key].half()
            if use_pad_trick:
                basis[key] = F.pad(basis[key], (0, 1))
            idx += 1
    return basis


_basis_mod.get_basis_script = _get_basis_no_jit


def _update_basis_with_fused_no_jit(
    basis: Dict[str, Tensor],
    max_degree: int,
    use_pad_trick: bool,
    fully_fused: bool,
) -> Dict[str, Tensor]:
    """Non-JIT replacement for update_basis_with_fused."""
    num_edges = basis['0,0'].shape[0]
    device = basis['0,0'].device
    dtype = basis['0,0'].dtype
    sum_dim = sum(degree_to_dim(d) for d in range(max_degree + 1))

    # Fused per output degree
    for d_out in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_out)) for d in range(max_degree + 1))
        basis_fused = torch.zeros(
            num_edges, sum_dim, sum_freq,
            degree_to_dim(d_out) + int(use_pad_trick),
            device=device, dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            dim_in = degree_to_dim(d_in)
            dim_out = degree_to_dim(d_out)
            dim_freq = degree_to_dim(min(d_out, d_in))
            basis_fused[
                :, acc_d:acc_d + dim_in, acc_f:acc_f + dim_freq, :dim_out
            ] = basis[f'{d_in},{d_out}'][:, :, :, :dim_out]
            acc_d += dim_in
            acc_f += dim_freq
        basis[f'out{d_out}_fused'] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_in)) for d in range(max_degree + 1))
        dim_in = degree_to_dim(d_in)
        basis_fused = torch.zeros(
            num_edges, dim_in, sum_freq, sum_dim,
            device=device, dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            dim_out = degree_to_dim(d_out)
            dim_freq = degree_to_dim(min(d_out, d_in))
            basis_fused[
                :, :, acc_f:acc_f + dim_freq, acc_d:acc_d + dim_out
            ] = basis[f'{d_in},{d_out}'][:, :, :, :dim_out]
            acc_d += dim_out
            acc_f += dim_freq
        basis[f'in{d_in}_fused'] = basis_fused

    if fully_fused:
        sum_freq = sum(
            sum(degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1))
            for d_out in range(max_degree + 1)
        )
        basis_fused = torch.zeros(
            num_edges, sum_dim, sum_freq, sum_dim,
            device=device, dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            b = basis[f'out{d_out}_fused']
            dim_out = degree_to_dim(d_out)
            basis_fused[
                :, :, acc_f:acc_f + b.shape[2], acc_d:acc_d + dim_out
            ] = b[:, :, :, :dim_out]
            acc_f += b.shape[2]
            acc_d += dim_out
        basis['fully_fused'] = basis_fused

    del basis['0,0']
    return basis


_basis_mod.update_basis_with_fused = _update_basis_with_fused_no_jit

# Re-bind get_basis to use patched functions
get_basis = _basis_mod.get_basis
update_basis_with_fused = _update_basis_with_fused_no_jit


# ============================================================================
# NVIDIA forward pass (no DGL) — supports all fuse levels
# ============================================================================

def nvidia_forward(
    conv: ConvSE3,
    features_dict: dict[str, Tensor],
    distances: Tensor,
    basis: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Replicate ConvSE3 forward without DGL graph aggregation.

    Supports NONE, PARTIAL (conv_out), and FULL fuse levels.
    """
    edge_feats = distances.unsqueeze(-1)  # (E, 1)
    lvals = conv.fiber_out.degrees

    if conv.used_fuse_level == ConvSE3FuseLevel.FULL:
        in_features_fused = torch.cat(
            [features_dict[str(d)] for d in conv.fiber_in.degrees], dim=-1
        )
        out_fused = conv.conv(in_features_fused, edge_feats, basis['fully_fused'])
        return unfuse_features(out_fused, lvals)

    elif conv.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(conv, 'conv_out'):
        in_features_fused = torch.cat(
            [features_dict[str(d)] for d in conv.fiber_in.degrees], dim=-1
        )
        out = {}
        for d_out in lvals:
            basis_used = basis[f'out{d_out}_fused']
            result = conv.conv_out[str(d_out)](
                in_features_fused, edge_feats, basis_used
            )
            out_dim = degree_to_dim(d_out)
            out[str(d_out)] = result[..., :out_dim]
        return out

    else:
        # NONE
        out = {}
        for d_out in lvals:
            acc = 0
            for d_in in conv.fiber_in.degrees:
                key = f'{d_in},{d_out}'
                basis_used = basis.get(key, None)
                result = conv.conv[key](
                    features_dict[str(d_in)], edge_feats, basis_used
                )
                if basis_used is not None:
                    out_dim = degree_to_dim(d_out)
                    result = result[..., :out_dim]
                acc = acc + result
            out[str(d_out)] = acc
        return out


# ============================================================================
# Weight extraction (reused from existing prototype)
# ============================================================================

def extract_block_diagonal_weights(
    Lambda: Tensor,
    lvals_in: list[int],
    lvals_out: list[int],
    product_repr: ProductRepr,
) -> Tensor:
    """Extract block-diagonal weights from a full matrix in m-first basis."""
    weight_dim = product_repr.nreps()
    weights = torch.zeros(weight_dim, dtype=Lambda.dtype, device=Lambda.device)

    mmax = max(max(lvals_in), max(lvals_out))
    in_off = 0
    out_off = 0
    w_off = 0

    for m in range(mmax + 1):
        n_in = sum(1 for l in lvals_in if l >= m)
        n_out = sum(1 for l in lvals_out if l >= m)
        if n_in == 0 or n_out == 0:
            continue

        if m == 0:
            block = Lambda[out_off:out_off + n_out, in_off:in_off + n_in]
            for o in range(n_out):
                for i in range(n_in):
                    weights[w_off + o * n_in + i] = block[o, i]
            in_off += n_in
            out_off += n_out
            w_off += n_out * n_in
        else:
            for o in range(n_out):
                for i in range(n_in):
                    a = Lambda[out_off + 2 * o, in_off + 2 * i]
                    b = Lambda[out_off + 2 * o, in_off + 2 * i + 1]
                    weights[w_off + (o * n_in + i) * 2] = a
                    weights[w_off + (o * n_in + i) * 2 + 1] = b
            in_off += 2 * n_in
            out_off += 2 * n_out
            w_off += 2 * n_out * n_in

    assert w_off == weight_dim, f"w_off={w_off} != weight_dim={weight_dim}"
    return weights


# ============================================================================
# Radial weight extraction — supports all fuse levels
# ============================================================================

def _extract_radial_weights_per_pair(
    conv: ConvSE3,
    lvals: list[int],
    bin_edges: Tensor,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Extract per-(d_in, d_out) radial weights from any fuse level.

    Returns dict mapping 'd_in,d_out' -> (num_bins+1, C_out, C_in, num_freq_pair).
    """
    C_in = conv.fiber_in[0]
    C_out = conv.fiber_out[0]
    num_bins_plus1 = bin_edges.shape[0]
    # Determine conv parameter dtype to avoid dtype mismatch in MLP forward
    conv_dtype = next(conv.parameters()).dtype
    edge_input = bin_edges.unsqueeze(-1).to(conv_dtype)
    radial_cache = {}

    with torch.no_grad():
        if conv.used_fuse_level == ConvSE3FuseLevel.NONE:
            for d_in in lvals:
                for d_out in lvals:
                    key = f'{d_in},{d_out}'
                    versatile = conv.conv[key]
                    w = versatile.radial_func(edge_input)
                    radial_cache[key] = w.view(
                        num_bins_plus1, C_out, C_in, versatile.freq_sum
                    )

        elif conv.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(conv, 'conv_out'):
            for d_out in lvals:
                versatile = conv.conv_out[str(d_out)]
                w_fused = versatile.radial_func(edge_input)
                # Raw shape: (B, C_out * C_in * sum_freq)
                # NVIDIA reshapes to (B, C_out, C_in * sum_freq)
                # where the C_in*sum_freq dim has C_in as outer, sum_freq as inner
                sum_freq = versatile.freq_sum
                w_fused = w_fused.view(num_bins_plus1, C_out, C_in, sum_freq)

                # Split along freq axis by d_in
                freq_offset = 0
                for d_in in lvals:
                    nf = degree_to_dim(min(d_in, d_out))
                    key = f'{d_in},{d_out}'
                    radial_cache[key] = w_fused[
                        :, :, :, freq_offset:freq_offset + nf
                    ].contiguous()
                    freq_offset += nf

        elif conv.used_fuse_level == ConvSE3FuseLevel.FULL:
            versatile = conv.conv
            w_fused = versatile.radial_func(edge_input)
            sum_freq = versatile.freq_sum
            w_fused = w_fused.view(num_bins_plus1, C_out, C_in, sum_freq)

            freq_offset = 0
            for d_out in lvals:
                for d_in in lvals:
                    nf = degree_to_dim(min(d_in, d_out))
                    key = f'{d_in},{d_out}'
                    radial_cache[key] = w_fused[
                        :, :, :, freq_offset:freq_offset + nf
                    ].contiguous()
                    freq_offset += nf

        else:
            raise ValueError(f"Unsupported fuse level: {conv.used_fuse_level}")

    return radial_cache


# ============================================================================
# Weight conversion
# ============================================================================

def convert_nvidia_to_flasheq_table(
    conv: ConvSE3,
    product_repr: ProductRepr,
    lvals: list[int],
    num_bins: int,
    min_dist: float,
    max_dist: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Convert NVIDIA ConvSE3 (any fuse level) radial weights to flash-eq table.

    Both the NVIDIA basis and flash-eq's P/Q use sphericart SH convention
    (via monkey-patch), so the conversion is a direct projection:
      1. Evaluate each pair's RadialProfile at bin edge distances
      2. Build dense kernel K(r, e_z) by contracting with CG basis
      3. Project into m-first basis: Lambda = (P^T @ K @ Q)^T
      4. Extract block-diagonal entries
    """
    C_in = conv.fiber_in[0]
    C_out = conv.fiber_out[0]
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = product_repr.nreps()

    # Reference direction e_z
    ez = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)

    # CG basis at e_z (now in sphericart convention via monkey-patch)
    print("  Computing CG basis at e_z (sphericart convention)...")
    basis_ez = get_basis(ez.float(), max_degree=max(lvals), use_pad_trick=False, amp=False)

    # Flash-eq P, Q at e_z (also sphericart convention)
    repr_in = Repr(lvals=lvals, mult=C_in)
    repr_out = Repr(lvals=lvals, mult=C_out)
    basis_computer = WignerDBasis([repr_in, repr_out]).to(device).to(dtype)
    P_ez, Q_ez = basis_computer(ez)
    P_ez = P_ez.squeeze(0)  # (dim, dim)
    Q_ez = Q_ez.squeeze(0)

    # Bin edges
    bin_edges = torch.linspace(min_dist, max_dist, num_bins + 1, device=device, dtype=dtype)

    # Precompute all radial weights at all bin edges (generic for any fuse level)
    print(f"  Evaluating NVIDIA radial MLPs at bin edges (fuse={conv.used_fuse_level.name})...")
    radial_cache = _extract_radial_weights_per_pair(conv, lvals, bin_edges, dtype)
    # Cast to target dtype (conv may be float32, but we project in float64)
    radial_cache = {k: v.to(dtype) for k, v in radial_cache.items()}

    # Precompute basis blocks at e_z
    basis_blocks = {}
    for d_in in lvals:
        for d_out in lvals:
            key = f'{d_in},{d_out}'
            if key in basis_ez:
                B = basis_ez[key].squeeze(0).to(dtype)  # (2*d_in+1, num_freq, 2*d_out+1)
                basis_blocks[key] = B

    # Offsets into dim axis
    offsets = {}
    off = 0
    for l in lvals:
        offsets[l] = off
        off += 2 * l + 1

    # Build weight table
    table = torch.zeros(num_bins + 1, C_out, C_in, weight_dim, device=device, dtype=dtype)

    print("  Projecting kernels into m-first basis...")
    for b in range(num_bins + 1):
        for co in range(C_out):
            for ci in range(C_in):
                K = torch.zeros(dim, dim, device=device, dtype=dtype)

                for d_in in lvals:
                    for d_out in lvals:
                        key = f'{d_in},{d_out}'
                        w = radial_cache[key][b, co, ci, :]  # (num_freq,)

                        if key in basis_blocks:
                            B = basis_blocks[key]
                            K_block = torch.einsum('j, ijl -> il', w, B)
                        else:
                            K_block = w.view(1, 1)

                        in_s = offsets[d_in]
                        out_s = offsets[d_out]
                        in_size = 2 * d_in + 1
                        out_size = 2 * d_out + 1
                        K[in_s:in_s + in_size, out_s:out_s + out_size] = K_block

                # Direct projection (no convention correction needed)
                Lambda = (P_ez.T @ K @ Q_ez).T

                table[b, co, ci, :] = extract_block_diagonal_weights(
                    Lambda, lvals, lvals, product_repr
                )

    return table, bin_edges


# ============================================================================
# Flash-eq forward with injected table
# ============================================================================

def flasheq_forward_with_table(
    table: Tensor,
    layer: EquivariantEdgewiseLinear,
    P: Tensor,
    Q: Tensor,
    features: Tensor,
    distances: Tensor,
) -> Tensor:
    """Run flash-eq forward pass with an injected weight table."""
    rw = layer.radial_weights
    bin_param1, bin_param2 = rw.binning_params()

    f_diag = torch.bmm(features, P)
    out_diag = block_diagonal_cuda(
        f_diag,
        table.to(features.dtype),
        distances,
        layer.product_repr,
        bin_param1=bin_param1,
        bin_param2=bin_param2,
        num_bins=rw.num_bins,
        log_bins=rw.log,
        sh_scale=0.0,
    )
    return torch.bmm(out_diag, Q.mT)


# ============================================================================
# Helper: compute basis for a given fuse level
# ============================================================================

def compute_basis_for_fuse_level(
    directions: Tensor,
    max_degree: int,
    fuse_level: ConvSE3FuseLevel,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Compute NVIDIA CG basis, optionally fusing for PARTIAL/FULL."""
    basis = get_basis(
        directions.float(), max_degree=max_degree,
        use_pad_trick=False, amp=False,
    )
    basis = {k: v.to(dtype) for k, v in basis.items()}

    if fuse_level in (ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL):
        basis = update_basis_with_fused(
            basis, max_degree, use_pad_trick=False,
            fully_fused=(fuse_level == ConvSE3FuseLevel.FULL),
        )

    return basis


# ============================================================================
# Correctness tests
# ============================================================================

def test_correctness():
    """Quick correctness test: small data, float64, all fuse levels."""
    device = torch.device("cuda")
    dtype = torch.float64

    lvals = [0, 1, 2]
    C = 4
    num_edges = 128
    num_bins = 500
    min_dist, max_dist = 0.0, 10.0
    max_degree = max(lvals)

    dim = sum(2 * l + 1 for l in lvals)
    repr_in = Repr(lvals=lvals, mult=C)
    repr_out = Repr(lvals=lvals, mult=C)
    product = ProductRepr(repr_in, repr_out)

    print(f"{'=' * 70}")
    print(f" CORRECTNESS TEST")
    print(f"{'=' * 70}")
    print(f"  lvals={lvals}, dim={dim}, C={C}, edges={num_edges}")
    print(f"  weight_dim={product.nreps()}, num_bins={num_bins}")
    print(f"  SH convention: sphericart (monkey-patched)")

    # Generate test data
    torch.manual_seed(42)
    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 9.0 + 0.5

    features_dict = {}
    for d in lvals:
        features_dict[str(d)] = torch.randn(
            num_edges, C, 2 * d + 1, device=device, dtype=dtype
        )
    features_cat = torch.cat([features_dict[str(d)] for d in lvals], dim=-1)

    # Build rotation infrastructure for equivariance checks
    from flash_eq import random_rotation
    axis, angle = random_rotation(device=device, dtype=dtype)
    wigner_in = WignerD(repr_in).to(device)
    wigner_out = WignerD(repr_out).to(device)
    axis_gen = axis[..., [2, 0, 1]]
    axis_f = axis_gen.float()
    angle_f = angle.float()
    D_in = wigner_in.rot(axis_f.unsqueeze(0), angle_f.unsqueeze(0)).squeeze(0).to(dtype)
    D_out = wigner_out.rot(axis_f.unsqueeze(0), angle_f.unsqueeze(0)).squeeze(0).to(dtype)

    sh_perm = [2, 0, 1]
    D1 = D_in[1:4, 1:4]
    R = D1[sh_perm][:, sh_perm]

    features_cat_rot = features_cat @ D_in.T.unsqueeze(0)
    directions_rot = directions @ R.T

    features_dict_rot = {}
    off = 0
    for d in lvals:
        size = 2 * d + 1
        features_dict_rot[str(d)] = features_cat_rot[:, :, off:off + size]
        off += size

    # Flash-eq layer (shared across fuse levels — weights get injected)
    flasheq = EquivariantEdgewiseLinear(
        repr_in, repr_out,
        num_bins=num_bins,
        min_dist=min_dist,
        max_dist=max_dist,
        solid_harmonic_scale=0.0,
    ).to(device).to(dtype)

    basis_computer = WignerDBasis([repr_in, repr_out]).to(device).to(dtype)
    P, Q = basis_computer(directions)
    P_rot, Q_rot = basis_computer(directions_rot)

    # Reference: store NONE-level output for cross-fuse-level comparison
    reference_nvidia_cat = None

    for fuse_level in [ConvSE3FuseLevel.NONE, ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL]:
        print(f"\n{'-' * 70}")
        print(f"  Fuse level: {fuse_level.name}")
        print(f"{'-' * 70}")

        # Build ConvSE3 at this fuse level
        fiber = Fiber.create(num_degrees=max_degree + 1, num_channels=C)
        conv = ConvSE3(
            fiber_in=fiber,
            fiber_out=fiber,
            fiber_edge=Fiber({}),
            pool=False,
            use_layer_norm=False,
            self_interaction=False,
            max_degree=max_degree,
            fuse_level=fuse_level,
        ).to(device).to(dtype)
        print(f"  Actual fuse level: {conv.used_fuse_level.name}")

        # Compute basis
        basis = compute_basis_for_fuse_level(directions, max_degree, conv.used_fuse_level, dtype)

        # NVIDIA forward
        with torch.no_grad():
            nvidia_out = nvidia_forward(conv, features_dict, distances, basis)
        nvidia_out_cat = torch.cat([nvidia_out[str(d)] for d in lvals], dim=-1)
        print(f"  NVIDIA output norm: {nvidia_out_cat.norm().item():.4f}")

        # Cross-fuse-level check (all fuse levels should give different outputs
        # since they have different random weights, but the structure should match)
        if reference_nvidia_cat is None:
            reference_nvidia_cat = nvidia_out_cat

        # Convert weights to flash-eq table
        table, bin_edges = convert_nvidia_to_flasheq_table(
            conv, product, lvals, num_bins, min_dist, max_dist, device, dtype
        )
        flasheq.radial_weights.register_buffer('bin_edges', bin_edges.float())

        # Flash-eq forward
        with torch.no_grad():
            feq_out = flasheq_forward_with_table(table, flasheq, P, Q, features_cat, distances)

        abs_err = (nvidia_out_cat - feq_out).abs()
        rel_err = abs_err / nvidia_out_cat.abs().clamp(min=1e-15)
        print(f"  Max abs error (NVIDIA vs FEQ): {abs_err.max().item():.2e}")
        print(f"  Mean abs error:                {abs_err.mean().item():.2e}")

        atol = 1e-5
        status = "PASS" if abs_err.max().item() < atol else "FAIL"
        print(f"  Output match: {status} (tol={atol:.0e})")

        # Equivariance check
        basis_rot = compute_basis_for_fuse_level(
            directions_rot, max_degree, conv.used_fuse_level, dtype
        )
        with torch.no_grad():
            nvidia_rot = nvidia_forward(conv, features_dict_rot, distances, basis_rot)
        nvidia_rot_cat = torch.cat([nvidia_rot[str(d)] for d in lvals], dim=-1)

        with torch.no_grad():
            feq_rot = flasheq_forward_with_table(
                table, flasheq, P_rot, Q_rot, features_cat_rot, distances
            )

        nvidia_expected = nvidia_out_cat @ D_out.T.unsqueeze(0)
        feq_expected = feq_out @ D_out.T.unsqueeze(0)

        nv_equiv_err = (nvidia_rot_cat - nvidia_expected).abs().max().item()
        feq_equiv_err = (feq_rot - feq_expected).abs().max().item()
        cross_err = (nvidia_rot_cat - feq_rot).abs().max().item()

        print(f"  NVIDIA equivariance error:  {nv_equiv_err:.2e}")
        print(f"  Flash-eq equivariance error: {feq_equiv_err:.2e}")
        print(f"  Cross-method (rotated):      {cross_err:.2e}")


# ============================================================================
# Benchmarks
# ============================================================================

def clear_memory():
    """Clear CUDA memory and reset peak stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _benchmark_fn(fn, n_warmup=3, n_iter=10):
    """Warmup, then time and measure peak memory."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    clear_memory()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iter):
        fn()
    end_evt.record()
    torch.cuda.synchronize()

    time_ms = start_evt.elapsed_time(end_evt) / n_iter
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    return time_ms, peak_mem_gb


def test_benchmarks():
    """Large-scale GPU memory + runtime benchmarks (fwd + bwd, matching README methodology)."""
    device = torch.device("cuda")
    dtype = torch.float32

    C = 32
    num_bins = 500
    min_dist, max_dist = 0.0, 10.0
    n_warmup = 3
    n_iter = 10

    print(f"\n{'=' * 90}")
    print(f" BENCHMARK: NVIDIA ConvSE3 (all fuse levels) vs Flash-eq — fwd + bwd")
    print(f"{'=' * 90}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  C={C}, num_bins={num_bins}, dtype={dtype}")
    print(f"  warmup={n_warmup}, iterations={n_iter}")

    configs = [
        # (lvals, num_edges)
        ([0, 1], 32_000),
        ([0, 1, 2], 32_000),
        ([0, 1, 2], 50_000),
    ]

    for lvals, num_edges in configs:
        max_degree = max(lvals)
        dim = sum(2 * l + 1 for l in lvals)
        repr_in = Repr(lvals=lvals, mult=C)
        repr_out = Repr(lvals=lvals, mult=C)
        product = ProductRepr(repr_in, repr_out)

        print(f"\n{'=' * 90}")
        print(f"  L={max_degree}, E={num_edges:,}, dim={dim}")
        print(f"{'=' * 90}")

        # Generate test data
        torch.manual_seed(42)
        directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 9.0 + 0.5

        features_dict = {}
        for d in lvals:
            features_dict[str(d)] = torch.randn(
                num_edges, C, 2 * d + 1, device=device, dtype=dtype,
                requires_grad=True,
            )
        features_cat = torch.cat(
            [features_dict[str(d)] for d in lvals], dim=-1
        ).detach().requires_grad_(True)

        target_dict = {}
        for d in lvals:
            target_dict[str(d)] = torch.randn(
                num_edges, C, 2 * d + 1, device=device, dtype=dtype,
            )
        target_cat = torch.cat([target_dict[str(d)] for d in lvals], dim=-1)

        results = []

        # Benchmark each NVIDIA fuse level
        for fuse_level in [ConvSE3FuseLevel.NONE, ConvSE3FuseLevel.PARTIAL, ConvSE3FuseLevel.FULL]:
            clear_memory()

            fiber = Fiber.create(num_degrees=max_degree + 1, num_channels=C)
            conv = ConvSE3(
                fiber_in=fiber,
                fiber_out=fiber,
                fiber_edge=Fiber({}),
                pool=False,
                use_layer_norm=False,
                self_interaction=False,
                max_degree=max_degree,
                fuse_level=fuse_level,
            ).to(device).to(dtype)

            basis = compute_basis_for_fuse_level(
                directions, max_degree, conv.used_fuse_level, dtype
            )
            optimizer = torch.optim.Adam(conv.parameters(), lr=1e-4)

            def nvidia_step(
                conv=conv, fd=features_dict, td=target_dict,
                d=distances, b=basis, opt=optimizer, _lvals=lvals,
            ):
                opt.zero_grad()
                out = nvidia_forward(conv, fd, d, b)
                loss = sum(
                    ((out[str(deg)] - td[str(deg)]) ** 2).mean()
                    for deg in _lvals
                )
                loss.backward()
                opt.step()

            try:
                time_ms, peak_mem_gb = _benchmark_fn(nvidia_step, n_warmup, n_iter)
                label = f"NVIDIA {conv.used_fuse_level.name}"
                results.append((label, time_ms, peak_mem_gb))
                print(f"  {label:<20s}  time={time_ms:8.2f} ms  peak_mem={peak_mem_gb:6.1f} GB")
            except torch.cuda.OutOfMemoryError:
                label = f"NVIDIA {conv.used_fuse_level.name}"
                results.append((label, None, None))
                print(f"  {label:<20s}  OOM")
                clear_memory()

            # Keep NONE conv for weight conversion
            if conv.used_fuse_level == ConvSE3FuseLevel.NONE:
                conv_none = conv

        # Benchmark flash-eq
        clear_memory()

        # Convert weights from NONE conv
        table, bin_edges = convert_nvidia_to_flasheq_table(
            conv_none, product, lvals, num_bins, min_dist, max_dist, device,
            dtype=torch.float64,
        )
        table = table.to(dtype)

        flasheq = EquivariantEdgewiseLinear(
            repr_in, repr_out,
            num_bins=num_bins,
            min_dist=min_dist,
            max_dist=max_dist,
            solid_harmonic_scale=0.0,
        ).to(device).to(dtype)
        flasheq.radial_weights.register_buffer('bin_edges', bin_edges.float())
        feq_optimizer = torch.optim.Adam(flasheq.parameters(), lr=1e-4)

        basis_computer = WignerDBasis([repr_in, repr_out]).to(device).to(dtype)
        P, Q = basis_computer(directions)

        def feq_step():
            feq_optimizer.zero_grad()
            out = flasheq_forward_with_table(
                table, flasheq, P, Q, features_cat, distances
            )
            loss = ((out - target_cat) ** 2).mean()
            loss.backward()
            feq_optimizer.step()

        try:
            time_ms, peak_mem_gb = _benchmark_fn(feq_step, n_warmup, n_iter)
            results.append(("Flash-eq", time_ms, peak_mem_gb))
            print(f"  {'Flash-eq':<20s}  time={time_ms:8.2f} ms  peak_mem={peak_mem_gb:6.1f} GB")
        except torch.cuda.OutOfMemoryError:
            results.append(("Flash-eq", None, None))
            print(f"  {'Flash-eq':<20s}  OOM")
            clear_memory()

        # Summary table
        feq_time, feq_mem = results[-1][1], results[-1][2]
        if feq_time is not None:
            print(f"\n  {'Method':<20s} {'Time (ms)':>10s} {'Mem (GB)':>10s} {'Speedup':>10s} {'Mem savings':>12s}")
            print(f"  {'-' * 62}")
            for label, t, m in results:
                if t is None:
                    print(f"  {label:<20s} {'OOM':>10s}")
                    continue
                speedup = t / feq_time if label != "Flash-eq" else 1.0
                mem_ratio = m / feq_mem if feq_mem > 0 else float('inf')
                print(
                    f"  {label:<20s} {t:10.2f} {m:10.1f} "
                    f"{speedup:10.2f}x {mem_ratio:12.1f}x"
                )


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NVIDIA SE(3)-T → Flash-eq weight conversion prototype"
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true",
        help="Run GPU memory/runtime benchmarks",
    )
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip correctness tests",
    )
    args = parser.parse_args()

    if not args.skip_correctness:
        test_correctness()

    if args.benchmark:
        test_benchmarks()
    elif not args.skip_correctness:
        print(f"\n{'=' * 70}")
        print("  Run with --benchmark (-b) to include GPU memory/runtime benchmarks.")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
