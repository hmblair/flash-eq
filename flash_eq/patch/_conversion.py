"""
Weight extraction and table conversion for NVIDIA ConvSE3 → flash-eq.

Evaluates the NVIDIA RadialProfile MLPs at bin-edge distances, builds
dense kernels using CG basis at e_z, projects into flash-eq's m-first
basis via Wigner-D matrices, and extracts block-diagonal entries.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn
from torch import Tensor

from flash_eq import ProductRepr, Repr, WignerDBasis


def _degree_to_dim(d: int) -> int:
    """Compute 2*d + 1."""
    return 2 * d + 1


def _extract_radial_weights_per_pair(
    conv: nn.Module,
    lvals: list[int],
    bin_edges: Tensor,
) -> dict[str, Tensor]:
    """Extract per-(d_in, d_out) radial weights from any fuse level.

    Returns dict mapping ``'d_in,d_out'`` to tensors of shape
    ``(num_bins+1, C_out, C_in, num_freq_pair)``.

    Handles NONE, PARTIAL (by output), and FULL fuse levels.
    """
    C_in = conv.fiber_in[0]
    C_out = conv.fiber_out[0]
    num_bins_plus1 = bin_edges.shape[0]
    conv_dtype = next(conv.parameters()).dtype
    edge_input = bin_edges.unsqueeze(-1).to(conv_dtype)
    radial_cache: dict[str, Tensor] = {}

    fuse_name = conv.used_fuse_level.name

    with torch.no_grad():
        if fuse_name == "NONE":
            for d_in in lvals:
                for d_out in lvals:
                    key = f"{d_in},{d_out}"
                    versatile = conv.conv[key]
                    w = versatile.radial_func(edge_input)
                    radial_cache[key] = w.view(
                        num_bins_plus1, C_out, C_in, versatile.freq_sum
                    )

        elif fuse_name == "PARTIAL" and hasattr(conv, "conv_out"):
            for d_out in lvals:
                versatile = conv.conv_out[str(d_out)]
                w_fused = versatile.radial_func(edge_input)
                sum_freq = versatile.freq_sum
                w_fused = w_fused.view(num_bins_plus1, C_out, C_in, sum_freq)

                freq_offset = 0
                for d_in in lvals:
                    nf = _degree_to_dim(min(d_in, d_out))
                    key = f"{d_in},{d_out}"
                    radial_cache[key] = w_fused[
                        :, :, :, freq_offset : freq_offset + nf
                    ].contiguous()
                    freq_offset += nf

        elif fuse_name == "FULL":
            versatile = conv.conv
            w_fused = versatile.radial_func(edge_input)
            sum_freq = versatile.freq_sum
            w_fused = w_fused.view(num_bins_plus1, C_out, C_in, sum_freq)

            freq_offset = 0
            for d_out in lvals:
                for d_in in lvals:
                    nf = _degree_to_dim(min(d_in, d_out))
                    key = f"{d_in},{d_out}"
                    radial_cache[key] = w_fused[
                        :, :, :, freq_offset : freq_offset + nf
                    ].contiguous()
                    freq_offset += nf

        else:
            raise ValueError(
                f"Unsupported fuse level: {conv.used_fuse_level}. "
                f"Expected NONE, PARTIAL (conv_out), or FULL."
            )

    return radial_cache


def _extract_block_diagonal_weights(
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
            block = Lambda[out_off : out_off + n_out, in_off : in_off + n_in]
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


def convert_conv_to_table(
    conv: nn.Module,
    basis_mod: ModuleType,
    num_bins: int,
    min_dist: float,
    max_dist: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, ProductRepr, Repr, Repr]:
    """Convert an NVIDIA ConvSE3 module to a flash-eq weight table.

    Both the NVIDIA basis and flash-eq's P/Q must use the same SH
    convention (via :func:`~flash_eq.patch._convention.apply_convention_patches`).

    The conversion pipeline:
      1. Evaluate each pair's RadialProfile MLP at bin-edge distances
      2. Build dense kernel K(r, e_z) by contracting with CG basis
      3. Project into m-first basis: ``Lambda = (P^T @ K @ Q)^T``
      4. Extract block-diagonal entries

    Args:
        conv: NVIDIA ``ConvSE3`` module (any fuse level).
        basis_mod: The patched NVIDIA basis module (from
            :func:`apply_convention_patches`).
        num_bins: Number of distance bins for the weight table.
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        device: Device for computation.
        dtype: Precision for the projection (float64 recommended).

    Returns:
        ``(table, product_repr, in_repr, out_repr)`` where *table* has
        shape ``(num_bins + 1, C_out, C_in, weight_dim)``.
    """
    lvals_in = list(conv.fiber_in.degrees)
    lvals_out = list(conv.fiber_out.degrees)
    C_in = conv.fiber_in[0]
    C_out = conv.fiber_out[0]
    dim_in = sum(2 * l + 1 for l in lvals_in)
    dim_out = sum(2 * l + 1 for l in lvals_out)

    in_repr = Repr(lvals=lvals_in, mult=C_in)
    out_repr = Repr(lvals=lvals_out, mult=C_out)
    product_repr = ProductRepr(in_repr, out_repr)
    weight_dim = product_repr.nreps()

    # Reference direction e_z
    ez = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)

    # CG basis at e_z (sphericart convention via monkey-patch)
    max_degree = max(max(lvals_in), max(lvals_out))
    basis_ez = basis_mod.get_basis(
        ez.float(), max_degree=max_degree, use_pad_trick=False, amp=False
    )

    # Flash-eq P, Q at e_z (also sphericart convention)
    basis_computer = WignerDBasis([in_repr, out_repr]).to(device).to(dtype)
    P_ez, Q_ez = basis_computer(ez)
    P_ez = P_ez.squeeze(0)  # (dim_in, dim_in)
    Q_ez = Q_ez.squeeze(0)  # (dim_out, dim_out)

    # Bin edges
    bin_edges = torch.linspace(
        min_dist, max_dist, num_bins + 1, device=device, dtype=dtype
    )

    # Evaluate NVIDIA radial MLPs at bin edges (supports all fuse levels)
    # Use all input degrees for extraction (the MLP was trained on these)
    lvals_for_extraction = list(range(max_degree + 1))
    radial_cache = _extract_radial_weights_per_pair(
        conv, lvals_for_extraction, bin_edges
    )
    radial_cache = {k: v.to(dtype) for k, v in radial_cache.items()}

    # Precompute basis blocks at e_z
    basis_blocks: dict[str, Tensor] = {}
    for d_in in lvals_for_extraction:
        for d_out in lvals_for_extraction:
            key = f"{d_in},{d_out}"
            if key in basis_ez:
                B = basis_ez[key].squeeze(0).to(dtype)
                basis_blocks[key] = B

    # Offsets into dim axes
    offsets_in: dict[int, int] = {}
    off = 0
    for l in lvals_in:
        offsets_in[l] = off
        off += 2 * l + 1

    offsets_out: dict[int, int] = {}
    off = 0
    for l in lvals_out:
        offsets_out[l] = off
        off += 2 * l + 1

    # Build weight table
    table = torch.zeros(
        num_bins + 1, C_out, C_in, weight_dim, device=device, dtype=dtype
    )

    for b in range(num_bins + 1):
        for co in range(C_out):
            for ci in range(C_in):
                K = torch.zeros(dim_in, dim_out, device=device, dtype=dtype)

                for d_in in lvals_in:
                    for d_out in lvals_out:
                        key = f"{d_in},{d_out}"
                        w = radial_cache[key][b, co, ci, :]

                        if key in basis_blocks:
                            B = basis_blocks[key]
                            K_block = torch.einsum("j, ijl -> il", w, B)
                        else:
                            K_block = w.view(1, 1)

                        in_s = offsets_in[d_in]
                        out_s = offsets_out[d_out]
                        in_size = 2 * d_in + 1
                        out_size = 2 * d_out + 1
                        K[in_s : in_s + in_size, out_s : out_s + out_size] = K_block

                Lambda = (P_ez.T @ K @ Q_ez).T

                table[b, co, ci, :] = _extract_block_diagonal_weights(
                    Lambda, lvals_in, lvals_out, product_repr
                )

    return table, product_repr, in_repr, out_repr
