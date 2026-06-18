"""
Patch NVIDIA SE(3)-Transformers for memory-efficient inference with flash-eq.

Usage::

    from flash_eq.patch import patch

    model = patch(model, num_bins=500, max_dist=10.0)

The ``patch()`` function converts all ``ConvSE3`` layers in-place to use
flash-eq's block-diagonal CUDA kernel with binned weight tables, replacing
the memory-heavy per-edge CG basis tensors with compact Wigner-D matrices.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import types
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from flash_eq import Repr, WignerDBasis

from ._convention import apply_convention_patches
from ._conversion import convert_conv_to_table
from ._patched_conv import PatchedConvSE3

__all__ = ["patch"]


def _find_modules_by_classname(
    model: nn.Module, classname: str
) -> list[tuple[str, nn.Module]]:
    """Find all submodules whose class name matches *classname*."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if type(mod).__name__ == classname
    ]


def _replace_module(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    """Replace a submodule at *dotted_name* inside *root*."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_mod)


def _validate_conv(conv: nn.Module, name: str) -> None:
    """Check that a ConvSE3 is compatible with flash-eq conversion."""
    # Check edge_dim == 1 (distance only)
    fuse_name = conv.used_fuse_level.name
    if fuse_name == "FULL":
        radial_net = conv.conv.radial_func.net
    elif fuse_name == "PARTIAL" and hasattr(conv, "conv_out"):
        first_key = next(iter(conv.conv_out))
        radial_net = conv.conv_out[first_key].radial_func.net
    elif fuse_name == "PARTIAL" and hasattr(conv, "conv_in"):
        raise ValueError(
            f"ConvSE3 '{name}' uses PARTIAL fuse by input degree, "
            f"which is not supported. Only NONE, PARTIAL (by output), "
            f"and FULL fuse levels are supported."
        )
    else:  # NONE
        first_key = next(iter(conv.conv))
        radial_net = conv.conv[first_key].radial_func.net

    # Find first Linear layer to check input dimension
    for layer in radial_net:
        if isinstance(layer, nn.Linear):
            if layer.in_features != 1:
                raise ValueError(
                    f"ConvSE3 '{name}' has edge_dim={layer.in_features} "
                    f"(expected 1). Only distance-only radial MLPs are "
                    f"supported. Additional invariant edge features cannot "
                    f"be converted to binned tables."
                )
            break

    # Check uniform channel count across degrees
    channels = set(conv.fiber_in.channels)
    if len(channels) > 1:
        raise ValueError(
            f"ConvSE3 '{name}' has non-uniform input channels "
            f"{dict(zip(conv.fiber_in.degrees, conv.fiber_in.channels))}. "
            f"All degrees must have the same channel count."
        )
    channels = set(conv.fiber_out.channels)
    if len(channels) > 1:
        raise ValueError(
            f"ConvSE3 '{name}' has non-uniform output channels "
            f"{dict(zip(conv.fiber_out.degrees, conv.fiber_out.channels))}. "
            f"All degrees must have the same channel count."
        )


def _get_populated_edge_features(
    relative_pos: Tensor,
    edge_features: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """Add relative position norms to edge features.

    Reimplementation of NVIDIA's ``get_populated_edge_features`` to
    avoid importing from the NVIDIA package at inference time.
    """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], r[..., None]], dim=1)
    else:
        edge_features["0"] = r[..., None]
    return edge_features


def patch(
    model: nn.Module,
    num_bins: int = 500,
    min_dist: float = 0.0,
    max_dist: float = 10.0,
    dtype: torch.dtype = torch.float64,
) -> nn.Module:
    """Patch an NVIDIA SE(3)-Transformer for memory-efficient inference.

    Converts all ``ConvSE3`` layers to use flash-eq's block-diagonal
    CUDA kernel. The model produces identical outputs within ~1e-4
    interpolation tolerance (with 500 bins).

    Args:
        model: Trained NVIDIA SE(3)-Transformer module.
        num_bins: Number of distance bins (more bins = higher accuracy).
        min_dist: Minimum distance for radial weight binning.
        max_dist: Maximum distance for radial weight binning.
        dtype: Precision for weight conversion (float64 recommended for
            accuracy; the resulting tables are stored in the model's
            original dtype).

    Returns:
        The patched model (modified in-place).

    Raises:
        ValueError: If no ``ConvSE3`` modules are found, or if any
            convolution has unsupported configuration (edge_dim > 1,
            PARTIAL fuse by input degree).
    """
    # Find all ConvSE3 modules
    convs = _find_modules_by_classname(model, "ConvSE3")
    if not convs:
        raise ValueError(
            "No ConvSE3 modules found in model. "
            "Ensure the model is an NVIDIA SE(3)-Transformer."
        )

    # Get the ConvSE3 class and apply convention patches
    conv_cls = type(convs[0][1])
    basis_mod = apply_convention_patches(conv_cls)

    # Determine device from model parameters
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Validate all convolutions before converting any
    for name, conv in convs:
        _validate_conv(conv, name)

    # Collect all unique lvals for WignerDBasis
    all_lvals: set[tuple[int, ...]] = set()
    for _, conv in convs:
        all_lvals.add(tuple(conv.fiber_in.degrees))
        all_lvals.add(tuple(conv.fiber_out.degrees))

    # Build WignerDBasis with all unique representations
    unique_reprs = [Repr(lvals=list(lv), mult=1) for lv in sorted(all_lvals)]
    wigner_basis = WignerDBasis(unique_reprs).to(device)

    # Create a mapping from lvals tuple -> index in WignerDBasis output
    lvals_to_idx: dict[tuple[int, ...], int] = {}
    for i, r in enumerate(unique_reprs):
        lvals_to_idx[tuple(r.lvals.tolist())] = i

    # Convert each ConvSE3 and replace
    print(f"Patching {len(convs)} ConvSE3 module(s)...")
    for name, conv in convs:
        print(f"  Converting {name} (fuse={conv.used_fuse_level.name})...")
        table, product_repr, in_repr, out_repr = convert_conv_to_table(
            conv,
            basis_mod,
            num_bins=num_bins,
            min_dist=min_dist,
            max_dist=max_dist,
            device=device,
            dtype=dtype,
        )

        # Store table in model's original dtype
        table = table.to(model_dtype)

        patched = PatchedConvSE3(
            original_conv=conv,
            table=table,
            product_repr=product_repr,
            in_repr=in_repr,
            out_repr=out_repr,
            num_bins=num_bins,
            min_dist=min_dist,
            max_dist=max_dist,
        )
        _replace_module(model, name, patched)

    # Patch SE3Transformer forward (if found)
    transformers = _find_modules_by_classname(model, "SE3Transformer")
    # Also check if the model itself is an SE3Transformer
    if type(model).__name__ == "SE3Transformer":
        transformers.append(("", model))
    # Also check for SE3TransformerPooled which wraps SE3Transformer
    pooled = _find_modules_by_classname(model, "SE3TransformerPooled")
    for pname, pmod in pooled:
        if hasattr(pmod, "transformer"):
            tname = f"{pname}.transformer" if pname else "transformer"
            if not any(n == tname for n, _ in transformers):
                transformers.append((tname, pmod.transformer))

    for tname, transformer in transformers:
        _patch_transformer_forward(transformer, wigner_basis, lvals_to_idx)

    print("Patching complete.")
    return model


def _patch_transformer_forward(
    transformer: nn.Module,
    wigner_basis: WignerDBasis,
    lvals_to_idx: dict[tuple[int, ...], int],
) -> None:
    """Replace an SE3Transformer's forward to compute P/Q instead of CG basis."""
    # Attach the WignerDBasis as a submodule so .to() works
    transformer._flasheq_basis = wigner_basis

    # Determine which P/Q indices to use
    # In standard SE3Transformer, all convolutions share the same lvals
    max_degree = transformer.max_degree
    lvals_key = tuple(range(max_degree + 1))
    idx = lvals_to_idx.get(lvals_key)
    if idx is None:
        raise ValueError(
            f"Cannot find WignerDBasis index for lvals={lvals_key}. "
            f"Available: {list(lvals_to_idx.keys())}"
        )
    transformer._flasheq_basis_idx = idx

    def patched_forward(
        self,
        graph: object,
        node_feats: Dict[str, Tensor],
        edge_feats: Optional[Dict[str, Tensor]] = None,
        basis: Optional[Dict[str, Tensor]] = None,
    ) -> object:
        rel_pos = graph.edata["rel_pos"]
        distances = rel_pos.norm(dim=-1)
        directions = rel_pos / distances.unsqueeze(-1).clamp(min=1e-8)

        # Compute Wigner-D matrices (replaces CG basis computation)
        all_matrices = self._flasheq_basis(directions)
        M = all_matrices[self._flasheq_basis_idx]
        basis = {"_P": M, "_Q": M, "_distances": distances}

        # Populate edge features (unchanged)
        edge_feats = _get_populated_edge_features(rel_pos, edge_feats)

        # Run graph modules (NormSE3 ignores basis via *args/**kwargs)
        node_feats = self.graph_modules(
            node_feats, edge_feats, graph=graph, basis=basis
        )

        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)
        if self.return_type is not None:
            return node_feats[str(self.return_type)]
        return node_feats

    transformer.forward = types.MethodType(patched_forward, transformer)
