"""
SH/CG convention alignment for NVIDIA SE(3)-Transformer patching.

Replaces NVIDIA's e3nn-based spherical harmonic and Clebsch-Gordan
computations with sphericart-convention equivalents. This ensures the
NVIDIA CG basis and flash-eq's Wigner-D matrices share the same SH
convention, making weight conversion a direct projection.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import sys
from functools import lru_cache
from types import ModuleType
from typing import Dict, List

import torch
from torch import Tensor

from flash_eq import lebedev_grid, real_spherical_harmonics


def _real_clebsch_gordan(l1: int, l2: int, l3: int) -> Tensor:
    """Compute real CG coefficients via Lebedev quadrature (sphericart convention).

    CG[m1, m2, m3] = integral Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} dOmega
    """
    points, weights = lebedev_grid(precision=131)
    points = points.to(torch.float64)
    weights = weights.to(torch.float64)

    lmax = max(l1, l2, l3)
    Y = real_spherical_harmonics(lmax, points).to(torch.float64)

    def get_Y(l: int) -> Tensor:
        return Y[:, l * l : (l + 1) ** 2]

    return torch.einsum("n,na,nb,nc->abc", weights, get_Y(l1), get_Y(l2), get_Y(l3))


@lru_cache(maxsize=None)
def _get_clebsch_gordon_sc(J: int, d_in: int, d_out: int, device: object) -> Tensor:
    """Sphericart-convention replacement for NVIDIA's get_clebsch_gordon.

    Returns shape (2*d_out+1, 2*d_in+1, 2*J+1) to match NVIDIA's index order.
    """
    cg = _real_clebsch_gordan(J, d_in, d_out)  # (2J+1, 2d_in+1, 2d_out+1)
    return cg.permute(2, 1, 0).to(device=device)  # (2d_out+1, 2d_in+1, 2J+1)


@lru_cache(maxsize=None)
def _get_all_clebsch_gordon_sc(max_degree: int, device: object) -> List[List[Tensor]]:
    """Sphericart-convention replacement for NVIDIA's get_all_clebsch_gordon."""
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(_get_clebsch_gordon_sc(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def _get_spherical_harmonics_sc(
    relative_pos: Tensor, max_degree: int
) -> List[Tensor]:
    """Sphericart-convention replacement for NVIDIA's get_spherical_harmonics."""
    all_degrees = list(range(2 * max_degree + 1))
    sh = real_spherical_harmonics(max(all_degrees), relative_pos.to(torch.float64))
    sh = sh.to(relative_pos.dtype)
    return [sh[:, d * d : (d + 1) ** 2] for d in all_degrees]


def _get_basis_no_jit(
    max_degree: int,
    use_pad_trick: bool,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
    amp: bool,
) -> Dict[str, Tensor]:
    """Non-JIT replacement for NVIDIA's get_basis_script."""
    import torch.nn.functional as F

    basis: Dict[str, Tensor] = {}
    idx = 0
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f"{d_in},{d_out}"
            K_Js = []
            for freq_idx, J in enumerate(
                range(abs(d_in - d_out), d_in + d_out + 1)
            ):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(
                    torch.einsum(
                        "n f, k l f -> n l k",
                        spherical_harmonics[J].float(),
                        Q_J.float(),
                    )
                )
            basis[key] = torch.stack(K_Js, 2)
            if amp:
                basis[key] = basis[key].half()
            if use_pad_trick:
                basis[key] = F.pad(basis[key], (0, 1))
            idx += 1
    return basis


def apply_convention_patches(conv_cls: type) -> ModuleType:
    """Monkey-patch the NVIDIA basis module to use sphericart SH conventions.

    Finds the basis module from the ConvSE3 class's package and replaces
    ``get_clebsch_gordon``, ``get_all_clebsch_gordon``,
    ``get_spherical_harmonics``, and ``get_basis_script`` with
    sphericart-convention equivalents.

    Args:
        conv_cls: The ConvSE3 class found in the model.

    Returns:
        The patched basis module (for calling ``get_basis()``).
    """
    # conv_cls.__module__ is e.g. 'se3_transformer.model.layers.convolution'
    # We need 'se3_transformer.model.basis'
    mod_path = conv_cls.__module__
    # Go up from .model.layers.convolution to .model
    parts = mod_path.split(".")
    # Find 'model' in the path and build basis module path
    try:
        model_idx = parts.index("model")
    except ValueError:
        raise ValueError(
            f"Cannot locate basis module from ConvSE3 at {mod_path}. "
            f"Expected 'model' in the module path."
        )
    basis_path = ".".join(parts[: model_idx + 1]) + ".basis"

    if basis_path not in sys.modules:
        raise ValueError(
            f"NVIDIA basis module '{basis_path}' not found in sys.modules. "
            f"Ensure the SE(3)-Transformer model is fully imported."
        )

    basis_mod = sys.modules[basis_path]

    basis_mod.get_clebsch_gordon = _get_clebsch_gordon_sc
    basis_mod.get_all_clebsch_gordon = _get_all_clebsch_gordon_sc
    basis_mod.get_spherical_harmonics = _get_spherical_harmonics_sc
    basis_mod.get_basis_script = _get_basis_no_jit

    return basis_mod
