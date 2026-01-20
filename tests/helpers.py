"""Shared test helpers for flash-eq tests."""

import torch

from flash_eq import Graph, Repr, WignerD
from flash_eq import random_rotation as _random_axis_angle


# Cached WignerD for l=1 (3D rotation matrices)
_WIGNER_L1: WignerD | None = None


def _get_wigner_l1() -> WignerD:
    """Get cached WignerD for l=1."""
    global _WIGNER_L1
    if _WIGNER_L1 is None:
        _WIGNER_L1 = WignerD(Repr([1]))
    return _WIGNER_L1


def random_rotation(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random SO(3) rotation.

    Returns:
        axis: (3,) rotation axis (unit vector)
        angle: () rotation angle in radians
        R: (3, 3) Cartesian rotation matrix
    """
    axis, angle = _random_axis_angle(device=device, dtype=dtype)

    # Get 3x3 rotation matrix using WignerD with l=1
    wigner = _get_wigner_l1().to(device)
    R = wigner.rot(axis, angle, cartesian=True)

    return axis, angle, R


def make_graph(
    num_nodes: int,
    num_edges: int,
    device: torch.device | str = "cpu",
) -> Graph:
    """Create random graph (may include self-loops).

    Returns:
        Graph with random edges.
    """
    return Graph.random(num_nodes, num_edges, device=device)


def check_equivariance(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-4,
    msg: str = "Equivariance check failed",
) -> None:
    """Check equivariance using relative difference.

    Args:
        output1: First output tensor (e.g., rotated after forward)
        output2: Second output tensor (e.g., forward after rotated input)
        rtol: Relative tolerance
        msg: Error message prefix
    """
    rel_diff = (output1 - output2).abs().max().item() / (output1.abs().max().item() + 1e-8)
    assert rel_diff < rtol, f"{msg}: rel_diff={rel_diff:.2e} (threshold={rtol:.0e})"
