"""Shared test helpers for flash-eq tests."""

import math
import torch
from scipy.spatial.transform import Rotation


def random_rotation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random SO(3) rotation.

    Returns:
        axis: (1, 3) rotation axis (unit vector)
        angle: (1,) rotation angle in radians
        R: (3, 3) Cartesian rotation matrix
    """
    axis_np = torch.randn(3).numpy()
    axis_np = axis_np / (axis_np ** 2).sum() ** 0.5
    angle_np = float(torch.rand(1) * 2 * math.pi)

    rotvec = axis_np * angle_np
    R_np = Rotation.from_rotvec(rotvec).as_matrix()

    axis = torch.tensor(axis_np, device=device, dtype=torch.float32).unsqueeze(0)
    angle = torch.tensor([angle_np], device=device, dtype=torch.float32)
    R = torch.tensor(R_np, device=device, dtype=torch.float32)

    return axis, angle, R


def make_graph(
    num_nodes: int,
    num_edges: int,
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random graph edge indices.

    Returns:
        src_indices: (num_edges,) source node for each edge
        dst_indices: (num_edges,) destination node for each edge
    """
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    return src, dst


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
