"""
Equivariance tests for flash-eq.

Tests that the layer satisfies SO(3)-equivariance:
    D_out @ f(x) = f(D_in @ x)

where D_in, D_out are Wigner-D rotation matrices.
"""

import math
import numpy as np
import torch
import pytest
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq import Repr, EquivariantEdgewiseLinear, WignerDBasis


def random_rotation(device: torch.device, use_identity: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random rotation.

    Returns:
        axis: (1, 3) rotation axis
        angle: (1,) rotation angle
        R: (3, 3) scipy-computed rotation matrix
    """
    if use_identity:
        # Identity rotation for debugging
        axis_np = np.array([0., 0., 1.])
        angle_np = 0.0
    else:
        # Generate random axis-angle
        axis_np = torch.randn(3).numpy()
        axis_np = axis_np / (axis_np ** 2).sum() ** 0.5
        angle_np = float(torch.rand(1) * 2 * math.pi)

    # Use scipy to compute rotation matrix
    rotvec = axis_np * angle_np
    R_np = Rotation.from_rotvec(rotvec).as_matrix()

    # Convert to torch
    axis = torch.tensor(axis_np, device=device, dtype=torch.float32).unsqueeze(0)
    angle = torch.tensor([angle_np], device=device, dtype=torch.float32)
    R = torch.tensor(R_np, device=device, dtype=torch.float32)

    return axis, angle, R


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_equivariance(layer, basis, in_repr, node_features, src_indices, distances, directions, device, use_identity=False):
    """
    Check equivariance for a random rotation.
    Returns (output_magnitude, relative_diff).
    """
    axis, angle, R = random_rotation(device, use_identity=use_identity)
    # D is the Wigner-D matrix for features
    D = in_repr.rot(axis, angle).squeeze(0)

    P, Q = basis(directions)

    # Method 1: Apply layer, then rotate output
    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

    # Method 2: Rotate input features and directions, then apply layer
    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    output_magnitude = output1.abs().mean().item()
    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)

    return output_magnitude, rel_diff


def test_equivariance_identity(device):
    """Test that identity rotation gives zero diff (sanity check)."""
    torch.manual_seed(42)

    lvals = [0, 1, 2]
    mult = 8
    num_nodes = 50
    num_edges = 200
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    node_features = torch.randn(num_nodes, mult, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    output_mag, rel_diff = check_equivariance(
        layer, basis, in_repr, node_features, src_indices, distances, directions, device,
        use_identity=True
    )

    # Output should be non-trivial
    assert output_mag > 0.01, f"Output too small: {output_mag:.2e}"
    # Identity should give zero diff
    assert rel_diff < 1e-5, f"Identity rotation failed: rel_diff={rel_diff:.2e}"


def test_equivariance_basic(device):
    """Test basic equivariance: D @ layer(x) = layer(D @ x)."""
    torch.manual_seed(42)

    lvals = [0, 1, 2]
    mult = 8
    num_nodes = 50
    num_edges = 200
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    node_features = torch.randn(num_nodes, mult, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    output_mag, rel_diff = check_equivariance(
        layer, basis, in_repr, node_features, src_indices, distances, directions, device
    )

    # Output should be non-trivial
    assert output_mag > 0.01, f"Output too small: {output_mag:.2e}"
    # Equivariance check
    assert rel_diff < 1e-4, f"Equivariance failed: rel_diff={rel_diff:.2e}"


def test_equivariance_multiple_rotations(device):
    """Test equivariance with multiple random rotations."""
    torch.manual_seed(123)

    lvals = [0, 1, 2, 3]
    mult = 8
    num_nodes = 30
    num_edges = 100
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    node_features = torch.randn(num_nodes, mult, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    for i in range(5):
        output_mag, rel_diff = check_equivariance(
            layer, basis, in_repr, node_features, src_indices, distances, directions, device
        )
        assert rel_diff < 1e-4, f"Rotation {i+1} failed: rel_diff={rel_diff:.2e}"


def test_equivariance_different_channels(device):
    """Test equivariance with different input/output channels."""
    torch.manual_seed(456)

    lvals = [0, 1, 2]
    mult_in = 16
    mult_out = 32
    num_nodes = 30
    num_edges = 100
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult_in).to(device)
    out_repr = Repr(lvals=lvals, mult=mult_out).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    node_features = torch.randn(num_nodes, mult_in, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    axis, angle, R = random_rotation(device)
    D_in = in_repr.rot(axis, angle).squeeze(0)
    D_out = out_repr.rot(axis, angle).squeeze(0)

    P, Q = basis(directions)

    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

    node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
    assert rel_diff < 1e-4, f"Equivariance failed: rel_diff={rel_diff:.2e}"


def test_equivariance_high_lmax(device):
    """Test equivariance with high angular momentum (lmax=5)."""
    torch.manual_seed(789)

    lvals = [0, 1, 2, 3, 4, 5]
    mult = 4
    num_nodes = 20
    num_edges = 50
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    node_features = torch.randn(num_nodes, mult, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    output_mag, rel_diff = check_equivariance(
        layer, basis, in_repr, node_features, src_indices, distances, directions, device
    )
    assert rel_diff < 1e-4, f"Equivariance failed: rel_diff={rel_diff:.2e}"


def test_equivariance_gradient(device):
    """Test that gradients are also equivariant."""
    torch.manual_seed(101)

    lvals = [0, 1, 2]
    mult = 4
    num_nodes = 10
    num_edges = 30
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    axis, angle, R = random_rotation(device)
    D = in_repr.rot(axis, angle).squeeze(0)

    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    P, Q = basis(directions)

    # Method 1: forward then rotate, compute gradient
    node_features1 = torch.randn(num_nodes, mult, dim, device=device, requires_grad=True)
    output1 = layer(P, Q, node_features1, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)
    loss1 = output1_rotated.sum()
    loss1.backward()
    grad1 = node_features1.grad.clone()

    # Sanity checks
    assert output1.abs().mean().item() > 1e-6, "Output too small"
    assert grad1.abs().mean().item() > 1e-6, "Gradient too small"

    # Method 2: rotate then forward, compute gradient
    node_features2 = node_features1.detach().clone().requires_grad_(True)

    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    node_features2_rotated = torch.einsum('ij,ncj->nci', D, node_features2)
    output2 = layer(P_rot, Q_rot, node_features2_rotated, distances, src_indices)
    loss2 = output2.sum()
    loss2.backward()

    # Compare rotated grad1 vs grad2
    grad1_rotated = torch.einsum('ij,ncj->nci', D, grad1)
    rel_diff = (grad1_rotated - node_features2.grad).abs().max().item() / (grad1_rotated.abs().max().item() + 1e-8)

    assert rel_diff < 1e-4, f"Gradient equivariance failed: rel_diff={rel_diff:.2e}"
