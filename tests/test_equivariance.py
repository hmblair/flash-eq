"""
Equivariance tests for flash-eq.

Tests that the layer satisfies SO(3)-equivariance:
    D_out @ f(x, d) = f(D_in @ x, R @ d)

where D_in, D_out are Wigner-D rotation matrices and R is the 3x3 rotation matrix.
"""

import math
import numpy as np
import torch
import pytest
from scipy.spatial.transform import Rotation

from flash_eq import Repr, WignerD, EquivariantEdgewiseLinear, WignerDBasis


def random_rotation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random rotation.

    Returns:
        axis: (1, 3) rotation axis
        angle: (1,) rotation angle
        R: (3, 3) rotation matrix
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


# Test configurations: (lvals_in, lvals_out)
LVALS_CONFIGS = [
    # Same in/out - single l
    ([0], [0]),
    ([1], [1]),
    ([2], [2]),
    # Same in/out - multiple l
    ([0, 1], [0, 1]),
    ([0, 1, 2], [0, 1, 2]),
    ([1, 2], [1, 2]),
    ([0, 2], [0, 2]),  # non-contiguous
    ([1, 2, 3], [1, 2, 3]),
    # Different in/out
    ([0], [0, 1]),
    ([0, 1], [0, 1, 2]),
    ([1], [0, 1]),
    ([1, 2], [0, 1, 2]),
    ([0, 1, 2], [1, 2]),
    ([2], [0, 1, 2]),
    ([0, 1], [2]),
]


@pytest.mark.parametrize("lvals_in,lvals_out", LVALS_CONFIGS)
def test_equivariance(cuda_device, lvals_in, lvals_out):
    """Test SO(3) equivariance for various representation configurations."""
    torch.manual_seed(42)

    mult = 4
    num_nodes = 20
    num_edges = 50
    dim_in = sum(2 * l + 1 for l in lvals_in)
    dim_out = sum(2 * l + 1 for l in lvals_out)

    in_repr = Repr(lvals=lvals_in, mult=mult)
    out_repr = Repr(lvals=lvals_out, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis(in_repr, out_repr).to(cuda_device)
    wigner_in = WignerD(in_repr).to(cuda_device)
    wigner_out = WignerD(out_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim_in, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Random rotation
    axis, angle, R = random_rotation(cuda_device)
    D_in = wigner_in.rot(axis, angle).squeeze(0)
    D_out = wigner_out.rot(axis, angle).squeeze(0)

    P, Q = basis(directions)

    # Method 1: Apply layer, then rotate output
    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

    # Method 2: Rotate input features and directions, then apply layer
    node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    # Check output is non-trivial
    output_mag = output1.abs().mean().item()
    assert output_mag > 0.01, f"Output too small: {output_mag:.2e}"

    # Check equivariance
    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
    assert rel_diff < 1e-4, f"Equivariance failed: rel_diff={rel_diff:.2e}"


@pytest.mark.parametrize("lvals_in,lvals_out", LVALS_CONFIGS[:5])  # Subset for speed
def test_equivariance_multiple_rotations(cuda_device, lvals_in, lvals_out):
    """Test equivariance with multiple random rotations."""
    torch.manual_seed(123)

    mult = 4
    num_nodes = 20
    num_edges = 50
    dim_in = sum(2 * l + 1 for l in lvals_in)
    dim_out = sum(2 * l + 1 for l in lvals_out)

    in_repr = Repr(lvals=lvals_in, mult=mult)
    out_repr = Repr(lvals=lvals_out, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis(in_repr, out_repr).to(cuda_device)
    wigner_in = WignerD(in_repr).to(cuda_device)
    wigner_out = WignerD(out_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim_in, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    for i in range(3):
        axis, angle, R = random_rotation(cuda_device)
        D_in = wigner_in.rot(axis, angle).squeeze(0)
        D_out = wigner_out.rot(axis, angle).squeeze(0)

        P, Q = basis(directions)
        output1 = layer(P, Q, node_features, distances, src_indices)
        output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

        node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        P_rot, Q_rot = basis(directions_rotated)
        output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

        rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
        assert rel_diff < 1e-4, f"Rotation {i+1} failed: rel_diff={rel_diff:.2e}"


def test_equivariance_high_lmax(cuda_device):
    """Test equivariance with high angular momentum (lmax=5)."""
    torch.manual_seed(789)

    lvals = [0, 1, 2, 3, 4, 5]
    mult = 2
    num_nodes = 10
    num_edges = 30
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult)
    out_repr = Repr(lvals=lvals, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis(in_repr, out_repr).to(cuda_device)
    wigner = WignerD(in_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    axis, angle, R = random_rotation(cuda_device)
    D = wigner.rot(axis, angle).squeeze(0)

    P, Q = basis(directions)
    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
    assert rel_diff < 1e-4, f"Equivariance failed: rel_diff={rel_diff:.2e}"


@pytest.mark.parametrize("lvals", [
    [0, 1, 2],
    [0, 1],
    [1, 2],
])
def test_gradient_equivariance(cuda_device, lvals):
    """Test that gradients are also equivariant.

    If the forward pass is equivariant: f(D@x, R@d) = D@f(x, d)
    Then gradients should transform: D @ grad_x L(f(x,d)) = grad_{x'} L(f(x',d'))
    where x' = D@x, d' = R@d.

    IMPORTANT: The loss function must be rotationally invariant for this to hold.
    We use L = ||output||^2 = (output**2).sum(), which is preserved under orthogonal
    rotations. Using L = output.sum() would NOT work since sum is not invariant.
    """
    torch.manual_seed(101)

    mult = 4
    num_nodes = 10
    num_edges = 30
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult)
    out_repr = Repr(lvals=lvals, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis(in_repr, out_repr).to(cuda_device)
    wigner = WignerD(in_repr).to(cuda_device)

    axis, angle, R = random_rotation(cuda_device)
    D = wigner.rot(axis, angle).squeeze(0)

    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    P, Q = basis(directions)

    # Method 1: Compute gradient, then rotate
    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device, requires_grad=True)
    output1 = layer(P, Q, node_features, distances, src_indices)
    # Use squared norm loss - invariant under rotation
    loss1 = (output1 ** 2).sum()
    loss1.backward()
    grad1 = node_features.grad.clone()

    # Sanity checks
    assert output1.abs().mean().item() > 1e-6, "Output too small"
    assert grad1.abs().mean().item() > 1e-6, "Gradient too small"

    # Rotate the gradient
    grad1_rotated = torch.einsum('ij,ncj->nci', D, grad1)

    # Method 2: Rotate input (as new leaf tensor), compute gradient w.r.t. rotated input
    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features.detach())
    node_features_rotated = node_features_rotated.clone().requires_grad_(True)

    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)

    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)
    # Use squared norm loss - invariant under rotation
    loss2 = (output2 ** 2).sum()
    loss2.backward()
    grad2 = node_features_rotated.grad

    # Verify losses are equal (sanity check for invariance)
    assert abs(loss1.item() - loss2.item()) < 1e-4 * abs(loss1.item()), \
        f"Losses should be equal for invariant loss: {loss1.item():.6f} vs {loss2.item():.6f}"

    # Compare: D @ grad1 should equal grad2
    rel_diff = (grad1_rotated - grad2).abs().max().item() / (grad1_rotated.abs().max().item() + 1e-8)

    assert rel_diff < 1e-4, f"Gradient equivariance failed: rel_diff={rel_diff:.2e}"
