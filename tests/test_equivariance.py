"""
Equivariance tests for flash-eq.

Tests that the layer satisfies SO(3)-equivariance:
    D_out @ f(x, d) = f(D_in @ x, R @ d)

where D_in, D_out are Wigner-D rotation matrices and R is the 3x3 rotation matrix.
"""

import torch
import pytest

from flash_eq import Repr, WignerD, EquivariantEdgewiseLinear, WignerDBasis

from .helpers import random_rotation, check_equivariance


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
    # Duplicate l values
    ([0, 0], [0, 0]),
    ([1, 1], [1, 1]),
    ([0, 1, 1], [0, 1, 1]),
    ([1, 2, 2], [1, 2, 2]),
    ([0, 1, 1], [0, 1, 2]),
    ([1, 2, 2], [0, 1]),
]


@pytest.mark.parametrize("lvals_in,lvals_out", LVALS_CONFIGS)
def test_equivariance(cuda_device, lvals_in, lvals_out):
    """Test SO(3) equivariance for various representation configurations."""
    torch.manual_seed(42)

    mult = 4
    num_nodes = 20
    num_edges = 50
    dim_in = sum(2 * l + 1 for l in lvals_in)
    sum(2 * l + 1 for l in lvals_out)

    in_repr = Repr(lvals=lvals_in, mult=mult)
    out_repr = Repr(lvals=lvals_out, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
    wigner_in = WignerD(in_repr).to(cuda_device)
    wigner_out = WignerD(out_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim_in, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Random rotation
    axis, angle, R = random_rotation(cuda_device)
    D_in = wigner_in.rot(axis, angle)
    D_out = wigner_out.rot(axis, angle)

    P, Q = basis(directions)

    # Gather node features to edges
    edge_features = node_features[src_indices]

    # Method 1: Apply layer, then rotate output
    output1 = layer(P, Q, edge_features, distances)
    output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

    # Method 2: Rotate input features and directions, then apply layer
    node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
    edge_features_rotated = node_features_rotated[src_indices]
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, edge_features_rotated, distances)

    # Check output is non-trivial
    output_mag = output1.abs().mean().item()
    assert output_mag > 0.01, f"Output too small: {output_mag:.2e}"

    # Check equivariance
    check_equivariance(output1_rotated, output2, rtol=1e-4)


@pytest.mark.parametrize("lvals_in,lvals_out", LVALS_CONFIGS[:5])  # Subset for speed
def test_equivariance_multiple_rotations(cuda_device, lvals_in, lvals_out):
    """Test equivariance with multiple random rotations."""
    torch.manual_seed(123)

    mult = 4
    num_nodes = 20
    num_edges = 50
    dim_in = sum(2 * l + 1 for l in lvals_in)
    sum(2 * l + 1 for l in lvals_out)

    in_repr = Repr(lvals=lvals_in, mult=mult)
    out_repr = Repr(lvals=lvals_out, mult=mult)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(cuda_device)
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
    wigner_in = WignerD(in_repr).to(cuda_device)
    wigner_out = WignerD(out_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim_in, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Gather node features to edges
    edge_features = node_features[src_indices]

    for i in range(3):
        axis, angle, R = random_rotation(cuda_device)
        D_in = wigner_in.rot(axis, angle)
        D_out = wigner_out.rot(axis, angle)

        P, Q = basis(directions)
        output1 = layer(P, Q, edge_features, distances)
        output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

        node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
        edge_features_rotated = node_features_rotated[src_indices]
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        P_rot, Q_rot = basis(directions_rotated)
        output2 = layer(P_rot, Q_rot, edge_features_rotated, distances)

        check_equivariance(output1_rotated, output2, rtol=1e-4, msg=f"Rotation {i+1}")


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
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
    wigner = WignerD(in_repr).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    axis, angle, R = random_rotation(cuda_device)
    D = wigner.rot(axis, angle)

    # Gather node features to edges
    edge_features = node_features[src_indices]

    P, Q = basis(directions)
    output1 = layer(P, Q, edge_features, distances)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
    edge_features_rotated = node_features_rotated[src_indices]
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, edge_features_rotated, distances)

    check_equivariance(output1_rotated, output2, rtol=1e-4, msg="High lmax equivariance")


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
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
    wigner = WignerD(in_repr).to(cuda_device)

    axis, angle, R = random_rotation(cuda_device)
    D = wigner.rot(axis, angle)

    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=cuda_device) * 5.0
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    P, Q = basis(directions)

    # Method 1: Compute gradient, then rotate
    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device, requires_grad=True)
    edge_features = node_features[src_indices]
    output1 = layer(P, Q, edge_features, distances)
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
    edge_features_rotated = node_features_rotated[src_indices]

    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)

    output2 = layer(P_rot, Q_rot, edge_features_rotated, distances)
    # Use squared norm loss - invariant under rotation
    loss2 = (output2 ** 2).sum()
    loss2.backward()
    grad2 = node_features_rotated.grad

    # Verify losses are equal (sanity check for invariance)
    assert abs(loss1.item() - loss2.item()) < 1e-4 * abs(loss1.item()), \
        f"Losses should be equal for invariant loss: {loss1.item():.6f} vs {loss2.item():.6f}"

    # Compare: D @ grad1 should equal grad2
    check_equivariance(grad1_rotated, grad2, rtol=1e-4, msg="Gradient equivariance")


@pytest.mark.parametrize("log_bins", [False, True])
def test_distance_gradient_flow(cuda_device, log_bins):
    """Test that gradients flow through to distances.

    The CUDA kernel computes binning internally from distances, which requires
    implementing the chain rule for gradient propagation. This test verifies
    that gradients are computed and non-zero for the distance input.
    """
    torch.manual_seed(42)

    lvals = [0, 1, 2]
    mult = 4
    num_nodes = 10
    num_edges = 30
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult)
    out_repr = Repr(lvals=lvals, mult=mult)

    # Use min_dist > 0 for log_bins
    min_dist = 0.5 if log_bins else 0.0
    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr,
        num_bins=50,
        min_dist=min_dist,
        max_dist=10.0,
        log_bins=log_bins,
    ).to(cuda_device)
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)

    # Distances with requires_grad=True
    distances = torch.rand(num_edges, device=cuda_device) * 5.0 + min_dist + 0.1
    distances = distances.requires_grad_(True)

    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    P, Q = basis(directions)
    edge_features = node_features[src_indices]

    output = layer(P, Q, edge_features, distances)
    loss = (output ** 2).sum()
    loss.backward()

    # Verify gradients exist and are non-zero
    assert distances.grad is not None, "distances.grad should not be None"
    assert distances.grad.shape == distances.shape, "grad shape mismatch"
    assert distances.grad.abs().sum() > 0, "distances.grad should be non-zero"

    # Verify gradients are finite
    assert torch.isfinite(distances.grad).all(), "distances.grad contains non-finite values"


@pytest.mark.parametrize("log_bins", [False, True])
def test_distance_gradient_numerical(cuda_device, log_bins):
    """Verify distance gradients numerically with finite differences.

    This checks that the analytical gradient computed by the CUDA kernel's
    chain rule implementation matches the numerical gradient.
    """
    torch.manual_seed(123)

    lvals = [0, 1]
    mult = 2
    num_nodes = 5
    num_edges = 10
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult)
    out_repr = Repr(lvals=lvals, mult=mult)

    min_dist = 0.5 if log_bins else 0.0
    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr,
        num_bins=50,
        min_dist=min_dist,
        max_dist=10.0,
        log_bins=log_bins,
    ).to(cuda_device)
    basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

    node_features = torch.randn(num_nodes, mult, dim, device=cuda_device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=cuda_device, dtype=torch.int64)
    directions = torch.randn(num_edges, 3, device=cuda_device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    P, Q = basis(directions)
    edge_features = node_features[src_indices]

    # Base distances
    distances = torch.rand(num_edges, device=cuda_device) * 5.0 + min_dist + 0.5

    # Compute analytical gradient
    distances_grad = distances.clone().requires_grad_(True)
    output = layer(P, Q, edge_features, distances_grad)
    loss = (output ** 2).sum()
    loss.backward()
    analytical_grad = distances_grad.grad.clone()

    # Compute numerical gradient via finite differences
    eps = 1e-4
    numerical_grad = torch.zeros_like(distances)
    for i in range(num_edges):
        # Forward
        distances_plus = distances.clone()
        distances_plus[i] += eps
        output_plus = layer(P, Q, edge_features, distances_plus)
        loss_plus = (output_plus ** 2).sum()

        # Backward
        distances_minus = distances.clone()
        distances_minus[i] -= eps
        output_minus = layer(P, Q, edge_features, distances_minus)
        loss_minus = (output_minus ** 2).sum()

        numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    # Compare
    rel_error = (analytical_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)
    max_rel_error = rel_error.max().item()

    assert max_rel_error < 0.05, (
        f"Distance gradient mismatch (log_bins={log_bins}): "
        f"max_rel_error={max_rel_error:.4f}"
    )
