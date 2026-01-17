"""
Explicit equivariance tests for flash-eq.

Tests that the flash-eq layer is SO(3)-equivariant:
    D_out @ f(x) = f(D_in @ x)

where D_in, D_out are Wigner-D rotation matrices for the representations.

The key property: rotating the input features and basis matrices should
give the same result as applying the layer then rotating the output.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq import Repr, EquivariantEdgewiseLinear, WignerDBasis


def random_direction(device: torch.device) -> torch.Tensor:
    """Generate a random unit direction vector."""
    d = torch.randn(3, device=device)
    return d / d.norm()


def test_equivariance_basic():
    """
    Test basic equivariance: D @ layer(x) = layer(D @ x).

    The test:
    1. Apply layer to original features, then rotate output
    2. Rotate input features, then apply layer
    3. Check that results match
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    lvals = [0, 1, 2]
    mult = 8
    num_nodes = 50
    num_edges = 200
    dim = sum(2 * l + 1 for l in lvals)  # 9

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    # Random rotation defined by direction
    # R is the 3x3 Cartesian rotation matrix for rotating directions
    # D is the Wigner-D matrix for rotating features in SH basis
    direction = random_direction(device)
    R = Repr.cartesian_rotation(direction.unsqueeze(0)).squeeze(0)
    D = in_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)

    # Input data
    node_features = torch.randn(num_nodes, mult, dim, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 5.0
    directions = torch.randn(num_edges, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Compute basis matrices
    P, Q = basis(directions)

    # Method 1: Apply layer, then rotate output
    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

    # Method 2: Rotate input features and directions, then apply layer
    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    # Compare
    diff = (output1_rotated - output2).abs()
    max_diff = diff.max().item()
    rel_diff = max_diff / (output1_rotated.abs().max().item() + 1e-8)

    print(f"  Basic equivariance test:")
    print(f"    Max absolute diff: {max_diff:.2e}")
    print(f"    Relative diff: {rel_diff:.2e}")

    passed = rel_diff < 1e-4
    print(f"    Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_equivariance_multiple_rotations():
    """Test equivariance with multiple random rotations."""
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"  Multiple rotation test (lvals={lvals}):")

    all_passed = True
    for i in range(5):
        direction = random_direction(device)
        R = Repr.cartesian_rotation(direction.unsqueeze(0)).squeeze(0)
        D = in_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)

        P, Q = basis(directions)

        # Method 1: layer then rotate
        output1 = layer(P, Q, node_features, distances, src_indices)
        output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

        # Method 2: rotate then layer
        node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        P_rot, Q_rot = basis(directions_rotated)
        output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

        rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
        passed = rel_diff < 1e-4

        print(f"    Rotation {i+1}: rel_diff={rel_diff:.2e} {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed

    return all_passed


def test_equivariance_different_channels():
    """Test equivariance with different input/output channels."""
    torch.manual_seed(456)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    direction = random_direction(device)
    R = Repr.cartesian_rotation(direction.unsqueeze(0)).squeeze(0)
    # Use out_repr for D since we rotate output
    D_out = out_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)
    # Use in_repr for rotating input
    D_in = in_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)

    P, Q = basis(directions)

    # Method 1: layer then rotate
    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D_out, output1)

    # Method 2: rotate then layer
    node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
    passed = rel_diff < 1e-4

    print(f"  Different channels test (cin={mult_in}, cout={mult_out}):")
    print(f"    Relative diff: {rel_diff:.2e}")
    print(f"    Status: {'PASS' if passed else 'FAIL'}")

    return passed


def test_equivariance_high_lmax():
    """Test equivariance with high angular momentum."""
    torch.manual_seed(789)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    direction = random_direction(device)
    R = Repr.cartesian_rotation(direction.unsqueeze(0)).squeeze(0)
    D = in_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)

    P, Q = basis(directions)

    output1 = layer(P, Q, node_features, distances, src_indices)
    output1_rotated = torch.einsum('ij,ecj->eci', D, output1)

    node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    output2 = layer(P_rot, Q_rot, node_features_rotated, distances, src_indices)

    rel_diff = (output1_rotated - output2).abs().max().item() / (output1_rotated.abs().max().item() + 1e-8)
    passed = rel_diff < 1e-4

    print(f"  High lmax test (lmax=5, dim={dim}):")
    print(f"    Relative diff: {rel_diff:.2e}")
    print(f"    Status: {'PASS' if passed else 'FAIL'}")

    return passed


def test_equivariance_gradient():
    """Test that gradients are also equivariant."""
    torch.manual_seed(101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lvals = [0, 1, 2]
    mult = 4
    num_nodes = 10
    num_edges = 30
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=mult).to(device)
    out_repr = Repr(lvals=lvals, mult=mult).to(device)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr, num_bins=50).to(device)
    basis = WignerDBasis(in_repr, out_repr).to(device)

    direction = random_direction(device)
    R = Repr.cartesian_rotation(direction.unsqueeze(0)).squeeze(0)
    D = in_repr.rot_to_ez(direction.unsqueeze(0)).squeeze(0)

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

    # Method 2: rotate then forward, compute gradient
    node_features2 = torch.randn(num_nodes, mult, dim, device=device, requires_grad=True)
    with torch.no_grad():
        node_features2.copy_(node_features1.detach())
    node_features2.requires_grad_(True)

    directions_rotated = torch.einsum('ij,ej->ei', R, directions)
    P_rot, Q_rot = basis(directions_rotated)
    node_features2_rotated = torch.einsum('ij,ncj->nci', D, node_features2)
    output2 = layer(P_rot, Q_rot, node_features2_rotated, distances, src_indices)
    loss2 = output2.sum()
    loss2.backward()

    # The gradient w.r.t. node_features2 needs to be compared with D @ grad1
    # Since node_features2_rotated = D @ node_features2,
    # grad2 = D^T @ (grad w.r.t. rotated) = D^T @ node_features2.grad
    # But we want to compare grad1 rotated vs node_features2.grad
    grad1_rotated = torch.einsum('ij,ncj->nci', D, grad1)

    rel_diff = (grad1_rotated - node_features2.grad).abs().max().item() / (grad1_rotated.abs().max().item() + 1e-8)
    passed = rel_diff < 1e-4

    print(f"  Gradient equivariance test:")
    print(f"    Relative diff: {rel_diff:.2e}")
    print(f"    Status: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    print("=" * 70)
    print("Flash-eq Equivariance Tests")
    print("=" * 70)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nDevice: {device_name}")

    all_passed = True

    print("\n" + "-" * 70)
    print("Test 1: Basic equivariance")
    print("-" * 70)
    all_passed &= test_equivariance_basic()

    print("\n" + "-" * 70)
    print("Test 2: Multiple random rotations")
    print("-" * 70)
    all_passed &= test_equivariance_multiple_rotations()

    print("\n" + "-" * 70)
    print("Test 3: Different input/output channels")
    print("-" * 70)
    all_passed &= test_equivariance_different_channels()

    print("\n" + "-" * 70)
    print("Test 4: High angular momentum (lmax=5)")
    print("-" * 70)
    all_passed &= test_equivariance_high_lmax()

    print("\n" + "-" * 70)
    print("Test 5: Gradient equivariance")
    print("-" * 70)
    all_passed &= test_equivariance_gradient()

    print("\n" + "=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
