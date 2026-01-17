"""Test the canonical reference implementation."""
import torch
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.reference import (
    reference_layer,
    reference_equivariance_test,
    get_weight_dim,
)


def test_reference_equivariance(lvals, cin=2, cout=3, num_nodes=10, num_edges=5):
    """Test that the reference implementation is equivariant for given lvals."""
    torch.manual_seed(42)
    device = 'cpu'
    dtype = torch.float64

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals)

    # Random inputs
    node_features = torch.randn(num_nodes, cin, dim, dtype=dtype, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, dtype=dtype, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Random weights
    compact_weights = torch.randn(cout, cin, weight_dim, dtype=dtype, device=device)

    # Random rotation
    R = torch.tensor(Rotation.random(random_state=42).as_matrix(), dtype=dtype, device=device)

    # Test equivariance
    out1, out2, error = reference_equivariance_test(
        node_features, src_indices, directions, compact_weights, lvals, R
    )

    return error


def main():
    """Run equivariance tests for various lvals configurations."""
    test_cases = [
        [0],
        [1],
        [2],
        [0, 1],
        [1, 2],
        [0, 1, 2],
        [0, 2],  # Non-contiguous
        [1, 2, 3],
    ]

    print("=" * 60)
    print("Reference Implementation Equivariance Tests")
    print("=" * 60)

    all_passed = True
    for lvals in test_cases:
        dim = sum(2 * l + 1 for l in lvals)
        weight_dim = get_weight_dim(lvals)
        error = test_reference_equivariance(lvals)
        status = "PASS" if error < 1e-6 else "FAIL"
        if error >= 1e-6:
            all_passed = False
        print(f"lvals={str(lvals):<12} dim={dim:<3} weight_dim={weight_dim:<3} error={error:.2e}  [{status}]")

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()
