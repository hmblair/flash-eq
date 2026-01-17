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


def test_reference_equivariance(lvals_in, lvals_out=None, cin=2, cout=3, num_nodes=10, num_edges=5):
    """Test that the reference implementation is equivariant for given lvals."""
    if lvals_out is None:
        lvals_out = lvals_in

    torch.manual_seed(42)
    device = 'cpu'
    dtype = torch.float64

    dim_in = sum(2 * l + 1 for l in lvals_in)
    dim_out = sum(2 * l + 1 for l in lvals_out)
    weight_dim = get_weight_dim(lvals_in, lvals_out)

    # Random inputs
    node_features = torch.randn(num_nodes, cin, dim_in, dtype=dtype, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, dtype=dtype, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Random weights
    compact_weights = torch.randn(cout, cin, weight_dim, dtype=dtype, device=device)

    # Random rotation
    R = torch.tensor(Rotation.random(random_state=42).as_matrix(), dtype=dtype, device=device)

    # Test equivariance
    out1, out2, error = reference_equivariance_test(
        node_features, src_indices, directions, compact_weights, lvals_in, lvals_out, R
    )

    return error, weight_dim


def main():
    """Run equivariance tests for various lvals configurations."""
    # Test cases: (lvals_in, lvals_out)
    test_cases = [
        # Same in/out
        ([0], None),
        ([1], None),
        ([2], None),
        ([0, 1], None),
        ([1, 2], None),
        ([0, 1, 2], None),
        ([0, 2], None),
        ([1, 2, 3], None),
        # Different in/out
        ([0], [0, 1]),
        ([0, 1], [0, 1, 2]),
        ([1], [0, 1]),
        ([1, 2], [0, 1, 2]),
        ([0, 1, 2], [1, 2]),
        ([2], [0, 1, 2]),
        ([0, 1], [2]),
    ]

    print("=" * 70)
    print("Reference Implementation Equivariance Tests")
    print("=" * 70)

    all_passed = True
    for lvals_in, lvals_out in test_cases:
        if lvals_out is None:
            lvals_out_display = lvals_in
        else:
            lvals_out_display = lvals_out

        dim_in = sum(2 * l + 1 for l in lvals_in)
        dim_out = sum(2 * l + 1 for l in lvals_out_display)
        error, weight_dim = test_reference_equivariance(lvals_in, lvals_out)
        status = "PASS" if error < 1e-6 else "FAIL"
        if error >= 1e-6:
            all_passed = False

        in_str = str(lvals_in)
        out_str = str(lvals_out_display)
        print(f"{in_str:<10} -> {out_str:<10}  dim={dim_in}->{dim_out:<2}  w={weight_dim:<3}  err={error:.2e}  [{status}]")

    print("=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()
