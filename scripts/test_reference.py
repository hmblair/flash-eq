"""Test the canonical reference implementation."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.representations import Repr, ProductRepr
from flash_eq.reference import reference_layer, equivariance_test


def test_equivariance(repr_in, repr_out=None, cin=2, cout=3, num_nodes=10, num_edges=5):
    """Test that the reference implementation is equivariant."""
    if repr_out is None:
        repr_out = repr_in

    torch.manual_seed(42)
    device = 'cpu'
    dtype = torch.float64

    prod = ProductRepr(repr_in, repr_out)
    weight_dim = prod.weight_dim()

    # Random inputs
    node_features = torch.randn(num_nodes, cin, repr_in.dim(), dtype=dtype, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, dtype=dtype, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Random weights
    compact_weights = torch.randn(cout, cin, weight_dim, dtype=dtype, device=device)

    # Random rotation (axis-angle)
    axis = torch.randn(3, dtype=dtype)
    axis = axis / axis.norm()
    angle = torch.tensor(1.23, dtype=dtype)  # arbitrary angle

    # Test equivariance
    _, _, error = equivariance_test(
        node_features, src_indices, directions, compact_weights,
        repr_in, repr_out, axis, angle
    )

    return error, weight_dim


def main():
    """Run equivariance tests for various representation configurations."""
    # Test cases: (repr_in, repr_out) - None means same as input
    test_cases = [
        # Same in/out
        (Repr([0]), None),
        (Repr([1]), None),
        (Repr([2]), None),
        (Repr([0, 1]), None),
        (Repr([1, 2]), None),
        (Repr([0, 1, 2]), None),
        (Repr([0, 2]), None),
        (Repr([1, 2, 3]), None),
        # Different in/out
        (Repr([0]), Repr([0, 1])),
        (Repr([0, 1]), Repr([0, 1, 2])),
        (Repr([1]), Repr([0, 1])),
        (Repr([1, 2]), Repr([0, 1, 2])),
        (Repr([0, 1, 2]), Repr([1, 2])),
        (Repr([2]), Repr([0, 1, 2])),
        (Repr([0, 1]), Repr([2])),
    ]

    print("=" * 70)
    print("Reference Implementation Equivariance Tests")
    print("=" * 70)

    all_passed = True
    for repr_in, repr_out in test_cases:
        repr_out_display = repr_out if repr_out is not None else repr_in

        error, weight_dim = test_equivariance(repr_in, repr_out)
        status = "PASS" if error < 1e-6 else "FAIL"
        if error >= 1e-6:
            all_passed = False

        in_str = str(repr_in.lvals)
        out_str = str(repr_out_display.lvals)
        dim_in = repr_in.dim()
        dim_out = repr_out_display.dim()
        print(f"{in_str:<10} -> {out_str:<10}  dim={dim_in}->{dim_out:<2}  w={weight_dim:<3}  err={error:.2e}  [{status}]")

    print("=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()
