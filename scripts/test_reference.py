"""Test the canonical reference implementation."""
import torch
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.reference import (
    reference_layer,
    reference_equivariance_test,
)


def test_reference_equivariance():
    """Test that the reference implementation is equivariant."""
    torch.manual_seed(42)
    device = 'cpu'
    dtype = torch.float64

    lvals = [1]
    cin, cout = 2, 3
    num_nodes = 10
    num_edges = 5
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = 1 + 2 * max(lvals)  # 1 for m=0, 2 for each m>0

    # Random inputs
    node_features = torch.randn(num_nodes, cin, dim, dtype=dtype, device=device)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    directions = torch.randn(num_edges, 3, dtype=dtype, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Non-identity but still equivariant weights
    # [λ₀, a, b] means m=0 scalar=λ₀, m=1 block=[[a,b],[-b,a]]
    compact_weights = torch.zeros(cout, cin, weight_dim, dtype=dtype, device=device)
    compact_weights[..., 0] = 2.0  # m=0 scalar (arbitrary)
    compact_weights[..., 1] = 3.0  # m=1 'a'
    compact_weights[..., 2] = 1.0  # m=1 'b' (non-zero!)

    # Random rotation
    R = torch.tensor(Rotation.random(random_state=42).as_matrix(), dtype=dtype, device=device)

    # Test equivariance
    out1, out2, error = reference_equivariance_test(
        node_features, src_indices, directions, compact_weights, lvals, R
    )

    print(f"lvals={lvals}, cin={cin}, cout={cout}")
    print(f"D @ layer(f, d):\n{out1[0]}")
    print(f"layer(D@f, R@d):\n{out2[0]}")
    print(f"\nRelative error: {error:.6e}")

    if error < 1e-6:
        print("PASSED!")
        return True
    else:
        print("FAILED!")
        return False


if __name__ == '__main__':
    test_reference_equivariance()
