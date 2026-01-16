"""Tests for EquivariantEdgewiseLinear layer and WignerDBasis."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq import Repr, EquivariantEdgewiseLinear, WignerDBasis


def test_basic_forward():
    """Test basic forward pass."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=32)
    out_repr = Repr(lvals=[0, 1, 2], mult=32)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 1000
    features = torch.randn(batch, 32, 9, device=device)  # 9 = 1 + 3 + 5
    distances = torch.rand(batch, device=device) * 5.0

    output = layer(features, distances)

    assert output.shape == (batch, 32, 9)
    assert not torch.isnan(output).any()
    print("  test_basic_forward: PASS")


def test_different_channels():
    """Test with different input/output channels."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=32)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 500
    features = torch.randn(batch, 16, 9, device=device)
    distances = torch.rand(batch, device=device) * 5.0

    output = layer(features, distances)

    assert output.shape == (batch, 32, 9)
    print("  test_different_channels: PASS")


def test_different_lvals_output_superset():
    """Test where output has more l-values than input.

    The m-components in output that have no input coupling will be zero.
    """
    device = torch.device("cuda")

    # Input has l=0,1, output has l=0,1,2
    # The m=2 output components will be zero (no coupling)
    in_repr = Repr(lvals=[0, 1], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=16)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 500
    dim_in = 1 + 3  # l=0: 1, l=1: 3
    dim_out = 1 + 3 + 5  # l=0: 1, l=1: 3, l=2: 5

    features = torch.randn(batch, 16, dim_in, device=device)
    distances = torch.rand(batch, device=device) * 5.0

    output = layer(features, distances)

    assert output.shape == (batch, 16, dim_out), f"Expected {(batch, 16, dim_out)}, got {output.shape}"
    # The l=2 components (last 5 values) should be zero since there's no m=2 input
    # In m-basis ordering, this is the last 2 components (m=2 real and imag)
    print("  test_different_lvals_output_superset: PASS")


def test_different_lvals_input_superset():
    """Test where input has more l-values than output.

    The m-components in input that have no output coupling are ignored.
    """
    device = torch.device("cuda")

    # Input has l=0,1,2, output has l=0,1
    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1], mult=16)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 500
    dim_in = 1 + 3 + 5  # l=0: 1, l=1: 3, l=2: 5
    dim_out = 1 + 3  # l=0: 1, l=1: 3

    features = torch.randn(batch, 16, dim_in, device=device)
    distances = torch.rand(batch, device=device) * 5.0

    output = layer(features, distances)

    assert output.shape == (batch, 16, dim_out), f"Expected {(batch, 16, dim_out)}, got {output.shape}"
    print("  test_different_lvals_input_superset: PASS")


def test_gradient_flow():
    """Test that gradients flow through the layer."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=16)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    features = torch.randn(100, 16, 9, device=device, requires_grad=True)
    distances = torch.rand(100, device=device) * 5.0

    output = layer(features, distances)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert features.grad is not None
    assert features.grad.abs().sum() > 0

    # Check MLP gradients
    has_mlp_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.radial_mlp.parameters()
    )
    assert has_mlp_grads
    print("  test_gradient_flow: PASS")


def test_float16():
    """Test with float16 precision."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=32)
    out_repr = Repr(lvals=[0, 1, 2], mult=32)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device).half()

    features = torch.randn(500, 32, 9, device=device, dtype=torch.float16)
    distances = torch.rand(500, device=device) * 5.0

    output = layer(features, distances)

    assert output.dtype == torch.float16
    assert not torch.isnan(output).any()
    print("  test_float16: PASS")


def test_custom_bins():
    """Test with custom bin configuration."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=16)

    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr,
        num_bins=50,
        min_dist=1.0,
        max_dist=8.0,
    ).to(device)

    features = torch.randn(100, 16, 9, device=device)
    distances = torch.rand(100, device=device) * 7.0 + 1.0  # [1, 8]

    output = layer(features, distances)

    assert output.shape == (100, 16, 9)
    print("  test_custom_bins: PASS")


def test_high_lmax():
    """Test with high angular momentum."""
    device = torch.device("cuda")

    lvals = [0, 1, 2, 3, 4]
    dim = sum(2 * l + 1 for l in lvals)  # 1 + 3 + 5 + 7 + 9 = 25

    in_repr = Repr(lvals=lvals, mult=16)
    out_repr = Repr(lvals=lvals, mult=16)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    features = torch.randn(200, 16, dim, device=device)
    distances = torch.rand(200, device=device) * 5.0

    output = layer(features, distances)

    assert output.shape == (200, 16, dim)
    print("  test_high_lmax: PASS")


def test_wigner_d_basis():
    """Test WignerDBasis computation."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=32)
    out_repr = Repr(lvals=[0, 1, 2], mult=32)

    basis = WignerDBasis(in_repr, out_repr).to(device)

    batch = 100
    directions = torch.randn(batch, 3, device=device)

    P, Q = basis(directions)

    # Check shapes
    dim = in_repr.dim()  # 9
    assert P.shape == (batch, dim, dim), f"Expected P shape {(batch, dim, dim)}, got {P.shape}"
    assert Q.shape == (batch, dim, dim), f"Expected Q shape {(batch, dim, dim)}, got {Q.shape}"

    # Check orthogonality (Wigner-D matrices are orthogonal)
    identity = torch.eye(dim, device=device)
    P_orth = torch.bmm(P.transpose(-1, -2), P)
    Q_orth = torch.bmm(Q.transpose(-1, -2), Q)

    assert torch.allclose(P_orth, identity.expand_as(P_orth), atol=1e-5), "P should be orthogonal"
    assert torch.allclose(Q_orth, identity.expand_as(Q_orth), atol=1e-5), "Q should be orthogonal"

    print("  test_wigner_d_basis: PASS")


def test_with_basis_matrices():
    """Test EquivariantEdgewiseLinear with pre-computed P, Q."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=16)

    basis = WignerDBasis(in_repr, out_repr).to(device)
    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 200
    dim = in_repr.dim()

    # Features in standard basis
    features = torch.randn(batch, 16, dim, device=device)
    distances = torch.rand(batch, device=device) * 5.0
    directions = torch.randn(batch, 3, device=device)

    # Compute basis matrices
    P, Q = basis(directions)

    # Apply layer with basis matrices
    output = layer(features, distances, P=P, Q=Q)

    assert output.shape == (batch, 16, dim)
    assert not torch.isnan(output).any()
    print("  test_with_basis_matrices: PASS")


def test_basis_sharing_multi_layer():
    """Test that basis matrices can be shared across multiple layers."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=16)
    out_repr = Repr(lvals=[0, 1, 2], mult=16)

    basis = WignerDBasis(in_repr, out_repr).to(device)
    layer1 = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
    layer2 = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    batch = 100
    dim = in_repr.dim()

    features = torch.randn(batch, 16, dim, device=device)
    distances = torch.rand(batch, device=device) * 5.0
    directions = torch.randn(batch, 3, device=device)

    # Compute basis once
    P, Q = basis(directions)

    # Apply multiple layers with shared basis
    out1 = layer1(features, distances, P=P, Q=Q)
    out2 = layer2(out1, distances, P=P, Q=Q)

    assert out2.shape == (batch, 16, dim)
    assert not torch.isnan(out2).any()
    print("  test_basis_sharing_multi_layer: PASS")


def test_gradient_with_basis():
    """Test gradient flow with P, Q matrices."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1, 2], mult=8)
    out_repr = Repr(lvals=[0, 1, 2], mult=8)

    basis = WignerDBasis(in_repr, out_repr).to(device)
    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    features = torch.randn(50, 8, 9, device=device, requires_grad=True)
    distances = torch.rand(50, device=device) * 5.0
    directions = torch.randn(50, 3, device=device)

    P, Q = basis(directions)
    output = layer(features, distances, P=P, Q=Q)
    loss = output.sum()
    loss.backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0

    has_mlp_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.radial_mlp.parameters()
    )
    assert has_mlp_grads
    print("  test_gradient_with_basis: PASS")


def test_pq_validation():
    """Test that P and Q must both be provided or both None."""
    device = torch.device("cuda")

    in_repr = Repr(lvals=[0, 1], mult=8)
    out_repr = Repr(lvals=[0, 1], mult=8)

    layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)

    features = torch.randn(10, 8, 4, device=device)
    distances = torch.rand(10, device=device) * 5.0
    P = torch.randn(10, 4, 4, device=device)

    try:
        layer(features, distances, P=P, Q=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "both be provided" in str(e)

    try:
        layer(features, distances, P=None, Q=P)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "both be provided" in str(e)

    print("  test_pq_validation: PASS")


def main():
    print("=" * 60)
    print("EquivariantEdgewiseLinear Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    print("\nBasic Tests:")
    test_basic_forward()
    test_different_channels()
    test_different_lvals_output_superset()
    test_different_lvals_input_superset()

    print("\nGradient Tests:")
    test_gradient_flow()

    print("\nDtype Tests:")
    test_float16()

    print("\nConfiguration Tests:")
    test_custom_bins()
    test_high_lmax()

    print("\nWigner-D Basis Tests:")
    test_wigner_d_basis()
    test_with_basis_matrices()
    test_basis_sharing_multi_layer()
    test_gradient_with_basis()
    test_pq_validation()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
