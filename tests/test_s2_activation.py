"""Tests for S2Activation layer.

Tests cover:
- Forward pass shape preservation
- Approximate SO(3) equivariance (controlled by Lebedev precision)
- Gradient flow
- Edge cases (batch dimensions, precisions, lmax values)

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

import torch
import pytest

from flash_eq import Repr, WignerD, S2Activation, SeparableS2Activation

from .helpers import random_rotation


# S2Activation supports non-contiguous l values
LVALS_CONFIGS = [
    [0],
    [1],
    [0, 1],
    [0, 1, 2],
    [1, 2],
    [0, 2],  # non-contiguous
    [0, 1, 2, 3],
    # Duplicate l values
    [0, 0],
    [1, 1],
    [0, 1, 1],
    [1, 2, 2],
]

MULT_CONFIGS = [1, 4, 16]

PRECISION_CONFIGS = [17, 47, 131]


class TestS2ActivationShape:
    """Tests for S2Activation output shapes."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_forward_shape(self, device, lvals, mult):
        """Output shape should match input shape."""
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()
        batch_size = 32

        act = S2Activation(repr).to(device)
        x = torch.randn(batch_size, mult, dim, device=device)
        y = act(x)

        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_multiple_batch_dims(self, device):
        """Test with multiple batch dimensions."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = S2Activation(repr).to(device)

        # 2D batch
        x = torch.randn(4, 8, 8, dim, device=device)
        y = act(x)
        assert y.shape == x.shape

        # 3D batch
        x = torch.randn(2, 3, 4, 8, dim, device=device)
        y = act(x)
        assert y.shape == x.shape

    def test_single_sample(self, device):
        """Test with batch size of 1."""
        repr = Repr(lvals=[0, 1], mult=4)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.randn(1, 4, dim, device=device)
        y = act(x)

        assert y.shape == x.shape


class TestS2ActivationEquivariance:
    """Tests for approximate SO(3) equivariance."""

    @pytest.mark.parametrize("lvals", [[0, 1], [0, 1, 2], [1, 2], [0, 2]])
    @pytest.mark.parametrize("precision", [47, 131])
    def test_approximate_equivariance(self, device, lvals, precision):
        """S2Activation should be approximately equivariant.

        The equivariance error depends on Lebedev precision.
        Higher precision = better equivariance.
        """
        torch.manual_seed(42)

        mult = 8
        batch_size = 16
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        act = S2Activation(repr, precision=precision).to(device)
        wigner = WignerD(repr).to(device)

        x = torch.randn(batch_size, mult, dim, device=device)

        # Get rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Method 1: forward then rotate
        y1 = act(x)
        y1_rot = torch.einsum('ij,bmj->bmi', D, y1)

        # Method 2: rotate then forward
        x_rot = torch.einsum('ij,bmj->bmi', D, x)
        y2 = act(x_rot)

        # Check approximate equivariance
        # Tolerance depends on precision: higher precision = tighter tolerance
        rtol = 0.05 if precision >= 131 else 0.15
        rel_diff = (y1_rot - y2).abs().max().item() / (y1_rot.abs().max().item() + 1e-8)
        assert rel_diff < rtol, f"Equivariance error {rel_diff:.3f} > {rtol} for precision {precision}"

    def test_equivariance_improves_with_precision(self, device):
        """Higher precision should give better equivariance."""
        torch.manual_seed(123)

        lvals = [0, 1, 2]
        mult = 8
        batch_size = 32
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        wigner = WignerD(repr).to(device)
        x = torch.randn(batch_size, mult, dim, device=device)

        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle)
        x_rot = torch.einsum('ij,bmj->bmi', D, x)

        errors = []
        for precision in [17, 47, 131]:
            act = S2Activation(repr, precision=precision).to(device)

            y1 = act(x)
            y1_rot = torch.einsum('ij,bmj->bmi', D, y1)
            y2 = act(x_rot)

            rel_diff = (y1_rot - y2).abs().max().item() / (y1_rot.abs().max().item() + 1e-8)
            errors.append(rel_diff)

        # Highest precision should be significantly better than lowest
        # (intermediate values may have noise when errors are very small)
        assert errors[2] < errors[0] * 0.5, (
            f"Precision 131 error {errors[2]:.6f} not significantly less than "
            f"precision 17 error {errors[0]:.6f}"
        )


class TestS2ActivationGradient:
    """Tests for gradient computation."""

    @pytest.mark.parametrize("lvals", [[0, 1], [0, 1, 2]])
    def test_gradient_flow(self, device, lvals):
        """Gradients should flow through the layer."""
        mult = 4
        batch_size = 8
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.randn(batch_size, mult, dim, device=device, requires_grad=True)

        y = act(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed for input"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert x.grad.abs().sum() > 0, "Zero gradients"

    def test_parameter_gradients(self, device):
        """MLP parameters should receive gradients."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.randn(16, 8, dim, device=device)

        y = act(x)
        loss = y.sum()
        loss.backward()

        for name, param in act.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestS2ActivationEdgeCases:
    """Tests for edge cases and special configurations."""

    def test_scalars_only(self, device):
        """Test with only scalar (l=0) representation."""
        repr = Repr(lvals=[0], mult=8)
        act = S2Activation(repr).to(device)

        x = torch.randn(16, 8, 1, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_no_scalars(self, device):
        """Test with no scalar (l>0 only) representation."""
        repr = Repr(lvals=[1, 2], mult=4)
        dim = repr.dim()  # 3 + 5 = 8

        act = S2Activation(repr).to(device)
        x = torch.randn(16, 4, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_high_lmax(self, device):
        """Test with high angular momentum."""
        repr = Repr(lvals=[0, 1, 2, 3, 4, 5, 6], mult=2)
        dim = repr.dim()  # 49

        act = S2Activation(repr, precision=131).to(device)
        x = torch.randn(8, 2, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    @pytest.mark.parametrize("precision", PRECISION_CONFIGS)
    def test_different_precisions(self, device, precision):
        """Test with different Lebedev precisions."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        dim = repr.dim()

        act = S2Activation(repr, precision=precision).to(device)
        x = torch.randn(16, 4, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_zero_input(self, device):
        """Test with zero input."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.zeros(8, 4, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_large_input(self, device):
        """Test numerical stability with large inputs."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.randn(8, 4, dim, device=device) * 100
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_small_input(self, device):
        """Test numerical stability with small inputs."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        dim = repr.dim()

        act = S2Activation(repr).to(device)
        x = torch.randn(8, 4, dim, device=device) * 1e-6
        y = act(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()


class TestS2ActivationProperties:
    """Tests for layer properties and configuration."""

    def test_n_points_matches_precision(self, device):
        """Number of grid points should match Lebedev precision."""
        repr = Repr(lvals=[0, 1, 2], mult=4)

        # Known point counts for each precision
        expected_points = {17: 110, 23: 194, 29: 302, 35: 434, 41: 590, 47: 770}

        for precision, n_points in expected_points.items():
            act = S2Activation(repr, precision=precision).to(device)
            assert act.n_points == n_points, f"Precision {precision}: expected {n_points}, got {act.n_points}"

    def test_extra_repr(self, device):
        """Test string representation."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        act = S2Activation(repr, precision=47).to(device)

        s = act.extra_repr()
        assert "mult=8" in s
        assert "dim=9" in s
        assert "n_points=770" in s
        assert "precision=47" in s

    def test_buffer_shapes(self, device):
        """Test that transform matrices have correct shapes."""
        lvals = [0, 1, 2]
        repr = Repr(lvals=lvals, mult=4)
        dim = repr.dim()

        act = S2Activation(repr, precision=47).to(device)

        # Y_T: (dim, n_points)
        assert act.Y_T.shape == (dim, 770)
        # Y_inv_T: (n_points, dim)
        assert act.Y_inv_T.shape == (770, dim)

    def test_custom_activation(self, device):
        """Test with custom activation function."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        dim = repr.dim()

        act = S2Activation(repr, activation=torch.nn.ReLU()).to(device)
        x = torch.randn(8, 4, dim, device=device)
        y = act(x)

        assert y.shape == x.shape

    def test_hidden_mult(self, device):
        """Test different hidden layer multipliers."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        for hidden_mult in [1, 2, 4]:
            act = S2Activation(repr, hidden_mult=hidden_mult).to(device)
            x = torch.randn(8, 8, dim, device=device)
            y = act(x)
            assert y.shape == x.shape


class TestSeparableS2ActivationShape:
    """Tests for SeparableS2Activation output shapes."""

    @pytest.mark.parametrize("lvals", [[0, 1], [0, 1, 2], [0, 2], [0, 1, 2, 3]])
    @pytest.mark.parametrize("mult", [4, 8, 16])
    def test_forward_shape(self, device, lvals, mult):
        """Output shape should match input shape."""
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()
        batch_size = 32

        act = SeparableS2Activation(repr).to(device)
        x = torch.randn(batch_size, mult, dim, device=device)
        y = act(x)

        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_scalars_only(self, device):
        """Test with only scalars (should skip S² path)."""
        repr = Repr(lvals=[0], mult=8)
        act = SeparableS2Activation(repr).to(device)

        x = torch.randn(16, 8, 1, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert act.s2_act is None  # No S² activation for scalars only

    def test_requires_scalars(self, device):
        """SeparableS2Activation requires l=0."""
        repr = Repr(lvals=[1, 2], mult=4)

        with pytest.raises(ValueError, match="requires l=0"):
            SeparableS2Activation(repr)

    def test_multiple_batch_dims(self, device):
        """Test with multiple batch dimensions."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr).to(device)

        x = torch.randn(4, 8, 8, dim, device=device)
        y = act(x)
        assert y.shape == x.shape


class TestSeparableS2ActivationEquivariance:
    """Tests for approximate SO(3) equivariance."""

    @pytest.mark.parametrize("lvals", [[0, 1], [0, 1, 2], [0, 2]])
    def test_approximate_equivariance(self, device, lvals):
        """SeparableS2Activation should be approximately equivariant."""
        torch.manual_seed(42)

        mult = 8
        batch_size = 16
        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        act = SeparableS2Activation(repr, precision=47).to(device)
        wigner = WignerD(repr).to(device)

        x = torch.randn(batch_size, mult, dim, device=device)

        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Method 1: forward then rotate
        y1 = act(x)
        y1_rot = torch.einsum('ij,bmj->bmi', D, y1)

        # Method 2: rotate then forward
        x_rot = torch.einsum('ij,bmj->bmi', D, x)
        y2 = act(x_rot)

        # Check approximate equivariance
        rtol = 0.15
        rel_diff = (y1_rot - y2).abs().max().item() / (y1_rot.abs().max().item() + 1e-8)
        assert rel_diff < rtol, f"Equivariance error {rel_diff:.3f} > {rtol}"


class TestSeparableS2ActivationGradient:
    """Tests for gradient computation."""

    def test_gradient_flow(self, device):
        """Gradients should flow through both scalar and higher-degree paths."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr).to(device)
        x = torch.randn(16, 8, dim, device=device, requires_grad=True)

        y = act(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().sum() > 0

    def test_parameter_gradients(self, device):
        """All parameters should receive gradients."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr, use_gate=True).to(device)
        x = torch.randn(16, 8, dim, device=device)

        y = act(x)
        loss = y.sum()
        loss.backward()

        for name, param in act.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestSeparableS2ActivationGating:
    """Tests for the optional gating mechanism."""

    def test_with_gating(self, device):
        """Test with gating enabled."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr, use_gate=True).to(device)
        x = torch.randn(16, 8, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert act.gate_linear is not None

    def test_without_gating(self, device):
        """Test with gating disabled."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr, use_gate=False).to(device)
        x = torch.randn(16, 8, dim, device=device)
        y = act(x)

        assert y.shape == x.shape
        assert act.gate_linear is None

    def test_gating_output_range(self, device):
        """Gating should produce bounded modifications."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        dim = repr.dim()

        act = SeparableS2Activation(repr, use_gate=True).to(device)
        x = torch.randn(16, 8, dim, device=device)
        y = act(x)

        # Output should be finite
        assert torch.isfinite(y).all()
