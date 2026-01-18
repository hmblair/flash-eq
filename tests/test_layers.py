"""
Tests for equivariant layers.

Tests rotation invariance (RepNorm) and equivariance (other layers).
"""

import math
import torch
import pytest
from scipy.spatial.transform import Rotation

from flash_eq import (
    Repr, WignerD,
    RepNorm, EquivariantLinear, EquivariantGating, EquivariantLayerNorm
)


def random_rotation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random rotation.

    Returns:
        axis: (1, 3) rotation axis
        angle: (1,) rotation angle
        D: Wigner-D matrix for the rotation
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


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


LVALS_CONFIGS = [
    [0],
    [1],
    [0, 1],
    [0, 1, 2],
    [1, 2],
    [0, 2],
]

MULT_CONFIGS = [1, 4, 8]


class TestRepNorm:
    """Tests for RepNorm layer."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_rotation_invariance(self, device, lvals, mult):
        """RepNorm output should be unchanged by rotation."""
        torch.manual_seed(42)

        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        norm = RepNorm(repr).to(device)
        wigner = WignerD(repr).to(device)

        # Input tensor: (batch, mult, dim)
        x = torch.randn(16, mult, dim, device=device)

        # Compute norms before rotation
        norms_before = norm(x)

        # Apply random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)
        x_rot = torch.einsum('ij,bmj->bmi', D, x)

        # Compute norms after rotation
        norms_after = norm(x_rot)

        # Norms should be invariant
        rel_diff = (norms_before - norms_after).abs().max().item() / (norms_before.abs().max().item() + 1e-8)
        assert rel_diff < 1e-5, f"RepNorm invariance failed: rel_diff={rel_diff:.2e}"

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        norm = RepNorm(repr).to(device)

        x = torch.randn(8, 4, 9, device=device)
        out = norm(x)

        assert out.shape == (8, 4, 3), f"Expected shape (8, 4, 3), got {out.shape}"

    def test_multiple_batch_dims(self, device):
        """Test with multiple batch dimensions."""
        repr = Repr(lvals=[0, 1], mult=2)
        norm = RepNorm(repr).to(device)

        x = torch.randn(2, 3, 4, 2, 4, device=device)  # (..., mult, dim)
        out = norm(x)

        assert out.shape == (2, 3, 4, 2, 2), f"Expected shape (2, 3, 4, 2, 2), got {out.shape}"


class TestEquivariantLinear:
    """Tests for EquivariantLinear layer."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_equivariance(self, device, lvals, mult):
        """D @ layer(x) should equal layer(D @ x)."""
        torch.manual_seed(42)

        in_repr = Repr(lvals=lvals, mult=mult)
        out_repr = Repr(lvals=lvals, mult=mult * 2)
        dim = in_repr.dim()

        layer = EquivariantLinear(in_repr, out_repr).to(device)
        wigner = WignerD(in_repr).to(device)

        # Input tensor
        x = torch.randn(16, mult, dim, device=device)

        # Random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)

        # Method 1: Apply layer, then rotate
        out1 = layer(x)
        out1_rot = torch.einsum('ij,bmj->bmi', D, out1)

        # Method 2: Rotate input, then apply layer
        x_rot = torch.einsum('ij,bmj->bmi', D, x)
        out2 = layer(x_rot)

        # Check equivariance
        rel_diff = (out1_rot - out2).abs().max().item() / (out1_rot.abs().max().item() + 1e-8)
        assert rel_diff < 1e-4, f"EquivariantLinear equivariance failed: rel_diff={rel_diff:.2e}"

    def test_lvals_mismatch_raises(self, device):
        """Should raise ValueError if lvals don't match."""
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        with pytest.raises(ValueError, match="cannot modify the degrees"):
            EquivariantLinear(in_repr, out_repr)

    def test_no_bias_for_no_scalars(self, device):
        """Bias should be None if no scalar components."""
        in_repr = Repr(lvals=[1, 2], mult=4)
        out_repr = Repr(lvals=[1, 2], mult=8)

        layer = EquivariantLinear(in_repr, out_repr, bias=True)
        assert layer.bias is None

    def test_bias_disabled(self, device):
        """Bias should be None if explicitly disabled."""
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=8)

        layer = EquivariantLinear(in_repr, out_repr, bias=False)
        assert layer.bias is None

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=8)

        layer = EquivariantLinear(in_repr, out_repr).to(device)
        x = torch.randn(16, 4, 9, device=device)
        out = layer(x)

        assert out.shape == (16, 8, 9), f"Expected shape (16, 8, 9), got {out.shape}"


class TestEquivariantGating:
    """Tests for EquivariantGating layer."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_equivariance(self, device, lvals, mult):
        """D @ layer(x) should equal layer(D @ x)."""
        torch.manual_seed(42)

        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        layer = EquivariantGating(repr).to(device)
        wigner = WignerD(repr).to(device)

        # Input tensor
        x = torch.randn(16, mult, dim, device=device)

        # Random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)

        # Method 1: Apply layer, then rotate
        out1 = layer(x)
        out1_rot = torch.einsum('ij,bmj->bmi', D, out1)

        # Method 2: Rotate input, then apply layer
        x_rot = torch.einsum('ij,bmj->bmi', D, x)
        out2 = layer(x_rot)

        # Check equivariance
        rel_diff = (out1_rot - out2).abs().max().item() / (out1_rot.abs().max().item() + 1e-8)
        assert rel_diff < 1e-4, f"EquivariantGating equivariance failed: rel_diff={rel_diff:.2e}"

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = EquivariantGating(repr).to(device)

        x = torch.randn(16, 4, 9, device=device)
        out = layer(x)

        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    def test_multiple_batch_dims(self, device):
        """Test with multiple batch dimensions."""
        repr = Repr(lvals=[0, 1], mult=2)
        layer = EquivariantGating(repr).to(device)

        x = torch.randn(2, 3, 2, 4, device=device)
        out = layer(x)

        assert out.shape == x.shape


class TestEquivariantLayerNorm:
    """Tests for EquivariantLayerNorm layer."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_equivariance(self, device, lvals, mult):
        """D @ layer(x) should equal layer(D @ x)."""
        torch.manual_seed(42)

        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        layer = EquivariantLayerNorm(repr).to(device)
        wigner = WignerD(repr).to(device)

        # Input tensor
        x = torch.randn(16, mult, dim, device=device)

        # Random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)

        # Method 1: Apply layer, then rotate
        out1 = layer(x)
        out1_rot = torch.einsum('ij,bmj->bmi', D, out1)

        # Method 2: Rotate input, then apply layer
        x_rot = torch.einsum('ij,bmj->bmi', D, x)
        out2 = layer(x_rot)

        # Check equivariance
        rel_diff = (out1_rot - out2).abs().max().item() / (out1_rot.abs().max().item() + 1e-8)
        assert rel_diff < 1e-4, f"EquivariantLayerNorm equivariance failed: rel_diff={rel_diff:.2e}"

    def test_mult_1_edge_case(self, device):
        """Test that mult=1 works (skips LayerNorm)."""
        repr = Repr(lvals=[0, 1, 2], mult=1)
        layer = EquivariantLayerNorm(repr).to(device)

        assert layer.lnorm is None

        x = torch.randn(16, 1, 9, device=device)
        out = layer(x)

        assert out.shape == x.shape

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = EquivariantLayerNorm(repr).to(device)

        x = torch.randn(16, 4, 9, device=device)
        out = layer(x)

        assert out.shape == x.shape


class TestReprMethods:
    """Tests for new Repr methods."""

    def test_indices(self):
        """Test indices() method."""
        repr = Repr(lvals=[0, 1, 2])
        indices = repr.indices()
        expected = [0, 1, 1, 1, 2, 2, 2, 2, 2]
        assert indices == expected, f"Expected {expected}, got {indices}"

    def test_indices_non_contiguous(self):
        """Test indices with non-contiguous lvals."""
        repr = Repr(lvals=[0, 2])
        indices = repr.indices()
        expected = [0, 1, 1, 1, 1, 1]  # 1 scalar + 5 for l=2
        assert indices == expected, f"Expected {expected}, got {indices}"

    def test_find_scalar_with_scalar(self):
        """Test find_scalar when scalar exists."""
        repr = Repr(lvals=[0, 1, 2])
        count, locs = repr.find_scalar()
        assert count == 1
        assert locs == [0]

    def test_find_scalar_without_scalar(self):
        """Test find_scalar when no scalar."""
        repr = Repr(lvals=[1, 2])
        count, locs = repr.find_scalar()
        assert count == 0
        assert locs == []

    def test_find_scalar_scalar_not_first(self):
        """Test find_scalar when scalar is not the first irrep."""
        repr = Repr(lvals=[1, 0, 2])
        count, locs = repr.find_scalar()
        assert count == 1
        assert locs == [3]  # After l=1 (3 dims)
