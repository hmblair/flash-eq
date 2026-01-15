"""Tests for SO(3) equivariance of EquivariantLinear."""

import torch
import pytest
from scipy.spatial.transform import Rotation

from flash_eq import Repr, EquivariantLinear

# Check if CUDA is available for testing
CUDA_AVAILABLE = torch.cuda.is_available()


def _reorder_axis(axis: torch.Tensor) -> torch.Tensor:
    """Reorder axis from standard (x,y,z) to generator ordering (y,z,x)."""
    return torch.stack([axis[..., 1], axis[..., 2], axis[..., 0]], dim=-1)


class TestEquivariantLinear:
    """Test suite for EquivariantLinear."""

    @pytest.fixture
    def layer(self):
        """Create a layer with lmax=2."""
        repr_in = Repr(lvals=[0, 1, 2])
        repr_out = Repr(lvals=[0, 1, 2])
        return EquivariantLinear(repr_in, repr_out)

    def test_output_shape(self, layer):
        """Test that output has correct shape."""
        batch, channels_in, channels_out = 10, 4, 8

        features = torch.randn(batch, channels_in, layer.repr_in.dim())
        directions = torch.randn(batch, 3)
        weights = torch.randn(batch, channels_out, channels_in, layer.weight_dim)

        output = layer(features, directions, weights)

        assert output.shape == (batch, channels_out, layer.repr_out.dim())

    def test_equivariance(self, layer):
        """Test SO(3) equivariance: f(D@x, D@d) = D @ f(x, d)."""
        torch.manual_seed(42)

        batch, channels_in, channels_out = 8, 4, 4
        features = torch.randn(batch, channels_in, layer.repr_in.dim())
        directions = torch.randn(batch, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        weights = torch.randn(batch, channels_out, channels_in, layer.weight_dim) * 0.1

        # Random rotation
        axis_std = torch.randn(3)
        axis_std = axis_std / axis_std.norm()
        angle = torch.tensor(0.7)

        axis_reordered = _reorder_axis(axis_std)
        D_in = layer.repr_in.rot(axis_reordered.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)
        D_out = layer.repr_out.rot(axis_reordered.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

        R = torch.from_numpy(
            Rotation.from_rotvec((axis_std * angle).numpy()).as_matrix()
        ).float()

        # Method 1: Compute output, then rotate
        out1 = layer(features, directions, weights)
        out1_rot = torch.einsum('ij,bcj->bci', D_out, out1)

        # Method 2: Rotate inputs, then compute
        features_rot = torch.einsum('ij,bcj->bci', D_in, features)
        directions_rot = directions @ R.T
        out2 = layer(features_rot, directions_rot, weights)

        # Should match
        diff = (out1_rot - out2).abs().max().item()
        assert diff < 1e-5, f"Equivariance failed: max diff = {diff:.2e}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_equivariance_cuda(self):
        """Test SO(3) equivariance with CUDA kernel: f(D@x, D@d) = D @ f(x, d)."""
        torch.manual_seed(42)

        repr_in = Repr(lvals=[0, 1, 2])
        repr_out = Repr(lvals=[0, 1, 2])
        layer = EquivariantLinear(repr_in, repr_out, use_cuda=True).cuda()

        batch, channels_in, channels_out = 8, 4, 4
        features = torch.randn(batch, channels_in, layer.repr_in.dim(), device='cuda')
        directions = torch.randn(batch, 3, device='cuda')
        directions = directions / directions.norm(dim=-1, keepdim=True)
        weights = torch.randn(batch, channels_out, channels_in, layer.weight_dim, device='cuda') * 0.1

        # Random rotation
        axis_std = torch.randn(3)
        axis_std = axis_std / axis_std.norm()
        angle = torch.tensor(0.7)

        axis_reordered = _reorder_axis(axis_std)
        D_in = layer.repr_in.rot(axis_reordered.unsqueeze(0), angle.unsqueeze(0)).squeeze(0).cuda()
        D_out = layer.repr_out.rot(axis_reordered.unsqueeze(0), angle.unsqueeze(0)).squeeze(0).cuda()

        R = torch.from_numpy(
            Rotation.from_rotvec((axis_std * angle).numpy()).as_matrix()
        ).float().cuda()

        # Method 1: Compute output, then rotate
        out1 = layer(features, directions, weights)
        out1_rot = torch.einsum('ij,bcj->bci', D_out, out1)

        # Method 2: Rotate inputs, then compute
        features_rot = torch.einsum('ij,bcj->bci', D_in, features)
        directions_rot = directions @ R.T
        out2 = layer(features_rot, directions_rot, weights)

        # Should match
        diff = (out1_rot - out2).abs().max().item()
        assert diff < 1e-4, f"CUDA equivariance failed: max diff = {diff:.2e}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_matches_python(self):
        """Test that CUDA kernel produces same output as Python implementation."""
        torch.manual_seed(123)

        repr_in = Repr(lvals=[0, 1, 2])
        repr_out = Repr(lvals=[0, 1, 2])

        layer_python = EquivariantLinear(repr_in, repr_out, use_cuda=False).cuda()
        layer_cuda = EquivariantLinear(repr_in, repr_out, use_cuda=True).cuda()

        batch, channels_in, channels_out = 8, 4, 4
        features = torch.randn(batch, channels_in, repr_in.dim(), device='cuda')
        directions = torch.randn(batch, 3, device='cuda')
        directions = directions / directions.norm(dim=-1, keepdim=True)
        weights = torch.randn(batch, channels_out, channels_in, layer_python.weight_dim, device='cuda')

        out_python = layer_python(features, directions, weights)
        out_cuda = layer_cuda(features, directions, weights)

        diff = (out_python - out_cuda).abs().max().item()
        rel_diff = diff / (out_python.abs().max().item() + 1e-8)
        assert rel_diff < 1e-5, f"CUDA vs Python mismatch: rel_diff = {rel_diff:.2e}"

    def test_weight_dim(self, layer):
        """Test that weight_dim matches expected formula."""
        # For lvals=[0,1,2], weight structure per (l1,l2) pair:
        # (0,0): 1, (0,1): 1, (0,2): 1
        # (1,0): 1, (1,1): 3, (1,2): 3
        # (2,0): 1, (2,1): 3, (2,2): 5
        # Total = 1+1+1 + 1+3+3 + 1+3+5 = 19
        expected = 19
        assert layer.weight_dim == expected, f"Expected {expected}, got {layer.weight_dim}"

    def test_different_repr(self):
        """Test with different input/output representations."""
        repr_in = Repr(lvals=[0, 1])
        repr_out = Repr(lvals=[1, 2])
        layer = EquivariantLinear(repr_in, repr_out)

        batch, channels_in, channels_out = 5, 2, 3
        features = torch.randn(batch, channels_in, repr_in.dim())
        directions = torch.randn(batch, 3)
        weights = torch.randn(batch, channels_out, channels_in, layer.weight_dim)

        output = layer(features, directions, weights)
        assert output.shape == (batch, channels_out, repr_out.dim())

    def test_scalar_only(self):
        """Test with scalar-only representation."""
        repr_in = Repr(lvals=[0])
        repr_out = Repr(lvals=[0])
        layer = EquivariantLinear(repr_in, repr_out)

        assert layer.weight_dim == 1

        batch = 4
        features = torch.randn(batch, 2, 1)
        directions = torch.randn(batch, 3)
        weights = torch.randn(batch, 3, 2, 1)

        output = layer(features, directions, weights)
        assert output.shape == (batch, 3, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
