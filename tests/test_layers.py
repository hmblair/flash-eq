"""
Tests for equivariant layers.

Tests rotation invariance (RepNorm) and equivariance (other layers).
"""

import math
import torch
import pytest
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from flash_eq import (
    Repr, WignerD,
    RepNorm, EquivariantLinear, EquivariantGating, EquivariantLayerNorm,
    SeparableEquivariantLayerNorm, GraphPooling,
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


class TestSeparableEquivariantLayerNorm:
    """Tests for SeparableEquivariantLayerNorm layer."""

    @pytest.mark.parametrize("lvals", LVALS_CONFIGS)
    @pytest.mark.parametrize("mult", MULT_CONFIGS)
    def test_equivariance(self, device, lvals, mult):
        """D @ layer(x) should equal layer(D @ x)."""
        torch.manual_seed(42)

        repr = Repr(lvals=lvals, mult=mult)
        dim = repr.dim()

        layer = SeparableEquivariantLayerNorm(repr).to(device)
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
        assert rel_diff < 1e-4, f"SeparableEquivariantLayerNorm equivariance failed: rel_diff={rel_diff:.2e}"

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        x = torch.randn(16, 4, 9, device=device)
        out = layer(x)

        assert out.shape == x.shape

    def test_no_scalars(self, device):
        """Test with representation containing no scalars."""
        repr = Repr(lvals=[1, 2], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        assert layer.nscalar == 0
        assert layer.nhigher == 2
        assert not hasattr(layer, 'scalar_norm')
        assert hasattr(layer, 'higher_norm')

        x = torch.randn(16, 4, 8, device=device)  # dim = 3 + 5 = 8
        out = layer(x)

        assert out.shape == x.shape

    def test_only_scalars(self, device):
        """Test with representation containing only scalars."""
        repr = Repr(lvals=[0], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        assert layer.nscalar == 1
        assert layer.nhigher == 0
        assert hasattr(layer, 'scalar_norm')
        assert not hasattr(layer, 'higher_norm')

        x = torch.randn(16, 4, 1, device=device)
        out = layer(x)

        assert out.shape == x.shape

    def test_preserves_higher_degree_ratios(self, device):
        """Test that L>0 components maintain relative magnitudes."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        # Create input where l=1 has 2x the magnitude of l=2
        x = torch.zeros(16, 4, 9, device=device)
        x[..., 1:4] = 2.0  # l=1 components
        x[..., 4:9] = 1.0  # l=2 components

        out = layer(x)

        # The ratio of l=1 to l=2 norms should be preserved
        l1_norm = out[..., 1:4].norm(dim=-1).mean()
        l2_norm = out[..., 4:9].norm(dim=-1).mean()
        ratio = l1_norm / l2_norm

        # Original ratio was sqrt(3*4) / sqrt(5*1) = sqrt(12/5) ≈ 1.55
        expected_ratio = math.sqrt(3 * 4.0 / (5 * 1.0))
        assert abs(ratio - expected_ratio) < 0.1, f"Ratio {ratio:.2f} != expected {expected_ratio:.2f}"

    def test_multiple_batch_dims(self, device):
        """Test with multiple batch dimensions."""
        repr = Repr(lvals=[0, 1], mult=2)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        x = torch.randn(2, 3, 2, 4, device=device)
        out = layer(x)

        assert out.shape == x.shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the layer."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        x = torch.randn(16, 4, 9, device=device, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_learnable_parameters(self, device):
        """Test that learnable parameters are created correctly."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = SeparableEquivariantLayerNorm(repr).to(device)

        # Check scalar norm params
        assert layer.scalar_norm.weight.shape == (4, 1)  # (mult, nscalar)
        assert layer.scalar_norm.bias.shape == (4, 1)

        # Check higher norm params
        assert layer.higher_norm.weight.shape == (4,)  # (mult,)


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


class TestGraphPooling:
    """Tests for GraphPooling layer."""

    @pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
    def test_output_shape(self, device, reduce):
        """Test that output shape is correct."""
        pool = GraphPooling(reduce=reduce)

        num_nodes, num_edges, channels, dim = 10, 100, 8, 9
        edge_features = torch.randn(num_edges, channels, dim, device=device)
        dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

        out = pool(edge_features, dst_indices, num_nodes)

        assert out.shape == (num_nodes, channels, dim)

    @pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
    def test_dtype_preservation(self, device, reduce):
        """Test that output dtype matches input dtype."""
        pool = GraphPooling(reduce=reduce)

        for dtype in [torch.float32, torch.float64]:
            edge_features = torch.randn(50, 4, 9, device=device, dtype=dtype)
            dst_indices = torch.randint(0, 10, (50,), device=device)

            out = pool(edge_features, dst_indices, 10)
            assert out.dtype == dtype

    def test_sum_correctness(self, device):
        """Test that sum pooling produces correct results."""
        pool = GraphPooling(reduce='sum')

        # Simple case: 4 edges -> 2 nodes
        edge_features = torch.tensor([
            [[1.0, 2.0]],  # edge 0 -> node 0
            [[3.0, 4.0]],  # edge 1 -> node 0
            [[5.0, 6.0]],  # edge 2 -> node 1
            [[7.0, 8.0]],  # edge 3 -> node 1
        ], device=device)
        dst_indices = torch.tensor([0, 0, 1, 1], device=device)

        out = pool(edge_features, dst_indices, num_nodes=2)

        expected = torch.tensor([
            [[4.0, 6.0]],   # 1+3, 2+4
            [[12.0, 14.0]], # 5+7, 6+8
        ], device=device)

        assert torch.allclose(out, expected)

    def test_mean_correctness(self, device):
        """Test that mean pooling produces correct results."""
        pool = GraphPooling(reduce='mean')

        edge_features = torch.tensor([
            [[2.0, 4.0]],  # edge 0 -> node 0
            [[4.0, 8.0]],  # edge 1 -> node 0
            [[9.0, 3.0]],  # edge 2 -> node 1
        ], device=device)
        dst_indices = torch.tensor([0, 0, 1], device=device)

        out = pool(edge_features, dst_indices, num_nodes=2)

        expected = torch.tensor([
            [[3.0, 6.0]],  # mean of [2,4] and [4,8]
            [[9.0, 3.0]],  # single edge
        ], device=device)

        assert torch.allclose(out, expected)

    def test_max_correctness(self, device):
        """Test that max pooling produces correct results."""
        pool = GraphPooling(reduce='max')

        edge_features = torch.tensor([
            [[1.0, 5.0]],  # edge 0 -> node 0
            [[3.0, 2.0]],  # edge 1 -> node 0
            [[7.0, 8.0]],  # edge 2 -> node 1
        ], device=device)
        dst_indices = torch.tensor([0, 0, 1], device=device)

        out = pool(edge_features, dst_indices, num_nodes=2)

        expected = torch.tensor([
            [[3.0, 5.0]],  # max(1,3), max(5,2)
            [[7.0, 8.0]],  # single edge
        ], device=device)

        assert torch.allclose(out, expected)

    def test_zero_edges(self, device):
        """Test with zero edges (empty graph)."""
        pool = GraphPooling(reduce='sum')

        edge_features = torch.zeros(0, 4, 9, device=device)
        dst_indices = torch.zeros(0, dtype=torch.long, device=device)

        out = pool(edge_features, dst_indices, num_nodes=5)

        assert out.shape == (5, 4, 9)
        assert (out == 0).all()

    def test_nodes_with_no_edges(self, device):
        """Test that nodes with no incoming edges get zeros."""
        pool_sum = GraphPooling(reduce='sum')
        pool_mean = GraphPooling(reduce='mean')
        pool_max = GraphPooling(reduce='max')

        # All edges go to node 0, nodes 1-4 have no edges
        edge_features = torch.randn(10, 4, 9, device=device)
        dst_indices = torch.zeros(10, dtype=torch.long, device=device)

        for pool in [pool_sum, pool_mean, pool_max]:
            out = pool(edge_features, dst_indices, num_nodes=5)

            # Nodes 1-4 should be zero
            assert (out[1:] == 0).all()
            # Node 0 should be non-zero (with high probability)
            assert out[0].abs().sum() > 0

    def test_all_edges_to_one_node(self, device):
        """Test when all edges point to a single node."""
        pool = GraphPooling(reduce='sum')

        num_edges = 100
        edge_features = torch.ones(num_edges, 2, 3, device=device)
        dst_indices = torch.zeros(num_edges, dtype=torch.long, device=device)

        out = pool(edge_features, dst_indices, num_nodes=10)

        # Node 0 should have sum = num_edges
        assert torch.allclose(out[0], torch.full((2, 3), float(num_edges), device=device))
        # Other nodes should be zero
        assert (out[1:] == 0).all()

    def test_single_edge(self, device):
        """Test with a single edge."""
        pool = GraphPooling(reduce='sum')

        edge_features = torch.tensor([[[1.0, 2.0, 3.0]]], device=device)
        dst_indices = torch.tensor([2], device=device)

        out = pool(edge_features, dst_indices, num_nodes=5)

        assert out.shape == (5, 1, 3)
        assert torch.allclose(out[2], edge_features[0])
        assert (out[:2] == 0).all()
        assert (out[3:] == 0).all()

    def test_invalid_reduce_raises(self):
        """Test that invalid reduce parameter raises ValueError."""
        with pytest.raises(ValueError, match="reduce must be one of"):
            GraphPooling(reduce='invalid')

    def test_extra_repr(self):
        """Test string representation."""
        pool = GraphPooling(reduce='mean')
        assert "mean" in pool.extra_repr()

    @pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
    def test_gradient_flow(self, device, reduce):
        """Test that gradients flow through pooling."""
        pool = GraphPooling(reduce=reduce)

        edge_features = torch.randn(50, 4, 9, device=device, requires_grad=True)
        dst_indices = torch.randint(0, 10, (50,), device=device)

        out = pool(edge_features, dst_indices, num_nodes=10)
        loss = out.sum()
        loss.backward()

        assert edge_features.grad is not None
        assert edge_features.grad.shape == edge_features.shape

    def test_large_scale(self, device):
        """Test with large number of edges."""
        pool = GraphPooling(reduce='sum')

        num_nodes, num_edges = 1000, 100000
        edge_features = torch.randn(num_edges, 32, 9, device=device)
        dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

        out = pool(edge_features, dst_indices, num_nodes)

        assert out.shape == (num_nodes, 32, 9)
        # Verify it ran without error and produced finite values
        assert torch.isfinite(out).all()
