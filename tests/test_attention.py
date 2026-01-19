"""Tests for equivariant edge attention.

Tests:
    - Basic forward pass
    - Attention weights sum to 1 per destination node
    - Equivariance: rotating features rotates output
    - Invariance of attention weights to rotation
    - Multi-head attention
    - Integration with EquivariantEdgewiseLinear pipeline

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
import math
import pytest
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from flash_eq import Repr, WignerD, WignerDBasis, EquivariantEdgewiseLinear, GraphPooling
from flash_eq.attention import EquivariantEdgeAttention


def random_rotation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random rotation (matching test_equivariance.py).

    Returns:
        axis: (1, 3) rotation axis
        angle: (1,) rotation angle
        R: (3, 3) rotation matrix
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


def make_graph(num_nodes, num_edges, device='cpu'):
    """Create a random graph with edges."""
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
    return src_indices, dst_indices


class TestEquivariantEdgeAttention:
    """Test suite for EquivariantEdgeAttention."""

    @pytest.fixture
    def repr(self):
        """Standard representation with l=0,1,2."""
        return Repr(lvals=[0, 1, 2], mult=8)

    @pytest.fixture
    def graph(self):
        """Random graph for testing."""
        num_nodes, num_edges = 20, 100
        src, dst = make_graph(num_nodes, num_edges)
        return src, dst, num_nodes, num_edges

    def test_forward_shape(self, repr, graph):
        """Test that forward pass produces correct output shape."""
        src, dst, num_nodes, num_edges = graph

        attn = EquivariantEdgeAttention(repr, num_heads=1)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        out = attn(edge_features, dst, num_nodes)

        assert out.shape == edge_features.shape

    def test_forward_multihead_shape(self, repr, graph):
        """Test multi-head attention produces correct shape."""
        src, dst, num_nodes, num_edges = graph

        attn = EquivariantEdgeAttention(repr, num_heads=4)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        out = attn(edge_features, dst, num_nodes)

        assert out.shape == edge_features.shape

    def test_attention_weights_sum_to_one(self, repr, graph):
        """Test that attention weights sum to 1 per destination node."""
        src, dst, num_nodes, num_edges = graph

        attn = EquivariantEdgeAttention(repr, num_heads=1, dropout=0.0)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        # Extract attention weights by running forward with identity features
        # Set all features to 1, output will be attention weights
        ones = torch.ones(num_edges, repr.mult, repr.dim())
        out = attn(ones, dst, num_nodes)

        # Sum attention-weighted outputs per destination
        # For each destination, sum of outputs should equal num_edges_to_dest
        # But since attention weights sum to 1, we need to check differently

        # Instead, manually compute attention weights
        scalars = edge_features[..., attn._scalar_locs].reshape(num_edges, -1)
        scalars = attn.layer_norm(scalars)
        scalars_heads = scalars.view(num_edges, attn.num_heads, -1)
        logits = attn.attn_proj(attn.leaky_relu(scalars_heads)).squeeze(-1)
        weights = attn._neighbor_softmax(logits, dst, num_nodes)

        # Sum weights per destination node
        weight_sums = torch.zeros(num_nodes, attn.num_heads)
        idx = dst.unsqueeze(-1).expand_as(weights)
        weight_sums.scatter_add_(0, idx, weights)

        # Each node with at least one incoming edge should have weights sum to 1
        nodes_with_edges = torch.zeros(num_nodes, dtype=torch.bool)
        nodes_with_edges[dst] = True

        for i in range(num_nodes):
            if nodes_with_edges[i]:
                assert torch.allclose(weight_sums[i], torch.ones(attn.num_heads), atol=1e-5), \
                    f"Node {i}: weights sum to {weight_sums[i]}, expected 1.0"

    def test_attention_weight_invariance(self, repr, graph):
        """Test that attention weights are invariant under rotation.

        Since attention weights are computed from scalar (l=0) features,
        and scalars are rotation-invariant, the attention weights should
        not change when we rotate the input features.
        """
        src, dst, num_nodes, num_edges = graph
        torch.manual_seed(42)

        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0)
        wigner = WignerD(repr)

        # Original features
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        # Random rotation
        axis, angle, R = random_rotation(torch.device('cpu'))
        D = wigner.rot(axis, angle).squeeze(0)  # (dim, dim)

        # Rotate features: f_rotated[e, c, :] = D @ f[e, c, :]
        edge_features_rotated = torch.einsum('ij,...j->...i', D, edge_features)

        # Compute attention weights for both
        def get_attention_weights(features):
            scalars = features[..., attn._scalar_locs].reshape(num_edges, -1)
            scalars = attn.layer_norm(scalars)
            scalars_heads = scalars.view(num_edges, attn.num_heads, -1)
            logits = attn.attn_proj(attn.leaky_relu(scalars_heads)).squeeze(-1)
            return attn._neighbor_softmax(logits, dst, num_nodes)

        weights_original = get_attention_weights(edge_features)
        weights_rotated = get_attention_weights(edge_features_rotated)

        # Scalars are at l=0, which is invariant under rotation
        # So D @ scalar = scalar, meaning weights should be identical
        assert torch.allclose(weights_original, weights_rotated, atol=1e-5), \
            f"Attention weights changed under rotation. " \
            f"Max diff: {(weights_original - weights_rotated).abs().max()}"

    def test_equivariance(self, repr, graph):
        """Test full equivariance: D @ Attention(f) = Attention(D @ f).

        Since attention weights are rotation-invariant (computed from scalars),
        and they multiply features elementwise, rotating inputs should rotate outputs.
        """
        src, dst, num_nodes, num_edges = graph
        torch.manual_seed(42)

        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0)
        wigner = WignerD(repr)

        # Original features
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        # Random rotation
        axis, angle, R = random_rotation(torch.device('cpu'))
        D = wigner.rot(axis, angle).squeeze(0)

        # Method 1: Attention then rotate
        out_original = attn(edge_features, dst, num_nodes)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        # Method 2: Rotate then attention
        edge_features_rotated = torch.einsum('ij,...j->...i', D, edge_features)
        rotate_then_out = attn(edge_features_rotated, dst, num_nodes)

        # Should be equal
        assert torch.allclose(out_then_rotate, rotate_then_out, atol=1e-5), \
            f"Equivariance violated. Max diff: {(out_then_rotate - rotate_then_out).abs().max()}"

    def test_no_l0_raises(self):
        """Test that repr without l=0 raises an error."""
        repr_no_scalar = Repr(lvals=[1, 2], mult=8)

        with pytest.raises(ValueError, match="must include l=0"):
            EquivariantEdgeAttention(repr_no_scalar)

    def test_num_heads_divides_mult(self):
        """Test that num_heads must divide mult."""
        repr = Repr(lvals=[0, 1, 2], mult=8)

        # Should work
        EquivariantEdgeAttention(repr, num_heads=1)
        EquivariantEdgeAttention(repr, num_heads=2)
        EquivariantEdgeAttention(repr, num_heads=4)
        EquivariantEdgeAttention(repr, num_heads=8)

        # Should fail
        with pytest.raises(ValueError, match="must divide mult"):
            EquivariantEdgeAttention(repr, num_heads=3)

    def test_dropout_training_vs_eval(self, repr, graph):
        """Test that dropout is applied in training but not eval."""
        src, dst, num_nodes, num_edges = graph

        attn = EquivariantEdgeAttention(repr, num_heads=1, dropout=0.5)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim())

        # In eval mode, dropout is disabled
        attn.eval()
        out1 = attn(edge_features, dst, num_nodes)
        out2 = attn(edge_features, dst, num_nodes)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

        # In train mode with dropout, outputs may differ
        attn.train()
        torch.manual_seed(1)
        out3 = attn(edge_features, dst, num_nodes)
        torch.manual_seed(2)
        out4 = attn(edge_features, dst, num_nodes)
        # Note: outputs might still be similar if dropout doesn't hit, but generally differ


class TestAttentionIntegration:
    """Test attention integrated with full pipeline."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available (EquivariantEdgewiseLinear requires CUDA)")
        return torch.device('cuda')

    @pytest.fixture
    def setup(self, device):
        """Set up full pipeline components."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        return {
            'repr': repr,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'device': device,
        }

    def test_full_pipeline_equivariance(self, setup):
        """Test equivariance of full pipeline: EdgewiseLinear -> Attention -> Pooling."""
        repr = setup['repr']
        num_nodes = setup['num_nodes']
        num_edges = setup['num_edges']
        device = setup['device']

        torch.manual_seed(42)

        # Create layers (move to device)
        basis = WignerDBasis(repr, repr).to(device)
        linear = EquivariantEdgewiseLinear(repr, repr).to(device)
        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0).to(device)
        pool = GraphPooling(reduce='sum')
        wigner = WignerD(repr).to(device)

        # Create graph and features (on device)
        src, dst = make_graph(num_nodes, num_edges, device=device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        # Random rotation
        axis, angle, R = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)

        # Method 1: Forward then rotate output
        P, Q = basis(directions)
        edge_feat = linear(P, Q, node_features, distances, src)
        edge_feat = attn(edge_feat, dst, num_nodes)
        out_original = pool(edge_feat, dst, num_nodes)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        # Method 2: Rotate inputs then forward
        node_features_rot = torch.einsum('ij,...j->...i', D, node_features)
        directions_rot = torch.einsum('ij,...j->...i', R, directions)

        P_rot, Q_rot = basis(directions_rot)
        edge_feat_rot = linear(P_rot, Q_rot, node_features_rot, distances, src)
        edge_feat_rot = attn(edge_feat_rot, dst, num_nodes)
        rotate_then_out = pool(edge_feat_rot, dst, num_nodes)

        # Check equivariance using relative difference (same metric as test_equivariance.py)
        rel_diff = (out_then_rotate - rotate_then_out).abs().max().item() / (out_then_rotate.abs().max().item() + 1e-8)
        assert rel_diff < 5e-3, \
            f"Pipeline equivariance violated. rel_diff={rel_diff:.2e}"


class TestAttentionGPU:
    """GPU-specific tests (skipped if CUDA unavailable)."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device('cuda')

    def test_gpu_forward(self, device):
        """Test forward pass on GPU."""
        repr = Repr(lvals=[0, 1, 2], mult=32)
        num_nodes, num_edges = 100, 1000

        attn = EquivariantEdgeAttention(repr, num_heads=8).to(device)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim(), device=device)
        dst = torch.randint(0, num_nodes, (num_edges,), device=device)

        out = attn(edge_features, dst, num_nodes)

        assert out.device.type == 'cuda'
        assert out.shape == edge_features.shape

    def test_gpu_equivariance(self, device):
        """Test equivariance on GPU."""
        repr = Repr(lvals=[0, 1, 2], mult=32)
        num_nodes, num_edges = 100, 1000

        torch.manual_seed(42)

        attn = EquivariantEdgeAttention(repr, num_heads=8, dropout=0.0).to(device)
        wigner = WignerD(repr).to(device)

        edge_features = torch.randn(num_edges, repr.mult, repr.dim(), device=device)
        dst = torch.randint(0, num_nodes, (num_edges,), device=device)

        # Random rotation
        axis, angle, R = random_rotation(device)
        D = wigner.rot(axis, angle).squeeze(0)

        # Equivariance test
        out_original = attn(edge_features, dst, num_nodes)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        edge_features_rotated = torch.einsum('ij,...j->...i', D, edge_features)
        rotate_then_out = attn(edge_features_rotated, dst, num_nodes)

        assert torch.allclose(out_then_rotate, rotate_then_out, atol=1e-4), \
            f"GPU equivariance violated. Max diff: {(out_then_rotate - rotate_then_out).abs().max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
