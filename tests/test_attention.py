"""Tests for equivariant edge attention.

Tests:
    - Forward pass shape and device handling
    - Attention weights sum to 1 per destination node
    - Attention weight invariance under rotation
    - Full equivariance: D @ Attention(f) = Attention(D @ f)
    - Multi-head attention
    - Integration with EquivariantEdgewiseLinear pipeline (CUDA only)

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
import pytest
import torch

from flash_eq import (
    Repr,
    WignerD,
    WignerDBasis,
    EquivariantEdgewiseLinear,
    GraphPooling,
)
from flash_eq.layers.attention import EquivariantEdgeAttention

from .helpers import random_rotation, make_graph, check_equivariance


# Test configurations
REPR_CONFIGS = [
    ([0, 1], 4),
    ([0, 1, 2], 8),
    ([0, 1, 2], 16),
    # Duplicate l values
    ([0, 0, 1], 4),
    ([0, 1, 1, 2], 8),
]

NUM_HEADS_CONFIGS = [1, 2, 4]


class TestEquivariantEdgeAttention:
    """Test suite for EquivariantEdgeAttention."""

    @pytest.mark.parametrize("lvals,mult", REPR_CONFIGS)
    def test_forward_shape(self, device, lvals, mult):
        """Test that forward pass produces correct output shape."""
        repr = Repr(lvals=lvals, mult=mult)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=1).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, mult, repr.dim(), device=device)

        out = attn(edge_features, dst, num_nodes)

        assert out.shape == edge_features.shape
        assert out.device.type == device.type

    @pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
    def test_multihead_shape(self, device, num_heads):
        """Test multi-head attention produces correct shape."""
        mult = 8  # Divisible by all NUM_HEADS_CONFIGS
        repr = Repr(lvals=[0, 1, 2], mult=mult)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=num_heads).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, mult, repr.dim(), device=device)

        out = attn(edge_features, dst, num_nodes)

        assert out.shape == edge_features.shape

    def test_attention_weights_sum_to_one(self, device):
        """Test that attention weights sum to 1 per destination node."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim(), device=device)

        # Manually compute attention weights
        scalars = edge_features[..., attn._scalar_locs].reshape(num_edges, -1)
        scalars = attn.layer_norm(scalars)
        scalars_heads = scalars.view(num_edges, attn.num_heads, -1)
        logits = attn.attn_proj(attn.leaky_relu(scalars_heads)).squeeze(-1)
        weights = attn._neighbor_softmax(logits, dst, num_nodes)

        # Sum weights per destination node
        weight_sums = torch.zeros(num_nodes, attn.num_heads, device=device)
        idx = dst.unsqueeze(-1).expand_as(weights)
        weight_sums.scatter_add_(0, idx, weights)

        # Each node with incoming edges should have weights sum to 1
        nodes_with_edges = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        nodes_with_edges[dst] = True

        for i in range(num_nodes):
            if nodes_with_edges[i]:
                assert torch.allclose(
                    weight_sums[i],
                    torch.ones(attn.num_heads, device=device),
                    atol=1e-5,
                ), f"Node {i}: weights sum to {weight_sums[i]}, expected 1.0"

    @pytest.mark.parametrize("lvals,mult", REPR_CONFIGS)
    def test_attention_weight_invariance(self, device, lvals, mult):
        """Test that attention weights are invariant under rotation.

        Attention weights are computed from scalar (l=0) features,
        which are rotation-invariant.
        """
        torch.manual_seed(42)
        repr = Repr(lvals=lvals, mult=mult)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0).to(device)
        wigner = WignerD(repr).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, mult, repr.dim(), device=device)

        # Random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Rotate features
        edge_features_rotated = torch.einsum('ij,...j->...i', D, edge_features)

        # Compute attention weights for both
        def get_weights(features):
            scalars = features[..., attn._scalar_locs].reshape(num_edges, -1)
            scalars = attn.layer_norm(scalars)
            scalars_heads = scalars.view(num_edges, attn.num_heads, -1)
            logits = attn.attn_proj(attn.leaky_relu(scalars_heads)).squeeze(-1)
            return attn._neighbor_softmax(logits, dst, num_nodes)

        weights_original = get_weights(edge_features)
        weights_rotated = get_weights(edge_features_rotated)

        assert torch.allclose(weights_original, weights_rotated, atol=1e-5), \
            f"Attention weights changed under rotation. " \
            f"Max diff: {(weights_original - weights_rotated).abs().max()}"

    @pytest.mark.parametrize("lvals,mult", REPR_CONFIGS)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
    def test_equivariance(self, device, lvals, mult, num_heads):
        """Test full equivariance: D @ Attention(f) = Attention(D @ f)."""
        if mult % num_heads != 0:
            pytest.skip(f"mult={mult} not divisible by num_heads={num_heads}")

        torch.manual_seed(42)
        repr = Repr(lvals=lvals, mult=mult)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=num_heads, dropout=0.0).to(device)
        wigner = WignerD(repr).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, mult, repr.dim(), device=device)

        # Random rotation
        axis, angle, _ = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Method 1: Attention then rotate
        out_original = attn(edge_features, dst, num_nodes)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        # Method 2: Rotate then attention
        edge_features_rotated = torch.einsum('ij,...j->...i', D, edge_features)
        rotate_then_out = attn(edge_features_rotated, dst, num_nodes)

        check_equivariance(out_then_rotate, rotate_then_out, rtol=1e-4)

    def test_no_l0_raises(self):
        """Test that repr without l=0 raises an error."""
        repr_no_scalar = Repr(lvals=[1, 2], mult=8)

        with pytest.raises(ValueError, match="must include l=0"):
            EquivariantEdgeAttention(repr_no_scalar)

    def test_num_heads_divides_mult(self):
        """Test that num_heads must divide mult."""
        repr = Repr(lvals=[0, 1, 2], mult=8)

        # Should work
        for h in [1, 2, 4, 8]:
            EquivariantEdgeAttention(repr, num_heads=h)

        # Should fail
        with pytest.raises(ValueError, match="must divide mult"):
            EquivariantEdgeAttention(repr, num_heads=3)

    def test_dropout_determinism(self, device):
        """Test that dropout is applied in training but not eval."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(repr, num_heads=1, dropout=0.5).to(device)
        _, dst = make_graph(num_nodes, num_edges, device)
        edge_features = torch.randn(num_edges, repr.mult, repr.dim(), device=device)

        # Eval mode should be deterministic
        attn.eval()
        out1 = attn(edge_features, dst, num_nodes)
        out2 = attn(edge_features, dst, num_nodes)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


class TestFullPipeline:
    """Test attention integrated with EquivariantEdgewiseLinear (CUDA only)."""

    def test_pipeline_equivariance(self, cuda_device):
        """Test equivariance: EdgewiseLinear -> Attention -> Pooling."""
        torch.manual_seed(42)
        device = cuda_device

        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        # Create layers
        basis = WignerDBasis([repr, repr]).to(device)
        linear = EquivariantEdgewiseLinear(repr, repr).to(device)
        attn = EquivariantEdgeAttention(repr, num_heads=2, dropout=0.0).to(device)
        pool = GraphPooling(reduce='sum')
        wigner = WignerD(repr).to(device)

        # Create graph and features
        src, dst = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        # Random rotation
        axis, angle, R = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Method 1: Forward then rotate output
        P, Q = basis(directions)
        edge_feat = node_features[src]  # gather to edges
        edge_feat = linear(P, Q, edge_feat, distances)
        edge_feat = attn(edge_feat, dst, num_nodes)
        out_original = pool(edge_feat, dst, num_nodes)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        # Method 2: Rotate inputs then forward
        node_features_rot = torch.einsum('ij,...j->...i', D, node_features)
        directions_rot = torch.einsum('ij,...j->...i', R, directions)

        P_rot, Q_rot = basis(directions_rot)
        edge_feat_rot = node_features_rot[src]  # gather to edges
        edge_feat_rot = linear(P_rot, Q_rot, edge_feat_rot, distances)
        edge_feat_rot = attn(edge_feat_rot, dst, num_nodes)
        rotate_then_out = pool(edge_feat_rot, dst, num_nodes)

        # Looser tolerance for full pipeline due to accumulated numerical error
        check_equivariance(out_then_rotate, rotate_then_out, rtol=5e-3, msg="Pipeline equivariance")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
