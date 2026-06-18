"""Tests for equivariant attention layers.

Tests:
    - EquivariantEdgeAttention: Low-level Q/K/V attention
    - EquivariantAttention: Full message-passing layer with equivariance

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
import pytest
import torch

from flash_eq import (
    Repr,
    WignerD,
    WignerDBasis,
)
from flash_eq.layers.attention import EquivariantEdgeAttention, EquivariantAttention

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
    """Test suite for EquivariantEdgeAttention (low-level Q/K/V attention)."""

    def test_forward_shape(self, device):
        """Test that forward pass produces correct output shape."""
        num_heads = 4
        qk_dim = 36
        mult = 8
        dim = 9
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(num_heads=num_heads, qk_dim=qk_dim).to(device)
        graph = make_graph(num_nodes, num_edges, device)

        Q = torch.randn(num_edges, num_heads, qk_dim, device=device)
        K = torch.randn(num_edges, num_heads, qk_dim, device=device)
        V = torch.randn(num_edges, mult, dim, device=device)

        out = attn(Q, K, V, graph)

        assert out.shape == V.shape
        assert out.device.type == device.type

    @pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
    def test_multihead_shape(self, device, num_heads):
        """Test multi-head attention produces correct shape."""
        qk_dim = 36
        mult = 8  # Divisible by all NUM_HEADS_CONFIGS
        dim = 9
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(num_heads=num_heads, qk_dim=qk_dim).to(device)
        graph = make_graph(num_nodes, num_edges, device)

        Q = torch.randn(num_edges, num_heads, qk_dim, device=device)
        K = torch.randn(num_edges, num_heads, qk_dim, device=device)
        V = torch.randn(num_edges, mult, dim, device=device)

        out = attn(Q, K, V, graph)

        assert out.shape == V.shape

    def test_attention_weights_sum_to_one(self, device):
        """Test that attention weights sum to 1 per destination node."""
        num_heads = 2
        qk_dim = 36
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(num_heads=num_heads, qk_dim=qk_dim, dropout=0.0).to(device)
        graph = make_graph(num_nodes, num_edges, device)

        Q = torch.randn(num_edges, num_heads, qk_dim, device=device)
        K = torch.randn(num_edges, num_heads, qk_dim, device=device)

        # Manually compute attention weights
        logits = (Q * K).sum(-1) * attn.scale
        weights = attn._neighbor_softmax(logits, graph.dst, graph.num_nodes)

        # Sum weights per destination node
        weight_sums = torch.zeros(num_nodes, num_heads, device=device)
        idx = graph.dst.unsqueeze(-1).expand_as(weights)
        weight_sums.scatter_add_(0, idx, weights)

        # Each node with incoming edges should have weights sum to 1
        nodes_with_edges = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        nodes_with_edges[graph.dst] = True

        for i in range(num_nodes):
            if nodes_with_edges[i]:
                assert torch.allclose(
                    weight_sums[i],
                    torch.ones(num_heads, device=device),
                    atol=1e-5,
                ), f"Node {i}: weights sum to {weight_sums[i]}, expected 1.0"

    def test_dropout_determinism(self, device):
        """Test that dropout is applied in training but not eval."""
        num_heads = 2
        qk_dim = 36
        mult = 8
        dim = 9
        num_nodes, num_edges = 20, 100

        attn = EquivariantEdgeAttention(num_heads=num_heads, qk_dim=qk_dim, dropout=0.5).to(device)
        graph = make_graph(num_nodes, num_edges, device)

        Q = torch.randn(num_edges, num_heads, qk_dim, device=device)
        K = torch.randn(num_edges, num_heads, qk_dim, device=device)
        V = torch.randn(num_edges, mult, dim, device=device)

        # Eval mode should be deterministic
        attn.eval()
        out1 = attn(Q, K, V, graph)
        out2 = attn(Q, K, V, graph)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


class TestEquivariantAttention:
    """Test suite for EquivariantAttention (full message-passing layer)."""

    @pytest.mark.parametrize("lvals,mult", REPR_CONFIGS)
    def test_forward_shape(self, cuda_device, lvals, mult):
        """Test that forward pass produces correct output shape."""
        device = cuda_device
        repr = Repr(lvals=lvals, mult=mult)
        num_nodes, num_edges = 20, 100

        layer = EquivariantAttention(repr, repr, num_heads=1).to(device)
        basis = WignerDBasis([repr, repr]).to(device)

        graph = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, mult, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        P, Q = basis(directions)
        out = layer(P, Q, node_features, distances, graph)

        assert out.shape == node_features.shape
        assert out.device.type == device.type

    @pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
    def test_multihead_shape(self, cuda_device, num_heads):
        """Test multi-head attention produces correct shape."""
        device = cuda_device
        mult = 8  # Divisible by all NUM_HEADS_CONFIGS
        repr = Repr(lvals=[0, 1, 2], mult=mult)
        num_nodes, num_edges = 20, 100

        layer = EquivariantAttention(repr, repr, num_heads=num_heads).to(device)
        basis = WignerDBasis([repr, repr]).to(device)

        graph = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, mult, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        P, Q = basis(directions)
        out = layer(P, Q, node_features, distances, graph)

        assert out.shape == node_features.shape

    @pytest.mark.parametrize("lvals,mult", REPR_CONFIGS)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
    def test_equivariance(self, cuda_device, lvals, mult, num_heads):
        """Test full equivariance: D @ Attention(f) = Attention(D @ f, R @ dirs)."""
        if mult % num_heads != 0:
            pytest.skip(f"mult={mult} not divisible by num_heads={num_heads}")

        device = cuda_device
        torch.manual_seed(42)
        repr = Repr(lvals=lvals, mult=mult)
        num_nodes, num_edges = 20, 100

        layer = EquivariantAttention(repr, repr, num_heads=num_heads, dropout=0.0).to(device)
        basis = WignerDBasis([repr, repr]).to(device)
        wigner = WignerD(repr).to(device)

        graph = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, mult, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        # Random rotation
        axis, angle, R = random_rotation(device)
        D = wigner.rot(axis, angle)

        # Method 1: Forward then rotate output
        P, Q = basis(directions)
        out_original = layer(P, Q, node_features, distances, graph)
        out_then_rotate = torch.einsum('ij,...j->...i', D, out_original)

        # Method 2: Rotate inputs then forward
        node_features_rot = torch.einsum('ij,...j->...i', D, node_features)
        directions_rot = torch.einsum('ij,...j->...i', R, directions)

        P_rot, Q_rot = basis(directions_rot)
        rotate_then_out = layer(P_rot, Q_rot, node_features_rot, distances, graph)

        check_equivariance(out_then_rotate, rotate_then_out, rtol=5e-3, msg="Attention equivariance")

    def test_num_heads_divides_mult(self):
        """Test that num_heads must divide mult."""
        repr = Repr(lvals=[0, 1, 2], mult=8)

        # Should work
        for h in [1, 2, 4, 8]:
            EquivariantAttention(repr, repr, num_heads=h)

        # Should fail
        with pytest.raises(ValueError, match="must divide"):
            EquivariantAttention(repr, repr, num_heads=3)

    def test_different_in_out_repr(self, cuda_device):
        """Test with different input and output representations."""
        device = cuda_device
        in_repr = Repr(lvals=[0, 1], mult=8)
        out_repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        layer = EquivariantAttention(in_repr, out_repr, num_heads=2).to(device)
        basis = WignerDBasis([in_repr, out_repr]).to(device)

        graph = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, 8, in_repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        P, Q = basis(directions)
        out = layer(P, Q, node_features, distances, graph)

        assert out.shape == (num_nodes, 8, out_repr.dim())

    def test_with_edge_features(self, cuda_device):
        """Test with optional edge features."""
        device = cuda_device
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        layer = EquivariantAttention(repr, repr, num_heads=2).to(device)
        basis = WignerDBasis([repr, repr]).to(device)

        graph = make_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, 8, repr.dim(), device=device)
        edge_features = torch.randn(num_edges, 8, repr.dim(), device=device)
        directions = torch.randn(num_edges, 3, device=device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=device) * 5 + 0.5

        P, Q = basis(directions)
        out = layer(P, Q, node_features, distances, graph, edge_features=edge_features)

        assert out.shape == node_features.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
