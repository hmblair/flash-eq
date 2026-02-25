"""Tests for SequencePositionEncoding."""

import torch
import pytest

import torch.nn as nn

from flash_eq import (
    SequencePositionEncoding,
    EquivariantTransformerBlock,
    Repr,
    WignerD,
    WignerDBasis,
)
from .helpers import random_rotation, make_graph, check_equivariance


class TestSequencePositionEncoding:
    """Tests for sinusoidal and learnable sequence encodings."""

    @pytest.mark.parametrize("dim", [8, 16, 32, 33])
    def test_output_shape(self, dim):
        enc = SequencePositionEncoding(dim=dim)
        seq_pos = torch.arange(50)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 0, 10, 49])
        out = enc(seq_pos, src, dst)
        assert out.shape == (4, dim)

    def test_symmetry(self):
        """Encoding depends on |i - j|, so swapping src/dst gives same result."""
        enc = SequencePositionEncoding(dim=16)
        seq_pos = torch.arange(100)
        src = torch.tensor([0, 5, 10])
        dst = torch.tensor([3, 20, 50])
        forward = enc(seq_pos, src, dst)
        backward = enc(seq_pos, dst, src)
        assert torch.allclose(forward, backward)

    def test_same_distance_same_encoding(self):
        """Edges with the same sequence distance get the same encoding."""
        enc = SequencePositionEncoding(dim=16)
        seq_pos = torch.arange(100)
        # All have |i - j| = 5
        src = torch.tensor([0, 10, 50])
        dst = torch.tensor([5, 15, 55])
        out = enc(seq_pos, src, dst)
        assert torch.allclose(out[0], out[1])
        assert torch.allclose(out[0], out[2])

    def test_different_distance_different_encoding(self):
        """Edges with different sequence distances get different encodings."""
        enc = SequencePositionEncoding(dim=16)
        seq_pos = torch.arange(100)
        src = torch.tensor([0, 0])
        dst = torch.tensor([1, 50])
        out = enc(seq_pos, src, dst)
        assert not torch.allclose(out[0], out[1])

    def test_zero_distance(self):
        """Self-loops (distance 0) produce a valid encoding."""
        enc = SequencePositionEncoding(dim=16)
        seq_pos = torch.arange(10)
        src = torch.tensor([0, 5])
        dst = torch.tensor([0, 5])
        out = enc(seq_pos, src, dst)
        assert out.shape == (2, 16)
        assert torch.allclose(out[0], out[1])

    def test_clipping(self):
        """Distances beyond max_seq_distance are clipped."""
        enc = SequencePositionEncoding(dim=16, max_seq_distance=10)
        seq_pos = torch.arange(200)
        src = torch.tensor([0, 0])
        dst = torch.tensor([10, 150])  # distance 10 and 150
        out = enc(seq_pos, src, dst)
        # Both should map to distance 10 (clipped)
        assert torch.allclose(out[0], out[1])

    def test_learnable_output_shape(self):
        enc = SequencePositionEncoding(dim=16, learnable=True)
        seq_pos = torch.arange(50)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([5, 10, 49])
        out = enc(seq_pos, src, dst)
        assert out.shape == (3, 16)

    def test_learnable_has_parameters(self):
        enc = SequencePositionEncoding(dim=16, learnable=True)
        params = list(enc.parameters())
        assert len(params) == 1
        assert params[0].shape == (129, 16)  # max_seq_distance + 1

    def test_sinusoidal_no_parameters(self):
        enc = SequencePositionEncoding(dim=16, learnable=False)
        params = list(enc.parameters())
        assert len(params) == 0

    def test_learnable_gradients(self):
        """Learnable encoding produces gradients."""
        enc = SequencePositionEncoding(dim=16, learnable=True)
        seq_pos = torch.arange(20)
        src = torch.tensor([0, 1])
        dst = torch.tensor([5, 10])
        out = enc(seq_pos, src, dst)
        out.sum().backward()
        assert enc.embedding.weight.grad is not None
        assert enc.embedding.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("dim", [1, 2, 3, 7, 64])
    def test_odd_even_dims(self, dim):
        """Works for both odd and even output dimensions."""
        enc = SequencePositionEncoding(dim=dim)
        seq_pos = torch.arange(10)
        src = torch.tensor([0])
        dst = torch.tensor([5])
        out = enc(seq_pos, src, dst)
        assert out.shape == (1, dim)
        assert torch.isfinite(out).all()


class TestSequenceEncodingIntegration:
    """Integration tests: SequencePositionEncoding + EquivariantTransformerBlock."""

    @staticmethod
    def _make_edge_features(seq_enc, seq_proj, seq_pos, graph, repr):
        """Project sequence encoding into equivariant edge features.

        The encoding is scalar (l=0 only), so we project to the channel
        dimension and zero-pad the l>0 components.
        """
        enc = seq_enc(seq_pos, graph.src, graph.dst)  # (E, enc_dim)
        scalar = seq_proj(enc)  # (E, mult)
        E, mult = scalar.shape
        # Pad with zeros for l>0 components: (E, mult, 1) -> (E, mult, dim)
        edge_features = scalar.new_zeros(E, mult, repr.dim())
        edge_features[..., :1] = scalar.unsqueeze(-1)
        return edge_features

    def test_forward_shape(self, cuda_device):
        """Block produces correct shape when given sequence edge features."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 30, 150

        seq_enc = SequencePositionEncoding(dim=16).to(cuda_device)
        seq_proj = nn.Linear(16, repr.mult, bias=False).to(cuda_device)

        block = EquivariantTransformerBlock(
            repr, repr, num_heads=2,
        ).to(cuda_device)
        basis = WignerDBasis([repr]).to(cuda_device)

        graph = make_graph(num_nodes, num_edges, cuda_device)
        seq_pos = torch.arange(num_nodes, device=cuda_device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        (M,) = basis(directions)
        edge_features = self._make_edge_features(seq_enc, seq_proj, seq_pos, graph, repr)
        output = block(M, M, node_features, distances, graph, edge_features=edge_features)

        assert output.shape == (num_nodes, repr.mult, repr.dim())
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, cuda_device):
        """Gradients flow through sequence encoding into the block."""
        repr = Repr(lvals=[0, 1], mult=8)
        num_nodes, num_edges = 20, 100

        seq_enc = SequencePositionEncoding(dim=16, learnable=True).to(cuda_device)
        seq_proj = nn.Linear(16, repr.mult, bias=False).to(cuda_device)

        block = EquivariantTransformerBlock(
            repr, repr, num_heads=2,
        ).to(cuda_device)
        basis = WignerDBasis([repr]).to(cuda_device)

        graph = make_graph(num_nodes, num_edges, cuda_device)
        seq_pos = torch.arange(num_nodes, device=cuda_device)
        node_features = torch.randn(
            num_nodes, repr.mult, repr.dim(), device=cuda_device, requires_grad=True,
        )
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        (M,) = basis(directions)
        edge_features = self._make_edge_features(seq_enc, seq_proj, seq_pos, graph, repr)
        output = block(M, M, node_features, distances, graph, edge_features=edge_features)

        loss = (output ** 2).sum()
        loss.backward()

        assert node_features.grad is not None
        assert torch.isfinite(node_features.grad).all()
        # Learnable encoding receives gradients
        assert seq_enc.embedding.weight.grad is not None
        assert seq_enc.embedding.weight.grad.abs().sum() > 0
        # Projection receives gradients
        assert seq_proj.weight.grad is not None

    def test_equivariance(self, cuda_device):
        """Scalar edge features are rotation-invariant, so block stays equivariant."""
        torch.manual_seed(42)

        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        seq_enc = SequencePositionEncoding(dim=16).to(cuda_device)
        seq_proj = nn.Linear(16, repr.mult, bias=False).to(cuda_device)

        block = EquivariantTransformerBlock(
            repr, repr, num_heads=2, dropout=0.0,
        ).to(cuda_device)
        block.eval()

        basis = WignerDBasis([repr]).to(cuda_device)
        wigner = WignerD(repr).to(cuda_device)

        graph = make_graph(num_nodes, num_edges, cuda_device)
        seq_pos = torch.arange(num_nodes, device=cuda_device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        axis, angle, R = random_rotation(cuda_device)
        D = wigner.rot(axis, angle)

        # Edge features are scalar (l=0 in slot 0, zeros for l>0).
        # Under rotation D, the l=0 component is invariant and l>0 zeros stay zero.
        edge_features = self._make_edge_features(seq_enc, seq_proj, seq_pos, graph, repr)

        # Forward then rotate
        (M,) = basis(directions)
        out1 = block(M, M, node_features, distances, graph, edge_features=edge_features)
        out1_rotated = torch.einsum('ij,ncj->nci', D, out1)

        # Rotate then forward
        node_features_rotated = torch.einsum('ij,ncj->nci', D, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        (M_rot,) = basis(directions_rotated)
        out2 = block(M_rot, M_rot, node_features_rotated, distances, graph,
                     edge_features=edge_features)

        check_equivariance(out1_rotated, out2, rtol=5e-3,
                          msg="Transformer + sequence encoding equivariance")

    def test_with_vs_without(self, cuda_device):
        """Sequence encoding changes the output (not a no-op)."""
        repr = Repr(lvals=[0, 1], mult=8)
        num_nodes, num_edges = 20, 100

        seq_enc = SequencePositionEncoding(dim=16).to(cuda_device)
        seq_proj = nn.Linear(16, repr.mult, bias=False).to(cuda_device)

        block = EquivariantTransformerBlock(
            repr, repr, num_heads=2,
        ).to(cuda_device)
        block.eval()
        basis = WignerDBasis([repr]).to(cuda_device)

        graph = make_graph(num_nodes, num_edges, cuda_device)
        seq_pos = torch.arange(num_nodes, device=cuda_device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        (M,) = basis(directions)

        with torch.no_grad():
            out_without = block(M, M, node_features, distances, graph)
            edge_features = self._make_edge_features(seq_enc, seq_proj, seq_pos, graph, repr)
            out_with = block(M, M, node_features, distances, graph, edge_features=edge_features)

        assert not torch.allclose(out_without, out_with, atol=1e-6)
