"""Tests for equivariant transformer blocks.

Tests:
    - Forward pass shape and device handling
    - Same repr (with residual) vs different repr (without residual)
    - Full equivariance: D @ Transformer(f) = Transformer(D @ f)
    - Gradient flow through the model
    - Full transformer stack

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
import pytest
import torch

from flash_eq import (
    Repr,
    WignerD,
    WignerDBasis,
    EquivariantTransformerBlock,
    EquivariantTransformer,
)

from .helpers import random_rotation, make_graph, check_equivariance


# Test configurations for single block
BLOCK_CONFIGS = [
    # (in_lvals, in_mult, out_lvals, out_mult, num_heads)
    # Same repr (uses residual)
    ([0, 1], 8, [0, 1], 8, 2),
    ([0, 1, 2], 16, [0, 1, 2], 16, 4),
    # Different mult (no residual)
    ([0, 1], 8, [0, 1], 16, 4),
    # Different lvals (no residual)
    ([0, 1], 8, [0, 1, 2], 8, 2),
    ([0, 1, 2], 8, [0], 8, 2),
    # Both different
    ([0, 1], 8, [0, 1, 2], 16, 4),
]


class TestEquivariantTransformerBlock:
    """Test suite for EquivariantTransformerBlock."""

    @pytest.mark.parametrize("in_lvals,in_mult,out_lvals,out_mult,num_heads", BLOCK_CONFIGS)
    def test_forward_shape(self, cuda_device, in_lvals, in_mult, out_lvals, out_mult, num_heads):
        """Test that forward pass produces correct output shape."""
        in_repr = Repr(lvals=in_lvals, mult=in_mult)
        out_repr = Repr(lvals=out_lvals, mult=out_mult)
        num_nodes, num_edges = 20, 100

        block = EquivariantTransformerBlock(
            in_repr, out_repr, num_heads=num_heads
        ).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        P, Q = basis(directions)
        output = block(P, Q, node_features, distances, src, dst, num_nodes)

        assert output.shape == (num_nodes, out_mult, out_repr.dim())
        assert output.device.type == cuda_device.type
        assert torch.isfinite(output).all()

    def test_residual_connection_same_repr(self, cuda_device):
        """Test that residual connection is used when in_repr == out_repr."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        block = EquivariantTransformerBlock(repr, repr, num_heads=2).to(cuda_device)

        # Check internal flag
        assert block._use_attn_residual is True

        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        basis = WignerDBasis([repr, repr]).to(cuda_device)
        P, Q = basis(directions)

        output = block(P, Q, node_features, distances, src, dst, num_nodes)

        # Output should have contribution from input (residual)
        # This is a weak test but verifies the path works
        assert torch.isfinite(output).all()

    def test_no_residual_different_repr(self, cuda_device):
        """Test that no residual is used when reprs differ."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        out_repr = Repr(lvals=[0, 1, 2], mult=16)

        block = EquivariantTransformerBlock(in_repr, out_repr, num_heads=4).to(cuda_device)

        # Check internal flag
        assert block._use_attn_residual is False

    @pytest.mark.parametrize("in_lvals,in_mult,out_lvals,out_mult,num_heads", [
        # Same repr cases (equivariance should hold)
        ([0, 1], 8, [0, 1], 8, 2),
        ([0, 1, 2], 8, [0, 1, 2], 8, 2),
    ])
    def test_equivariance_same_repr(self, cuda_device, in_lvals, in_mult, out_lvals, out_mult, num_heads):
        """Test SO(3) equivariance when in_repr == out_repr."""
        torch.manual_seed(42)

        in_repr = Repr(lvals=in_lvals, mult=in_mult)
        out_repr = Repr(lvals=out_lvals, mult=out_mult)
        num_nodes, num_edges = 20, 100

        block = EquivariantTransformerBlock(
            in_repr, out_repr, num_heads=num_heads, dropout=0.0
        ).to(cuda_device)
        block.eval()  # Disable dropout

        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
        wigner_in = WignerD(in_repr).to(cuda_device)
        wigner_out = WignerD(out_repr).to(cuda_device)

        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        # Random rotation
        axis, angle, R = random_rotation(cuda_device)
        D_in = wigner_in.rot(axis, angle)
        D_out = wigner_out.rot(axis, angle)

        # Method 1: Forward then rotate output
        P, Q = basis(directions)
        output1 = block(P, Q, node_features, distances, src, dst, num_nodes)
        output1_rotated = torch.einsum('ij,ncj->nci', D_out, output1)

        # Method 2: Rotate inputs then forward
        node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        P_rot, Q_rot = basis(directions_rotated)
        output2 = block(P_rot, Q_rot, node_features_rotated, distances, src, dst, num_nodes)

        # Check equivariance (looser tolerance for full block)
        check_equivariance(output1_rotated, output2, rtol=5e-3, msg="Transformer block equivariance")

    @pytest.mark.parametrize("in_lvals,in_mult,out_lvals,out_mult,num_heads", [
        # Different repr cases
        ([0, 1], 8, [0, 1, 2], 16, 4),
        ([0, 1, 2], 16, [0], 8, 2),
    ])
    def test_equivariance_different_repr(self, cuda_device, in_lvals, in_mult, out_lvals, out_mult, num_heads):
        """Test SO(3) equivariance when in_repr != out_repr."""
        torch.manual_seed(42)

        in_repr = Repr(lvals=in_lvals, mult=in_mult)
        out_repr = Repr(lvals=out_lvals, mult=out_mult)
        num_nodes, num_edges = 20, 100

        block = EquivariantTransformerBlock(
            in_repr, out_repr, num_heads=num_heads, dropout=0.0
        ).to(cuda_device)
        block.eval()

        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)
        wigner_in = WignerD(in_repr).to(cuda_device)
        wigner_out = WignerD(out_repr).to(cuda_device)

        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        # Random rotation
        axis, angle, R = random_rotation(cuda_device)
        D_in = wigner_in.rot(axis, angle)
        D_out = wigner_out.rot(axis, angle)

        # Method 1: Forward then rotate output
        P, Q = basis(directions)
        output1 = block(P, Q, node_features, distances, src, dst, num_nodes)
        output1_rotated = torch.einsum('ij,ncj->nci', D_out, output1)

        # Method 2: Rotate inputs then forward
        node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        P_rot, Q_rot = basis(directions_rotated)
        output2 = block(P_rot, Q_rot, node_features_rotated, distances, src, dst, num_nodes)

        check_equivariance(output1_rotated, output2, rtol=5e-3, msg="Transformer block equivariance (different repr)")

    def test_gradient_flow(self, cuda_device):
        """Test that gradients flow through the entire block."""
        in_repr = Repr(lvals=[0, 1, 2], mult=8)
        out_repr = Repr(lvals=[0, 1, 2], mult=8)
        num_nodes, num_edges = 20, 100

        block = EquivariantTransformerBlock(in_repr, out_repr, num_heads=2).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(
            num_nodes, in_repr.mult, in_repr.dim(),
            device=cuda_device, requires_grad=True
        )
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        P, Q = basis(directions)
        output = block(P, Q, node_features, distances, src, dst, num_nodes)

        # Use invariant loss (squared norm)
        loss = (output ** 2).sum()
        loss.backward()

        # Check gradients exist and are finite
        assert node_features.grad is not None
        assert torch.isfinite(node_features.grad).all()
        assert node_features.grad.abs().mean() > 1e-8, "Gradients too small"

        # Check all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_mlp_ratio(self, cuda_device):
        """Test that mlp_ratio affects hidden dimension."""
        repr = Repr(lvals=[0, 1, 2], mult=8)

        block_1x = EquivariantTransformerBlock(repr, repr, mlp_ratio=1).to(cuda_device)
        block_4x = EquivariantTransformerBlock(repr, repr, mlp_ratio=4).to(cuda_device)

        # Check MLP hidden dimensions
        assert block_1x.mlp_up.weight.shape[0] == 8 * 3  # 8 mult * 3 irreps = 24
        assert block_4x.mlp_up.weight.shape[0] == 32 * 3  # 32 mult * 3 irreps = 96


class TestEquivariantTransformer:
    """Test suite for EquivariantTransformer (full stack)."""

    def test_forward_shape(self, cuda_device):
        """Test full transformer forward pass shape."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=16)
        out_repr = Repr(lvals=[0], mult=4)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=3, num_heads=4
        ).to(cuda_device)

        num_nodes, num_edges = 20, 100
        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        basis = WignerDBasis(model.get_basis_reprs()).to(cuda_device)
        matrices = basis(directions)

        output = model(matrices, node_features, distances, src, dst, num_nodes)

        assert output.shape == (num_nodes, out_repr.mult, out_repr.dim())
        assert torch.isfinite(output).all()

    def test_get_basis_reprs(self, cuda_device):
        """Test that get_basis_reprs returns correct representations."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=16)
        out_repr = Repr(lvals=[0], mult=4)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=3, num_heads=4
        )

        reprs = model.get_basis_reprs()
        assert len(reprs) == 3
        assert reprs[0] is in_repr
        assert reprs[1] is hidden_repr
        assert reprs[2] is out_repr

    def test_layer_structure(self, cuda_device):
        """Test that layers have correct repr configurations."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=16)
        out_repr = Repr(lvals=[0], mult=4)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=4, num_heads=4
        )

        # Layer 0: in -> hidden
        assert model.layers[0].in_repr is in_repr
        assert model.layers[0].out_repr is hidden_repr

        # Layer 1: hidden -> hidden
        assert model.layers[1].in_repr is hidden_repr
        assert model.layers[1].out_repr is hidden_repr

        # Layer 2: hidden -> hidden
        assert model.layers[2].in_repr is hidden_repr
        assert model.layers[2].out_repr is hidden_repr

        # Layer 3: hidden -> out
        assert model.layers[3].in_repr is hidden_repr
        assert model.layers[3].out_repr is out_repr

    def test_equivariance(self, cuda_device):
        """Test SO(3) equivariance of full transformer."""
        torch.manual_seed(42)

        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=8)
        out_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=2, num_heads=2, dropout=0.0
        ).to(cuda_device)
        model.eval()

        wigner_in = WignerD(in_repr).to(cuda_device)
        wigner_out = WignerD(out_repr).to(cuda_device)

        num_nodes, num_edges = 20, 100
        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        basis = WignerDBasis(model.get_basis_reprs()).to(cuda_device)

        # Random rotation
        axis, angle, R = random_rotation(cuda_device)
        D_in = wigner_in.rot(axis, angle)
        D_out = wigner_out.rot(axis, angle)

        # Method 1: Forward then rotate
        matrices = basis(directions)
        output1 = model(matrices, node_features, distances, src, dst, num_nodes)
        output1_rotated = torch.einsum('ij,ncj->nci', D_out, output1)

        # Method 2: Rotate then forward
        node_features_rotated = torch.einsum('ij,ncj->nci', D_in, node_features)
        directions_rotated = torch.einsum('ij,ej->ei', R, directions)
        matrices_rot = basis(directions_rotated)
        output2 = model(matrices_rot, node_features_rotated, distances, src, dst, num_nodes)

        # Looser tolerance for multi-layer model
        check_equivariance(output1_rotated, output2, rtol=1e-2, msg="Full transformer equivariance")

    def test_gradient_flow(self, cuda_device):
        """Test gradient flow through full transformer."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=8)
        out_repr = Repr(lvals=[0], mult=4)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=3, num_heads=2
        ).to(cuda_device)

        num_nodes, num_edges = 20, 100
        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(
            num_nodes, in_repr.mult, in_repr.dim(),
            device=cuda_device, requires_grad=True
        )
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        basis = WignerDBasis(model.get_basis_reprs()).to(cuda_device)
        matrices = basis(directions)

        output = model(matrices, node_features, distances, src, dst, num_nodes)
        loss = (output ** 2).sum()
        loss.backward()

        # Check input gradients
        assert node_features.grad is not None
        assert torch.isfinite(node_features.grad).all()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_single_layer(self, cuda_device):
        """Test transformer with single layer (in -> out directly)."""
        in_repr = Repr(lvals=[0, 1], mult=8)
        hidden_repr = Repr(lvals=[0, 1, 2], mult=16)
        out_repr = Repr(lvals=[0], mult=4)

        model = EquivariantTransformer(
            in_repr, hidden_repr, out_repr,
            num_layers=1, num_heads=4
        ).to(cuda_device)

        # With num_layers=1, single layer goes directly in -> out
        assert len(model.layers) == 1
        assert model.layers[0].in_repr is in_repr
        assert model.layers[0].out_repr is out_repr

        num_nodes, num_edges = 20, 100
        src, dst = make_graph(num_nodes, num_edges, cuda_device)
        node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim(), device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        distances = torch.rand(num_edges, device=cuda_device) * 5 + 0.5

        basis = WignerDBasis(model.get_basis_reprs()).to(cuda_device)
        matrices = basis(directions)

        output = model(matrices, node_features, distances, src, dst, num_nodes)
        # Output should be out_repr shape
        assert output.shape == (num_nodes, out_repr.mult, out_repr.dim())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
