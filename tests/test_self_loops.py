"""
Tests for self-loop handling in flash-eq.

Self-loops (src == dst) create zero-length edge vectors with undefined directions.
The solid harmonic scaling in the CUDA kernel suppresses l>0 components at zero
distance, following the structure of solid harmonics where r^l Y_l^m vanishes
at r=0 for l>0.
"""

import torch

from flash_eq import (
    Graph,
    Repr,
    EquivariantEdgewiseLinear,
    EquivariantTransformerBlock,
    WignerDBasis,
    random_rotation,
)
from flash_eq.representations import WignerD


class TestSelfLoopBasics:
    """Basic tests for self-loop handling."""

    def test_self_loop_no_nan(self, cuda_device):
        """Self-loops should not produce NaN/Inf."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 10
        edge_features = torch.randn(num_edges, 8, 9, device=cuda_device)
        directions = torch.zeros(num_edges, 3, device=cuda_device)  # All self-loops
        distances = torch.zeros(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all(), "Self-loops produced NaN/Inf"

    def test_self_loop_suppresses_higher_l(self, cuda_device):
        """Self-loops (distance=0) should zero out l>0 output components."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 10
        # Input with non-zero l>0 components
        edge_features = torch.randn(num_edges, 8, 9, device=cuda_device)
        directions = torch.zeros(num_edges, 3, device=cuda_device)
        distances = torch.zeros(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        # l=0 at index 0, l=1 at indices 1:4, l=2 at indices 4:9
        l0_output = output[..., 0]
        l1_output = output[..., 1:4]
        l2_output = output[..., 4:9]

        # l>0 should be zero for self-loops
        assert l1_output.abs().max() < 1e-6, f"l=1 not suppressed: {l1_output.abs().max()}"
        assert l2_output.abs().max() < 1e-6, f"l=2 not suppressed: {l2_output.abs().max()}"
        # l=0 can be non-zero
        assert l0_output.abs().max() > 0, "l=0 should not be zero"

    def test_self_loop_scalar_passthrough(self, cuda_device):
        """Scalar (l=0) input should pass through self-loops."""
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 5
        # Input with only l=0 (scalars), l=1 zeroed
        edge_features = torch.randn(num_edges, 4, 4, device=cuda_device)
        edge_features[..., 1:4] = 0  # Zero l=1

        directions = torch.zeros(num_edges, 3, device=cuda_device)
        distances = torch.zeros(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        # Output l=0 should be non-zero (transformed scalars)
        l0_output = output[..., 0]
        assert l0_output.abs().max() > 0, "Scalar should pass through"


class TestSelfLoopEquivariance:
    """Equivariance tests specifically for self-loops."""

    def test_self_loop_equivariance_scalar_only(self, cuda_device):
        """Self-loops with scalar-only input should be perfectly equivariant."""
        repr = Repr(lvals=[0, 1], mult=8)
        block = EquivariantTransformerBlock(repr, repr, num_heads=2, dropout=0.0).to(cuda_device)
        block.eval()

        num_nodes = 10
        src = torch.arange(num_nodes, device=cuda_device)
        dst = torch.arange(num_nodes, device=cuda_device)
        graph = Graph(src=src, dst=dst, num_nodes=num_nodes)

        # Zero distances and directions (self-loops)
        distances = torch.zeros(num_nodes, device=cuda_device)
        directions = torch.zeros(num_nodes, 3, device=cuda_device)

        # Input with only l=0 features
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)
        node_features[..., 1:4] = 0  # Zero l=1

        # Random rotation
        axis, angle = random_rotation(device=cuda_device)
        wigner = WignerD(repr).to(cuda_device)
        D = wigner.rot(axis, angle)

        # Rotated features
        features_rot = torch.einsum('ij,ncj->nci', D, node_features)

        # Basis matrices (same for zero directions)
        basis = WignerDBasis([repr, repr]).to(cuda_device)
        P, Q = basis(directions)

        with torch.no_grad():
            out1 = block(P, Q, node_features, distances, graph)
            out1_rot = torch.einsum('ij,ncj->nci', D, out1)

            out2 = block(P, Q, features_rot, distances, graph)

        rel_diff = (out1_rot - out2).abs().max() / (out2.abs().max() + 1e-8)
        assert rel_diff < 1e-5, f"Self-loop equivariance failed: rel_diff={rel_diff:.2e}"

    def test_self_loop_equivariance_full(self, cuda_device):
        """Self-loops with full input should be equivariant (l>0 suppressed)."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        block = EquivariantTransformerBlock(repr, repr, num_heads=2, dropout=0.0).to(cuda_device)
        block.eval()

        num_nodes = 10
        src = torch.arange(num_nodes, device=cuda_device)
        dst = torch.arange(num_nodes, device=cuda_device)
        graph = Graph(src=src, dst=dst, num_nodes=num_nodes)

        distances = torch.zeros(num_nodes, device=cuda_device)
        directions = torch.zeros(num_nodes, 3, device=cuda_device)

        # Full input (non-zero l>0)
        node_features = torch.randn(num_nodes, repr.mult, repr.dim(), device=cuda_device)

        # Random rotation
        axis, angle = random_rotation(device=cuda_device)
        wigner = WignerD(repr).to(cuda_device)
        D = wigner.rot(axis, angle)

        features_rot = torch.einsum('ij,ncj->nci', D, node_features)

        basis = WignerDBasis([repr, repr]).to(cuda_device)
        P, Q = basis(directions)

        with torch.no_grad():
            out1 = block(P, Q, node_features, distances, graph)
            out1_rot = torch.einsum('ij,ncj->nci', D, out1)

            out2 = block(P, Q, features_rot, distances, graph)

        rel_diff = (out1_rot - out2).abs().max() / (out2.abs().max() + 1e-8)
        assert rel_diff < 1e-4, f"Self-loop equivariance failed: rel_diff={rel_diff:.2e}"


class TestMixedEdges:
    """Tests for graphs with both self-loops and regular edges."""

    def test_mixed_edges_no_nan(self, cuda_device):
        """Mixed self-loops and regular edges should not produce NaN."""
        repr = Repr(lvals=[0, 1, 2], mult=8)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 20
        edge_features = torch.randn(num_edges, 8, 9, device=cuda_device)

        # Mix of self-loops (zero) and regular edges
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions[:5] = 0  # First 5 are self-loops

        distances = torch.rand(num_edges, device=cuda_device) * 5.0
        distances[:5] = 0  # Self-loops have zero distance

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all(), "Mixed edges produced NaN/Inf"

    def test_mixed_edges_equivariance(self, cuda_device):
        """Equivariance with mixed self-loops and regular edges."""
        repr = Repr(lvals=[0, 1], mult=8)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 20
        edge_features = torch.randn(num_edges, 8, 4, device=cuda_device)

        # Mix of self-loops and regular edges
        directions = torch.randn(num_edges, 3, device=cuda_device)
        directions[:5] = 0

        distances = torch.rand(num_edges, device=cuda_device) * 5.0
        distances[:5] = 0

        # Random rotation
        axis, angle = random_rotation(device=cuda_device)
        wigner = WignerD(repr).to(cuda_device)
        D = wigner.rot(axis, angle)

        # Rotate features
        features_rot = torch.einsum('ij,ncj->nci', D, edge_features)

        # Rotate directions (only non-self-loop directions)
        # Use l=1 Wigner-D with cartesian=True to get 3x3 rotation matrix
        wigner_l1 = WignerD(Repr([1])).to(cuda_device)
        R = wigner_l1.rot(axis, angle, cartesian=True)
        directions_rot = directions.clone()
        directions_rot[5:] = directions[5:] @ R.T

        # Compute outputs
        P1, Q1 = basis(directions)
        P2, Q2 = basis(directions_rot)

        out1 = layer(P1, Q1, edge_features, distances)
        out1_rot = torch.einsum('ij,ncj->nci', D, out1)

        out2 = layer(P2, Q2, features_rot, distances)

        rel_diff = (out1_rot - out2).abs().max() / (out2.abs().max() + 1e-8)
        assert rel_diff < 1e-4, f"Mixed edges equivariance failed: rel_diff={rel_diff:.2e}"


class TestSelfLoopGradients:
    """Gradient tests for self-loops."""

    def test_self_loop_gradient_flow(self, cuda_device):
        """Gradients should flow through self-loops without NaN."""
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 10
        edge_features = torch.randn(
            num_edges, 4, 4, device=cuda_device, requires_grad=True
        )
        directions = torch.zeros(num_edges, 3, device=cuda_device)
        distances = torch.zeros(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)
        loss = output.sum()
        loss.backward()

        assert edge_features.grad is not None, "No gradient computed"
        assert torch.isfinite(edge_features.grad).all(), "Gradient contains NaN/Inf"

    def test_self_loop_gradient_zero_for_higher_l(self, cuda_device):
        """Gradient w.r.t. l>0 input should be zero for self-loops at distance=0."""
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantEdgewiseLinear(repr, repr).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        num_edges = 5
        edge_features = torch.randn(
            num_edges, 4, 4, device=cuda_device, requires_grad=True
        )
        directions = torch.zeros(num_edges, 3, device=cuda_device)
        distances = torch.zeros(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        # Loss only on l=0 output
        loss = output[..., 0].sum()
        loss.backward()

        # Gradient w.r.t. l>0 input should be zero (no contribution to l=0 output)
        grad_l1 = edge_features.grad[..., 1:4]
        assert grad_l1.abs().max() < 1e-6, f"l=1 gradient not zero: {grad_l1.abs().max()}"


class TestSolidHarmonicScaling:
    """Tests for the solid harmonic scaling behavior."""

    def test_scaling_at_various_distances(self, cuda_device):
        """l>0 should be increasingly suppressed as distance decreases."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        layer = EquivariantEdgewiseLinear(repr, repr, solid_harmonic_scale=1.0).to(cuda_device)
        basis = WignerDBasis([repr, repr]).to(cuda_device)

        # Same features, different distances
        edge_features = torch.randn(5, 4, 9, device=cuda_device)
        edge_features = edge_features.expand(4, 5, 4, 9).clone()  # 4 distance values, 5 edges each

        directions = torch.randn(5, 3, device=cuda_device)
        directions = directions.expand(4, 5, 3).reshape(-1, 3)

        # Distances: 0, 0.5, 1.0, 2.0
        distances = torch.tensor([0.0, 0.5, 1.0, 2.0], device=cuda_device)
        distances = distances.repeat_interleave(5)

        P, Q = basis(directions)
        edge_features_flat = edge_features.reshape(-1, 4, 9)
        output = layer(P, Q, edge_features_flat, distances)
        output = output.reshape(4, 5, 4, 9)

        # Get l=2 output norms for each distance
        l2_norms = output[..., 4:9].norm(dim=-1).mean(dim=(1, 2))

        # l=2 should increase with distance (less suppression)
        for i in range(1, 4):
            assert l2_norms[i] > l2_norms[i-1] * 0.9, \
                f"l=2 not increasing with distance: {l2_norms.tolist()}"

        # At distance=0, l=2 should be essentially zero
        assert l2_norms[0] < 1e-6, f"l=2 not zero at distance=0: {l2_norms[0]}"
