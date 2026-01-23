"""
Edge case tests for flash-eq.

Tests boundary conditions and degenerate inputs that could cause runtime failures.
"""

import torch

from flash_eq import (
    Repr, EquivariantEdgewiseLinear, WignerDBasis,
    RepNorm, EquivariantLinear, EquivariantGating, EquivariantLayerNorm,
    RadialBasisFunctions,
)


class TestEmptyInputs:
    """Tests for empty tensor inputs (CUDA-only due to EquivariantEdgewiseLinear)."""

    def test_zero_edges(self, cuda_device):
        """Layer should handle zero edges gracefully."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 0

        edge_features = torch.randn(num_edges, 4, 9, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.randn(num_edges, device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert output.shape == (0, 4, 9)

    def test_single_edge(self, cuda_device):
        """Layer should handle single edge."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 1

        edge_features = torch.randn(num_edges, 4, 9, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.rand(num_edges, device=cuda_device) * 5.0

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert output.shape == (1, 4, 9)
        assert torch.isfinite(output).all()

    def test_single_node(self, cuda_device):
        """Layer should handle single node with self-loop."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_nodes = 1
        num_edges = 5

        node_features = torch.randn(num_nodes, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.rand(num_edges, device=cuda_device) * 5.0
        src_indices = torch.zeros(num_edges, device=cuda_device, dtype=torch.long)

        # Gather node features to edges
        edge_features = node_features[src_indices]

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert output.shape == (5, 2, 4)
        assert torch.isfinite(output).all()


class TestNumericalEdgeCases:
    """Tests for numerical edge cases."""

    def test_zero_directions(self, device):
        """Directions with zero norm should not cause NaN."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        basis = WignerDBasis([in_repr, out_repr]).to(device)

        # Include zero direction
        directions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], device=device)

        P, Q = basis(directions)

        assert torch.isfinite(P).all(), "P contains NaN/Inf for zero direction"
        assert torch.isfinite(Q).all(), "Q contains NaN/Inf for zero direction"

    def test_near_zero_directions(self, device):
        """Very small directions should not cause NaN."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        basis = WignerDBasis([in_repr, in_repr]).to(device)

        directions = torch.tensor([
            [1e-10, 1e-10, 1e-10],
            [1e-20, 0.0, 0.0],
        ], device=device)

        P, Q = basis(directions)

        assert torch.isfinite(P).all(), "P contains NaN/Inf for near-zero direction"
        assert torch.isfinite(Q).all(), "Q contains NaN/Inf for near-zero direction"

    def test_axis_aligned_directions(self, device):
        """Axis-aligned directions (edge case for cross product)."""
        in_repr = Repr(lvals=[0, 1, 2], mult=2)
        basis = WignerDBasis([in_repr, in_repr]).to(device)

        # +z and -z are edge cases (cross product with e_z is zero)
        directions = torch.tensor([
            [0.0, 0.0, 1.0],   # +z
            [0.0, 0.0, -1.0],  # -z
            [1.0, 0.0, 0.0],   # +x
            [0.0, 1.0, 0.0],   # +y
        ], device=device)

        P, Q = basis(directions)

        assert torch.isfinite(P).all(), "P contains NaN/Inf for axis-aligned direction"
        assert torch.isfinite(Q).all(), "Q contains NaN/Inf for axis-aligned direction"

    def test_zero_distance(self, cuda_device):
        """Distance of exactly zero (CUDA-only)."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 5
        edge_features = torch.randn(num_edges, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.zeros(num_edges, device=cuda_device)  # All zero

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all(), "Output contains NaN/Inf for zero distance"

    def test_distance_outside_range(self, cuda_device):
        """Distances outside the binned range (CUDA-only)."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        # Layer with range [0, 10]
        layer = EquivariantEdgewiseLinear(
            in_repr, out_repr, min_dist=0.0, max_dist=10.0
        ).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 4
        edge_features = torch.randn(num_edges, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        # Distances: negative, zero, in range, above range
        distances = torch.tensor([-5.0, 0.0, 5.0, 100.0], device=cuda_device)

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all(), "Output contains NaN/Inf for out-of-range distance"

    def test_large_features(self, device):
        """Very large feature values."""
        repr = Repr(lvals=[0, 1], mult=2)
        layer = EquivariantLayerNorm(repr).to(device)

        x = torch.randn(8, 2, 4, device=device) * 1e6
        out = layer(x)

        assert torch.isfinite(out).all(), "LayerNorm output contains NaN/Inf for large inputs"

    def test_small_features(self, device):
        """Very small feature values."""
        repr = Repr(lvals=[0, 1], mult=2)
        layer = EquivariantLayerNorm(repr).to(device)

        x = torch.randn(8, 2, 4, device=device) * 1e-6
        out = layer(x)

        assert torch.isfinite(out).all(), "LayerNorm output contains NaN/Inf for small inputs"

    def test_zero_features(self, device):
        """All-zero feature vectors."""
        repr = Repr(lvals=[0, 1], mult=2)
        norm = RepNorm(repr).to(device)

        x = torch.zeros(8, 2, 4, device=device)
        out = norm(x)

        assert torch.isfinite(out).all(), "RepNorm output contains NaN/Inf for zero input"
        # RepNorm returns sqrt(epsilon) for zero input to avoid NaN gradients
        assert (out < 1e-3).all(), "RepNorm of zeros should be near-zero"


class TestDtypes:
    """Tests for different data types."""

    def test_fp16_layer(self, cuda_device):
        """Test layer with FP16 inputs (CUDA-only)."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device).half()
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 50
        edge_features = torch.randn(num_edges, 4, 9, device=cuda_device, dtype=torch.float16)
        directions = torch.randn(num_edges, 3, device=cuda_device, dtype=torch.float16)
        distances = torch.rand(num_edges, device=cuda_device, dtype=torch.float16) * 5.0

        P, Q = basis(directions.float())  # Basis computed in FP32
        P, Q = P.half(), Q.half()
        output = layer(P, Q, edge_features, distances)

        assert output.dtype == torch.float16
        assert torch.isfinite(output).all(), "FP16 output contains NaN/Inf"

    def test_fp16_building_blocks(self, cuda_device):
        """Test building block layers with FP16 (CUDA-only)."""
        repr = Repr(lvals=[0, 1, 2], mult=4)

        layers = [
            RepNorm(repr),
            EquivariantLinear(repr, repr),
            EquivariantGating(repr),
            EquivariantLayerNorm(repr),
        ]

        x = torch.randn(16, 4, 9, device=cuda_device, dtype=torch.float16)

        for layer in layers:
            layer = layer.to(cuda_device).half()
            out = layer(x)
            assert torch.isfinite(out).all(), f"{layer.__class__.__name__} FP16 output contains NaN/Inf"

    def test_fp64_layer(self, cuda_device):
        """Test layer with FP64 inputs (CUDA-only)."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device).double()
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device).double()

        num_edges = 20
        edge_features = torch.randn(num_edges, 2, 4, device=cuda_device, dtype=torch.float64)
        directions = torch.randn(num_edges, 3, device=cuda_device, dtype=torch.float64)
        distances = torch.rand(num_edges, device=cuda_device, dtype=torch.float64) * 5.0

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert output.dtype == torch.float64
        assert torch.isfinite(output).all()


class TestRadialBasis:
    """Tests for radial basis functions edge cases."""

    def test_single_basis_function(self, device):
        """Single basis function should work."""
        rbf = RadialBasisFunctions(num_functions=1).to(device)
        x = torch.rand(100, 1, device=device) * 10
        out = rbf(x)

        assert out.shape == (100, 1)
        assert torch.isfinite(out).all()

    def test_negative_distances(self, device):
        """Negative distances (physically meaningless but shouldn't crash)."""
        rbf = RadialBasisFunctions(num_functions=16).to(device)
        x = torch.tensor([[-5.0], [-1.0], [0.0], [5.0]], device=device)
        out = rbf(x)

        assert torch.isfinite(out).all()

class TestDegenerateGraphs:
    """Tests for degenerate graph structures (CUDA-only)."""

    def test_all_self_loops(self, cuda_device):
        """All edges are self-loops."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_nodes = 5
        num_edges = 10

        node_features = torch.randn(num_nodes, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.rand(num_edges, device=cuda_device) * 5.0
        # All edges from node i to node i
        src_indices = torch.arange(num_edges, device=cuda_device) % num_nodes

        # Gather node features to edges
        edge_features = node_features[src_indices]

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all()

    def test_all_edges_same_source(self, cuda_device):
        """All edges originate from the same node."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_nodes = 10
        num_edges = 50

        node_features = torch.randn(num_nodes, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.rand(num_edges, device=cuda_device) * 5.0
        src_indices = torch.zeros(num_edges, device=cuda_device, dtype=torch.long)  # All from node 0

        # Gather node features to edges
        edge_features = node_features[src_indices]

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all()

    def test_identical_directions(self, cuda_device):
        """All edges have the same direction."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 20
        edge_features = torch.randn(num_edges, 4, 9, device=cuda_device)
        # All same direction
        directions = torch.tensor([[1.0, 0.0, 0.0]], device=cuda_device).expand(num_edges, 3)
        distances = torch.rand(num_edges, device=cuda_device) * 5.0

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all()

    def test_identical_distances(self, cuda_device):
        """All edges have the same distance."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(cuda_device)
        basis = WignerDBasis([in_repr, out_repr]).to(cuda_device)

        num_edges = 20
        edge_features = torch.randn(num_edges, 2, 4, device=cuda_device)
        directions = torch.randn(num_edges, 3, device=cuda_device)
        distances = torch.full((num_edges,), 5.0, device=cuda_device)  # All same distance

        P, Q = basis(directions)
        output = layer(P, Q, edge_features, distances)

        assert torch.isfinite(output).all()
