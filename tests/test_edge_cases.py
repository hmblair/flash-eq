"""
Edge case tests for flash-eq.

Tests boundary conditions and degenerate inputs that could cause runtime failures.
"""

import torch
import pytest

from flash_eq import (
    Repr, EquivariantEdgewiseLinear, WignerDBasis,
    RepNorm, EquivariantLinear, EquivariantGating, EquivariantLayerNorm,
    RadialBasisFunctions, RadialMLP, BinnedModule,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEmptyInputs:
    """Tests for empty tensor inputs."""

    def test_zero_edges(self, device):
        """Layer should handle zero edges gracefully."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_nodes = 10
        num_edges = 0

        node_features = torch.randn(num_nodes, 4, 9, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.randn(num_edges, device=device)
        src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert output.shape == (0, 4, 9)

    def test_single_edge(self, device):
        """Layer should handle single edge."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_nodes = 10
        num_edges = 1

        node_features = torch.randn(num_nodes, 4, 9, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.rand(num_edges, device=device) * 5.0
        src_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert output.shape == (1, 4, 9)
        assert torch.isfinite(output).all()

    def test_single_node(self, device):
        """Layer should handle single node with self-loop."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_nodes = 1
        num_edges = 5

        node_features = torch.randn(num_nodes, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.rand(num_edges, device=device) * 5.0
        src_indices = torch.zeros(num_edges, device=device, dtype=torch.long)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert output.shape == (5, 2, 4)
        assert torch.isfinite(output).all()


class TestNumericalEdgeCases:
    """Tests for numerical edge cases."""

    def test_zero_directions(self, device):
        """Directions with zero norm should not cause NaN."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        basis = WignerDBasis(in_repr, out_repr).to(device)

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
        basis = WignerDBasis(in_repr, in_repr).to(device)

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
        basis = WignerDBasis(in_repr, in_repr).to(device)

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

    def test_zero_distance(self, device):
        """Distance of exactly zero."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 5
        node_features = torch.randn(10, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.zeros(num_edges, device=device)  # All zero
        src_indices = torch.randint(0, 10, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert torch.isfinite(output).all(), "Output contains NaN/Inf for zero distance"

    def test_distance_outside_range(self, device):
        """Distances outside the binned range."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        # Layer with range [0, 10]
        layer = EquivariantEdgewiseLinear(
            in_repr, out_repr, min_dist=0.0, max_dist=10.0
        ).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 4
        node_features = torch.randn(10, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        # Distances: negative, zero, in range, above range
        distances = torch.tensor([-5.0, 0.0, 5.0, 100.0], device=device)
        src_indices = torch.randint(0, 10, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

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
        assert (out == 0).all(), "RepNorm of zeros should be zeros"


class TestDtypes:
    """Tests for different data types."""

    def test_fp16_layer(self, device):
        """Test layer with FP16 inputs."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 test requires CUDA")

        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device).half()
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 50
        node_features = torch.randn(20, 4, 9, device=device, dtype=torch.float16)
        directions = torch.randn(num_edges, 3, device=device, dtype=torch.float16)
        distances = torch.rand(num_edges, device=device, dtype=torch.float16) * 5.0
        src_indices = torch.randint(0, 20, (num_edges,), device=device)

        P, Q = basis(directions.float())  # Basis computed in FP32
        P, Q = P.half(), Q.half()
        output = layer(P, Q, node_features, distances, src_indices)

        assert output.dtype == torch.float16
        assert torch.isfinite(output).all(), "FP16 output contains NaN/Inf"

    def test_fp16_building_blocks(self, device):
        """Test building block layers with FP16."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 test requires CUDA")

        repr = Repr(lvals=[0, 1, 2], mult=4)

        layers = [
            RepNorm(repr),
            EquivariantLinear(repr, repr),
            EquivariantGating(repr),
            EquivariantLayerNorm(repr),
        ]

        x = torch.randn(16, 4, 9, device=device, dtype=torch.float16)

        for layer in layers:
            layer = layer.to(device).half()
            out = layer(x)
            assert torch.isfinite(out).all(), f"{layer.__class__.__name__} FP16 output contains NaN/Inf"

    def test_fp64_layer(self, device):
        """Test layer with FP64 inputs."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device).double()
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 20
        node_features = torch.randn(10, 2, 4, device=device, dtype=torch.float64)
        directions = torch.randn(num_edges, 3, device=device, dtype=torch.float64)
        distances = torch.rand(num_edges, device=device, dtype=torch.float64) * 5.0
        src_indices = torch.randint(0, 10, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

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

    def test_binned_module_edge_values(self, device):
        """Test binned module at exact bin edges."""
        mlp = RadialMLP(hidden_dim=32, num_basis=10, in_mult=2, out_mult=2).to(device)
        binned = BinnedModule(mlp, num_bins=10, min_val=0.0, max_val=10.0).to(device)

        # Exact bin edge values
        values = torch.tensor([0.0, 1.0, 5.0, 9.0, 10.0], device=device)
        bin_lo, interp = binned.bin_indices(values)

        assert torch.isfinite(bin_lo.float()).all()
        assert torch.isfinite(interp).all()
        assert (interp >= 0).all() and (interp <= 1).all()


class TestDegenerateGraphs:
    """Tests for degenerate graph structures."""

    def test_all_self_loops(self, device):
        """All edges are self-loops."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_nodes = 5
        num_edges = 10

        node_features = torch.randn(num_nodes, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.rand(num_edges, device=device) * 5.0
        # All edges from node i to node i
        src_indices = torch.arange(num_edges, device=device) % num_nodes

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert torch.isfinite(output).all()

    def test_all_edges_same_source(self, device):
        """All edges originate from the same node."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_nodes = 10
        num_edges = 50

        node_features = torch.randn(num_nodes, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.rand(num_edges, device=device) * 5.0
        src_indices = torch.zeros(num_edges, device=device, dtype=torch.long)  # All from node 0

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert torch.isfinite(output).all()

    def test_identical_directions(self, device):
        """All edges have the same direction."""
        in_repr = Repr(lvals=[0, 1, 2], mult=4)
        out_repr = Repr(lvals=[0, 1, 2], mult=4)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 20
        node_features = torch.randn(10, 4, 9, device=device)
        # All same direction
        directions = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(num_edges, 3)
        distances = torch.rand(num_edges, device=device) * 5.0
        src_indices = torch.randint(0, 10, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert torch.isfinite(output).all()

    def test_identical_distances(self, device):
        """All edges have the same distance."""
        in_repr = Repr(lvals=[0, 1], mult=2)
        out_repr = Repr(lvals=[0, 1], mult=2)

        layer = EquivariantEdgewiseLinear(in_repr, out_repr).to(device)
        basis = WignerDBasis(in_repr, out_repr).to(device)

        num_edges = 20
        node_features = torch.randn(10, 2, 4, device=device)
        directions = torch.randn(num_edges, 3, device=device)
        distances = torch.full((num_edges,), 5.0, device=device)  # All same distance
        src_indices = torch.randint(0, 10, (num_edges,), device=device)

        P, Q = basis(directions)
        output = layer(P, Q, node_features, distances, src_indices)

        assert torch.isfinite(output).all()
