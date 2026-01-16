"""
Tests for binned radial weights implementation.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.binned_weights import (
    RadialBinning,
    BinData,
    BinnedRadialEmbedding,
    interpolate_weights,
)
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)


def assert_close(a, b, rtol=1e-4, atol=1e-4):
    """Assert two tensors are close."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = (a - b).abs()
        raise AssertionError(f"Tensors not close. Max diff: {diff.max():.6e}")


def test_radial_binning_creation():
    """Test RadialBinning creation."""
    binning = RadialBinning(num_bins=100, min_dist=0.0, max_dist=10.0)
    assert binning.num_bins == 100
    assert len(binning.bin_edges) == 101
    assert binning.table_size == 101
    print("  test_radial_binning_creation: PASS")


def test_radial_binning_edges():
    """Test bin edges."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    edges = binning.bin_edges
    assert edges[0] == 0.0
    assert edges[-1] == 10.0
    assert len(edges) == 11
    print("  test_radial_binning_edges: PASS")


def test_radial_binning_centers():
    """Test bin centers."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    centers = binning.bin_centers
    assert len(centers) == 10
    assert torch.allclose(centers[0], torch.tensor(0.5))
    assert torch.allclose(centers[-1], torch.tensor(9.5))
    print("  test_radial_binning_centers: PASS")


def test_compute_indices():
    """Test index computation."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    distances = torch.tensor([0.5, 1.5, 5.0, 9.5])
    indices = binning.compute_indices(distances)
    assert indices.tolist() == [0, 1, 5, 9]
    print("  test_compute_indices: PASS")


def test_compute_indices_clamping():
    """Test index clamping for out-of-range values."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    distances = torch.tensor([-1.0, 15.0])
    indices = binning.compute_indices(distances)
    assert indices[0] == 0
    assert indices[1] == 9
    print("  test_compute_indices_clamping: PASS")


def test_compute_bins():
    """Test full bin computation."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    distances = torch.tensor([0.5, 5.5])
    bin_data = binning.compute_bins(distances)

    assert isinstance(bin_data, BinData)
    assert bin_data.lo.dtype == torch.int32
    assert bin_data.hi.dtype == torch.int32
    assert bin_data.weight.shape == distances.shape
    print("  test_compute_bins: PASS")


def test_interpolation_weights():
    """Test interpolation weight computation."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)
    distances = torch.tensor([0.5])
    bin_data = binning.compute_bins(distances)
    assert torch.allclose(bin_data.weight, torch.tensor([0.5]), atol=0.01)
    print("  test_interpolation_weights: PASS")


def test_create_table():
    """Test lookup table creation."""
    binning = RadialBinning(num_bins=10, min_dist=0.0, max_dist=10.0)

    def radial_fn(d):
        return d.unsqueeze(-1) * torch.ones(5)

    table = binning.create_table(radial_fn)
    assert table.shape == (11, 5)
    print("  test_create_table: PASS")


def test_validation():
    """Test input validation."""
    try:
        RadialBinning(num_bins=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        RadialBinning(min_dist=10.0, max_dist=5.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  test_validation: PASS")


class SimpleRadialNet(nn.Module):
    """Simple radial net that outputs (N, cout, cin, weight_dim) for testing."""
    def __init__(self, cout, cin, weight_dim, hidden=32):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.weight_dim = weight_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cout * cin * weight_dim)
        )

    def forward(self, distances):
        # distances: (N,) -> (N, cout, cin, weight_dim)
        out = self.net(distances.unsqueeze(-1))
        return out.view(-1, self.cout, self.cin, self.weight_dim)


def test_embedding_creation():
    """Test BinnedRadialEmbedding creation."""
    cout, cin, weight_dim = 4, 4, 16
    radial_net = SimpleRadialNet(cout, cin, weight_dim)
    embedding = BinnedRadialEmbedding(radial_net, cout=cout, cin=cin, weight_dim=weight_dim, num_bins=50)
    assert embedding.weight_dim == weight_dim
    assert embedding.cout == cout
    assert embedding.cin == cin
    assert embedding.binning.num_bins == 50
    print("  test_embedding_creation: PASS")


def test_embedding_get_table():
    """Test lookup table retrieval."""
    cout, cin, weight_dim = 4, 4, 16
    radial_net = SimpleRadialNet(cout, cin, weight_dim)
    embedding = BinnedRadialEmbedding(radial_net, cout=cout, cin=cin, weight_dim=weight_dim, num_bins=10)
    table = embedding.get_table()
    assert table.shape == (11, cout, cin, weight_dim)
    print("  test_embedding_get_table: PASS")


def test_embedding_caching():
    """Test table caching."""
    cout, cin, weight_dim = 4, 4, 16
    radial_net = SimpleRadialNet(cout, cin, weight_dim)
    embedding = BinnedRadialEmbedding(radial_net, cout=cout, cin=cin, weight_dim=weight_dim, num_bins=10)
    table1 = embedding.get_table()
    table2 = embedding.get_table()
    assert table1 is table2
    print("  test_embedding_caching: PASS")


def test_interpolate_weights():
    """Test Python weight interpolation."""
    # table shape: (num_bins, cout, cin, weight_dim) = (10, 2, 2, 3)
    num_bins, cout, cin, weight_dim = 10, 2, 2, 3
    # Create table where values increase with bin index for easy verification
    table = torch.arange(num_bins).float().view(num_bins, 1, 1, 1).expand(num_bins, cout, cin, weight_dim)
    bin_data = BinData(
        lo=torch.tensor([0, 2, 4]),
        hi=torch.tensor([1, 3, 5]),
        weight=torch.tensor([0.5, 0.5, 0.5]),
    )
    result = interpolate_weights(table, bin_data)
    # Expected: interpolation gives 0.5, 2.5, 4.5 for all (cout, cin, weight_dim)
    assert result.shape == (3, cout, cin, weight_dim)
    assert torch.allclose(result[0], torch.full((cout, cin, weight_dim), 0.5))
    assert torch.allclose(result[1], torch.full((cout, cin, weight_dim), 2.5))
    assert torch.allclose(result[2], torch.full((cout, cin, weight_dim), 4.5))
    print("  test_interpolate_weights: PASS")


def test_binned_vs_full_correctness():
    """Verify binned kernel matches full kernel with expanded weights."""
    device = torch.device('cuda')
    lvals = [0, 1, 2]
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    batch, cin, cout = 32, 16, 16
    num_bins = 50

    metadata = build_block_metadata(lvals, lvals, device)

    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    features = torch.randn(batch, cin, dim, device=device)

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    distances = torch.rand(batch, device=device) * 10.0
    # Table shape: (num_bins + 1, cout, cin, weight_dim)
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device)
    bin_data = binning.compute_bins(distances)

    # Binned kernel
    output_binned = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight,
        cout, metadata
    )

    # Full kernel with expanded weights
    # interpolate_weights returns (batch, cout, cin, weight_dim)
    weights_interp = interpolate_weights(radial_table, bin_data)
    output_full = block_diagonal_cuda(features, weights_interp, metadata)

    assert_close(output_binned, output_full)
    print("  test_binned_vs_full_correctness: PASS")


def test_binned_output_shape():
    """Verify output shape is correct."""
    device = torch.device('cuda')
    lvals = [0, 1, 2, 3]
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    batch, cin, cout = 64, 32, 32
    num_bins = 100

    metadata = build_block_metadata(lvals, lvals, device)
    features = torch.randn(batch, cin, dim, device=device)

    binning = RadialBinning(num_bins=num_bins, device=device)
    # Table shape: (num_bins + 1, cout, cin, weight_dim)
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device)
    bin_data = binning.compute_bins(torch.rand(batch, device=device) * 10.0)

    output = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight,
        cout, metadata
    )

    assert output.shape == (batch, cout, dim)
    print("  test_binned_output_shape: PASS")


def test_binned_dtypes():
    """Test binned kernel with different dtypes."""
    device = torch.device('cuda')
    lvals = [0, 1, 2]
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    batch, cin, cout = 32, 16, 16
    num_bins = 50

    metadata = build_block_metadata(lvals, lvals, device)
    binning = RadialBinning(num_bins=num_bins, device=device)

    for dtype in [torch.float32, torch.float16]:
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
        # Table shape: (num_bins + 1, cout, cin, weight_dim)
        radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)
        bin_data = binning.compute_bins(torch.rand(batch, device=device) * 10.0)
        bin_data.weight = bin_data.weight.to(dtype)

        output = block_diagonal_binned_interp_cuda(
            features, radial_table,
            bin_data.lo, bin_data.hi, bin_data.weight,
            cout, metadata
        )

        assert output.dtype == dtype
        assert not torch.isnan(output).any()

    print("  test_binned_dtypes: PASS")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Binned Weights Tests")
    print("=" * 60)

    print("\nRadialBinning Tests (CPU):")
    test_radial_binning_creation()
    test_radial_binning_edges()
    test_radial_binning_centers()
    test_compute_indices()
    test_compute_indices_clamping()
    test_compute_bins()
    test_interpolation_weights()
    test_create_table()
    test_validation()

    print("\nBinnedRadialEmbedding Tests (CPU):")
    test_embedding_creation()
    test_embedding_get_table()
    test_embedding_caching()

    print("\nInterpolation Tests (CPU):")
    test_interpolate_weights()

    if torch.cuda.is_available():
        print("\nCUDA Kernel Tests:")
        test_binned_vs_full_correctness()
        test_binned_output_shape()
        test_binned_dtypes()
    else:
        print("\nSkipping CUDA tests (no GPU available)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
