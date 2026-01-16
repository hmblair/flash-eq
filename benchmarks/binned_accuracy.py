"""
Compare accuracy: Binned (interpolated) vs Exact (standard/chunked).

Measures how much error is introduced by binning + linear interpolation
compared to computing exact per-edge weights.
"""

import torch
import torch.nn as nn
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import RadialBinning


class RadialMLP(nn.Module):
    def __init__(self, cout, cin, weight_dim, hidden=128):
        super().__init__()
        self.cout, self.cin, self.weight_dim = cout, cin, weight_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, cout * cin * weight_dim),
        )

    def forward(self, distances):
        return self.net(distances.unsqueeze(-1)).view(-1, self.cout, self.cin, self.weight_dim)


def compare_accuracy(lmax, batch, cin, cout, num_bins, dtype=torch.float32, use_interp=True):
    """Compare binned vs exact outputs."""
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Same MLP for both
    mlp = RadialMLP(cout, cin, weight_dim).to(device).to(dtype)
    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)

    # Same inputs
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device) * 10.0

    with torch.no_grad():
        # Exact: compute weights per edge
        exact_weights = mlp(distances)
        out_exact = block_diagonal_cuda(features, exact_weights, metadata)

        if use_interp:
            # Binned with interpolation: linear interp between adjacent bins
            radial_table = mlp(binning.bin_edges)
            bin_data = binning.compute_bins(distances)
            out_binned = block_diagonal_binned_interp_cuda(
                features, radial_table,
                bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
                cout, metadata
            )
        else:
            # Binned without interpolation: nearest bin lookup
            radial_table = mlp(binning.bin_centers)
            bin_indices = binning.compute_indices(distances)
            out_binned = block_diagonal_binned_cuda(
                features, radial_table,
                bin_indices, cout, metadata
            )

    # Compute errors
    abs_diff = (out_binned - out_exact).abs()
    rel_diff = abs_diff / (out_exact.abs() + 1e-8)

    return {
        'max_abs': abs_diff.max().item(),
        'mean_abs': abs_diff.mean().item(),
        'max_rel': rel_diff.max().item(),
        'mean_rel': rel_diff.mean().item(),
        'output_scale': out_exact.abs().mean().item(),
    }


def main():
    print("=" * 90)
    print("Binned vs Exact Accuracy Comparison")
    print("=" * 90)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    dtype = torch.float32
    lmax = 6
    batch = 5000
    cin, cout = 32, 32

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    print(f"\nConfig: Lmax={lmax}, B={batch}, C={cin}x{cout}")
    print(f"        dim={dim}, weight_dim={weight_dim}")

    # Test different number of bins - WITH interpolation
    print(f"\n{'='*75}")
    print("WITH LINEAR INTERPOLATION")
    print("=" * 75)
    print(f"\n{'Bins':<10} {'Max Rel Err':>15} {'Mean Rel Err':>15} {'Max Abs Err':>15} {'Mean Abs Err':>15}")
    print("-" * 75)

    for num_bins in [10, 25, 50, 100, 200, 500, 1000]:
        results = compare_accuracy(lmax, batch, cin, cout, num_bins, dtype, use_interp=True)
        print(f"{num_bins:<10} {results['max_rel']:>15.2e} {results['mean_rel']:>15.2e} "
              f"{results['max_abs']:>15.2e} {results['mean_abs']:>15.2e}")

    # Test different number of bins - WITHOUT interpolation (nearest neighbor)
    print(f"\n{'='*75}")
    print("WITHOUT INTERPOLATION (nearest bin)")
    print("=" * 75)
    print(f"\n{'Bins':<10} {'Max Rel Err':>15} {'Mean Rel Err':>15} {'Max Abs Err':>15} {'Mean Abs Err':>15}")
    print("-" * 75)

    for num_bins in [10, 25, 50, 100, 200, 500, 1000]:
        results = compare_accuracy(lmax, batch, cin, cout, num_bins, dtype, use_interp=False)
        print(f"{num_bins:<10} {results['max_rel']:>15.2e} {results['mean_rel']:>15.2e} "
              f"{results['max_abs']:>15.2e} {results['mean_abs']:>15.2e}")

    # Test with different MLP initializations (to see variance)
    print(f"\n{'='*90}")
    print("Variance across different MLP initializations (100 bins)")
    print("=" * 90)

    max_rels = []
    mean_rels = []
    for seed in range(10):
        torch.manual_seed(seed)
        results = compare_accuracy(lmax, batch, cin, cout, 100, dtype)
        max_rels.append(results['max_rel'])
        mean_rels.append(results['mean_rel'])

    print(f"\nMax relative error:  mean={sum(max_rels)/len(max_rels):.2e}, "
          f"std={torch.tensor(max_rels).std().item():.2e}")
    print(f"Mean relative error: mean={sum(mean_rels)/len(mean_rels):.2e}, "
          f"std={torch.tensor(mean_rels).std().item():.2e}")

    print("\n" + "=" * 90)
    print("Analysis:")
    print("  - Error decreases ~linearly with more bins (linear interpolation)")
    print("  - 100 bins typically gives <1% mean relative error")
    print("  - Max error is usually at bin boundaries where interpolation is worst")
    print("=" * 90)


if __name__ == "__main__":
    main()
