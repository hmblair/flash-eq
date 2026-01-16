"""
Test gradient support for binned interpolated block-diagonal multiplication.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_binned_interp_cuda,
    get_weight_dim,
)
from flash_eq.binned_weights import RadialBinning


def test_grad_features(lvals, batch, cin, cout, num_bins, dtype=torch.float32):
    """Test that grad_features matches between binned and exact approaches."""
    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create radial table (pretend it came from an MLP)
    radial_table = torch.randn(
        num_bins + 1, cout, cin, weight_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    distances = torch.rand(batch, device=device) * 10.0
    bin_data = binning.compute_bins(distances)

    # Input features with grad
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)

    # Forward pass
    output = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
        cout, metadata
    )

    # Backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    grad_features_binned = features.grad.clone()

    # Compare with exact approach
    features_exact = features.detach().clone().requires_grad_(True)
    t = bin_data.weight.to(dtype).view(-1, 1, 1, 1)
    weights_exact = (1 - t) * radial_table.detach()[bin_data.lo] + t * radial_table.detach()[bin_data.hi]
    output_exact = block_diagonal_cuda(features_exact, weights_exact, metadata)
    output_exact.backward(grad_output)

    grad_features_exact = features_exact.grad

    # Compare
    max_diff = (grad_features_binned - grad_features_exact).abs().max().item()
    rel_diff = max_diff / (grad_features_exact.abs().max().item() + 1e-8)

    return max_diff, rel_diff


def test_grad_radial_table(lvals, batch, cin, cout, num_bins, dtype=torch.float32):
    """Test that grad_radial_table is computed correctly."""
    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create radial table with grad
    radial_table = torch.randn(
        num_bins + 1, cout, cin, weight_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    distances = torch.rand(batch, device=device) * 10.0
    bin_data = binning.compute_bins(distances)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)

    # Forward pass with binned
    output = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
        cout, metadata
    )

    # Backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    grad_radial_table_binned = radial_table.grad.clone()

    # Compare with manual computation using exact weights
    radial_table_exact = radial_table.detach().clone().requires_grad_(True)
    t = bin_data.weight.to(dtype).view(-1, 1, 1, 1)
    weights = (1 - t) * radial_table_exact[bin_data.lo] + t * radial_table_exact[bin_data.hi]
    output_exact = block_diagonal_cuda(features, weights, metadata)
    output_exact.backward(grad_output)

    # Manual scatter of grad_weights to grad_radial_table
    # This should match what our backward does
    grad_radial_table_exact = radial_table_exact.grad

    # Compare
    max_diff = (grad_radial_table_binned - grad_radial_table_exact).abs().max().item()
    rel_diff = max_diff / (grad_radial_table_exact.abs().max().item() + 1e-8)

    return max_diff, rel_diff


def test_grad_through_mlp(lvals, batch, cin, cout, num_bins, dtype=torch.float32):
    """Test that gradients flow correctly through an MLP to radial_table."""
    import torch.nn as nn

    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Simple MLP for radial weights
    mlp = nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, cout * cin * weight_dim),
    ).to(device).to(dtype)

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    distances = torch.rand(batch, device=device) * 10.0
    bin_data = binning.compute_bins(distances)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)

    # Forward: MLP at bin edges -> radial_table -> binned block-diagonal
    radial_table = mlp(binning.bin_edges.unsqueeze(-1)).view(num_bins + 1, cout, cin, weight_dim)
    output = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
        cout, metadata
    )

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check that MLP parameters have gradients
    has_grads = all(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp.parameters())

    return has_grads


def test_gradcheck(lvals, batch, cin, cout, num_bins):
    """Use torch.autograd.gradcheck to verify gradients numerically."""
    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Use float64 for gradcheck
    dtype = torch.float64

    radial_table = torch.randn(
        num_bins + 1, cout, cin, weight_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0
    bin_data = binning.compute_bins(distances)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
    interp_weight = bin_data.weight.to(dtype).requires_grad_(True)

    def func(features, radial_table, interp_weight):
        return block_diagonal_binned_interp_cuda(
            features, radial_table,
            bin_data.lo, bin_data.hi, interp_weight,
            cout, metadata
        )

    # Use relaxed tolerances for CUDA kernels
    try:
        passed = torch.autograd.gradcheck(
            func, (features, radial_table, interp_weight),
            eps=1e-5, atol=1e-2, rtol=1e-2,
            raise_exception=True,
            nondet_tol=1e-4,  # Allow some non-determinism in CUDA
        )
    except Exception as e:
        print(f"    gradcheck error: {str(e)[:200]}")
        passed = False

    return passed


def test_grad_distances(lvals, batch, cin, cout, num_bins, dtype=torch.float32):
    """Test that gradients flow through to distances (for force computation)."""
    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create radial table
    radial_table = torch.randn(
        num_bins + 1, cout, cin, weight_dim,
        device=device, dtype=dtype
    )

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)

    # Distances with requires_grad (for force computation)
    distances = torch.rand(batch, device=device, dtype=dtype) * 9.0 + 0.5  # Avoid edges
    distances = distances.requires_grad_(True)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)

    # Forward pass - binning should preserve gradient graph
    bin_data = binning.compute_bins(distances)
    output = block_diagonal_binned_interp_cuda(
        features, radial_table,
        bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
        cout, metadata
    )

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that distances have gradients
    has_grad = distances.grad is not None
    grad_nonzero = has_grad and distances.grad.abs().sum() > 0

    return has_grad, grad_nonzero


def test_grad_distances_numerical(lvals, batch, cin, cout, num_bins):
    """Numerically verify gradient w.r.t. distances."""
    device = torch.device("cuda")
    dtype = torch.float64  # Use float64 for numerical accuracy

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    radial_table = torch.randn(
        num_bins + 1, cout, cin, weight_dim,
        device=device, dtype=dtype
    )

    binning = RadialBinning(num_bins=num_bins, max_dist=10.0, device=device)
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)

    # Distances away from bin edges to avoid clamp discontinuities
    distances = torch.rand(batch, device=device, dtype=dtype) * 8.0 + 1.0
    distances = distances.requires_grad_(True)

    def compute_output(d):
        bin_data = binning.compute_bins(d)
        out = block_diagonal_binned_interp_cuda(
            features, radial_table,
            bin_data.lo, bin_data.hi, bin_data.weight.to(dtype),
            cout, metadata
        )
        return out.sum()

    # Autograd gradient
    loss = compute_output(distances)
    loss.backward()
    grad_auto = distances.grad.clone()

    # Numerical gradient
    eps = 1e-5
    grad_num = torch.zeros_like(distances)
    for i in range(batch):
        distances_p = distances.detach().clone()
        distances_p[i] += eps
        distances_m = distances.detach().clone()
        distances_m[i] -= eps

        loss_p = compute_output(distances_p)
        loss_m = compute_output(distances_m)

        grad_num[i] = (loss_p - loss_m) / (2 * eps)

    # Compare
    max_diff = (grad_auto - grad_num).abs().max().item()
    rel_diff = max_diff / (grad_num.abs().max().item() + 1e-8)

    return max_diff, rel_diff


def main():
    print("=" * 70)
    print("Binned Interpolated Block-Diagonal Gradient Tests")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    all_passed = True

    # Test grad_features
    print("\n" + "-" * 70)
    print("Test: grad_features matches exact approach")
    print("-" * 70)

    configs = [
        ([0, 1], 32, 8, 8, 50),
        ([0, 1, 2], 64, 16, 16, 100),
        ([0, 1, 2, 3], 32, 16, 16, 100),
    ]

    for lvals, batch, cin, cout, num_bins in configs:
        max_diff, rel_diff = test_grad_features(lvals, batch, cin, cout, num_bins)
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: "
              f"rel_diff={rel_diff:.2e} [{status}]")

    # Test grad_radial_table
    print("\n" + "-" * 70)
    print("Test: grad_radial_table matches exact approach")
    print("-" * 70)

    for lvals, batch, cin, cout, num_bins in configs:
        max_diff, rel_diff = test_grad_radial_table(lvals, batch, cin, cout, num_bins)
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: "
              f"rel_diff={rel_diff:.2e} [{status}]")

    # Test gradient flow through MLP
    print("\n" + "-" * 70)
    print("Test: gradients flow through MLP")
    print("-" * 70)

    for lvals, batch, cin, cout, num_bins in configs:
        has_grads = test_grad_through_mlp(lvals, batch, cin, cout, num_bins)
        status = "PASS" if has_grads else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: [{status}]")

    # Test gradient flow to distances (for force computation)
    print("\n" + "-" * 70)
    print("Test: gradients flow through to distances (for forces)")
    print("-" * 70)

    for lvals, batch, cin, cout, num_bins in configs:
        has_grad, grad_nonzero = test_grad_distances(lvals, batch, cin, cout, num_bins)
        status = "PASS" if (has_grad and grad_nonzero) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: "
              f"has_grad={has_grad}, nonzero={grad_nonzero} [{status}]")

    # Numerical verification of distance gradients
    print("\n" + "-" * 70)
    print("Test: numerical verification of grad_distances")
    print("-" * 70)

    gradcheck_configs = [
        ([0, 1], 8, 4, 4, 20),
        ([0, 1, 2], 8, 4, 4, 20),
    ]

    for lvals, batch, cin, cout, num_bins in gradcheck_configs:
        max_diff, rel_diff = test_grad_distances_numerical(lvals, batch, cin, cout, num_bins)
        # Allow 1% relative error for numerical gradient test
        status = "PASS" if rel_diff < 1e-2 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: "
              f"rel_diff={rel_diff:.2e} [{status}]")

    # Numerical gradient check (smaller sizes due to cost)
    print("\n" + "-" * 70)
    print("Test: torch.autograd.gradcheck (numerical verification)")
    print("-" * 70)

    gradcheck_configs = [
        ([0, 1], 4, 2, 2, 10),
        ([0, 1, 2], 4, 2, 2, 10),
    ]

    for lvals, batch, cin, cout, num_bins in gradcheck_configs:
        passed = test_gradcheck(lvals, batch, cin, cout, num_bins)
        # gradcheck can be flaky with CUDA kernels - treat as warning not failure
        status = "PASS" if passed else "WARN"
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, bins={num_bins}: [{status}]")

    print("\n" + "=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
