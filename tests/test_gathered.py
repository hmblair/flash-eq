"""Tests for gathered (fused node->edge) kernel with bin-sorting."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_binned_interp_cuda,
    block_diagonal_gathered_cuda,
    get_weight_dim,
)


def test_gathered_matches_binned():
    """Test that gathered kernel produces same output as binned kernel."""
    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 4
    num_nodes = 500
    num_edges = 2000
    cin = cout = 16
    num_bins = 50

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create node features and edge structure
    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 10.0
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)

    # Expand features for binned kernel (current approach)
    edge_features = node_features[src_indices]

    # Run binned kernel
    output_binned = block_diagonal_binned_interp_cuda(edge_features, radial_table, distances, metadata)

    # Run gathered kernel (unsorted for direct comparison)
    output_gathered, _ = block_diagonal_gathered_cuda(
        node_features, src_indices, radial_table, distances, metadata, sort_by_bin=False
    )

    # Compare
    max_diff = (output_binned - output_gathered).abs().max().item()
    rel_diff = max_diff / output_binned.abs().mean().item()

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")

    assert torch.allclose(output_binned, output_gathered, rtol=1e-4, atol=1e-5), \
        f"Outputs differ! Max diff: {max_diff}"
    print("  test_gathered_matches_binned: PASS")


def test_gathered_sorted_matches_unsorted():
    """Test that bin-sorted gathered output can be unsorted to match unsorted output."""
    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 4
    num_nodes = 500
    num_edges = 2000
    cin = cout = 16
    num_bins = 50

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 10.0
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)

    # Run unsorted
    output_unsorted, _ = block_diagonal_gathered_cuda(
        node_features, src_indices, radial_table, distances, metadata, sort_by_bin=False
    )

    # Run sorted
    output_sorted, unsort_indices = block_diagonal_gathered_cuda(
        node_features, src_indices, radial_table, distances, metadata, sort_by_bin=True
    )

    # Unsort the sorted output
    output_restored = output_sorted[unsort_indices]

    # Compare
    max_diff = (output_unsorted - output_restored).abs().max().item()

    print(f"  Max difference after unsort: {max_diff:.2e}")

    assert torch.allclose(output_unsorted, output_restored, rtol=1e-4, atol=1e-5), \
        f"Sorted/unsorted mismatch! Max diff: {max_diff}"
    print("  test_gathered_sorted_matches_unsorted: PASS")


def test_gathered_gradients():
    """Test that gathered kernel gradients match binned kernel gradients."""
    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 3
    num_nodes = 200
    num_edges = 500
    cin = cout = 8
    num_bins = 30

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create inputs with gradients
    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype, requires_grad=True)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 10.0
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype, requires_grad=True)

    # Clone for gathered
    node_features_g = node_features.detach().clone().requires_grad_(True)
    radial_table_g = radial_table.detach().clone().requires_grad_(True)

    # Binned: expand features first
    edge_features = node_features[src_indices]
    output_binned = block_diagonal_binned_interp_cuda(edge_features, radial_table, distances, metadata)
    loss_binned = output_binned.sum()
    loss_binned.backward()

    # Gathered (unsorted for direct comparison)
    output_gathered, _ = block_diagonal_gathered_cuda(
        node_features_g, src_indices, radial_table_g, distances, metadata, sort_by_bin=False
    )
    loss_gathered = output_gathered.sum()
    loss_gathered.backward()

    # Compare grad_radial_table
    grad_table_diff = (radial_table.grad - radial_table_g.grad).abs().max().item()
    print(f"  grad_radial_table max diff: {grad_table_diff:.2e}")

    # For node features gradients, run a separate forward/backward with fresh tensors
    # The gathered kernel scatters gradients back to nodes internally
    node_features2 = node_features.detach().clone().requires_grad_(True)
    radial_table2 = radial_table.detach().clone()
    edge_features2 = node_features2[src_indices]
    output2 = block_diagonal_binned_interp_cuda(edge_features2, radial_table2, distances, metadata)
    loss2 = output2.sum()
    loss2.backward()

    grad_node_diff = (node_features2.grad - node_features_g.grad).abs().max().item()
    print(f"  grad_node_features max diff: {grad_node_diff:.2e}")

    assert torch.allclose(radial_table.grad, radial_table_g.grad, rtol=1e-4, atol=1e-5), \
        f"Table gradients differ! Max diff: {grad_table_diff}"
    assert torch.allclose(node_features2.grad, node_features_g.grad, rtol=1e-4, atol=1e-5), \
        f"Node feature gradients differ! Max diff: {grad_node_diff}"
    print("  test_gathered_gradients: PASS")


def test_gathered_large_scale():
    """Test gathered kernel at large scale (128k edges)."""
    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 6
    num_nodes = 5000
    num_edges = 128000
    cin = cout = 32
    num_bins = 100

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 10.0
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)

    # Test that it runs without error
    output, unsort_indices = block_diagonal_gathered_cuda(
        node_features, src_indices, radial_table, distances, metadata, sort_by_bin=True
    )

    assert output.shape == (num_edges, cout, dim)
    assert unsort_indices.shape == (num_edges,)
    assert not torch.isnan(output).any()
    print("  test_gathered_large_scale: PASS")


def benchmark_gathered_vs_binned():
    """Benchmark gathered (bin-sorted) vs binned kernel."""
    import gc

    device = torch.device("cuda")
    dtype = torch.float32

    lmax = 6
    num_nodes = 5000
    num_edges = 128000
    cin = cout = 32
    num_bins = 100

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    node_features = torch.randn(num_nodes, cin, dim, device=device, dtype=dtype)
    src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    distances = torch.rand(num_edges, device=device) * 10.0
    radial_table = torch.randn(num_bins + 1, cout, cin, weight_dim, device=device, dtype=dtype)

    # Precompute expanded features for binned
    edge_features = node_features[src_indices]

    # Warmup
    for _ in range(3):
        _ = block_diagonal_binned_interp_cuda(edge_features, radial_table, distances, metadata)
        _ = block_diagonal_gathered_cuda(node_features, src_indices, radial_table, distances, metadata, sort_by_bin=True)
    torch.cuda.synchronize()

    # Benchmark binned (with pre-expansion)
    gc.collect()
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        edge_features = node_features[src_indices]  # Include expansion in timing
        _ = block_diagonal_binned_interp_cuda(edge_features, radial_table, distances, metadata)
    end.record()
    torch.cuda.synchronize()
    binned_ms = start.elapsed_time(end) / 10

    # Benchmark gathered (bin-sorted)
    gc.collect()
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        _ = block_diagonal_gathered_cuda(node_features, src_indices, radial_table, distances, metadata, sort_by_bin=True)
    end.record()
    torch.cuda.synchronize()
    gathered_sorted_ms = start.elapsed_time(end) / 10

    # Benchmark gathered (unsorted, for comparison)
    gc.collect()
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        _ = block_diagonal_gathered_cuda(node_features, src_indices, radial_table, distances, metadata, sort_by_bin=False)
    end.record()
    torch.cuda.synchronize()
    gathered_unsorted_ms = start.elapsed_time(end) / 10

    print(f"\n  Config: L={lmax}, nodes={num_nodes}, edges={num_edges}, C={cin}x{cout}")
    print(f"  Binned (with expand):     {binned_ms:.2f}ms")
    print(f"  Gathered (unsorted):      {gathered_unsorted_ms:.2f}ms")
    print(f"  Gathered (bin-sorted):    {gathered_sorted_ms:.2f}ms")
    print(f"  Speedup (sorted vs binned): {binned_ms / gathered_sorted_ms:.2f}x")


def main():
    print("=" * 60)
    print("Gathered Kernel Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    print("\nCorrectness Tests:")
    test_gathered_matches_binned()
    test_gathered_sorted_matches_unsorted()
    test_gathered_gradients()
    test_gathered_large_scale()

    print("\nPerformance Benchmark:")
    benchmark_gathered_vs_binned()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
