"""
Test FusedRadialBlockDiagonal - verify chunked implementation matches original CUDA.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    get_weight_dim,
)
from flash_eq.fused_radial import FusedRadialBlockDiagonal


def test_chunked_matches_original(lvals, batch, cin, cout, chunk_size, dtype=torch.float32):
    """
    Verify that FusedRadialBlockDiagonal.forward() matches the original
    block_diagonal_cuda() implementation.
    """
    device = torch.device('cuda')

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    # Create layer
    layer = FusedRadialBlockDiagonal(
        cout, cin, weight_dim, hidden_dim=64, chunk_size=chunk_size
    ).to(device).to(dtype)
    layer.set_metadata(metadata)

    # Test inputs
    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    distances = torch.rand(batch, device=device, dtype=dtype) * 10.0

    with torch.no_grad():
        # Chunked implementation
        out_chunked = layer(features, distances)

        # Original implementation (compute full weights, then block_diagonal)
        out_reference = layer.forward_reference(features, distances)

    # Compare
    max_diff = (out_chunked - out_reference).abs().max().item()
    rel_diff = max_diff / (out_reference.abs().max().item() + 1e-8)

    return max_diff, rel_diff


def test_different_chunk_sizes():
    """Test that all chunk sizes produce identical results."""
    device = torch.device('cuda')
    lvals = [0, 1, 2]
    batch, cin, cout = 32, 16, 16

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    features = torch.randn(batch, cin, dim, device=device)
    distances = torch.rand(batch, device=device) * 10.0

    # Create reference layer (large chunk = no chunking effect)
    layer_ref = FusedRadialBlockDiagonal(cout, cin, weight_dim, chunk_size=cout).to(device)
    layer_ref.set_metadata(metadata)

    with torch.no_grad():
        out_ref = layer_ref.forward_reference(features, distances)

    results = []
    for chunk_size in [1, 2, 4, 8, 16]:
        if chunk_size > cout:
            continue

        # Create layer with same weights
        layer = FusedRadialBlockDiagonal(cout, cin, weight_dim, chunk_size=chunk_size).to(device)
        layer.load_state_dict(layer_ref.state_dict())
        layer.set_metadata(metadata)

        with torch.no_grad():
            out_chunked = layer(features, distances)

        rel_diff = (out_chunked - out_ref).abs().max() / out_ref.abs().max()
        results.append((chunk_size, rel_diff.item()))

    return results


def main():
    print("=" * 70)
    print("FusedRadialBlockDiagonal Tests")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    all_passed = True

    # Test chunked matches original for various configs
    print("\n" + "-" * 70)
    print("Chunked vs Original CUDA Implementation")
    print("-" * 70)

    configs = [
        ([0, 1], 32, 16, 16, 4),
        ([0, 1, 2], 32, 16, 16, 8),
        ([0, 1, 2], 64, 32, 32, 8),
        ([0, 1, 2, 3], 32, 32, 32, 8),
        ([0, 1, 2, 3], 32, 32, 32, 1),  # chunk_size=1
    ]

    for lvals, batch, cin, cout, chunk_size in configs:
        max_diff, rel_diff = test_chunked_matches_original(
            lvals, batch, cin, cout, chunk_size
        )
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, chunk={chunk_size}: "
              f"rel_diff={rel_diff:.2e} [{status}]")

    # Test all chunk sizes produce same result
    print("\n" + "-" * 70)
    print("Different Chunk Sizes Produce Same Result")
    print("-" * 70)

    results = test_different_chunk_sizes()
    for chunk_size, rel_diff in results:
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"chunk_size={chunk_size}: rel_diff={rel_diff:.2e} [{status}]")

    # Test FP16
    print("\n" + "-" * 70)
    print("FP16 Correctness")
    print("-" * 70)

    for lvals, batch, cin, cout, chunk_size in configs[:3]:
        max_diff, rel_diff = test_chunked_matches_original(
            lvals, batch, cin, cout, chunk_size, dtype=torch.float16
        )
        status = "PASS" if rel_diff < 1e-2 else "FAIL"  # Looser for FP16
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}, chunk={chunk_size}: "
              f"rel_diff={rel_diff:.2e} [{status}]")

    print("\n" + "=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
