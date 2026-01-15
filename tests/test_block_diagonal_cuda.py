"""
Test CUDA block-diagonal kernel - correctness and performance.
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    get_weight_dim,
)


def block_diagonal_python(features, weights, lvals_in, lvals_out):
    """Pure Python reference implementation."""
    lmax = max(max(lvals_in), max(lvals_out))
    count = lambda lvals, m: sum(1 for l in lvals if l >= m)

    batch, channels_in, _ = features.shape
    channels_out = weights.shape[1]

    # Build blocks
    blocks = []
    in_off = out_off = w_off = 0
    for m in range(lmax + 1):
        n_in, n_out = count(lvals_in, m), count(lvals_out, m)
        if n_in > 0 and n_out > 0:
            blocks.append({'m': m, 'n_in': n_in, 'n_out': n_out,
                          'in_off': in_off, 'out_off': out_off, 'w_off': w_off})
            mult = 1 if m == 0 else 2
            in_off += mult * n_in
            out_off += mult * n_out
            w_off += mult * n_out * n_in

    dim_out = out_off
    out = torch.zeros(batch, channels_out, dim_out, device=features.device, dtype=features.dtype)

    for blk in blocks:
        m = blk['m']
        n_in, n_out = blk['n_in'], blk['n_out']
        in_s, out_s, w_off = blk['in_off'], blk['out_off'], blk['w_off']

        if m == 0:
            f_m = features[:, :, in_s:in_s + n_in]
            w_m = weights[:, :, :, w_off:w_off + n_out * n_in].view(
                batch, channels_out, channels_in, n_out, n_in)
            out[:, :, out_s:out_s + n_out] = torch.einsum('bocji,bci->boj', w_m, f_m)
        else:
            f_re = features[:, :, in_s:in_s + n_in]
            f_im = features[:, :, in_s + n_in:in_s + 2*n_in]
            w_m = weights[:, :, :, w_off:w_off + 2*n_out*n_in].view(
                batch, channels_out, channels_in, n_out, n_in, 2)
            a, b = w_m[..., 0], w_m[..., 1]
            out[:, :, out_s:out_s + n_out] = (
                torch.einsum('bocji,bci->boj', a, f_re) +
                torch.einsum('bocji,bci->boj', b, f_im)
            )
            out[:, :, out_s + n_out:out_s + 2*n_out] = (
                torch.einsum('bocji,bci->boj', a, f_im) -
                torch.einsum('bocji,bci->boj', b, f_re)
            )

    return out


def test_correctness(lvals, batch, cin, cout, dtype=torch.float32):
    """Test CUDA kernel matches Python reference."""
    device = torch.device('cuda')

    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)

    # Use FP32 for reference to avoid numerical issues
    out_ref = block_diagonal_python(
        features.float(), weights.float(), lvals, lvals
    ).to(dtype)
    out_cuda = block_diagonal_cuda(features, weights, metadata)

    max_diff = (out_ref - out_cuda).abs().max().item()
    rel_diff = max_diff / (out_ref.abs().max().item() + 1e-8)

    return max_diff, rel_diff


def test_backward(lvals, batch, cin, cout, dtype=torch.float32):
    """Test backward pass gradients."""
    device = torch.device('cuda')

    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
    weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype, requires_grad=True)

    out = block_diagonal_cuda(features, weights, metadata)
    loss = out.sum()
    loss.backward()

    # Check gradients exist and have correct shape
    assert features.grad is not None
    assert weights.grad is not None
    assert features.grad.shape == features.shape
    assert weights.grad.shape == weights.shape

    return True


def benchmark(lvals, batch, cin, cout, dtype=torch.float32, n_warmup=10, n_iter=100):
    """Benchmark CUDA kernel."""
    device = torch.device('cuda')

    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)

    features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        _ = block_diagonal_cuda(features, weights, metadata)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = block_diagonal_cuda(features, weights, metadata)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / n_iter * 1000

    return cuda_time


def main():
    print("=" * 70)
    print("Block-Diagonal CUDA Kernel Tests")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test configurations
    configs = [
        ([0, 1], 32, 64, 64),
        ([0, 1, 2], 32, 64, 64),
        ([0, 1, 2, 3], 32, 128, 128),
        ([0, 1, 2, 3, 4], 16, 128, 128),
    ]

    all_passed = True

    # FP32 Tests
    print("\n" + "-" * 70)
    print("FP32 Correctness Tests")
    print("-" * 70)

    for lvals, batch, cin, cout in configs:
        max_diff, rel_diff = test_correctness(lvals, batch, cin, cout, torch.float32)
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}: rel_diff={rel_diff:.2e} [{status}]")

    # FP16 Tests
    print("\n" + "-" * 70)
    print("FP16 Correctness Tests")
    print("-" * 70)

    for lvals, batch, cin, cout in configs:
        max_diff, rel_diff = test_correctness(lvals, batch, cin, cout, torch.float16)
        status = "PASS" if rel_diff < 1e-2 else "FAIL"  # Looser tolerance for FP16
        if status == "FAIL":
            all_passed = False
        print(f"L={lvals}, B={batch}, C={cin}x{cout}: rel_diff={rel_diff:.2e} [{status}]")

    # Backward Tests
    print("\n" + "-" * 70)
    print("Backward Pass Tests")
    print("-" * 70)

    for lvals, batch, cin, cout in configs[:2]:
        for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            try:
                test_backward(lvals, batch, cin, cout, dtype)
                print(f"L={lvals}, {name}: PASS")
            except Exception as e:
                print(f"L={lvals}, {name}: FAIL - {e}")
                all_passed = False

    # Performance
    print("\n" + "-" * 70)
    print("Performance Benchmarks")
    print("-" * 70)

    for lvals, batch, cin, cout in configs:
        fp32_time = benchmark(lvals, batch, cin, cout, torch.float32)
        fp16_time = benchmark(lvals, batch, cin, cout, torch.float16)
        print(f"L={lvals}, B={batch}, C={cin}x{cout}: FP32={fp32_time:.3f}ms, FP16={fp16_time:.3f}ms")

    print("\n" + "=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
