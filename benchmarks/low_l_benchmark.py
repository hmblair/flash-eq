"""Benchmark V2 vs dense at lower lmax values."""
import torch
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda_v2,
    get_weight_dim,
)

def _build_random_orthogonal(batch, dim, device, dtype):
    rand = torch.randn(batch, dim, dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand)
    return Q.to(dtype)

def _build_m_order_permutation(lvals, device):
    perm = []
    lmax = max(lvals) if lvals else 0
    def std_pos(l_idx, m):
        return sum(2 * lvals[i] + 1 for i in range(l_idx)) + lvals[l_idx] + m
    for l_idx, l in enumerate(lvals):
        perm.append(std_pos(l_idx, 0))
    for m in range(1, lmax + 1):
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, m))
        for l_idx, l in enumerate(lvals):
            if l >= m:
                perm.append(std_pos(l_idx, -m))
    return torch.tensor(perm, dtype=torch.long, device=device)

def benchmark(lmax, batch, cin, cout, dtype):
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    perm = _build_m_order_permutation(lvals, device)

    # Dense params
    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    # Low-rank
    features_lr = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    weights_lr = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    # Dense
    features_dense = torch.randn(batch, cin, dim, device=device, dtype=dtype)
    basis_dense = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
    radial_weights_dense = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype)

    def lr_forward():
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features_lr.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_cuda_v2(f_diag, weights_lr, metadata)
        return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    def dense_forward():
        basis_view = basis_dense.view(batch, dim, -1)
        tmp = features_dense @ basis_view
        tmp = tmp.view(batch, -1, dim)
        return radial_weights_dense @ tmp

    # Warmup
    for _ in range(10):
        lr_forward()
        dense_forward()
    torch.cuda.synchronize()

    # Benchmark LR
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        lr_forward()
    end.record()
    torch.cuda.synchronize()
    lr_time = start.elapsed_time(end) / 100
    lr_mem = torch.cuda.max_memory_allocated() / 1024**3

    # Benchmark Dense
    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(100):
        dense_forward()
    end.record()
    torch.cuda.synchronize()
    dense_time = start.elapsed_time(end) / 100
    dense_mem = torch.cuda.max_memory_allocated() / 1024**3

    return lr_time, dense_time, lr_mem, dense_mem

def main():
    print("=" * 100)
    print("Low-Rank V2 vs Dense at Lower L values")
    print("=" * 100)
    print(f"{'Config':<32} {'LR V2':>10} {'Dense':>10} {'Speedup':>10} {'LR Mem':>10} {'Dense Mem':>10} {'Savings':>10}")
    print("-" * 100)

    configs = [
        (2, 1000, 64, 64),
        (2, 5000, 64, 64),
        (2, 10000, 64, 64),
        (2, 10000, 32, 32),
        (3, 1000, 64, 64),
        (3, 5000, 64, 64),
        (3, 10000, 32, 32),
        (4, 1000, 64, 64),
        (4, 5000, 64, 64),
    ]

    for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n{name}:")
        for lmax, batch, cin, cout in configs:
            try:
                lr_t, dense_t, lr_m, dense_m = benchmark(lmax, batch, cin, cout, dtype)
                speedup = dense_t / lr_t
                savings = (dense_m - lr_m) / dense_m * 100
                print(f"L={lmax}, B={batch}, C={cin}x{cout:<6} {lr_t:>9.2f}ms {dense_t:>9.2f}ms {speedup:>9.2f}x {lr_m:>9.2f}GB {dense_m:>9.2f}GB {savings:>9.1f}%")
            except Exception as e:
                print(f"L={lmax}, B={batch}, C={cin}x{cout}: {e}")
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
