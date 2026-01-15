"""
Benchmark forward AND backward pass: Low-rank vs Dense.
"""

import torch
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
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


def benchmark(lmax, batch, cin, cout, dtype, n_warmup=5, n_iter=20):
    device = torch.device("cuda")
    lvals = list(range(lmax + 1))
    dim = sum(2*l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    metadata = build_block_metadata(lvals, lvals, device)
    perm = _build_m_order_permutation(lvals, device)

    # Dense params
    degrees = list(range(lmax + 1))
    freq_sum = sum((2 * min(d_in, d_out) + 1) for d_in in degrees for d_out in degrees)

    # Pre-build P and Q (not optimized, just used for transform)
    P = _build_random_orthogonal(batch, dim, device, dtype)
    Q = _build_random_orthogonal(batch, dim, device, dtype)
    P_perm = P[:, :, perm]
    Q_perm = Q[:, :, perm]

    def lr_forward_backward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
        weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype, requires_grad=True)
        f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        out_diag = block_diagonal_cuda(f_diag, weights, metadata)
        out = torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)
        loss = out.sum()
        loss.backward()

    def dense_forward_backward():
        features = torch.randn(batch, cin, dim, device=device, dtype=dtype, requires_grad=True)
        basis = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype, requires_grad=True)
        radial_weights = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype, requires_grad=True)
        basis_view = basis.view(batch, dim, -1)
        tmp = features @ basis_view
        tmp = tmp.view(batch, -1, dim)
        out = radial_weights @ tmp
        loss = out.sum()
        loss.backward()

    def lr_forward_only():
        with torch.no_grad():
            features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
            weights = torch.randn(batch, cout, cin, weight_dim, device=device, dtype=dtype)
            f_diag = torch.bmm(P_perm.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
            out_diag = block_diagonal_cuda(f_diag, weights, metadata)
            return torch.bmm(Q_perm, out_diag.transpose(-1, -2)).transpose(-1, -2)

    def dense_forward_only():
        with torch.no_grad():
            features = torch.randn(batch, cin, dim, device=device, dtype=dtype)
            basis = torch.randn(batch, dim, freq_sum, dim, device=device, dtype=dtype)
            radial_weights = torch.randn(batch, cout, cin * freq_sum, device=device, dtype=dtype)
            basis_view = basis.view(batch, dim, -1)
            tmp = features @ basis_view
            tmp = tmp.view(batch, -1, dim)
            return radial_weights @ tmp

    # Warmup
    for _ in range(n_warmup):
        lr_forward_backward()
        dense_forward_backward()
    torch.cuda.synchronize()

    # Benchmark LR forward+backward
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        lr_forward_backward()
    end.record()
    torch.cuda.synchronize()
    lr_time = start.elapsed_time(end) / n_iter

    # Benchmark Dense forward+backward
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        dense_forward_backward()
    end.record()
    torch.cuda.synchronize()
    dense_time = start.elapsed_time(end) / n_iter

    # Forward-only benchmarks
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        lr_forward_only()
    end.record()
    torch.cuda.synchronize()
    lr_fwd_time = start.elapsed_time(end) / n_iter

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        dense_forward_only()
    end.record()
    torch.cuda.synchronize()
    dense_fwd_time = start.elapsed_time(end) / n_iter

    return {
        'lr_fwd': lr_fwd_time,
        'lr_fwd_bwd': lr_time,
        'dense_fwd': dense_fwd_time,
        'dense_fwd_bwd': dense_time,
    }


def main():
    print("=" * 110)
    print("Forward + Backward Benchmark: Low-Rank V2 vs Dense")
    print("=" * 110)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    configs = [
        (4, 500, 64, 64),
        (6, 500, 64, 64),
        (6, 1000, 64, 64),
        (6, 2000, 32, 32),
        (6, 5000, 32, 32),
    ]

    for dtype, name in [(torch.float32, "FP32")]:
        print(f"\n{'-' * 110}")
        print(f"{name} Results")
        print(f"{'-' * 110}")
        print(f"{'Config':<28} {'LR Fwd':>10} {'Dense Fwd':>10} {'Fwd Speedup':>12} {'LR Fwd+Bwd':>12} {'Dense Fwd+Bwd':>14} {'Total Speedup':>14}")
        print(f"{'-' * 110}")

        for lmax, batch, cin, cout in configs:
            try:
                r = benchmark(lmax, batch, cin, cout, dtype)
                fwd_speedup = r['dense_fwd'] / r['lr_fwd']
                total_speedup = r['dense_fwd_bwd'] / r['lr_fwd_bwd']

                config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
                print(f"{config_str:<28} {r['lr_fwd']:>9.2f}ms {r['dense_fwd']:>9.2f}ms {fwd_speedup:>11.2f}x "
                      f"{r['lr_fwd_bwd']:>11.2f}ms {r['dense_fwd_bwd']:>13.2f}ms {total_speedup:>13.2f}x")
            except Exception as e:
                print(f"L={lmax}, B={batch}, C={cin}x{cout}: {e}")
                torch.cuda.empty_cache()

    print("\n" + "=" * 110)
    print("Note: Both forward and backward use V2 kernel (m-block parallelization with shared memory)")
    print("=" * 110)


if __name__ == "__main__":
    main()
