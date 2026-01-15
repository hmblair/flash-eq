"""
Benchmark individual components of the low-rank equivariant pipeline.

Pipeline: output = Q @ block_diag(P^T @ features, weights)

Components:
1. P^T @ features (batched matmul)
2. block_diagonal_cuda (custom kernel)
3. Q @ result (batched matmul)
"""

import torch
import torch.utils.benchmark as benchmark
from flash_eq.block_diagonal_cuda import (
    build_block_metadata,
    block_diagonal_cuda,
    block_diagonal_cuda_v2,
    get_weight_dim,
)


def benchmark_component(fn, name, num_iters=100, warmup=10):
    """Benchmark a single component."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def run_benchmark(lvals, batch_size, channels_in, channels_out, dtype=torch.float32):
    """Run component-wise benchmark for given config."""
    device = torch.device("cuda")

    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)

    # Create inputs
    features = torch.randn(batch_size, channels_in, dim, device=device, dtype=dtype)
    P = torch.randn(batch_size, dim, dim, device=device, dtype=dtype)
    Q = torch.randn(batch_size, dim, dim, device=device, dtype=dtype)
    weights = torch.randn(batch_size, channels_out, channels_in, weight_dim, device=device, dtype=dtype)
    metadata = build_block_metadata(lvals, lvals, device)

    # Intermediate tensors (pre-allocated for fair timing)
    f_diag = torch.empty(batch_size, channels_in, dim, device=device, dtype=dtype)
    f_out = torch.empty(batch_size, channels_out, dim, device=device, dtype=dtype)
    output = torch.empty(batch_size, channels_out, dim, device=device, dtype=dtype)

    # Component functions
    def step1_pt_matmul():
        torch.bmm(P.transpose(-1, -2), features.transpose(-1, -2), out=f_diag.transpose(-1, -2))

    def step2_block_diag_v1():
        return block_diagonal_cuda(f_diag, weights, metadata)

    def step2_block_diag_v2():
        return block_diagonal_cuda_v2(f_diag, weights, metadata)

    def step3_q_matmul():
        torch.bmm(Q, f_out.transpose(-1, -2), out=output.transpose(-1, -2))

    def full_pipeline_v1():
        f_d = torch.bmm(P.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        f_o = block_diagonal_cuda(f_d, weights, metadata)
        return torch.bmm(Q, f_o.transpose(-1, -2)).transpose(-1, -2)

    def full_pipeline_v2():
        f_d = torch.bmm(P.transpose(-1, -2), features.transpose(-1, -2)).transpose(-1, -2)
        f_o = block_diagonal_cuda_v2(f_d, weights, metadata)
        return torch.bmm(Q, f_o.transpose(-1, -2)).transpose(-1, -2)

    # Run step2 once to populate f_out for step3
    f_out_tmp = step2_block_diag_v1()
    f_out.copy_(f_out_tmp)

    # Benchmark each component
    t1 = benchmark_component(step1_pt_matmul, "P^T @ features")
    t2_v1 = benchmark_component(step2_block_diag_v1, "block_diag V1")
    t2_v2 = benchmark_component(step2_block_diag_v2, "block_diag V2")
    t3 = benchmark_component(step3_q_matmul, "Q @ result")
    t_full_v1 = benchmark_component(full_pipeline_v1, "full pipeline V1")
    t_full_v2 = benchmark_component(full_pipeline_v2, "full pipeline V2")

    return {
        'P^T @ features': t1,
        'block_diag_v1': t2_v1,
        'block_diag_v2': t2_v2,
        'Q @ result': t3,
        'full_v1': t_full_v1,
        'full_v2': t_full_v2,
    }


def main():
    print("=" * 80)
    print("Component-wise Pipeline Timing")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print()

    configs = [
        # (lmax, batch, cin, cout)
        (3, 1000, 64, 64),
        (4, 1000, 64, 64),
        (6, 1000, 64, 64),
        (6, 2000, 64, 64),
        (6, 5000, 64, 64),
        (6, 5000, 32, 32),
        (6, 10000, 32, 32),
    ]

    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n{'-' * 100}")
        print(f"{dtype_name} Results")
        print(f"{'-' * 100}")
        print(f"{'Config':<32} {'P^T@f':>8} {'BD-V1':>10} {'BD-V2':>10} {'Q@out':>8} {'Full-V1':>10} {'Full-V2':>10} {'Speedup':>8}")
        print(f"{'-' * 100}")

        for lmax, batch, cin, cout in configs:
            lvals = list(range(lmax + 1))
            dim = sum(2 * l + 1 for l in lvals)

            try:
                results = run_benchmark(lvals, batch, cin, cout, dtype)

                speedup = results['full_v1'] / results['full_v2']
                config_str = f"L={lmax}, B={batch}, C={cin}x{cout}"
                print(f"{config_str:<32} "
                      f"{results['P^T @ features']:>7.2f}ms "
                      f"{results['block_diag_v1']:>9.2f}ms "
                      f"{results['block_diag_v2']:>9.2f}ms "
                      f"{results['Q @ result']:>7.2f}ms "
                      f"{results['full_v1']:>9.2f}ms "
                      f"{results['full_v2']:>9.2f}ms "
                      f"{speedup:>7.2f}x")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"L={lmax}, B={batch}, C={cin}x{cout}: OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

    print()
    print("=" * 100)
    print("BD-V1: Original kernel (per-output parallelization)")
    print("BD-V2: New kernel (per-m-block parallelization with shared memory)")
    print("=" * 100)


if __name__ == "__main__":
    main()
