"""Profile time distribution between m=0 and m>0 kernels."""

import torch
import torch.nn as nn
from flash_eq import EquivariantEdgewiseLinear, WignerDBasis, Repr


def profile_kernels(lmax, num_edges, cin, cout, num_bins=100, dtype=torch.float16, n_warmup=5, n_iter=20):
    """Profile forward pass kernel timing using CUDA events."""
    device = torch.device("cuda")

    lvals = list(range(lmax + 1))
    dim = sum(2 * l + 1 for l in lvals)

    in_repr = Repr(lvals=lvals, mult=cin)
    out_repr = Repr(lvals=lvals, mult=cout)

    layer = EquivariantEdgewiseLinear(
        in_repr, out_repr,
        num_bins=num_bins,
        min_dist=0.0,
        max_dist=10.0,
    ).to(device).to(dtype)

    basis = WignerDBasis([in_repr, out_repr]).to(device)

    edge_features = torch.randn(num_edges, cin, dim, device=device, dtype=dtype)

    directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distances = torch.rand(num_edges, device=device, dtype=dtype) * 10.0

    P, Q = basis(directions)

    # Warmup
    for _ in range(n_warmup):
        with torch.amp.autocast('cuda'):
            _ = layer(P, Q, edge_features, distances)
    torch.cuda.synchronize()

    # Profile with nsys markers or just total time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        with torch.amp.autocast('cuda'):
            _ = layer(P, Q, edge_features, distances)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end) / n_iter

    # Compute theoretical work distribution
    # m=0: n_out(0) * n_in(0) weights per (cout, cin) pair
    # m>0: 2 * n_out(m) * n_in(m) weights per (cout, cin) pair

    def count_l_geq_m(m):
        return sum(1 for l in lvals if l >= m)

    m0_work = count_l_geq_m(0) * count_l_geq_m(0)  # n_out * n_in for m=0
    mpos_work = sum(2 * count_l_geq_m(m) * count_l_geq_m(m) for m in range(1, lmax + 1))
    total_work = m0_work + mpos_work

    m0_frac = m0_work / total_work
    mpos_frac = mpos_work / total_work

    return {
        'total_ms': total_ms,
        'm0_work_frac': m0_frac,
        'mpos_work_frac': mpos_frac,
        'lmax': lmax,
        'num_edges': num_edges,
        'mmax': lmax,
        'n_in_m0': count_l_geq_m(0),
        'n_out_m0': count_l_geq_m(0),
    }


def main():
    print("Kernel Time Distribution Analysis")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    configs = [
        (1, 32000, 32, 32),
        (2, 32000, 32, 32),
        (4, 5000, 32, 32),
        (6, 5000, 32, 32),
        (4, 20000, 32, 32),
        (6, 20000, 32, 32),
    ]

    print(f"{'Config':<30} {'Time (ms)':<12} {'m=0 work %':<12} {'m>0 work %':<12}")
    print("-" * 80)

    for lmax, num_edges, cin, cout in configs:
        r = profile_kernels(lmax, num_edges, cin, cout)
        config_str = f"L={lmax}, E={num_edges}"
        print(f"{config_str:<30} {r['total_ms']:<12.2f} {r['m0_work_frac']*100:<12.1f} {r['mpos_work_frac']*100:<12.1f}")

    print()
    print("Note: Work fractions are theoretical based on weight counts.")
    print("Actual kernel time split requires nsys profiling.")
    print()
    print("To get actual kernel timing, run with nsys:")
    print("  nsys profile -o profile python scripts/profile_kernels.py")
    print("  nsys stats profile.nsys-rep")


if __name__ == "__main__":
    main()
