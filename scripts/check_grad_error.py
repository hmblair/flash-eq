"""Print gradient relative error for both log_bins settings and dtypes."""
import torch
from flash_eq import Repr, EquivariantEdgewiseLinear, WignerDBasis

lvals = [0, 1]
mult = 2
num_nodes = 5
num_edges = 10
dim = sum(2 * l + 1 for l in lvals)

in_repr = Repr(lvals=lvals, mult=mult)
out_repr = Repr(lvals=lvals, mult=mult)

device = torch.device("cuda")

for dtype_name, dtype in [("float32", torch.float32), ("float64", torch.float64)]:
    print(f"\n=== {dtype_name} ===")
    for log_bins in [False, True]:
        torch.manual_seed(123)
        min_dist = 0.5 if log_bins else 0.0
        layer = EquivariantEdgewiseLinear(
            in_repr, out_repr,
            num_bins=50, min_dist=min_dist, max_dist=10.0,
            log_bins=log_bins,
        ).to(device).to(dtype)
        basis = WignerDBasis([in_repr, out_repr]).to(device).to(dtype)

        node_features = torch.randn(num_nodes, mult, dim, device=device, dtype=dtype)
        src_indices = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
        directions = torch.randn(num_edges, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        P, Q = basis(directions)
        edge_features = node_features[src_indices]
        distances = torch.rand(num_edges, device=device, dtype=dtype) * 5.0 + min_dist + 0.5

        distances_grad = distances.clone().requires_grad_(True)
        output = layer(P, Q, edge_features, distances_grad)
        loss = (output ** 2).sum()
        loss.backward()
        analytical = distances_grad.grad.clone()

        eps = 1e-4 if dtype == torch.float32 else 1e-6
        numerical = torch.zeros_like(distances)
        for i in range(num_edges):
            dp = distances.clone(); dp[i] += eps
            dm = distances.clone(); dm[i] -= eps
            numerical[i] = ((layer(P, Q, edge_features, dp) ** 2).sum()
                          - (layer(P, Q, edge_features, dm) ** 2).sum()) / (2 * eps)

        rel_err = (analytical - numerical).abs() / (numerical.abs() + 1e-8)
        print(f"  log_bins={log_bins}: max_rel_error = {rel_err.max().item():.6f}  mean_rel_error = {rel_err.mean().item():.6f}")
