"""
Memory-efficient radial MLP + block-diagonal multiplication.

Instead of:
    radial_weights = MLP(edge_features)  # (B, cout, cin, weight_dim) - LARGE
    output = block_diagonal(features, radial_weights)

We use chunked computation:
    for chunk in output_channels:
        weights_chunk = hidden2 @ W3[chunk] + b3[chunk]  # Uses cuBLAS
        output[chunk] = block_diagonal(features, weights_chunk)

This avoids ever materializing the full (B, cout, cin, weight_dim) tensor.
With chunk_size=8, achieves ~0.94-0.98x speed with 2.5-5x memory reduction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FusedRadialMLP(nn.Module):
    """
    MLP structured for fused kernel computation.

    The standard MLP computes:
        hidden1 = silu(edge_feat @ w1 + b1)     # (batch, hidden_dim)
        hidden2 = silu(hidden1 @ W2 + b2)       # (batch, hidden_dim)
        output = hidden2 @ W3 + b3              # (batch, cout * cin * weight_dim)

    This module restructures W3 to (cout, cin, hidden_dim, weight_dim) so the
    fused kernel can compute weights per (co, ci) pair on-demand.
    """

    def __init__(
        self,
        cout: int,
        cin: int,
        weight_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim

        # First layer: edge_feat -> hidden1 (edge_dim is typically 1 for distance)
        # For a single distance input, w1 is just (hidden_dim,) not a matrix
        self.w1 = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        # Second layer: hidden1 -> hidden2
        self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * (2.0 / hidden_dim) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        # Third layer: hidden2 -> weights, structured as (cout, cin, hidden_dim, weight_dim)
        self.W3 = nn.Parameter(
            torch.randn(cout, cin, hidden_dim, weight_dim) * (2.0 / hidden_dim) ** 0.5
        )
        self.b3 = nn.Parameter(torch.zeros(cout, cin, weight_dim))

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (for reference/testing).

        Args:
            edge_features: (batch,) or (batch, 1) distances

        Returns:
            weights: (batch, cout, cin, weight_dim)
        """
        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        # hidden1 = silu(edge_feat * w1 + b1)
        hidden1 = edge_features * self.w1 + self.b1  # (batch, hidden_dim)
        hidden1 = torch.nn.functional.silu(hidden1)

        # hidden2 = silu(hidden1 @ W2 + b2)
        hidden2 = hidden1 @ self.W2 + self.b2  # (batch, hidden_dim)
        hidden2 = torch.nn.functional.silu(hidden2)

        # output = hidden2 @ W3.view(hidden_dim, -1) + b3.view(-1)
        # Reshape for batch matmul
        batch = hidden2.size(0)
        W3_flat = self.W3.view(self.cout * self.cin, self.hidden_dim, self.weight_dim)
        b3_flat = self.b3.view(self.cout * self.cin, self.weight_dim)

        # (batch, hidden_dim) @ (cout*cin, hidden_dim, weight_dim) -> (batch, cout*cin, weight_dim)
        output = torch.einsum('bh,cdhw->bcdw', hidden2, self.W3) + self.b3

        return output  # (batch, cout, cin, weight_dim)

    def get_fused_params(self) -> Tuple[torch.Tensor, ...]:
        """Get parameters in the format expected by the fused CUDA kernel."""
        return (
            self.w1.data,
            self.b1.data,
            self.W2.data,
            self.b2.data,
            self.W3.data,
            self.b3.data,
        )


class FusedRadialBlockDiagonal(nn.Module):
    """
    Memory-efficient radial MLP + block-diagonal multiplication.

    Uses chunked computation to avoid materializing the large (B, cout, cin, weight_dim)
    intermediate tensor, reducing memory by 4-40x depending on chunk_size.

    Example:
        layer = FusedRadialBlockDiagonal(
            cout=64, cin=64, weight_dim=441,
            hidden_dim=128,
            chunk_size=8,  # Trade-off: larger = faster, smaller = less memory
        )

        # Setup metadata once
        metadata = build_block_metadata(lvals, lvals, device)
        layer.set_metadata(metadata)

        # Forward pass - just works
        output = layer(features, distances)

    Args:
        cout: Number of output channels
        cin: Number of input channels
        weight_dim: Dimension of weight tensor (from get_weight_dim)
        hidden_dim: MLP hidden dimension (default 128)
        chunk_size: Output channels processed per chunk (default 8).
            - Larger = faster (more cuBLAS efficiency) but more memory
            - Smaller = slower but less memory
            - chunk_size=8 gives ~0.94-0.98x speed with 4-8x memory reduction
            - chunk_size=1 gives ~0.4-0.6x speed with 20-40x memory reduction
    """

    def __init__(
        self,
        cout: int,
        cin: int,
        weight_dim: int,
        hidden_dim: int = 128,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.weight_dim = weight_dim
        self.chunk_size = chunk_size

        self.mlp = FusedRadialMLP(cout, cin, weight_dim, hidden_dim)
        self._metadata: Optional[Tuple[torch.Tensor, ...]] = None

    def set_metadata(self, metadata: Tuple[torch.Tensor, ...]):
        """Set block metadata (call once after moving to device)."""
        self._metadata = metadata

    def forward(
        self,
        features: torch.Tensor,
        edge_features: torch.Tensor,
        metadata: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with memory-efficient chunked computation.

        Args:
            features: (batch, cin, dim) features in diagonal basis
            edge_features: (batch,) edge distances
            metadata: Block metadata (optional if set via set_metadata)

        Returns:
            output: (batch, cout, dim)
        """
        from flash_eq.block_diagonal_cuda import block_diagonal_fused_broadcast_cuda

        meta = metadata if metadata is not None else self._metadata
        if meta is None:
            raise ValueError("metadata must be provided via set_metadata() or forward()")

        # Compute MLP hidden layers using cuBLAS
        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        hidden1 = torch.nn.functional.silu(edge_features * self.mlp.w1 + self.mlp.b1)
        hidden2 = torch.nn.functional.silu(hidden1 @ self.mlp.W2 + self.mlp.b2)

        # Chunked final projection + block-diagonal
        return block_diagonal_fused_broadcast_cuda(
            features, hidden2,
            self.mlp.W3, self.mlp.b3,
            self.cout, meta,
            chunk_size=self.chunk_size
        )

    def forward_reference(
        self,
        features: torch.Tensor,
        edge_features: torch.Tensor,
        metadata: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """
        Reference forward pass (standard MLP + block_diagonal, for testing).

        This materializes the full weight tensor - use only for correctness testing.
        """
        from flash_eq.block_diagonal_cuda import block_diagonal_cuda

        meta = metadata if metadata is not None else self._metadata
        if meta is None:
            raise ValueError("metadata must be provided")

        weights = self.mlp(edge_features)  # (batch, cout, cin, weight_dim)
        return block_diagonal_cuda(features, weights, meta)


def test_correctness():
    """Test that chunked forward matches reference implementation."""
    from flash_eq.block_diagonal_cuda import build_block_metadata, get_weight_dim

    print("Testing chunked kernel correctness...")

    device = torch.device("cuda")
    lvals = [0, 1, 2]
    dim = sum(2 * l + 1 for l in lvals)
    weight_dim = get_weight_dim(lvals, lvals)
    batch, cin, cout = 32, 16, 16

    metadata = build_block_metadata(lvals, lvals, device)

    # Test different chunk sizes
    for chunk_size in [1, 4, 8, 16]:
        layer = FusedRadialBlockDiagonal(
            cout, cin, weight_dim, hidden_dim=64, chunk_size=chunk_size
        ).to(device)
        layer.set_metadata(metadata)

        features = torch.randn(batch, cin, dim, device=device)
        distances = torch.rand(batch, device=device) * 10.0

        with torch.no_grad():
            out_chunked = layer(features, distances)
            out_ref = layer.forward_reference(features, distances)

        diff = (out_chunked - out_ref).abs()
        rel_diff = diff.max() / out_ref.abs().max()

        status = "PASS" if rel_diff < 1e-4 else "FAIL"
        print(f"  chunk_size={chunk_size}: rel_diff={rel_diff:.2e} [{status}]")

        if rel_diff >= 1e-4:
            return False

    return True


def benchmark_memory():
    """Compare memory usage of chunked vs standard approach."""
    import gc
    from flash_eq.block_diagonal_cuda import build_block_metadata, get_weight_dim

    print("\nMemory comparison (chunk_size=8 vs standard):")

    device = torch.device("cuda")

    configs = [
        (4, 5000, 32, 32),
        (6, 5000, 32, 32),
        (6, 5000, 64, 64),
        (6, 10000, 32, 32),
    ]

    for lmax, batch, cin, cout in configs:
        lvals = list(range(lmax + 1))
        dim = sum(2 * l + 1 for l in lvals)
        weight_dim = get_weight_dim(lvals, lvals)

        weight_tensor_mb = batch * cout * cin * weight_dim * 4 / 1024**2

        print(f"\n  Lmax={lmax}, B={batch}, C={cin}x{cout}: weight tensor = {weight_tensor_mb:.1f} MB")

        metadata = build_block_metadata(lvals, lvals, device)
        features = torch.randn(batch, cin, dim, device=device)
        distances = torch.rand(batch, device=device) * 10.0

        # Standard approach (materializes full weight tensor)
        layer_std = FusedRadialBlockDiagonal(
            cout, cin, weight_dim, hidden_dim=128, chunk_size=cout  # chunk_size=cout = no chunking
        ).to(device)
        layer_std.set_metadata(metadata)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = layer_std.forward_reference(features, distances)

        torch.cuda.synchronize()
        std_mem = torch.cuda.max_memory_allocated() / 1024**2

        del layer_std
        gc.collect()
        torch.cuda.empty_cache()

        # Chunked approach (chunk_size=8)
        layer_chunked = FusedRadialBlockDiagonal(
            cout, cin, weight_dim, hidden_dim=128, chunk_size=8
        ).to(device)
        layer_chunked.set_metadata(metadata)

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = layer_chunked(features, distances)

        torch.cuda.synchronize()
        chunked_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"    Standard: {std_mem:.1f} MB, Chunked(8): {chunked_mem:.1f} MB ({std_mem/chunked_mem:.1f}x reduction)")


def benchmark_speed():
    """Compare speed of different chunk sizes."""
    from flash_eq.block_diagonal_cuda import build_block_metadata, get_weight_dim

    print("\nSpeed comparison (various chunk sizes):")

    device = torch.device("cuda")
    n_warmup, n_iter = 5, 20

    configs = [
        (4, 5000, 32, 32),
        (6, 5000, 32, 32),
        (6, 5000, 64, 64),
    ]

    for lmax, batch, cin, cout in configs:
        lvals = list(range(lmax + 1))
        dim = sum(2 * l + 1 for l in lvals)
        weight_dim = get_weight_dim(lvals, lvals)

        print(f"\n  Lmax={lmax}, B={batch}, C={cin}x{cout}:")

        metadata = build_block_metadata(lvals, lvals, device)
        features = torch.randn(batch, cin, dim, device=device)
        distances = torch.rand(batch, device=device) * 10.0

        # Create layers with different chunk sizes
        chunk_sizes = [1, 2, 4, 8, 16, cout]  # cout = full (reference)
        layers = {}
        for cs in chunk_sizes:
            if cs > cout:
                continue
            layers[cs] = FusedRadialBlockDiagonal(
                cout, cin, weight_dim, hidden_dim=128, chunk_size=cs
            ).to(device)
            layers[cs].set_metadata(metadata)

        # Warmup all
        for cs, layer in layers.items():
            for _ in range(n_warmup):
                with torch.no_grad():
                    if cs == cout:
                        _ = layer.forward_reference(features, distances)
                    else:
                        _ = layer(features, distances)
        torch.cuda.synchronize()

        # Benchmark reference (full weights)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            with torch.no_grad():
                _ = layers[cout].forward_reference(features, distances)
        end.record()
        torch.cuda.synchronize()
        std_time = start.elapsed_time(end) / n_iter

        # Test various chunk sizes
        results = []
        for cs in [1, 2, 4, 8, 16]:
            if cs > cout:
                continue
            start.record()
            for _ in range(n_iter):
                with torch.no_grad():
                    _ = layers[cs](features, distances)
            end.record()
            torch.cuda.synchronize()
            cs_time = start.elapsed_time(end) / n_iter
            results.append(f"C{cs}:{cs_time:.1f}ms({std_time/cs_time:.2f}x)")

        print(f"    Std: {std_time:.1f}ms | " + " | ".join(results))


if __name__ == "__main__":
    print("=" * 80)
    print("FusedRadialBlockDiagonal - Memory-Efficient Radial MLP + Block-Diagonal")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    if test_correctness():
        benchmark_memory()
        benchmark_speed()
