"""Graph pooling for edge-to-node aggregation.

This module provides graph pooling operations for aggregating edge features
to nodes in equivariant graph neural networks.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..graph import Graph


def _expand_indices(dst_indices: torch.Tensor, channels: int, dim: int) -> torch.Tensor:
    """Expand (num_edges,) indices to (num_edges, channels, dim) for scatter."""
    return dst_indices.view(-1, 1, 1).expand(-1, channels, dim)


def _pool_sum(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sum pooling: aggregate edge features to nodes via summation."""
    _, channels, dim = edge_features.shape
    idx = _expand_indices(dst_indices, channels, dim)
    out = torch.zeros(num_nodes, channels, dim, device=edge_features.device, dtype=edge_features.dtype)
    return out.scatter_add_(0, idx, edge_features)


def _pool_mean(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean pooling: aggregate edge features to nodes via averaging."""
    out = _pool_sum(edge_features, dst_indices, num_nodes)

    # Count edges per node and divide
    counts = torch.zeros(num_nodes, device=out.device, dtype=out.dtype)
    counts.scatter_add_(0, dst_indices, torch.ones_like(dst_indices, dtype=out.dtype))
    return out / counts.clamp(min=1).view(-1, 1, 1)


def _pool_max(
    edge_features: torch.Tensor,
    dst_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Max pooling: aggregate edge features to nodes via maximum."""
    _, channels, dim = edge_features.shape
    idx = _expand_indices(dst_indices, channels, dim)
    out = torch.full((num_nodes, channels, dim), float('-inf'), device=edge_features.device, dtype=edge_features.dtype)
    out.scatter_reduce_(0, idx, edge_features, reduce='amax')
    return out.nan_to_num(neginf=0.0)


_POOL_FUNCTIONS = {
    'sum': _pool_sum,
    'mean': _pool_mean,
    'max': _pool_max,
}


class GraphPooling(nn.Module):
    """Aggregate edge features to nodes.

    Supports sum, mean, and max pooling. Uses PyTorch's scatter operations
    for GPU-efficient aggregation without external graph library dependencies.

    Args:
        reduce: Aggregation method ('sum', 'mean', or 'max').

    Example:
        >>> pool = GraphPooling(reduce='sum')
        >>> graph = Graph.random(num_nodes=100, num_edges=1000)
        >>> edge_features = torch.randn(1000, 32, 9)  # 1000 edges, 32 channels, dim=9
        >>> node_features = pool(edge_features, graph)
        >>> node_features.shape
        torch.Size([100, 32, 9])
    """

    def __init__(self, reduce: str = 'sum') -> None:
        super().__init__()
        if reduce not in _POOL_FUNCTIONS:
            raise ValueError(f"reduce must be one of {list(_POOL_FUNCTIONS.keys())}, got '{reduce}'")
        self.reduce = reduce
        self._pool_fn = _POOL_FUNCTIONS[reduce]

    def forward(
        self,
        edge_features: torch.Tensor,
        graph: Graph,
    ) -> torch.Tensor:
        """Aggregate edge features to destination nodes.

        Args:
            edge_features: (num_edges, channels, dim) edge feature tensor.
            graph: Graph containing edge indices and node count.

        Returns:
            node_features: (num_nodes, channels, dim) aggregated node features.
        """
        return self._pool_fn(edge_features, graph.dst, graph.num_nodes)

    def extra_repr(self) -> str:
        return f"reduce='{self.reduce}'"
