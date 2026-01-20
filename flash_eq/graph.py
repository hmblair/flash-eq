"""Graph data structure for equivariant GNNs.

Provides a lightweight graph container that bundles edge indices and node count,
simplifying function signatures throughout the codebase.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Graph:
    """Lightweight graph container for edge indices and node count.

    Bundles src_indices, dst_indices, and num_nodes into a single object,
    reducing function signature complexity. Uses COO format (coordinate list).

    Args:
        src: (num_edges,) source node index for each edge.
        dst: (num_edges,) destination node index for each edge.
        num_nodes: Total number of nodes in the graph.

    Example:
        >>> graph = Graph(
        ...     src=torch.tensor([0, 1, 2]),
        ...     dst=torch.tensor([1, 2, 0]),
        ...     num_nodes=3,
        ... )
        >>> graph.num_edges
        3
        >>> graph.device
        device(type='cpu')
    """

    src: torch.Tensor
    dst: torch.Tensor
    num_nodes: int

    def __post_init__(self) -> None:
        """Validate graph structure."""
        if self.src.dim() != 1:
            raise ValueError(f"src must be 1D, got shape {self.src.shape}")
        if self.dst.dim() != 1:
            raise ValueError(f"dst must be 1D, got shape {self.dst.shape}")
        if self.src.shape[0] != self.dst.shape[0]:
            raise ValueError(
                f"src and dst must have same length, got {self.src.shape[0]} and {self.dst.shape[0]}"
            )
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.src.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the edge tensors."""
        return self.src.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the edge index tensors."""
        return self.src.dtype

    def to(self, device: torch.device | str) -> Graph:
        """Move graph to device."""
        return Graph(
            src=self.src.to(device),
            dst=self.dst.to(device),
            num_nodes=self.num_nodes,
        )

    @staticmethod
    def random(
        num_nodes: int,
        num_edges: int,
        device: torch.device | str = "cpu",
    ) -> Graph:
        """Create a random graph (may include self-loops and multi-edges).

        Args:
            num_nodes: Number of nodes.
            num_edges: Number of edges.
            device: Device for tensors.

        Returns:
            Random graph with given size.
        """
        src = torch.randint(0, num_nodes, (num_edges,), device=device)
        dst = torch.randint(0, num_nodes, (num_edges,), device=device)
        return Graph(src=src, dst=dst, num_nodes=num_nodes)
