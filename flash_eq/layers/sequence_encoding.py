"""Sequence position encoding for polymer/protein graphs.

Encodes relative sequence distance (|i - j|) between connected nodes,
allowing the model to distinguish covalent neighbors from tertiary contacts.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SequencePositionEncoding(nn.Module):
    """Sinusoidal encoding of relative sequence distance for edges.

    For polymers and proteins, the distance in sequence space between two
    residues is informative:
    - Sequential neighbors (|i-j| = 1) are covalently bonded
    - Distant in sequence but close in space = tertiary/quaternary contacts

    This module computes an encoding of |seq_pos[src] - seq_pos[dst]| for
    each edge, producing a scalar feature vector that can be projected
    into equivariant edge features.

    Args:
        dim: Output dimension of the encoding.
        max_seq_distance: Maximum sequence distance to encode. Distances
            beyond this are clipped. Default 128 covers most local patterns.
        learnable: If True, use learnable embeddings instead of fixed
            sinusoidal encoding.

    Example:
        >>> enc = SequencePositionEncoding(dim=32)
        >>> seq_pos = torch.arange(100)  # residue indices
        >>> src = torch.tensor([0, 0, 1, 1, 2])
        >>> dst = torch.tensor([1, 5, 0, 2, 10])
        >>> features = enc(seq_pos, src, dst)  # (5, 32)
    """

    def __init__(
        self,
        dim: int,
        max_seq_distance: int = 128,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_distance = max_seq_distance
        self.learnable = learnable

        if learnable:
            self.embedding = nn.Embedding(max_seq_distance + 1, dim)
        else:
            position = torch.arange(max_seq_distance + 1).float().unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
            )
            pe = torch.zeros(max_seq_distance + 1, dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            if dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: dim // 2])
            self.register_buffer("pe", pe)

    def forward(
        self,
        seq_pos: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sequence position encoding for edges.

        Args:
            seq_pos: (num_nodes,) sequence position for each node (e.g.,
                residue index).
            src: (num_edges,) source node index for each edge.
            dst: (num_edges,) destination node index for each edge.

        Returns:
            (num_edges, dim) position encoding per edge.
        """
        seq_dist = (seq_pos[src] - seq_pos[dst]).abs()
        seq_dist = seq_dist.clamp(max=self.max_seq_distance)

        if self.learnable:
            return self.embedding(seq_dist.long())
        return self.pe[seq_dist.long()]

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_seq_distance={self.max_seq_distance}, "
            f"learnable={self.learnable}"
        )
