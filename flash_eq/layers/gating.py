"""Equivariant gating nonlinearity for spherical tensors.

This module provides norm-based gating for SO(3)-equivariant networks.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr
from .norm import RepNorm


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EquivariantGating(nn.Module):
    """Norm-based gating for spherical tensors.

    Computes norms of each irrep component, processes them through
    a linear layer and sigmoid, then uses the result to gate the
    original tensor. This provides a learnable, equivariant nonlinearity.

    Args:
        repr: The representation of input tensors.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> gate = EquivariantGating(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = gate(x)  # shape: (32, 8, 9)
    """

    def __init__(self, repr: Repr) -> None:
        super().__init__()

        self.repr = repr
        self.norm = RepNorm(repr)

        # Linear layer for processing norms
        self.linear = nn.Linear(
            repr.nreps() * repr.mult,
            repr.nreps() * repr.mult,
        )

        # Indices mapping norms back to full dimension
        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

        self.outdims = (repr.mult, repr.nreps())
        self.activation = nn.Sigmoid()

        self.apply(_init_weights)

    def forward(self, st: torch.Tensor) -> torch.Tensor:
        """Apply gating to spherical tensor.

        Args:
            st: Spherical tensor of shape (..., mult, dim).

        Returns:
            Gated tensor of shape (..., mult, dim).
        """
        # Compute norms
        norms = self.norm(st)

        *b, _, _ = norms.size()
        norms = norms.flatten(-2, -1)

        # Process through linear layer
        norms = self.linear(norms).view(*b, *self.outdims)

        # Apply activation
        norms = self.activation(norms)

        # Gate the input
        return st * norms[..., self.ix]
