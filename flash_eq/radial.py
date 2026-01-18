"""Radial weight networks for distance-dependent equivariant operations.

This module provides neural networks that map distances to tensor product weights
for equivariant operations, with optional binning for memory efficiency.

Classes:
    RadialBasisFunctions: Learnable radial basis function expansion.
    RadialMLP: MLP mapping scalar distance to weight vector.
    BinnedModule: Wrapper that precomputes module outputs at bin edges.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .utils import get_epsilon


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RadialBasisFunctions(nn.Module):
    """Learnable Gaussian radial basis function expansion.

    Expands scalar distance values into Gaussian basis functions with
    learnable centers and widths. Useful for encoding distances in
    equivariant networks where edge features should be rotation-invariant.

    Args:
        num_functions: Number of basis functions.
        r_min: Minimum distance for center initialization.
        r_max: Maximum distance for center initialization.

    Example:
        >>> rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0)
        >>> distances = torch.rand(100, 1) * 10
        >>> features = rbf(distances)  # shape: (100, 16)
    """

    def __init__(
        self,
        num_functions: int,
        r_min: float = 0.0,
        r_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.num_functions = num_functions
        self.mu = nn.Parameter(torch.linspace(r_min, r_max, num_functions))
        spacing = (r_max - r_min) / max(num_functions - 1, 1)
        self.sigma = nn.Parameter(torch.full((num_functions,), spacing))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian RBFs at input distances.

        Args:
            x: Input distances of shape (N, 1).

        Returns:
            Basis function values of shape (N, num_functions).
        """
        if x.dim() > 1 and x.size(-1) == 1:
            x = x.squeeze(-1)

        eps = get_epsilon(self.sigma.dtype)
        diff = (x[..., None] - self.mu) / self.sigma.abs().clamp(min=eps)
        return torch.exp(-diff ** 2)


class RadialMLP(nn.Module):
    """MLP that maps scalar distance to weight tensor.

    Uses radial basis functions (RBF) as the first layer to expand scalar
    distances into a richer feature space before the MLP.

    Takes distance as input (shape (N, 1)) and outputs weights with shape
    (N, out_mult, in_mult, num_basis).

    Args:
        hidden_dim: Hidden layer dimension (also used as RBF dimension).
        num_basis: Number of weight elements (weight_dim for block-diagonal).
        in_mult: Input multiplicity (channels_in).
        out_mult: Output multiplicity (channels_out).
        num_layers: Number of hidden layers (default: 2).
        r_max: Maximum distance for RBF initialization (default: 10.0).

    Example:
        >>> mlp = RadialMLP(hidden_dim=64, num_basis=44, in_mult=8, out_mult=8)
        >>> distances = torch.rand(100, 1)
        >>> weights = mlp(distances)  # (100, 8, 8, 44)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_basis: int,
        in_mult: int,
        out_mult: int,
        num_layers: int = 2,
        r_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.in_mult = in_mult
        self.out_mult = out_mult

        # RBF expansion as first layer
        self.rbf = RadialBasisFunctions(hidden_dim, r_min=0.0, r_max=r_max)

        # MLP after RBF: hidden_dim -> hidden -> ... -> output
        output_dim = out_mult * in_mult * num_basis
        layers = [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weights from distances.

        Args:
            x: Distances of shape (N, 1).

        Returns:
            Weights of shape (N, out_mult, in_mult, num_basis).
        """
        batch_size = x.size(0)
        rbf_features = self.rbf(x)  # (N, hidden_dim)
        out = self.mlp(rbf_features)
        return out.view(batch_size, self.out_mult, self.in_mult, self.num_basis)


class BinnedModule(nn.Module):
    """Wraps a module to precompute outputs at bin edges for memory efficiency.

    Instead of computing outputs per-edge, this module evaluates the wrapped
    module at bin edges during forward(). The CUDA kernel handles interpolation
    at runtime. This reduces memory from O(edges) to O(bins).

    The wrapped module should take input of shape (N, 1) and return (N, ...).

    Args:
        module: Module to wrap. Takes (N, 1) input, returns (N, ...).
        num_bins: Number of bins (default: 100).
        min_val: Minimum value for binning (default: 0.0).
        max_val: Maximum value for binning (default: 10.0).

    Example:
        >>> mlp = RadialMLP(hidden_dim=64, num_basis=44, in_mult=8, out_mult=8)
        >>> binned = BinnedModule(mlp, num_bins=100, min_val=0.0, max_val=10.0)
        >>> table = binned()  # (101, 8, 8, 44) - pass to CUDA kernel
    """

    def __init__(
        self,
        module: nn.Module,
        num_bins: int = 100,
        min_val: float = 0.0,
        max_val: float = 10.0,
    ) -> None:
        super().__init__()

        self.module = module
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        self.register_buffer(
            'bin_edges',
            torch.linspace(min_val, max_val, num_bins + 1)
        )

    def forward(self) -> torch.Tensor:
        """Compute output table at bin edges.

        Returns:
            Table of shape (num_bins+1, ...) where ... is the module output shape.
        """
        return self.module(self.bin_edges.unsqueeze(-1))

    def bin_indices(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bin indices and interpolation weights for given values.

        Args:
            values: (N,) values to bin.

        Returns:
            bin_lo: (N,) lower bin indices as int32.
            interp_weight: (N,) interpolation weights in [0, 1].
        """
        inv_bin_width = self.num_bins / (self.max_val - self.min_val)
        normalized = (values.float() - self.min_val) * inv_bin_width
        normalized = normalized.clamp(0.0, self.num_bins)
        bin_lo = normalized.floor().int().clamp(max=self.num_bins - 1)
        interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0)
        return bin_lo, interp_weight
