"""Radial weight networks for distance-dependent equivariant operations.

This module provides neural networks that map distances to tensor product weights
for equivariant operations, with optional binning for memory efficiency.

Classes:
    RadialBasisFunctions: Learnable radial basis function expansion.
    RadialMLP: MLP mapping scalar distance to weight vector.
    SeparableRadialNet: Parameter-efficient separable radial weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import get_epsilon, init_linear_weights


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

        self.apply(init_linear_weights)

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


class SeparableRadialNet(nn.Module):
    """Parameter-efficient radial weights with separable channel/radial structure.

    Decomposes the weight tensor as:

        W[o, i, w] = Σ_k C_k[o, i] × f_k(r, w)

    where:
    - C_k: K learned channel mixing matrices (distance-independent)
    - f_k(r, w): radial functions output by MLP, varying per weight_dim position

    This is physically motivated: channel interactions don't need to vary with
    distance, only the strength/pattern of the interaction varies radially.

    Parameters scale as O(K × out × in + hidden × K × weight_dim), which is
    much smaller than O(num_bins × out × in × weight_dim) for direct methods.

    Args:
        num_bins: Number of bins for output table.
        out_mult: Output multiplicity dimension.
        in_mult: Input multiplicity dimension.
        weight_dim: Weight dimension (e.g., from block-diagonal structure).
        rank: Number of channel mixing patterns K (default: 4).
        num_freqs: Number of sinusoidal frequencies (default: 16).
        hidden_dim: MLP hidden dimension (default: 64).
        min_val: Minimum distance value (default: 0.0).
        max_val: Maximum distance value (default: 10.0).
        log: If True, use logarithmic bin spacing.

    Example:
        >>> radial = SeparableRadialNet(
        ...     num_bins=100, out_mult=16, in_mult=16, weight_dim=7, rank=4
        ... )
        >>> table = radial()  # (101, 16, 16, 7)
    """

    bin_edges: torch.Tensor
    frequencies: torch.Tensor

    def __init__(
        self,
        num_bins: int = 100,
        out_mult: int = 16,
        in_mult: int = 16,
        weight_dim: int = 7,
        rank: int = 4,
        num_freqs: int = 16,
        hidden_dim: int = 64,
        min_val: float = 0.0,
        max_val: float = 10.0,
        log: bool = False,
    ) -> None:
        super().__init__()

        if log and min_val <= 0:
            raise ValueError("min_val must be > 0 for log spacing")

        self.num_bins = num_bins
        self.out_mult = out_mult
        self.in_mult = in_mult
        self.weight_dim = weight_dim
        self.rank = rank
        self.min_val = min_val
        self.max_val = max_val
        self.log = log

        # Channel mixing patterns C_k[out, in] - distance independent
        self.channel_patterns = nn.Parameter(
            torch.empty(rank, out_mult, in_mult)
        )
        nn.init.xavier_uniform_(self.channel_patterns)

        # Fixed sinusoidal frequencies
        self.register_buffer(
            'frequencies',
            torch.arange(1, num_freqs + 1, dtype=torch.float32) * torch.pi
        )
        basis_dim = 2 * num_freqs

        # MLP: distance features -> f_k(r, w) for all k and w
        self.trunk = nn.Sequential(
            nn.Linear(basis_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Output head: produces (rank × weight_dim) values
        self.head = nn.Linear(hidden_dim, rank * weight_dim)

        # Bin edges for distance-to-index mapping
        if log:
            self.register_buffer(
                'bin_edges',
                torch.logspace(
                    torch.tensor(min_val).log10().item(),
                    torch.tensor(max_val).log10().item(),
                    num_bins + 1,
                )
            )
            self._log_min = torch.tensor(min_val).log().item()
            self._inv_log_range = 1.0 / (torch.tensor(max_val / min_val).log().item())
        else:
            self.register_buffer(
                'bin_edges',
                torch.linspace(min_val, max_val, num_bins + 1)
            )

        self.apply(init_linear_weights)

    def _sinusoidal_basis(self, r: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal basis with polynomial envelope."""
        normalized = (r - self.min_val) / (self.max_val - self.min_val)
        normalized = normalized.clamp(0.0, 1.0)

        # Polynomial envelope for smooth cutoff
        x = normalized
        envelope = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5

        angles = normalized.unsqueeze(-1) * self.frequencies
        sin_features = torch.sin(angles) * envelope.unsqueeze(-1)
        cos_features = torch.cos(angles) * envelope.unsqueeze(-1)

        return torch.cat([sin_features, cos_features], dim=-1)

    def forward(self) -> torch.Tensor:
        """Compute binned weight table.

        Returns:
            Table of shape (num_bins+1, out_mult, in_mult, weight_dim).
        """
        # Compute radial functions f_k(r, w) at bin edges
        basis = self._sinusoidal_basis(self.bin_edges)  # (B, basis_dim)
        hidden = self.trunk(basis)  # (B, hidden_dim)
        f = self.head(hidden)  # (B, rank * weight_dim)
        f = f.view(-1, self.rank, self.weight_dim)  # (B, K, W)

        # Combine: W[b,o,i,w] = Σ_k C[k,o,i] × f[b,k,w]
        # C: (K, O, I), f: (B, K, W) -> output: (B, O, I, W)
        table = torch.einsum('koi,bkw->boiw', self.channel_patterns, f)

        return table

    def binning_params(self) -> tuple[float, float]:
        """Return binning parameters for the CUDA kernel."""
        if self.log:
            return (self._log_min, self._inv_log_range)
        else:
            inv_bin_width = self.num_bins / (self.max_val - self.min_val)
            return (self.min_val, inv_bin_width)

    def bin_indices(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bin indices and interpolation weights for given values."""
        if self.log:
            normalized = (values.float().clamp(min=self.min_val).log() - self._log_min) * self._inv_log_range * self.num_bins
        else:
            inv_bin_width = self.num_bins / (self.max_val - self.min_val)
            normalized = (values.float() - self.min_val) * inv_bin_width

        normalized = normalized.clamp(0.0, self.num_bins)
        bin_lo = normalized.floor().int().clamp(max=self.num_bins - 1)
        interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0)
        return bin_lo, interp_weight
