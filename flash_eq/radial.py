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
    """Learnable lookup table with Gaussian smoothing for radial weights.

    Stores a learnable table of values at bin edges. During forward(), applies
    Gaussian smoothing along the bin dimension, which:
    1. Provides implicit regularization (smooth radial functions)
    2. Spreads gradients to neighboring bins during backprop

    The CUDA kernel handles interpolation between bins at runtime.

    Args:
        num_bins: Number of bins (default: 100).
        shape: Shape of output per bin, e.g., (out_mult, in_mult, weight_dim).
        min_val: Minimum distance value (default: 0.0).
        max_val: Maximum distance value (default: 10.0).
        log: If True, use logarithmic bin spacing (density ~ 1/r).
            Requires min_val > 0. Smoothing is uniform in log-space.
        sigma: Gaussian kernel width in bin units (default: 1.0).
            Larger values = more smoothing. The kernel covers ~3*sigma bins.

    Example:
        >>> binned = BinnedModule(num_bins=100, shape=(8, 8, 44))
        >>> table = binned()  # (101, 8, 8, 44) - pass to CUDA kernel
        >>> # Log spacing for molecular data
        >>> binned_log = BinnedModule(
        ...     num_bins=100, shape=(8, 8, 44),
        ...     min_val=0.5, max_val=10.0, log=True
        ... )
    """

    def __init__(
        self,
        num_bins: int = 100,
        shape: tuple[int, ...] = (),
        min_val: float = 0.0,
        max_val: float = 10.0,
        log: bool = False,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()

        if log and min_val <= 0:
            raise ValueError("min_val must be > 0 for log spacing")

        self.num_bins = num_bins
        self.shape = shape
        self.min_val = min_val
        self.max_val = max_val
        self.log = log
        self.sigma = sigma

        # Learnable table: (num_bins+1, *shape)
        # Xavier init treating shape[0] as fan_out, rest as fan_in
        self.raw_table = nn.Parameter(torch.empty(num_bins + 1, *shape))
        if shape:
            nn.init.xavier_uniform_(self.raw_table.view(num_bins + 1, shape[0], -1))
        else:
            nn.init.zeros_(self.raw_table)

        # Fixed Gaussian smoothing kernel
        radius = max(1, int(3 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        self._kernel: torch.Tensor
        self.register_buffer('_kernel', kernel / kernel.sum())

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

    def forward(self) -> torch.Tensor:
        """Return smoothed lookup table.

        Returns:
            Table of shape (num_bins+1, *shape).
        """
        # Reshape for conv1d: (num_bins+1, *shape) -> (prod(shape), 1, num_bins+1)
        n_bins = self.num_bins + 1
        flat = self.raw_table.permute(*range(1, len(self.shape) + 1), 0)
        flat = flat.reshape(-1, 1, n_bins)

        # Apply Gaussian smoothing with replicate padding
        pad = self._kernel.shape[0] // 2
        padded = torch.nn.functional.pad(flat, (pad, pad), mode='replicate')
        smoothed = torch.nn.functional.conv1d(padded, self._kernel.view(1, 1, -1))

        # Reshape back: (prod(shape), 1, num_bins+1) -> (num_bins+1, *shape)
        smoothed = smoothed.reshape(*self.shape, n_bins)
        return smoothed.permute(len(self.shape), *range(len(self.shape)))

    def bin_indices(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bin indices and interpolation weights for given values.

        Args:
            values: (N,) distance values to bin.

        Returns:
            bin_lo: (N,) lower bin indices as int32.
            interp_weight: (N,) interpolation weights in [0, 1].
        """
        if self.log:
            normalized = (values.float().clamp(min=self.min_val).log() - self._log_min) * self._inv_log_range * self.num_bins
        else:
            inv_bin_width = self.num_bins / (self.max_val - self.min_val)
            normalized = (values.float() - self.min_val) * inv_bin_width

        normalized = normalized.clamp(0.0, self.num_bins)
        bin_lo = normalized.floor().int().clamp(max=self.num_bins - 1)
        interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0)
        return bin_lo, interp_weight


class BinnedRadialBasis(nn.Module):
    """Radial basis function weights with binned output for CUDA kernel.

    Uses K radial basis functions to parameterize the weight table, reducing
    parameters from O(num_bins × out × in × weight_dim) to O(K × out × in × weight_dim).

    The basis functions are evaluated at bin centers to produce the full table
    needed by the CUDA kernel. This enforces smooth radial functions while
    maintaining computational efficiency.

    Args:
        num_bins: Number of bins (default: 100).
        shape: Shape of output per bin, e.g., (out_mult, in_mult, weight_dim).
        num_bases: Number of radial basis functions (default: 16).
        min_val: Minimum distance value (default: 0.0).
        max_val: Maximum distance value (default: 10.0).
        log: If True, use logarithmic bin spacing.
        trainable_bases: If True, basis centers/widths are learnable (default: False).

    Example:
        >>> # Instead of 100 × 64 × 64 × 85 = 34.8M params
        >>> # Uses 16 × 64 × 64 × 85 = 5.6M params
        >>> binned = BinnedRadialBasis(
        ...     num_bins=100, shape=(64, 64, 85), num_bases=16
        ... )
        >>> table = binned()  # (101, 64, 64, 85) - same interface as BinnedModule
    """

    centers: torch.Tensor
    widths: torch.Tensor
    bin_edges: torch.Tensor

    def __init__(
        self,
        num_bins: int = 100,
        shape: tuple[int, ...] = (),
        num_bases: int = 16,
        min_val: float = 0.0,
        max_val: float = 10.0,
        log: bool = False,
        trainable_bases: bool = False,
    ) -> None:
        super().__init__()

        if log and min_val <= 0:
            raise ValueError("min_val must be > 0 for log spacing")

        self.num_bins = num_bins
        self.shape = shape
        self.num_bases = num_bases
        self.min_val = min_val
        self.max_val = max_val
        self.log = log

        # Radial basis function centers and widths
        if log:
            centers = torch.logspace(
                torch.tensor(min_val).log10().item(),
                torch.tensor(max_val).log10().item(),
                num_bases,
            )
        else:
            centers = torch.linspace(min_val, max_val, num_bases)

        spacing = (max_val - min_val) / max(num_bases - 1, 1)
        widths = torch.full((num_bases,), spacing)

        if trainable_bases:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

        # Learnable coefficients: (num_bases, *shape)
        self.coefficients = nn.Parameter(torch.empty(num_bases, *shape))
        if shape:
            # Xavier init treating shape[0] as fan_out
            nn.init.xavier_uniform_(self.coefficients.view(num_bases, shape[0], -1))
        else:
            nn.init.zeros_(self.coefficients)

        # Precompute bin edge positions
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

    def _evaluate_basis(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian basis functions at given distances.

        Args:
            r: (N,) distance values.

        Returns:
            (N, num_bases) basis function values.
        """
        eps = get_epsilon(r.dtype)
        # Ensure widths are positive
        widths = self.widths.abs().clamp(min=eps) if isinstance(self.widths, nn.Parameter) else self.widths.clamp(min=eps)
        diff = (r.unsqueeze(-1) - self.centers) / widths
        return torch.exp(-diff ** 2)

    def forward(self) -> torch.Tensor:
        """Compute binned weight table from basis functions.

        Returns:
            Table of shape (num_bins+1, *shape).
        """
        # Evaluate basis functions at bin edges: (num_bins+1, num_bases)
        basis_values = self._evaluate_basis(self.bin_edges)

        # Weighted sum of coefficients: (num_bins+1, *shape)
        # basis_values: (num_bins+1, K), coefficients: (K, *shape)
        # Result: (num_bins+1, *shape)
        table = torch.einsum('bk,k...->b...', basis_values, self.coefficients)

        return table

    def bin_indices(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bin indices and interpolation weights for given values.

        Args:
            values: (N,) distance values to bin.

        Returns:
            bin_lo: (N,) lower bin indices as int32.
            interp_weight: (N,) interpolation weights in [0, 1].
        """
        if self.log:
            normalized = (values.float().clamp(min=self.min_val).log() - self._log_min) * self._inv_log_range * self.num_bins
        else:
            inv_bin_width = self.num_bins / (self.max_val - self.min_val)
            normalized = (values.float() - self.min_val) * inv_bin_width

        normalized = normalized.clamp(0.0, self.num_bins)
        bin_lo = normalized.floor().int().clamp(max=self.num_bins - 1)
        interp_weight = (normalized - bin_lo.float()).clamp(0.0, 1.0)
        return bin_lo, interp_weight
