"""S² activation for SO(3)-equivariant networks.

Implements the S² activation from EquiformerV2, which applies nonlinearities
in the spatial domain on the sphere rather than directly on SH coefficients.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr
from ..spherical import S2Grid


class S2Activation(nn.Module):
    """S² activation: nonlinearity via spherical grid sampling.

    Transforms spherical harmonic coefficients to function values on a
    Lebedev quadrature grid, applies a channel-wise MLP, then transforms back
    to SH coefficients. This provides a learnable, approximately equivariant
    nonlinearity that mixes information across degrees.

    The forward pass:
        1. f_grid = f_coeffs @ Y^T  (SH → grid, shape: ..., mult, n_points)
        2. f_grid = MLP(f_grid)     (channel-wise nonlinearity)
        3. f_out = f_grid @ Y_inv^T (grid → SH, shape: ..., mult, dim)

    Supports non-contiguous l values (e.g., lvals=[1, 2] without l=0).

    Args:
        repr: The representation of input tensors.
        hidden_mult: Hidden layer size as multiple of channels (default: 2).
        activation: Activation function (default: SiLU).
        precision: Lebedev precision (default: 47, 770 points).
            Higher precision gives better equivariance but costs more.
            Available: 17 (110 pts), 23 (194), 29 (302), ..., 131 (5810 pts).

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> act = S2Activation(repr)
        >>> x = torch.randn(100, 32, 9)  # (batch, mult, dim)
        >>> y = act(x)  # same shape
    """

    def __init__(
        self,
        repr: Repr,
        hidden_mult: int = 2,
        activation: nn.Module | None = None,
        precision: int = 47,
    ) -> None:
        super().__init__()

        self.repr = repr
        self.mult = repr.mult
        self.dim = repr.dim()
        l_max = repr.lmax()

        # Initialize S² grid with Lebedev quadrature
        grid = S2Grid(l_max=l_max, precision=precision)
        self.n_points = grid.n_points
        self.precision = precision

        # Extract only the columns of Y corresponding to the l values in repr.
        # For each l, columns l² to (l+1)²-1 contain the 2l+1 SH coefficients.
        lvals = repr.lvals.tolist()
        indices = []
        for l in lvals:
            start = l * l
            end = (l + 1) * (l + 1)
            indices.extend(range(start, end))

        Y_subset = grid.Y[:, indices]  # (n_points, dim)
        weights = grid.weights

        # Recompute inverse transform for the subset
        W = torch.diag(weights)
        YtW = Y_subset.T @ W
        YtWY = YtW @ Y_subset
        Y_inv_subset = torch.linalg.solve(YtWY, YtW)  # (dim, n_points)

        # Store transposed matrices to avoid transpose in forward pass
        self.register_buffer('Y_T', Y_subset.T.float().contiguous())        # (dim, n_points)
        self.register_buffer('Y_inv_T', Y_inv_subset.T.float().contiguous())  # (n_points, dim)

        # Channel-wise MLP using Conv1d (avoids transpose operations)
        # Conv1d with kernel_size=1 mixes channels at each grid point
        hidden_dim = self.mult * hidden_mult
        act_fn = activation if activation is not None else nn.SiLU()

        self.mlp = nn.Sequential(
            nn.Conv1d(self.mult, hidden_dim, 1),
            act_fn,
            nn.Conv1d(hidden_dim, self.mult, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply S² activation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Activated tensor of shape (..., mult, dim).
        """
        # To grid: (..., mult, n_points)
        f_grid = f @ self.Y_T

        # Apply Conv1d MLP across channels at each grid point
        # Conv1d expects (batch, channels, length) = (batch, mult, n_points)
        batch_shape = f_grid.shape[:-2]
        f_grid = f_grid.view(-1, self.mult, self.n_points)
        f_grid = self.mlp(f_grid)
        f_grid = f_grid.view(*batch_shape, self.mult, self.n_points)

        # Back to SH: (..., mult, dim)
        return f_grid @ self.Y_inv_T

    def extra_repr(self) -> str:
        return f"mult={self.mult}, dim={self.dim}, n_points={self.n_points}, precision={self.precision}"
