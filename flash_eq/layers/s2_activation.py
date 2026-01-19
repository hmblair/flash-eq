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
        # Store transposed matrices to avoid transpose in forward pass
        self.register_buffer('Y_T', grid.Y.T.contiguous())        # (dim, n_points)
        self.register_buffer('Y_inv_T', grid.Y_inv.T.contiguous())  # (n_points, dim)

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


class SeparableS2Activation(nn.Module):
    """Separable S² activation (EquiformerV2 style).

    Treats scalar (l=0) and higher-degree components separately:
    - Scalars: Direct SiLU activation (fast, exact)
    - Full tensor: S² activation (includes degree mixing)

    The final output combines scalar activations with higher-degree outputs
    from the S² path.

    Args:
        repr: The representation of input tensors.
        hidden_mult: Hidden layer size as multiple of channels (default: 2).
        scalar_activation: Activation for scalars (default: SiLU).
        s2_activation: Activation for S² MLP (default: SiLU).
        precision: Lebedev precision (default: 47, 770 points).

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> act = SeparableS2Activation(repr)
        >>> x = torch.randn(100, 32, 9)
        >>> y = act(x)
    """

    def __init__(
        self,
        repr: Repr,
        hidden_mult: int = 2,
        scalar_activation: nn.Module | None = None,
        s2_activation: nn.Module | None = None,
        precision: int = 47,
    ) -> None:
        super().__init__()

        self.repr = repr

        # Find scalar and higher-degree components
        nscalar, scalar_locs = repr.find_scalar()
        self.nscalar = nscalar
        self.scalar_locs = scalar_locs

        # Mask for higher-degree components
        higher_mask = torch.ones(repr.dim(), dtype=torch.bool)
        higher_mask[scalar_locs] = False
        self.register_buffer('higher_mask', higher_mask)
        self.nhigher = repr.dim() - nscalar

        # Scalar activation (direct, no grid)
        self.scalar_act = scalar_activation if scalar_activation is not None else nn.SiLU()

        # S² activation for the full representation
        # We apply S² to everything, but only use higher-degree outputs
        self.s2_act: S2Activation | None = None
        if self.nhigher > 0:
            self.s2_act = S2Activation(
                repr,
                hidden_mult=hidden_mult,
                activation=s2_activation,
                precision=precision,
            )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply separable S² activation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Activated tensor of shape (..., mult, dim).
        """
        out = f.clone()

        # Scalar activation (direct)
        if self.nscalar > 0:
            out[..., self.scalar_locs] = self.scalar_act(f[..., self.scalar_locs])

        # S² activation for higher degrees
        if self.s2_act is not None:
            f_s2 = self.s2_act(f)
            out[..., self.higher_mask] = f_s2[..., self.higher_mask].to(out.dtype)  # type: ignore[index]

        return out

    def extra_repr(self) -> str:
        return f"nscalar={self.nscalar}, nhigher={self.nhigher}"
