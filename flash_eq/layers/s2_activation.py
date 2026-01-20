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
from ..utils import init_linear_weights


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

    Y_T: torch.Tensor
    Y_inv_T: torch.Tensor

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
        indices: list[int] = []
        for l in lvals:
            start = l * l
            end = (l + 1) * (l + 1)
            indices.extend(range(start, end))

        Y_subset = grid.Y[:, indices]  # (n_points, dim)
        weights = grid.weights

        # Recompute inverse transform for the subset
        # Use lstsq to handle duplicate l values (which create linearly dependent columns)
        W = torch.diag(weights)
        YtW = Y_subset.T @ W
        YtWY = YtW @ Y_subset
        Y_inv_subset = torch.linalg.lstsq(YtWY, YtW).solution  # (dim, n_points)

        # Store transposed matrices to avoid transpose in forward pass
        self.register_buffer('Y_T', Y_subset.T.float().contiguous())        # (dim, n_points)
        self.register_buffer('Y_inv_T', Y_inv_subset.T.float().contiguous())  # (n_points, dim)

        # Channel-wise MLP using Linear (better fusion under torch.compile)
        hidden_dim = self.mult * hidden_mult
        act_fn = activation if activation is not None else nn.SiLU()

        self.mlp = nn.Sequential(
            nn.Linear(self.mult, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, self.mult),
        )

        self.apply(init_linear_weights)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply S² activation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Activated tensor of shape (..., mult, dim).
        """
        # To grid: (..., mult, n_points)
        f_grid = f @ self.Y_T

        # Apply MLP across channels at each grid point
        # Linear operates on last dim, so transpose: (..., mult, n_points) -> (..., n_points, mult)
        f_grid = self.mlp(f_grid.transpose(-1, -2)).transpose(-1, -2)

        # Back to SH: (..., mult, dim)
        return f_grid @ self.Y_inv_T

    def extra_repr(self) -> str:
        return f"mult={self.mult}, dim={self.dim}, n_points={self.n_points}, precision={self.precision}"


class SeparableS2Activation(nn.Module):
    """Separable S² activation following EquiformerV2.

    Applies different nonlinearities to scalar and higher-degree features:
        - Scalars (l=0): Standard SiLU activation
        - Higher degrees (l>0): S² activation on spherical grid

    Optionally gates higher-degree outputs using scalar features.

    This separable design is more efficient than applying S² to all degrees,
    and provides better gradient flow through the scalar path.

    Args:
        repr: The representation of input tensors. Must include l=0.
        hidden_mult: Hidden layer size multiplier for S² MLP (default: 2).
        use_gate: Gate higher degrees by scalar features (default: True).
        precision: Lebedev precision for S² grid (default: 47, 770 points).

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=32)
        >>> act = SeparableS2Activation(repr)
        >>> x = torch.randn(100, 32, 9)  # (batch, mult, dim)
        >>> y = act(x)  # same shape

    Reference:
        EquiformerV2 (Liao et al., ICLR 2024), Section 3.3
    """

    _scalar_indices: torch.Tensor
    _higher_indices: torch.Tensor

    def __init__(
        self,
        repr: Repr,
        hidden_mult: int = 2,
        use_gate: bool = True,
        precision: int = 47,
    ) -> None:
        super().__init__()

        self.repr = repr
        self.mult = repr.mult
        self.use_gate = use_gate

        # Find scalar (l=0) and higher-degree indices
        lvals = repr.lvals.tolist()
        scalar_indices: list[int] = []
        higher_indices: list[int] = []
        higher_lvals: list[int] = []

        idx = 0
        for l in lvals:
            dim_l = 2 * l + 1
            if l == 0:
                scalar_indices.extend(range(idx, idx + dim_l))
            else:
                higher_indices.extend(range(idx, idx + dim_l))
                higher_lvals.append(l)
            idx += dim_l

        self.n_scalars = len(scalar_indices)
        self.n_higher = len(higher_indices)

        if self.n_scalars == 0:
            raise ValueError(
                f"SeparableS2Activation requires l=0 in repr. "
                f"Got lvals={lvals}"
            )

        self.register_buffer(
            '_scalar_indices',
            torch.tensor(scalar_indices, dtype=torch.long)
        )
        self.register_buffer(
            '_higher_indices',
            torch.tensor(higher_indices, dtype=torch.long)
        )

        # Scalar path: simple SiLU (no parameters needed beyond gating)
        self.scalar_act = nn.SiLU()

        # Higher-degree path: S² activation (only if we have higher degrees)
        self.s2_act: S2Activation | None = None
        if self.n_higher > 0:
            higher_repr = Repr(lvals=higher_lvals, mult=repr.mult)
            self.s2_act = S2Activation(
                higher_repr,
                hidden_mult=hidden_mult,
                precision=precision,
            )

        # Gating: use scalar features to gate higher degrees
        self.gate_linear: nn.Linear | None = None
        if use_gate and self.n_higher > 0:
            # Project scalar features to gate values for each higher-degree irrep
            n_higher_irreps = len(higher_lvals)
            self.gate_linear = nn.Linear(repr.mult, repr.mult * n_higher_irreps)
            self.gate_act = nn.Sigmoid()

            # Create indices to broadcast gate values to higher-degree dims
            gate_indices: list[int] = []
            for i, l in enumerate(higher_lvals):
                gate_indices.extend([i] * (2 * l + 1))
            self.register_buffer(
                '_gate_indices',
                torch.tensor(gate_indices, dtype=torch.long)
            )
            self.n_higher_irreps = n_higher_irreps

        self.apply(init_linear_weights)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply separable S² activation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Activated tensor of shape (..., mult, dim).
        """
        # Extract scalar and higher-degree components
        f_scalar = f[..., self._scalar_indices]  # (..., mult, n_scalars)
        f_scalar = self.scalar_act(f_scalar)

        if self.n_higher == 0:
            # Only scalars, return directly
            return f_scalar

        f_higher = f[..., self._higher_indices]  # (..., mult, n_higher)

        # Apply S² activation to higher degrees
        assert self.s2_act is not None  # guaranteed when n_higher > 0
        f_higher = self.s2_act(f_higher)

        # Optional gating by scalar features
        if self.gate_linear is not None:
            # Use mean of scalar features for gating
            scalar_mean = f_scalar.mean(dim=-1)  # (..., mult)
            gates = self.gate_act(self.gate_linear(scalar_mean))  # (..., mult * n_irreps)
            gates = gates.view(*gates.shape[:-1], self.mult, self.n_higher_irreps)
            # Broadcast gates to each m component: (..., mult, n_higher)
            gates = gates[..., self._gate_indices]
            f_higher = f_higher * gates

        # Reconstruct full tensor
        out = f.new_empty(f.shape)
        out[..., self._scalar_indices] = f_scalar
        out[..., self._higher_indices] = f_higher

        return out

    def extra_repr(self) -> str:
        return (
            f"mult={self.mult}, n_scalars={self.n_scalars}, "
            f"n_higher={self.n_higher}, use_gate={self.use_gate}"
        )
