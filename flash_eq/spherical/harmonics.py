"""Spherical harmonics and quadrature utilities.

This module provides:
- Lebedev quadrature grids for integration on S²
- Real spherical harmonic evaluation (via sphericart)

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import math
import torch
import sphericart.torch as sph

from .lebedev import LEBEDEV_RULES, get_available_precisions


# =============================================================================
# Lebedev Quadrature Grid
# =============================================================================

def lebedev_grid(precision: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get Lebedev quadrature points and weights.

    Lebedev grids are optimal for spherical integration, using ~2/3 the points
    of tensor-product grids for equivalent accuracy.

    Args:
        precision: Lebedev precision (17, 23, 29, 35, 41, 47, ...).
                  Higher = more points, higher accuracy.

    Returns:
        points: (n_points, 3) tensor of unit vectors
        weights: (n_points,) tensor of quadrature weights (sum to 4π)

    Example:
        >>> points, weights = lebedev_grid(precision=29)  # 302 points
    """
    available = get_available_precisions()

    if precision not in LEBEDEV_RULES:
        raise ValueError(
            f"Unknown Lebedev precision {precision}. Available: {available}"
        )

    n_points, max_l, point_list = LEBEDEV_RULES[precision]

    points = torch.tensor(
        [[p[0], p[1], p[2]] for p in point_list],
        dtype=torch.float64
    )
    weights = torch.tensor(
        [p[3] for p in point_list],
        dtype=torch.float64
    )

    # Scale weights to integrate to 4π (sphere surface area)
    weights = weights * 4 * math.pi

    return points, weights


# =============================================================================
# Real Spherical Harmonics
# =============================================================================

def real_spherical_harmonics(l_max: int, points: torch.Tensor) -> torch.Tensor:
    """Evaluate real spherical harmonics at given points.

    Uses sphericart for numerically stable evaluation at high l.

    Ordering: for each l, m goes from -l to +l.

    Args:
        l_max: Maximum degree
        points: (n_points, 3) tensor of unit vectors

    Returns:
        Y: (n_points, (l_max+1)²) tensor of SH values
    """
    calculator = sph.SphericalHarmonics(l_max=l_max)
    return calculator.compute(points)


# =============================================================================
# S² Grid: Precomputed Transform Matrices
# =============================================================================

class S2Grid:
    """Precomputed spherical harmonic transform matrices for quadrature.

    Stores the forward (SH → grid) and inverse (grid → SH) transform matrices
    for efficient S² activation using Lebedev quadrature.

    Args:
        l_max: Maximum spherical harmonic degree
        precision: Lebedev precision (17, 23, ..., 131).

    Example:
        >>> grid = S2Grid(l_max=6, precision=47)
        >>> # Forward: coefficients to grid values
        >>> f_grid = f_coeffs @ grid.Y.T
        >>> # Inverse: grid values to coefficients
        >>> f_coeffs = f_grid @ grid.Y_inv.T
    """

    def __init__(
        self,
        l_max: int,
        precision: int,
    ) -> None:
        self.l_max = l_max
        self.n_sh = (l_max + 1) ** 2
        self.precision = precision

        # Get Lebedev quadrature points and weights
        points, weights = lebedev_grid(precision)

        self.n_points = points.shape[0]

        # Compute SH values at quadrature points
        Y = real_spherical_harmonics(l_max, points)  # (n_points, n_sh)

        # Compute inverse transform via weighted least squares
        # Y_inv = (Y^T W Y)^{-1} Y^T W
        W = torch.diag(weights)
        YtW = Y.T @ W
        YtWY = YtW @ Y
        Y_inv = torch.linalg.solve(YtWY, YtW)  # (n_sh, n_points)

        # Store as float32 for efficiency
        self.Y = Y.float()
        self.Y_inv = Y_inv.float()
        self.points = points.float()
        self.weights = weights.float()

    def __repr__(self) -> str:
        return f"S2Grid(l_max={self.l_max}, n_points={self.n_points}, precision={self.precision})"
