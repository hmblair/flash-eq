"""Spherical harmonics and quadrature utilities.

This module provides:
- Lebedev quadrature grids for integration on S²
- Real spherical harmonic evaluation

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

import math
import torch
from functools import lru_cache

from .lebedev_tables import LEBEDEV_RULES, get_available_precisions


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

@lru_cache(maxsize=64)
def _factorial(n: int) -> float:
    """Cached factorial computation."""
    if n <= 1:
        return 1.0
    return n * _factorial(n - 1)


def _associated_legendre_batch(l_max: int, cos_theta: torch.Tensor) -> torch.Tensor:
    """Compute all associated Legendre polynomials up to l_max.

    Returns normalized associated Legendre functions using stable recurrence.

    Args:
        l_max: Maximum degree
        cos_theta: (n_points,) tensor of cos(θ) values

    Returns:
        P: (n_points, n_sh) tensor where n_sh = (l_max+1)²
           Indexed as P[:, l² + l + m] for each (l, m) with -l <= m <= l
    """
    n_points = cos_theta.shape[0]
    n_sh = (l_max + 1) ** 2

    x = cos_theta
    sin_theta = torch.sqrt(torch.clamp(1 - x * x, min=0))

    # Storage for P_l^m (only m >= 0, we'll handle m < 0 via symmetry)
    # Index: P_store[l, m] for m >= 0
    P_store = torch.zeros(l_max + 1, l_max + 1, n_points, dtype=cos_theta.dtype, device=cos_theta.device)

    # Initial values
    P_store[0, 0] = 1.0

    # Recurrence for P_l^l: P_l^l = -(2l-1) * sin(θ) * P_{l-1}^{l-1}
    for l in range(1, l_max + 1):
        P_store[l, l] = -(2 * l - 1) * sin_theta * P_store[l - 1, l - 1]

    # Recurrence for P_l^{l-1}: P_l^{l-1} = (2l-1) * cos(θ) * P_{l-1}^{l-1}
    for l in range(1, l_max + 1):
        P_store[l, l - 1] = (2 * l - 1) * x * P_store[l - 1, l - 1]

    # Recurrence for P_l^m (m < l-1):
    # (l-m) P_l^m = (2l-1) cos(θ) P_{l-1}^m - (l+m-1) P_{l-2}^m
    for l in range(2, l_max + 1):
        for m in range(0, l - 1):
            P_store[l, m] = ((2 * l - 1) * x * P_store[l - 1, m] - (l + m - 1) * P_store[l - 2, m]) / (l - m)

    # Now build the full output array with proper normalization
    P = torch.zeros(n_points, n_sh, dtype=cos_theta.dtype, device=cos_theta.device)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)

            # Normalization: sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * _factorial(l - am) / _factorial(l + am))

            P[:, idx] = norm * P_store[l, am]
            idx += 1

    return P


def real_spherical_harmonics(l_max: int, points: torch.Tensor) -> torch.Tensor:
    """Evaluate real spherical harmonics at given points.

    Real spherical harmonics are defined as:
        Y_l^0     = N_l^0 P_l^0(cos θ)
        Y_l^m     = √2 N_l^m P_l^m(cos θ) cos(mφ)   for m > 0
        Y_l^{-m}  = √2 N_l^m P_l^m(cos θ) sin(mφ)   for m > 0

    where N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)

    Ordering: for each l, m goes from -l to +l.

    Args:
        l_max: Maximum degree
        points: (n_points, 3) tensor of unit vectors

    Returns:
        Y: (n_points, (l_max+1)²) tensor of SH values
    """
    device = points.device
    dtype = points.dtype
    n_points = points.shape[0]
    n_sh = (l_max + 1) ** 2

    # Convert to spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cos_theta = z
    phi = torch.atan2(y, x)

    # Get normalized associated Legendre polynomials
    P = _associated_legendre_batch(l_max, cos_theta)

    # Build real spherical harmonics
    Y = torch.zeros(n_points, n_sh, device=device, dtype=dtype)

    sqrt2 = math.sqrt(2.0)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m == 0:
                Y[:, idx] = P[:, idx]
            elif m > 0:
                Y[:, idx] = sqrt2 * P[:, idx] * torch.cos(m * phi)
            else:  # m < 0
                Y[:, idx] = sqrt2 * P[:, idx] * torch.sin(abs(m) * phi)
            idx += 1

    return Y


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
