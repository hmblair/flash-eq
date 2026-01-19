"""Spherical harmonics and quadrature utilities.

This subpackage provides:
- Lebedev quadrature grids for integration on S²
- Real spherical harmonic evaluation (via sphericart)
- S2Grid for precomputed transform matrices

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

from .harmonics import S2Grid, lebedev_grid, real_spherical_harmonics
from .lebedev import LEBEDEV_RULES, get_available_precisions, get_lebedev_rule

__all__ = [
    "S2Grid",
    "lebedev_grid",
    "real_spherical_harmonics",
    "LEBEDEV_RULES",
    "get_available_precisions",
    "get_lebedev_rule",
]
