"""
flash-eq: Fast, memory-efficient SO(3)-equivariant linear layers.

Uses Wigner-D diagonalization to reduce memory from O(L^4) to O(L^2).

Main API:
    - EquivariantEdgewiseLinear: Production layer with binned weights
    - WignerDBasis: Computes P, Q basis matrices from edge directions

Representations:
    - Repr: Spherical representation (lvals + multiplicity)
    - ProductRepr: Product of two representations
    - Irrep, ProductIrrep: Single irreducible representations

Radial weight helpers:
    - BinnedRadialWeight: Binned radial weights with interpolation
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr, WignerD
from .layer import EquivariantEdgewiseLinear
from .basis import WignerDBasis
from .radial import BinnedRadialWeight

__all__ = [
    # Representations
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    "WignerD",
    # Layers
    "EquivariantEdgewiseLinear",
    # Basis
    "WignerDBasis",
    # Radial weights
    "BinnedRadialWeight",
]
__version__ = "0.1.0"
