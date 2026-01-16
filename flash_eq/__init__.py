"""
flash-eq: Fast, memory-efficient SO(3)-equivariant linear layers.

Uses Wigner-D diagonalization to reduce memory from O(L^4) to O(L^2).
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr
from .linear import EquivariantLinear
from .edgewise import EquivariantEdgewiseLinear
from .basis import WignerDBasis

__all__ = [
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    "EquivariantLinear",
    "EquivariantEdgewiseLinear",
    "WignerDBasis",
]
__version__ = "0.1.0"
