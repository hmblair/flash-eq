"""
flash-eq: Fast, memory-efficient SO(3)-equivariant linear layers.

Uses Wigner-D diagonalization to reduce memory from O(L^4) to O(L^2).
"""

from .linear import EquivariantLinear

__all__ = ["EquivariantLinear"]
__version__ = "0.1.0"
