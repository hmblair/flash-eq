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

Radial weights:
    - RadialMLP: MLP mapping distance to weight tensor
    - BinnedModule: Wrapper for precomputing module outputs at bin edges
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr, WignerD
from .basis import WignerDBasis
from .radial import RadialBasisFunctions, RadialMLP, BinnedModule
from .layers import (
    EquivariantEdgewiseLinear,
    EquivariantAttention,
    EquivariantEdgeAttention,
    RepNorm,
    EquivariantLinear,
    EquivariantGating,
    EquivariantLayerNorm,
    SeparableEquivariantLayerNorm,
    GraphPooling,
)
from .cuda import CUDANotAvailableError

__all__ = [
    # Representations
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    "WignerD",
    # Layers
    "EquivariantEdgewiseLinear",
    "EquivariantAttention",
    "RepNorm",
    "EquivariantLinear",
    "EquivariantGating",
    "EquivariantLayerNorm",
    "SeparableEquivariantLayerNorm",
    "GraphPooling",
    "EquivariantEdgeAttention",
    # Basis
    "WignerDBasis",
    # Radial weights
    "RadialBasisFunctions",
    "RadialMLP",
    "BinnedModule",
    # Exceptions
    "CUDANotAvailableError",
]
__version__ = "0.1.0"
