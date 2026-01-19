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
    - BinnedRadialBasis: Parameter-efficient radial basis function weights
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr, WignerD
from .basis import WignerDBasis
from .radial import RadialBasisFunctions, RadialMLP, BinnedModule, BinnedRadialBasis
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
    S2Activation,
    EquivariantTransformerBlock,
    EquivariantTransformer,
)
from .spherical import S2Grid, real_spherical_harmonics, lebedev_grid
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
    "S2Activation",
    # Transformer
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
    # Basis
    "WignerDBasis",
    # Radial weights
    "RadialBasisFunctions",
    "RadialMLP",
    "BinnedModule",
    "BinnedRadialBasis",
    # Spherical utilities
    "S2Grid",
    "real_spherical_harmonics",
    "lebedev_grid",
    # Exceptions
    "CUDANotAvailableError",
]
__version__ = "0.1.0"
