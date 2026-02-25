"""
flash-eq: Fast, memory-efficient SO(3)-equivariant linear layers.

Uses Wigner-D diagonalization to reduce memory from O(L^4) to O(L^2).

Main API:
    - EquivariantEdgewiseLinear: Production layer with distance-dependent weights
    - EquivariantTransformer: Full transformer stack
    - WignerDBasis: Computes P, Q basis matrices from edge directions

Representations:
    - Repr: Spherical representation (lvals + multiplicity)
    - ProductRepr: Product of two representations
    - Irrep, ProductIrrep: Single irreducible representations
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr, WignerD, WignerDBasis, random_rotation
from .graph import Graph
from .layers.radial import RadialBasisFunctions, RadialMLP, SeparableRadialNet
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
    SeparableS2Activation,
    EquivariantTransformerBlock,
    EquivariantTransformer,
    SequencePositionEncoding,
)
from .spherical import S2Grid, real_spherical_harmonics, lebedev_grid
from .cuda import CUDANotAvailableError
from .patch import patch

__all__ = [
    # Representations
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    "WignerD",
    "random_rotation",
    # Graph
    "Graph",
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
    "SeparableS2Activation",
    # Transformer
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
    # Basis
    "WignerDBasis",
    # Radial weights
    "RadialBasisFunctions",
    "RadialMLP",
    "SeparableRadialNet",
    # Encoding
    "SequencePositionEncoding",
    # Spherical utilities
    "S2Grid",
    "real_spherical_harmonics",
    "lebedev_grid",
    # Exceptions
    "CUDANotAvailableError",
    # Patching
    "patch",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"
