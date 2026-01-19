"""Equivariant neural network layers for spherical tensors.

This package provides building blocks for SO(3)-equivariant networks:

Linear layers:
    - EquivariantEdgewiseLinear: Edgewise linear with distance-dependent weights
    - EquivariantLinear: Linear layer preserving spherical tensor structure

Attention:
    - EquivariantEdgeAttention: Attention mechanism for equivariant edge features
    - EquivariantAttention: Full attention layer (linear + attention + pooling)

Normalization:
    - RepNorm: Compute rotation-invariant norms per irrep
    - EquivariantLayerNorm: Equivariant layer normalization
    - SeparableEquivariantLayerNorm: Separable normalization (EquiformerV2)

Pooling:
    - GraphPooling: Aggregate edge features to nodes (sum/mean/max)

Gating:
    - EquivariantGating: Norm-based gating nonlinearity

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from .linear import EquivariantEdgewiseLinear, EquivariantLinear
from .attention import EquivariantEdgeAttention, EquivariantAttention
from .norm import RepNorm, EquivariantLayerNorm, SeparableEquivariantLayerNorm
from .pooling import GraphPooling
from .gating import EquivariantGating

__all__ = [
    # Linear
    "EquivariantEdgewiseLinear",
    "EquivariantLinear",
    # Attention
    "EquivariantEdgeAttention",
    "EquivariantAttention",
    # Normalization
    "RepNorm",
    "EquivariantLayerNorm",
    "SeparableEquivariantLayerNorm",
    # Pooling
    "GraphPooling",
    # Gating
    "EquivariantGating",
]
