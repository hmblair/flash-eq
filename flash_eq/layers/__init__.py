"""Equivariant neural network layers for spherical tensors.

This package provides building blocks for SO(3)-equivariant networks:

Linear layers:
    - EquivariantEdgewiseLinear: Edgewise linear with distance-dependent weights
    - EquivariantLinear: Linear layer preserving spherical tensor structure

Attention:
    - EquivariantEdgeAttention: Attention mechanism for equivariant edge features
    - EquivariantAttention: Full attention layer (linear + attention + pooling)
    - GeometricEquivariantAttention: Attention with distance-aware Q/K projections

Normalization:
    - RepNorm: Compute rotation-invariant norms per irrep
    - EquivariantLayerNorm: Equivariant layer normalization
    - SeparableEquivariantLayerNorm: Separable normalization (EquiformerV2)

Pooling:
    - GraphPooling: Aggregate edge features to nodes (sum/mean/max)

Gating:
    - EquivariantGating: Norm-based gating nonlinearity

Transformer:
    - EquivariantTransformerBlock: Single transformer block
    - EquivariantTransformer: Full transformer stack

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from .linear import EquivariantEdgewiseLinear, EquivariantLinear
from .attention import EquivariantEdgeAttention, EquivariantAttention, GeometricEquivariantAttention
from .norm import RepNorm, EquivariantLayerNorm, SeparableEquivariantLayerNorm
from .pooling import GraphPooling
from .gating import EquivariantGating
from .s2_activation import S2Activation, SeparableS2Activation
from .transformer import EquivariantTransformerBlock, EquivariantTransformer
from .radial import RadialBasisFunctions, RadialMLP, SeparableRadialNet

__all__ = [
    # Linear
    "EquivariantEdgewiseLinear",
    "EquivariantLinear",
    # Attention
    "EquivariantEdgeAttention",
    "EquivariantAttention",
    "GeometricEquivariantAttention",
    # Normalization
    "RepNorm",
    "EquivariantLayerNorm",
    "SeparableEquivariantLayerNorm",
    # Pooling
    "GraphPooling",
    # Gating
    "EquivariantGating",
    # S² Activation
    "S2Activation",
    "SeparableS2Activation",
    # Transformer
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
    # Radial
    "RadialBasisFunctions",
    "RadialMLP",
    "SeparableRadialNet",
]
