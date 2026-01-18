# flash-eq

Fast, memory-efficient SO(3)-equivariant linear layers using Wigner-D diagonalization.

## Installation

```bash
pip install flash-eq
```

Requires PyTorch >= 2.0 with CUDA support. The CUDA kernel is JIT-compiled on first use.

## Quick Start

```python
import torch
from flash_eq import EquivariantEdgewiseLinear, WignerDBasis, Repr

# Define representations (l=0,1,2 with 32 channels)
in_repr = Repr(lvals=[0, 1, 2], mult=32)
out_repr = Repr(lvals=[0, 1, 2], mult=32)

# Create layer and basis
layer = EquivariantEdgewiseLinear(in_repr, out_repr).cuda()
basis = WignerDBasis(in_repr, out_repr).cuda()

# Input data
num_nodes, num_edges = 1000, 5000
node_features = torch.randn(num_nodes, 32, in_repr.dim, device='cuda')
src_indices = torch.randint(0, num_nodes, (num_edges,), device='cuda')
directions = torch.randn(num_edges, 3, device='cuda')
directions = directions / directions.norm(dim=-1, keepdim=True)
distances = torch.rand(num_edges, device='cuda') * 10.0

# Compute basis matrices (once per forward pass, shared across layers)
P, Q = basis(directions)

# Apply layer
output = layer(P, Q, node_features, distances, src_indices)
# output.shape = (num_edges, 32, out_repr.dim)
```

## API Reference

### Repr

Defines a spherical representation with angular momentum values and multiplicity.

```python
from flash_eq import Repr

# Scalars (l=0), vectors (l=1), and rank-2 tensors (l=2) with 32 channels each
repr = Repr(lvals=[0, 1, 2], mult=32)

repr.dim      # Total dimension: 1 + 3 + 5 = 9
repr.lvals    # [0, 1, 2]
repr.mult     # 32
```

### WignerDBasis

Computes the Wigner-D basis matrices P and Q from edge directions.

```python
from flash_eq import WignerDBasis

basis = WignerDBasis(in_repr, out_repr).cuda()

# directions: (num_edges, 3) unit vectors
P, Q = basis(directions)
# P: (num_edges, dim_in, dim_in)
# Q: (num_edges, dim_out, dim_out)
```

The basis matrices are computed once and shared across all layers in a network.

### EquivariantEdgewiseLinear

SO(3)-equivariant linear layer with distance-dependent weights.

```python
from flash_eq import EquivariantEdgewiseLinear

layer = EquivariantEdgewiseLinear(
    in_repr,                # Input representation
    out_repr,               # Output representation
    num_bins=100,           # Distance bins for radial weights (default: 100)
    min_dist=0.0,           # Minimum distance (default: 0.0)
    max_dist=10.0,          # Maximum distance (default: 10.0)
    radial_hidden=64,       # Hidden dim for radial MLP (default: 64)
    radial_layers=2,        # Hidden layers in radial MLP (default: 2)
)

output = layer(P, Q, node_features, distances, src_indices)
```

**Arguments:**
- `P`: Input basis matrix from `WignerDBasis`
- `Q`: Output basis matrix from `WignerDBasis`
- `node_features`: `(num_nodes, channels_in, dim_in)` node features
- `distances`: `(num_edges,)` edge distances
- `src_indices`: `(num_edges,)` source node index for each edge

**Returns:**
- `output`: `(num_edges, channels_out, dim_out)` edge features

### Equivariant Layers

Basic building blocks for SO(3)-equivariant networks that operate on spherical tensors.

#### RepNorm

Computes rotation-invariant norms for each irrep component.

```python
from flash_eq import RepNorm, Repr

repr = Repr(lvals=[0, 1, 2], mult=8)
norm = RepNorm(repr)

x = torch.randn(batch, 8, 9)  # (batch, mult, dim)
norms = norm(x)               # (batch, 8, 3) - one norm per irrep
```

#### EquivariantLinear

Linear layer that changes multiplicity while preserving angular momentum structure.

```python
from flash_eq import EquivariantLinear, Repr

in_repr = Repr(lvals=[0, 1, 2], mult=8)
out_repr = Repr(lvals=[0, 1, 2], mult=16)  # Must have same lvals

layer = EquivariantLinear(in_repr, out_repr, bias=True)

x = torch.randn(batch, 8, 9)
y = layer(x)  # (batch, 16, 9)
```

#### EquivariantGating

Norm-based gating nonlinearity that preserves equivariance.

```python
from flash_eq import EquivariantGating, Repr

repr = Repr(lvals=[0, 1, 2], mult=8)
gate = EquivariantGating(repr)

x = torch.randn(batch, 8, 9)
y = gate(x)  # (batch, 8, 9)
```

#### EquivariantLayerNorm

Layer normalization that preserves SO(3) equivariance.

```python
from flash_eq import EquivariantLayerNorm, Repr

repr = Repr(lvals=[0, 1, 2], mult=8)
ln = EquivariantLayerNorm(repr)

x = torch.randn(batch, 8, 9)
y = ln(x)  # (batch, 8, 9)
```

## Benchmark Results

Comparison with SE3-Transformer on NVIDIA H100 (80GB). Forward + backward pass with 32 input/output channels.

### FP32

| Config | SE3-Transformer | Flash-eq | Memory Savings | Speedup |
|--------|-----------------|----------|----------------|---------|
| L=1, E=32k | 5.8ms / 1.7GB | 5.3ms / 0.2GB | 8.1x | 1.1x |
| L=2, E=32k | 14.9ms / 4.9GB | 11.5ms / 0.4GB | 12.3x | 1.3x |
| L=4, E=5k | 10.6ms / 4.7GB | 7.1ms / 0.3GB | 13.9x | 1.5x |
| L=6, E=5k | 35.8ms / 20.7GB | 14.7ms / 0.7GB | 28.1x | 2.4x |
| L=4, E=20k | 40.1ms / 18.4GB | 21.5ms / 0.8GB | 24.0x | 1.9x |
| L=6, E=20k | OOM | 50.6ms / 1.7GB | - | - |
| L=4, E=50k | 135.0ms / 45.8GB | 50.1ms / 1.7GB | 27.7x | 2.7x |
| L=6, E=50k | OOM | 123.5ms / 3.7GB | - | - |
| L=4, E=128k | OOM | 125.6ms / 3.9GB | - | - |
| L=6, E=128k | OOM | 309.7ms / 8.9GB | - | - |

### FP16 (AMP)

| Config | SE3-Transformer | Flash-eq | Memory Savings | Speedup |
|--------|-----------------|----------|----------------|---------|
| L=1, E=32k | 2.5ms / 1.0GB | 5.4ms / 0.2GB | 5.1x | 0.5x |
| L=2, E=32k | 7.6ms / 2.7GB | 10.5ms / 0.4GB | 7.2x | 0.7x |
| L=4, E=5k | 6.1ms / 3.1GB | 6.6ms / 0.3GB | 9.9x | 0.9x |
| L=6, E=5k | 22.8ms / 16.8GB | 13.6ms / 0.7GB | 24.6x | 1.7x |
| L=4, E=20k | 22.5ms / 12.0GB | 19.7ms / 0.7GB | 16.0x | 1.1x |
| L=6, E=20k | 128.7ms / 66.8GB | 46.2ms / 1.7GB | 38.6x | 2.8x |
| L=4, E=50k | 53.9ms / 29.7GB | 45.3ms / 1.6GB | 18.4x | 1.2x |
| L=6, E=50k | OOM | 110.9ms / 3.9GB | - | - |
| L=4, E=128k | 189.7ms / 76.0GB | 112.9ms / 3.8GB | 19.7x | 1.7x |
| L=6, E=128k | OOM | 279.8ms / 9.3GB | - | - |

Note: Flash-eq is optimized for high angular momentum (L≥4) and large edge counts. At low L with FP16, SE3-Transformer's Tensor Core utilization gives it an advantage. Improving FP16 performance at low L is an active area of development.

## Theory

See [`docs/theory.tex`](docs/theory.tex) for the mathematical details.
