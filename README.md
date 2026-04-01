# flash-eq

Fast, memory-efficient SO(3)-equivariant graph neural networks using Wigner-D diagonalization.

## Installation

Pre-built wheels (Python 3.10-3.13, CUDA 12.4+, A100/L40/H100 + PTX for newer GPUs) are available from [GitHub Releases](https://github.com/hmblair/flash-eq/releases). Download the wheel for your Python version and install:

```bash
pip install flash_eq-<version>-cp<pyver>-cp<pyver>-linux_x86_64.whl
```

Or install from source (requires CUDA toolkit):

```bash
pip install git+https://github.com/hmblair/flash-eq.git
```

Requires PyTorch >= 2.0 with CUDA support.

## Quick Start

```python
import torch
from flash_eq import EquivariantTransformer, Graph, Repr

# Define representations
in_repr = Repr(lvals=[0, 1], mult=32)       # Input: scalars + vectors
hidden_repr = Repr(lvals=[0, 1, 2], mult=64) # Hidden: up to L=2
out_repr = Repr(lvals=[0], mult=1)           # Output: scalar prediction

# Create model
model = EquivariantTransformer(
    in_repr, hidden_repr, out_repr,
    num_layers=4,
    num_heads=8,
).cuda()

# Input data
num_nodes = 100
coordinates = torch.randn(num_nodes, 3, device='cuda')
node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim(), device='cuda')

# Graph connectivity (e.g., k-nearest neighbors)
graph = Graph.random(num_nodes, num_edges=500, device='cuda')
# Or from explicit indices:
# graph = Graph(src=src_indices, dst=dst_indices, num_nodes=num_nodes)

# Forward pass
output = model(coordinates, node_features, graph)
# output.shape = (num_nodes, 1, 1)
```

The model automatically:
- Computes edge vectors and distances from coordinates
- Constructs Wigner-D basis matrices for SO(3) equivariance
- Applies distance-dependent radial weights
- Pools edge features back to nodes

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

### Graph

Lightweight container for graph connectivity in COO format.

```python
from flash_eq import Graph

# From explicit indices
graph = Graph(
    src=src_indices,    # (num_edges,) source node indices
    dst=dst_indices,    # (num_edges,) destination node indices
    num_nodes=100,
)

# Random graph for testing
graph = Graph.random(num_nodes=100, num_edges=500, device='cuda')

graph.num_edges  # Number of edges
graph.device     # Device of tensors
graph.to('cuda') # Move to device
```

### EquivariantTransformer

Full equivariant transformer model with input projection, transformer blocks, and output projection.

```python
from flash_eq import EquivariantTransformer, Repr, Graph

model = EquivariantTransformer(
    in_repr=Repr([0, 1], mult=32),        # Input representation
    hidden_repr=Repr([0, 1, 2], mult=64), # Hidden representation
    out_repr=Repr([0], mult=1),           # Output representation
    num_layers=6,                          # Number of transformer blocks
    num_heads=8,                           # Attention heads
    num_bases=16,                          # Radial basis functions (recommended)
    max_dist=10.0,                         # Maximum distance for radial weights
).cuda()

# Forward pass
output = model(coordinates, node_features, graph)
```

**Arguments:**
- `coordinates`: `(num_nodes, 3)` node positions
- `node_features`: `(num_nodes, mult_in, dim_in)` input features
- `graph`: Graph with edge connectivity

**Returns:**
- `output`: `(num_nodes, mult_out, dim_out)` transformed features

### EquivariantTransformerBlock

Single transformer block for building custom architectures.

```python
from flash_eq import EquivariantTransformerBlock, WignerDBasis, Repr, Graph

block = EquivariantTransformerBlock(
    in_repr=Repr([0, 1, 2], mult=64),
    out_repr=Repr([0, 1, 2], mult=64),
    num_heads=8,
).cuda()

# Compute basis matrices
basis = WignerDBasis([in_repr, out_repr]).cuda()
P, Q = basis(directions)

# Forward pass
output = block(P, Q, node_features, distances, graph)
```

### WignerDBasis

Computes Wigner-D basis matrices from edge directions. Takes a list of representations
and returns one matrix per representation, deduplicating by `lvals` for efficiency.

```python
from flash_eq import WignerDBasis

# For a simple case with matching in/out representations:
basis = WignerDBasis([in_repr, out_repr]).cuda()
P, Q = basis(directions)  # One matrix per repr

# For a transformer with different layer configurations:
repr_in = Repr(lvals=[0, 1], mult=32)
repr_hidden = Repr(lvals=[0, 1, 2, 3, 4], mult=64)
repr_out = Repr(lvals=[0], mult=1)

basis = WignerDBasis([repr_in, repr_hidden, repr_out]).cuda()
M_in, M_hidden, M_out = basis(directions)

# Layer 1 (in -> hidden): P=M_in, Q=M_hidden
# Hidden layers (hidden -> hidden): P=M_hidden, Q=M_hidden
# Final layer (hidden -> out): P=M_hidden, Q=M_out
```

The basis matrices are computed once and shared across all layers in a network.

### EquivariantEdgewiseLinear

SO(3)-equivariant linear layer with distance-dependent weights. This is an edge-to-edge
transformation; callers are responsible for gathering node features to edges if needed.

```python
from flash_eq import EquivariantEdgewiseLinear

layer = EquivariantEdgewiseLinear(
    in_repr,                # Input representation
    out_repr,               # Output representation
    num_bins=100,           # Distance bins for radial weights (default: 100)
    min_dist=0.0,           # Minimum distance (default: 0.0)
    max_dist=10.0,          # Maximum distance (default: 10.0)
)

# Gather node features to edges, then apply transformation
edge_features = node_features[src_indices]
output = layer(P, Q, edge_features, distances)
```

**Arguments:**
- `P`: Input basis matrix from `WignerDBasis`
- `Q`: Output basis matrix from `WignerDBasis`
- `edge_features`: `(num_edges, channels_in, dim_in)` edge features
- `distances`: `(num_edges,)` edge distances

**Returns:**
- `output`: `(num_edges, channels_out, dim_out)` transformed edge features

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

## Patching NVIDIA SE(3)-Transformers

Already have a trained NVIDIA SE(3)-Transformer? `flash_eq.patch` converts it
in-place for memory-efficient inference — no retraining required.

```python
from flash_eq import patch

model = load_trained_nvidia_model()  # any nn.Module containing ConvSE3 layers
model = patch(model, num_bins=500, max_dist=10.0)

# Use exactly as before — same inputs, same outputs
output = model(graph, node_feats)
```

### What it does

`patch()` walks the module tree, finds all `ConvSE3` layers, and replaces each
with a `PatchedConvSE3` that uses flash-eq's block-diagonal CUDA kernel. The
per-edge Clebsch-Gordan basis tensors (the dominant memory cost) are replaced
with compact Wigner-D matrices, reducing memory from O(E * L^4) to O(E * L^2).

The conversion is exact up to distance-binning interpolation error (~1e-4 with
500 bins in float64). The patched model produces numerically identical outputs
to the original.

If an `SE3Transformer` or `SE3TransformerPooled` module is found, its `forward`
method is also patched to compute Wigner-D matrices from `graph.edata['rel_pos']`
instead of calling the original `get_basis()` / `update_basis_with_fused()`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_bins` | 500 | Distance bins for radial weight tables. More bins = higher accuracy. |
| `min_dist` | 0.0 | Minimum distance for binning. |
| `max_dist` | 10.0 | Maximum distance. Edges beyond this use the last bin (extrapolation). |
| `dtype` | `torch.float64` | Precision for weight conversion. Tables are stored in the model's original dtype. |

### Requirements

- The model must contain `ConvSE3` layers from NVIDIA's SE(3)-Transformer
- Radial MLPs must be distance-only (`edge_dim=1`); additional invariant edge
  features are not supported
- All degrees within a fiber must have the same channel count
- DGL is still required for attention and pooling operations
- **Inference only** — patched layers do not support training. For training,
  use flash-eq's native layers directly.

### Supported configurations

| Fuse level | Supported |
|-----------|-----------|
| `NONE` | Yes |
| `PARTIAL` (by output degree) | Yes |
| `FULL` | Yes |
| `PARTIAL` (by input degree) | No |

Self-interaction and pooling are preserved from the original `ConvSE3`.

### Patch benchmark

Single `ConvSE3` layer (FULL fuse, C=32, 500 bins) on NVIDIA H100. Max interpolation error < 7e-4 across all configs.

| Config | SE(3)-T Memory | Patched Memory | Savings | SE(3)-T Time | Patched Time | Speedup |
|--------|---------------|----------------|---------|-------------|-------------|---------|
| L=2, 4K edges | 485 MB | 169 MB | 2.9x | 0.71 ms | 1.03 ms | 0.69x |
| L=2, 16K edges | 1.8 GB | 459 MB | 4.0x | 2.20 ms | 2.41 ms | 0.91x |
| L=2, 65K edges | 7.1 GB | 1.6 GB | 4.5x | 8.87 ms | 7.70 ms | 1.15x |
| L=3, 4K edges | 1.4 GB | 476 MB | 2.9x | 1.48 ms | 1.42 ms | 1.04x |
| L=3, 16K edges | 5.3 GB | 1.5 GB | 3.6x | 5.42 ms | 3.86 ms | 1.40x |
| L=3, 65K edges | 21.0 GB | 5.6 GB | 3.8x | 21.80 ms | 13.39 ms | 1.63x |

Memory savings grow with edge count and angular momentum. Runtime improves at larger scales where the custom kernel's O(L^2) complexity outweighs cuBLAS overhead.

