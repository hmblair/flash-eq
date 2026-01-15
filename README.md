# flash-eq

Fast, memory-efficient SO(3)-equivariant linear layers using Wigner-D diagonalization.

## Installation

```bash
pip install flash-eq
```

## Usage

```python
import torch
from ciffy.nn.geometric.representations import Repr
from flash_eq import EquivariantLinear

# Define representations
repr_in = Repr(lvals=[0, 1, 2])   # scalars + vectors + rank-2
repr_out = Repr(lvals=[0, 1, 2])

# Create layer
layer = EquivariantLinear(repr_in, repr_out)

# Inputs
batch, channels_in, channels_out = 100, 8, 16
features = torch.randn(batch, channels_in, repr_in.dim())
directions = torch.randn(batch, 3)  # will be normalized internally
weights = torch.randn(batch, channels_out, channels_in, layer.weight_dim)

# Forward pass
output = layer(features, directions, weights)
# output.shape = (100, 16, 9)
```

## Memory Savings

The layer stores O(L²) weights per degree pair instead of O(L⁴) for dense matrices:

| lmax | Dense weights | flash-eq weights | Savings |
|------|---------------|------------------|---------|
| 2    | 81            | 19               | 4.3×    |
| 3    | 256           | 44               | 5.8×    |
| 4    | 625           | 85               | 7.4×    |

## Theory

See [`docs/theory.tex`](docs/theory.tex) for the mathematical details. The key insight is that SO(3)-equivariant weight matrices W(x) are diagonalized by Wigner-D matrices:

```
W(x) = D(g_x) Λ D(g_x)^T
```

where Λ is block-diagonal with 1×1 blocks for m=0 and 2×2 blocks for m>0.
