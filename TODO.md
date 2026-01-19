# TODO

## Known Performance Limitations

### Float16 performance at low L values

The custom CUDA kernel underperforms cuBLAS-based approaches (e.g., SE3-Transformer) at low L values when using float16/AMP:

| L | H100 Speedup (AMP) | A100 Speedup (AMP) |
|---|--------------------|--------------------|
| 1 | 0.46x | 0.49x |
| 2 | 0.73x | 0.62x |
| 4 | 0.93x | 0.64x |
| 6 | 1.67x | 1.88x |

**Root cause:** The kernel performs all computation in float32, converting half inputs at load time and back at store time. This means we don't benefit from:
- Half-precision compute units (2x throughput vs FP32)
- Tensor Cores (16x+ throughput for matrix ops)

cuBLAS GEMMs with AMP use Tensor Cores efficiently, giving SE3-Transformer a significant advantage at low L where the O(L^2) vs O(L^4) scaling hasn't yet favored our approach.

**Attempted optimizations that didn't help:**
- Storing shared memory in native half instead of float: No improvement because the conversion still happens in the inner loop, and the bottleneck is global memory access for weight interpolation.
- half2 vectorized loads: Alignment issues due to odd dimensions (n_in can be odd for certain L values, and offsets may not be 4-byte aligned).

**Potential solutions:**
1. Batch edges by radial bin and use WMMA/Tensor Cores for dense GEMM
2. Accept the tradeoff: Flash-eq wins on memory (5-8x savings) even when slower
3. Hybrid approach: use cuBLAS for low L, custom kernel for high L

**Note:** Flash-eq still provides 5-40x memory savings across all configurations, enabling larger batch sizes and models that would otherwise OOM.

## FP16 Numerical Stability in Wigner-D Computation

**RESOLVED:** FP16 inputs are now promoted to FP32 for `matrix_exp`, then cast back. The `_apply` override keeps generators in FP32 when `.half()` is called, while FP64 inputs use FP64 generators for full precision. See commit `2ce4693`.

## Integrate Separable S² Activation into Transformer Block

The current `EquivariantTransformerBlock` uses `EquivariantGating` for nonlinearity, but EquiformerV2 demonstrates that **Separable S² activation** improves force MAE by ~5% on OC20 S2EF.

**Current state:** `S2Activation` exists in `flash_eq/layers/s2_activation.py` but is not integrated into the transformer.

**EquiformerV2's Separable S² pattern:**
1. **Separate scalar and higher-degree paths:**
   - Scalars (l=0): Standard SiLU activation
   - Higher degrees (l>0): S²[MLP → SiLU → MLP] on spherical grid
2. **Gate higher degrees by scalar features** after S² activation

**Implementation options:**
1. Replace `EquivariantGating` with Separable S² in the MLP (recommended)
2. Add as configurable option (`activation="s2"` vs `activation="gate"`)
3. Use both: gating for attention path, S² for MLP

**Considerations:**
- S²Activation has higher memory cost due to grid expansion (770 points at precision=47)
- Benchmark shows ~2-5ms per 1K-2K nodes, adding ~15-25% overhead to block time
- May need gradient checkpointing for large graphs
- The separable design reduces cost vs applying S² to all degrees

**Sources:**
- EquiformerV2 (ICLR 2024), Section 3.3: https://arxiv.org/abs/2306.12059
- Complete Guide to Spherical Equivariant Graph Transformers: https://arxiv.org/abs/2512.13927

## Add Stochastic Depth Regularization

EquiformerV2 uses stochastic depth (drop path) for regularization, which is standard in modern vision transformers but missing from the current implementation.

**What it does:** Randomly drops entire residual blocks during training with probability p, scaling surviving paths by 1/(1-p). At inference, all paths are active.

**EquiformerV2 settings:**
- Drop rate: 0.05-0.1 (increases linearly with depth)
- Applied to both attention and MLP residual paths

**Implementation:**
```python
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x + residual
        keep_prob = 1.0 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x + residual * mask / keep_prob
```

**In EquivariantTransformerBlock:**
- Add `drop_path` parameter (default 0.0 for backward compatibility)
- Apply to `h = h + attn_out` and `h = h + mlp_out` lines
- In `EquivariantTransformer`, compute drop rates per layer: `[drop_path * i / (num_layers - 1) for i in range(num_layers)]`

**Sources:**
- EquiformerV2 (ICLR 2024): https://arxiv.org/abs/2306.12059
- Deep Networks with Stochastic Depth (Huang et al., 2016): https://arxiv.org/abs/1603.09382

## Redesign FFN with Separable Pattern

The current MLP in `EquivariantTransformerBlock` applies `EquivariantLinear → EquivariantGating → EquivariantLinear` uniformly to all degrees. EquiformerV2 uses a more expressive separable design.

**Current implementation:**
```python
self.mlp = nn.Sequential(
    EquivariantLinear(repr, hidden_repr),
    EquivariantGating(hidden_repr),
    EquivariantLinear(hidden_repr, repr),
)
```

**EquiformerV2's separable FFN:**
1. **Split features by degree:**
   - Extract scalar (l=0) and higher-degree (l>0) components
2. **Process separately:**
   - Scalars: Linear → SiLU → Linear (standard MLP)
   - Higher degrees: Linear → S²[MLP → SiLU → MLP] → Linear
3. **Gate higher degrees by scalars** (optional, adds expressivity)
4. **Concatenate outputs**

**Benefits:**
- More expressive nonlinearity for higher degrees via S² sampling
- Scalars get efficient standard activation (no grid overhead)
- Better gradient flow through scalar path

**Implementation sketch:**
```python
class SeparableFFN(nn.Module):
    def __init__(self, repr: Repr, hidden_mult: int = 4):
        # Scalar MLP
        scalar_dim = repr.mult  # l=0 has mult channels
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(scalar_dim * hidden_mult, scalar_dim),
        )
        # Higher-degree path with S² activation
        higher_repr = Repr(lvals=repr.lvals[repr.lvals > 0], mult=repr.mult)
        self.higher_linear1 = EquivariantLinear(higher_repr, higher_repr)
        self.s2_act = S2Activation(higher_repr, hidden_mult=2)
        self.higher_linear2 = EquivariantLinear(higher_repr, higher_repr)
```

**Sources:**
- EquiformerV2 (ICLR 2024), Section 3.3: https://arxiv.org/abs/2306.12059
- EGraFFBench (2024): https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00027g

## NVIDIA SE3-Transformer Weight Transfer

### Goal
Load pretrained NVIDIA SE3-Transformer weights (e.g., from RF-Diffusion) into Flash-EQ without retraining.

### Findings

**Basis representations are fundamentally different:**
- NVIDIA: CG coefficients contracted with spherical harmonics: `Σ_J w_J * [CG_J ⊗ Y_J(dir)]`
- Flash-EQ: Wigner-D rotation to m-diagonal basis: `Q(dir) @ Λ @ P(dir)^T`

Both have the same parameter count per channel pair (`num_J = 2*l + 1` for `l_in = l_out = l`).

**Analytical conversion fails for l > 0:**
Tested whether direction-independent Λ exists. Result: only works for l=0 (scalars). For l>0, the fitted Λ varies with direction, meaning no closed-form mapping exists.

**Gradient-based optimization also fails for l > 0:**
Optimizing Flash-EQ weights to match NVIDIA outputs:
- l=0: Converges (rel error ~1e-4)
- l=1, l=2: Stuck at ~4-5x relative error

This suggests Flash-EQ's block-diagonal structure cannot represent the same function space as NVIDIA's CG-basis for l>0.

### Unexplored Options

1. **Full model distillation** - Train Flash-EQ model end-to-end to match NVIDIA model outputs (not layer-wise). May work if the overall function is learnable even though individual layers differ.

2. **Hybrid architecture** - Keep NVIDIA's CG-basis computation but use Flash-EQ's binning/memory optimization for weight storage. Would require new kernel that applies CG⊗Y basis with binned radial weights.

3. **Accept approximation** - Use Flash-EQ with best-fit weights, accepting some accuracy loss. May be acceptable depending on downstream task tolerance.

4. **Different basis for Flash-EQ** - Modify Flash-EQ to use CG-basis instead of Wigner-D. Would lose the block-diagonal structure but enable direct weight loading.

5. **Sparse conversion matrix** - Investigate whether a sparse/structured transformation exists between the two weight spaces that we missed.

## Training Validation

### Denoising Training Script

Create a minimal training script to verify the model can learn, using ciffy for data loading.

**Task:** Coordinate denoising
- Add Gaussian noise to atom coordinates
- Predict displacement vectors to recover original positions
- Loss: MSE between predicted and true displacements

**Data pipeline (using ciffy):**
1. Load CIF structures via `ciffy.load(path, backend="torch")`
2. Build k-NN graph with `build_knn_graph(coords, k=16)`
3. Extract edges: `src`, `dst`, `directions`, `distances`
4. Compute spherical harmonic features from directions via `WignerDBasis`

**Model:**
- Input: noisy coordinates → spherical tensor features (l=0,1,2)
- Layers: 2-4 `EquivariantEdgewiseLinear` blocks
- Output: displacement vectors (l=1 only, equivariant)

**Training loop:**
1. Sample batch of structures
2. Add noise: `noisy_coords = coords + noise * randn`
3. Forward pass → predicted displacements
4. Loss: `MSE(pred_displacement, -noise)`
5. Verify loss decreases over epochs

**Success criteria:**
- Overfit single structure (loss → 0)
- Generalize across multiple structures (loss decreases)
- Equivariance preserved (rotate input → output rotates accordingly)
