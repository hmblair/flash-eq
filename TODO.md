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

## Add S2Activation to EquivariantTransformerBlock

The current `EquivariantTransformerBlock` uses `EquivariantGating` for nonlinearity, but `S2Activation` (spherical grid sampling + MLP) may provide better expressivity as used in EquiformerV2.

**Options:**
1. Replace `EquivariantGating` with `S2Activation` in the MLP
2. Add `S2Activation` as an optional nonlinearity (configurable)
3. Use both: gating for the attention path, S2Activation for the MLP

**Considerations:**
- S2Activation has higher memory cost due to grid expansion (770 points at precision=47)
- Benchmark shows ~2-5ms per 1K-2K nodes, adding ~15-25% overhead to block time
- May need gradient checkpointing for large graphs

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
