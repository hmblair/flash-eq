/**
 * @file block_diagonal.cu
 * @brief Block-diagonal multiplication with binned radial weights.
 *
 * Implements: out = Λ(r) @ f
 *
 * where:
 *   f: input features (num_edges, channels_in, dim_in) in m-first basis
 *   Λ(r): block-diagonal weights interpolated from radial table
 *   out: output features (num_edges, channels_out, dim_out) in m-first basis
 *
 * Split into separate kernels for m=0 (real scalars) and m>0 (complex-like 2x2).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_texture_types.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Enable texture memory for radial table access (set to 0 to use __ldg instead)
// Benchmarking shows __ldg is faster: texture adds 5-7% overhead for our access pattern
// because L2 cache is already efficient for deterministic strided access.
#define USE_TEXTURE_MEMORY 0

constexpr int THREADS = 256;


//------------------------------------------------------------------------------
// Device Helper Types
//------------------------------------------------------------------------------

/**
 * Complex number stored as real/imaginary pair.
 * Used for m>0 spherical harmonic components.
 */
struct Complex {
    float re, im;
};

/**
 * Interpolation state for binned radial weights.
 */
struct InterpState {
    int idx_lo;
    int idx_hi;
    float t;
    float one_minus_t;
};

/**
 * Binning parameters for distance-to-bin conversion.
 *
 * Supports two binning modes controlled by LOG_BINS template parameter:
 *   - Linear (LOG_BINS=false): normalized = (dist - param1) * param2
 *     where param1 = min_distance, param2 = 1/bin_width
 *   - Logarithmic (LOG_BINS=true): normalized = (log(dist) - param1) * param2 * num_bins
 *     where param1 = log(min_distance), param2 = 1/log(max/min)
 *
 * The normalized value is clamped to [0, num_bins] and used for linear interpolation
 * between adjacent bin edges in the radial weight table.
 */
struct BinningParams {
    float param1;   ///< min_val (linear) or log_min (log)
    float param2;   ///< inv_bin_width (linear) or inv_log_range (log)
    int num_bins;   ///< Number of bins (table has num_bins + 1 entries)
};

/**
 * Compute binning from distance value.
 * Template parameter LOG_BINS selects linear (false) or logarithmic (true) spacing.
 */
template <bool LOG_BINS>
[[nodiscard]] __device__ __forceinline__ InterpState compute_binning(float dist, BinningParams params) {
    float normalized;
    if constexpr (LOG_BINS) {
        // Log spacing: normalized = (log(dist) - log_min) * inv_log_range * num_bins
        normalized = (logf(fmaxf(dist, expf(params.param1))) - params.param1)
                     * params.param2 * params.num_bins;
    } else {
        // Linear spacing: normalized = (dist - min_val) * inv_bin_width
        normalized = (dist - params.param1) * params.param2;
    }

    normalized = fminf(fmaxf(normalized, 0.0f), static_cast<float>(params.num_bins));

    InterpState state;
    state.idx_lo = min(static_cast<int>(floorf(normalized)), params.num_bins - 1);
    state.idx_hi = min(state.idx_lo + 1, params.num_bins);
    state.t = normalized - static_cast<float>(state.idx_lo);
    state.one_minus_t = 1.0f - state.t;
    return state;
}

/**
 * Compute derivative of interpolation weight w.r.t. distance.
 * Used in backward pass to convert grad_interp_weight to grad_distance.
 */
template <bool LOG_BINS>
[[nodiscard]] __device__ __forceinline__ float binning_derivative(float dist, BinningParams params) {
    if constexpr (LOG_BINS) {
        // d(normalized)/d(dist) = inv_log_range * num_bins / dist
        return params.param2 * params.num_bins / fmaxf(dist, expf(params.param1));
    } else {
        // d(normalized)/d(dist) = inv_bin_width
        return params.param2;
    }
}

/**
 * Block parameters describing layout of one m-block in the weight matrix.
 * Precomputed on CPU and passed to kernels to avoid redundant computation.
 */
struct BlockParams {
    int m;         // Magnetic quantum number
    int n_in;      // Number of input l-values contributing to this m
    int n_out;     // Number of output l-values contributing to this m
    int in_off;    // Offset into input feature vector
    int out_off;   // Offset into output feature vector
    int w_off;     // Offset into weight vector
};

/**
 * Dimension information bundled for cleaner kernel signatures.
 * Reduces parameter count from 6 individual ints to 1 struct.
 */
struct DimInfo {
    int64_t num_edges;  // Number of edges (batch size)
    int Cin;            // Input channels
    int Cout;           // Output channels
    int Din;            // Input feature dimension
    int Dout;           // Output feature dimension
    int Wdim;           // Weight dimension per channel pair
};

/**
 * Maximum supported l-value + 1 for precomputed lookup tables.
 * Supports representations up to l=16 (289 components per channel).
 * Increase this value to support higher angular momentum, but note that
 * shared memory usage scales with MAX_LVALS^2.
 */
constexpr int MAX_LVALS = 17;

/**
 * Number of int32 values per m-block in the precomputed BlockParams array.
 * Layout: [m, n_in, n_out, in_off, out_off, w_off]
 */
constexpr int BLOCK_PARAMS_STRIDE = 6;


//------------------------------------------------------------------------------
// Device Helper Functions
//------------------------------------------------------------------------------

/**
 * Get the l-value for the idx-th element within an m-block using precomputed lookup table.
 * The lookup table is precomputed on CPU: l_lookup[m * MAX_LVALS + idx] = l-value
 *
 * @param l_lookup Precomputed lookup table
 * @param m The magnetic quantum number for this block
 * @param idx Index within the m-block (0-indexed)
 * @return The l-value corresponding to this index
 */
[[nodiscard]] __device__ __forceinline__ int get_l_for_index_fast(const int* __restrict__ l_lookup, int m, int idx) {
    return __ldg(&l_lookup[m * MAX_LVALS + idx]);
}

/**
 * Integer power via repeated multiplication. Much faster than powf for small l.
 */
[[nodiscard]] __device__ __forceinline__ float ipowf(float base, int exp) {
    float result = 1.0f;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

/**
 * Compute solid harmonic scale factor: (r / (r + scale))^l
 * Returns 1.0 for l=0, smoothly suppresses higher l at short distances.
 */
[[nodiscard]] __device__ __forceinline__ float solid_harmonic_scale(float distance, float scale, int l) {
    if (l == 0) return 1.0f;
    float weight = distance / (distance + scale);
    return ipowf(weight, l);
}

/**
 * Compute derivative of solid harmonic scale factor w.r.t. distance.
 * d/dr[(r/(r+s))^l] = (r/(r+s))^l * l * s / (r * (r+s))
 * Returns 0.0 for l=0 since scaling is constant.
 */
[[nodiscard]] __device__ __forceinline__ float solid_harmonic_scale_derivative(float distance, float scale, int l) {
    if (l == 0) return 0.0f;
    float r_plus_s = distance + scale;
    float sh_scale = ipowf(distance / r_plus_s, l);
    return sh_scale * l * scale / (distance * r_plus_s);
}

/**
 * Load a complex number from interleaved storage.
 * Storage format: [re_0, im_0, re_1, im_1, ...]
 */
[[nodiscard]] __device__ __forceinline__ Complex load_complex(const float* ptr, int idx) {
    return {ptr[2 * idx], ptr[2 * idx + 1]};
}

/**
 * Store a complex number to interleaved storage.
 */
template <typename scalar_t>
__device__ __forceinline__ void store_complex(scalar_t* ptr, int idx, Complex c) {
    ptr[2 * idx] = static_cast<scalar_t>(c.re);
    ptr[2 * idx + 1] = static_cast<scalar_t>(c.im);
}

/**
 * Load precomputed block parameters for magnetic quantum number m.
 * BlockParams are stored as 6 consecutive ints per m-value:
 *   [m, n_in, n_out, in_off, out_off, w_off]
 *
 * @param block_params_flat Flattened precomputed block params tensor
 * @param m The magnetic quantum number
 * @return BlockParams structure for this m-value
 */
[[nodiscard]] __device__ __forceinline__ BlockParams load_block_params(const int* __restrict__ block_params_flat, int m) {
    const int* ptr = block_params_flat + m * BLOCK_PARAMS_STRIDE;
    BlockParams bp;
    bp.m = __ldg(&ptr[0]);
    bp.n_in = __ldg(&ptr[1]);
    bp.n_out = __ldg(&ptr[2]);
    bp.in_off = __ldg(&ptr[3]);
    bp.out_off = __ldg(&ptr[4]);
    bp.w_off = __ldg(&ptr[5]);
    return bp;
}


/**
 * Linearly interpolate a weight from the radial table using __ldg (L2 cache).
 */
template <typename scalar_t>
[[nodiscard]] __device__ __forceinline__ float lerp_weight(
    const scalar_t* __restrict__ table,
    int idx_lo,
    int idx_hi,
    float t,
    float one_minus_t
) {
    return one_minus_t * static_cast<float>(__ldg(&table[idx_lo]))
         + t * static_cast<float>(__ldg(&table[idx_hi]));
}

#if USE_TEXTURE_MEMORY
/**
 * Linearly interpolate a weight from the radial table using texture memory.
 * Texture cache is replicated per SM, potentially reducing L2 contention
 * when many SMs access the same read-only data.
 */
[[nodiscard]] __device__ __forceinline__ float lerp_weight_tex(
    cudaTextureObject_t tex,
    int idx_lo,
    int idx_hi,
    float t,
    float one_minus_t
) {
    return one_minus_t * tex1Dfetch<float>(tex, idx_lo)
         + t * tex1Dfetch<float>(tex, idx_hi);
}

/**
 * Unified lerp_weight that uses texture when available (float32 only).
 * Template parameter USE_TEX enables texture path.
 */
template <typename scalar_t, bool USE_TEX>
[[nodiscard]] __device__ __forceinline__ float lerp_weight_unified(
    const scalar_t* __restrict__ table,
    cudaTextureObject_t tex,
    int idx_lo,
    int idx_hi,
    float t,
    float one_minus_t
) {
    if constexpr (USE_TEX && std::is_same_v<scalar_t, float>) {
        return lerp_weight_tex(tex, idx_lo, idx_hi, t, one_minus_t);
    } else {
        return lerp_weight(table, idx_lo, idx_hi, t, one_minus_t);
    }
}

/**
 * Create a CUDA texture object for float data.
 * Returns 0 (null texture) if creation fails.
 */
inline cudaTextureObject_t create_texture_object(const float* data, size_t num_elements) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<float*>(data);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;  // 32-bit float
    resDesc.res.linear.sizeInBytes = num_elements * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;

    cudaTextureObject_t tex = 0;
    cudaError_t err = cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        // Fall back to __ldg if texture creation fails
        return 0;
    }
    return tex;
}

inline void destroy_texture_object(cudaTextureObject_t tex) {
    if (tex != 0) {
        cudaDestroyTextureObject(tex);
    }
}
#endif

/**
 * Compute weight index for 2x2 rotation block.
 * Weights are stored as [a_00, b_00, a_01, b_01, ...] where each (a,b) pair
 * defines the 2x2 rotation matrix [[a, b], [-b, a]].
 *
 * @param o Output index within this m-block
 * @param i Input index within this m-block
 * @param n_in Number of input indices
 * @param component 0 for 'a' weight, 1 for 'b' weight
 */
[[nodiscard]] __device__ __forceinline__ int weight_idx_2x2(int o, int i, int n_in, int component) {
    return (o * n_in + i) * 2 + component;
}

/**
 * Complex multiply-accumulate for m>0 blocks (forward pass).
 * Computes: acc += rotation(a, b) * f
 *
 * The 2x2 rotation matrix structure:
 *   [a  b] [f.re]   [a*f.re + b*f.im]
 *   [-b a] [f.im] = [a*f.im - b*f.re]
 */
__device__ __forceinline__ void complex_mul_add(float a, float b, Complex f, Complex& acc) {
    acc.re += a * f.re + b * f.im;
    acc.im += a * f.im - b * f.re;
}

/**
 * Transpose rotation multiply-accumulate (backward pass for features).
 * Computes: acc += rotation(a, b)^T * grad_out
 *
 * Transpose of rotation matrix:
 *   [a  b]^T   [a  -b]
 *   [-b a]   = [b   a]
 *
 * So: grad_f.re += a * go.re - b * go.im
 *     grad_f.im += b * go.re + a * go.im
 */
__device__ __forceinline__ void complex_mul_add_transpose(float a, float b, Complex go, Complex& grad) {
    grad.re += a * go.re - b * go.im;
    grad.im += b * go.re + a * go.im;
}

/**
 * Compute gradient w.r.t. rotation weights (backward pass for table).
 * Given: out = rotation(a, b) * f, and grad_out
 * Returns: (grad_a, grad_b)
 *
 * grad_a = f.re * go.re + f.im * go.im  (derivative of both diagonal terms)
 * grad_b = f.im * go.re - f.re * go.im  (derivative of both off-diagonal terms)
 */
__device__ __forceinline__ void compute_weight_gradient(Complex f, Complex go, float& grad_a, float& grad_b) {
    grad_a = f.re * go.re + f.im * go.im;
    grad_b = f.im * go.re - f.re * go.im;
}

/**
 * Warp-level sum reduction.
 */
[[nodiscard]] __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


//------------------------------------------------------------------------------
// Forward Kernels
//------------------------------------------------------------------------------

/**
 * Forward kernel for m=0 block (real scalars).
 * One CUDA block per edge.
 * Includes solid harmonic scaling: weights multiplied by (r/(r+scale))^(l_in+l_out)
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors for (l_in, l_out) pairs are precomputed in shared memory.
 *
 * Template parameter USE_TEX enables texture memory for radial table (float32 only).
 */
template <typename scalar_t, bool LOG_BINS, bool USE_TEX = false>
__global__ void forward_m0_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    DimInfo dims,
    cudaTextureObject_t radial_tex = 0
) {
    const int64_t edge = blockIdx.x;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;

    // Load precomputed block parameters for m=0
    const BlockParams bp = load_block_params(block_params_flat, 0);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    // Shared memory layout: [features | scale_factors]
    extern __shared__ float shared_mem[];
    float* feat_shared = shared_mem;
    float* scale_shared = shared_mem + dims.Cin * bp.n_in;

    // Precompute scale factors for all (i, o) pairs: ipowf(sh_weight, l_in + l_out)
    // Optimization #8: avoid redundant ipowf calls in inner loop
    const int scale_size = bp.n_in * bp.n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, 0, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, 0, o);
        const int l_total = l_in + l_out;
        scale_shared[idx] = (l_total > 0) ? ipowf(sh_weight, l_total) : 1.0f;
    }

    // Load features into shared memory
    const int64_t feat_base = edge * dims.Cin * dims.Din;
    for (int i = tid; i < dims.Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * dims.Din + bp.in_off + local_idx]);
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < dims.Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        float acc = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;

        for (int ci = 0; ci < dims.Cin; ci++) {
            const float* f_ptr = feat_shared + ci * bp.n_in;
            const int w_ci_off = ci * dims.Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const int w_idx = w_ci_off + o * bp.n_in + i;
#if USE_TEXTURE_MEMORY
                float w = lerp_weight_unified<scalar_t, USE_TEX>(
                    radial_table, radial_tex, w_base_lo + w_idx, w_base_hi + w_idx,
                    interp.t, interp.one_minus_t);
#else
                float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);
#endif

                // Apply precomputed solid harmonic scaling
                w *= scale_shared[o * bp.n_in + i];

                acc += w * f_ptr[i];
            }
        }

        output[edge * dims.Cout * dims.Dout + co * dims.Dout + bp.out_off + o] = static_cast<scalar_t>(acc);
    }
}


/**
 * Forward kernel for m>0 blocks (complex-like 2x2 structure).
 * One CUDA block per (edge, m) pair where m ranges from 1 to mmax.
 * Includes solid harmonic scaling: weights multiplied by (r/(r+scale))^(l_in+l_out)
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors for (l_in, l_out) pairs are precomputed in shared memory.
 *
 * Template parameter USE_TEX enables texture memory for radial table (float32 only).
 */
template <typename scalar_t, bool LOG_BINS, bool USE_TEX = false>
__global__ void forward_mpos_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    int mmax,
    DimInfo dims,
    cudaTextureObject_t radial_tex = 0
) {
    // blk indexes m values from 1 to mmax
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;  // m=1, 2, ..., mmax

    // Load precomputed block parameters for this m
    const BlockParams bp = load_block_params(block_params_flat, m);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int in_size = 2 * bp.n_in;  // Complex pairs

    // Shared memory layout: [features | scale_factors]
    extern __shared__ float shared_mem[];
    float* feat_shared = shared_mem;
    float* scale_shared = shared_mem + dims.Cin * in_size;

    // Precompute scale factors for all (i, o) pairs: ipowf(sh_weight, l_in + l_out)
    // Optimization #8: avoid redundant ipowf calls in inner loop
    const int scale_size = bp.n_in * bp.n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, m, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, m, o);
        const int l_total = l_in + l_out;
        // l_total >= 2 for m>0 blocks (since l >= m >= 1 for both in and out)
        scale_shared[idx] = ipowf(sh_weight, l_total);
    }

    // Load features into shared memory
    const int64_t feat_base = edge * dims.Cin * dims.Din + bp.in_off;
    for (int i = tid; i < dims.Cin * in_size; i += THREADS) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * dims.Din + local_idx]);
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute outputs: out[o] = sum over channels and inputs of rotation(a,b) * f[i]
    for (int out_idx = tid; out_idx < dims.Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        Complex acc = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;

        for (int ci = 0; ci < dims.Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * dims.Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const Complex f = load_complex(f_ptr, i);
                const int w_idx = w_ci_off + weight_idx_2x2(o, i, bp.n_in, 0);

#if USE_TEXTURE_MEMORY
                float a = lerp_weight_unified<scalar_t, USE_TEX>(
                    radial_table, radial_tex, w_base_lo + w_idx, w_base_hi + w_idx,
                    interp.t, interp.one_minus_t);
                float b = lerp_weight_unified<scalar_t, USE_TEX>(
                    radial_table, radial_tex, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                    interp.t, interp.one_minus_t);
#else
                float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);
                float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      interp.t, interp.one_minus_t);
#endif

                // Apply precomputed solid harmonic scaling
                const float scale_factor = scale_shared[o * bp.n_in + i];
                a *= scale_factor;
                b *= scale_factor;

                complex_mul_add(a, b, f, acc);
            }
        }

        // Store output: output[edge, co, out_off + 2*o : out_off + 2*o + 2]
        scalar_t* out_ptr = &output[edge * dims.Cout * dims.Dout + co * dims.Dout + bp.out_off];
        store_complex(out_ptr, o, acc);
    }
}


//------------------------------------------------------------------------------
// Backward Feature Kernels
//------------------------------------------------------------------------------

/**
 * Backward kernel for m=0: compute grad_features.
 * Includes solid harmonic scaling on weights.
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors for (l_in, l_out) pairs are precomputed in shared memory.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_features_m0_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    DimInfo dims
) {
    const int64_t edge = blockIdx.x;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;

    // Load precomputed block parameters for m=0
    const BlockParams bp = load_block_params(block_params_flat, 0);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    // Shared memory layout: [grad_output | scale_factors]
    extern __shared__ float shared_mem[];
    float* grad_shared = shared_mem;
    float* scale_shared = shared_mem + dims.Cout * bp.n_out;

    // Precompute scale factors for all (i, o) pairs: ipowf(sh_weight, l_in + l_out)
    // Optimization #8: avoid redundant ipowf calls in inner loop
    const int scale_size = bp.n_in * bp.n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, 0, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, 0, o);
        const int l_total = l_in + l_out;
        scale_shared[idx] = (l_total > 0) ? ipowf(sh_weight, l_total) : 1.0f;
    }

    // Load grad_output into shared memory
    const int64_t grad_base = edge * dims.Cout * dims.Dout;
    for (int i = tid; i < dims.Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * dims.Dout + bp.out_off + local_idx]);
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < dims.Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        float grad = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + ci * dims.Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * dims.Wdim + bp.w_off;

        for (int co = 0; co < dims.Cout; co++) {
            const float* go_ptr = grad_shared + co * bp.n_out;
            const int w_co_off = co * dims.Cin * dims.Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const int w_idx = w_co_off + o * bp.n_in + i;
                float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);

                // Apply precomputed solid harmonic scaling
                w *= scale_shared[o * bp.n_in + i];

                grad += w * go_ptr[o];
            }
        }

        grad_features[edge * dims.Cin * dims.Din + ci * dims.Din + bp.in_off + i] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward kernel for m>0: compute grad_features.
 * Computes: grad_f = W^T @ grad_out (transpose of rotation).
 * Includes solid harmonic scaling on weights.
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors for (l_in, l_out) pairs are precomputed in shared memory.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_features_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    int mmax,
    DimInfo dims
) {
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;

    // Load precomputed block parameters for this m
    const BlockParams bp = load_block_params(block_params_flat, m);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int out_size = 2 * bp.n_out;

    // Shared memory layout: [grad_output | scale_factors]
    extern __shared__ float shared_mem[];
    float* grad_shared = shared_mem;
    float* scale_shared = shared_mem + dims.Cout * out_size;

    // Precompute scale factors for all (i, o) pairs: ipowf(sh_weight, l_in + l_out)
    // Optimization #8: avoid redundant ipowf calls in inner loop
    const int scale_size = bp.n_in * bp.n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, m, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, m, o);
        const int l_total = l_in + l_out;
        // l_total >= 2 for m>0 blocks (since l >= m >= 1 for both in and out)
        scale_shared[idx] = ipowf(sh_weight, l_total);
    }

    // Load grad_output into shared memory
    const int64_t grad_base = edge * dims.Cout * dims.Dout + bp.out_off;
    for (int i = tid; i < dims.Cout * out_size; i += THREADS) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * dims.Dout + local_idx]);
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute grad_features for each complex input
    for (int in_idx = tid; in_idx < dims.Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        Complex grad_f = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + ci * dims.Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * dims.Wdim + bp.w_off;

        for (int co = 0; co < dims.Cout; co++) {
            const float* go_ptr = grad_shared + co * out_size;
            const int w_co_off = co * dims.Cin * dims.Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const Complex go = load_complex(go_ptr, o);
                const int w_idx = w_co_off + weight_idx_2x2(o, i, bp.n_in, 0);

                float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);
                float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      interp.t, interp.one_minus_t);

                // Apply precomputed solid harmonic scaling
                const float scale_factor = scale_shared[o * bp.n_in + i];
                a *= scale_factor;
                b *= scale_factor;

                complex_mul_add_transpose(a, b, go, grad_f);
            }
        }

        // Store both real and imaginary gradients
        scalar_t* gf_ptr = &grad_features[edge * dims.Cin * dims.Din + ci * dims.Din + bp.in_off];
        store_complex(gf_ptr, i, grad_f);
    }
}


//------------------------------------------------------------------------------
// Backward Table Kernels
//------------------------------------------------------------------------------

/**
 * Backward kernel for m=0: compute grad_radial_table and grad_distance.
 * Includes solid harmonic scaling in gradient computation.
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors and l_total values are precomputed in shared memory.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_table_m0_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_distances,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    DimInfo dims
) {
    const int64_t edge = blockIdx.x;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;

    // Load precomputed block parameters for m=0
    const BlockParams bp = load_block_params(block_params_flat, 0);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;
    const int w_block_size = bp.n_out * bp.n_in;
    const int scale_size = bp.n_in * bp.n_out;

    // Shared memory layout: [features | grad_output | scale_factors | l_totals]
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + dims.Cin * bp.n_in;
    float* scale_shared = grad_shared + dims.Cout * bp.n_out;
    int* l_total_shared = reinterpret_cast<int*>(scale_shared + scale_size);

    // Precompute scale factors and l_totals for all (i, o) pairs
    // Optimization #8: avoid redundant ipowf calls in inner loop
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, 0, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, 0, o);
        const int l_total = l_in + l_out;
        l_total_shared[idx] = l_total;
        scale_shared[idx] = (l_total > 0) ? ipowf(sh_weight, l_total) : 1.0f;
    }

    // Load features
    const int64_t feat_base = edge * dims.Cin * dims.Din;
    for (int i = tid; i < dims.Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * dims.Din + bp.in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * dims.Cout * dims.Dout;
    for (int i = tid; i < dims.Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * dims.Dout + bp.out_off + local_idx]);
    }
    __syncthreads();

    // Precompute derivative factor: sh_scale / (dist * (dist + sh_scale))
    // d/dr[(r/(r+s))^l] = (r/(r+s))^l * l * s / (r * (r+s))
    // = scale_factor * l_total * deriv_factor
    const float deriv_factor = sh_scale / (dist * (dist + sh_scale));

    float local_grad_t = 0.0f;
    float local_grad_sh = 0.0f;  // Gradient through solid harmonic scaling

    // Compute weight gradients
    for (int w_idx = tid; w_idx < dims.Cout * dims.Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (dims.Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % dims.Cin;
        const int w_local = w_idx % w_block_size;
        const int o = w_local / bp.n_in;
        const int i = w_local % bp.n_in;

        // Use precomputed scale factor and l_total
        const int scale_idx = o * bp.n_in + i;
        const float scale_factor = scale_shared[scale_idx];
        const int l_total = l_total_shared[scale_idx];

        const float f = feat_shared[ci * bp.n_in + i];
        const float go = grad_shared[co * bp.n_out + o];
        // grad_w includes the scale factor since output = w * scale_factor * f
        const float grad_w = scale_factor * f * go;

        const int table_idx = co * dims.Cin * dims.Wdim + ci * dims.Wdim + bp.w_off + w_local;
        const int addr_lo = interp.idx_lo * table_stride + table_idx;
        const int addr_hi = interp.idx_hi * table_stride + table_idx;

        atomicAdd(&grad_radial_table[addr_lo], interp.one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[addr_hi], interp.t * grad_w);

        const float w_lo = static_cast<float>(__ldg(&radial_table[addr_lo]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[addr_hi]));
        const float interp_weight = interp.one_minus_t * w_lo + interp.t * w_hi;

        // Gradient through interpolation: d(output)/d(t) * d(t)/d(dist)
        local_grad_t += (w_hi - w_lo) * grad_w;

        // Gradient through solid harmonic scaling using precomputed values
        // d(scale)/d(dist) = scale_factor * l_total * deriv_factor
        if (l_total > 0) {
            local_grad_sh += interp_weight * f * go * scale_factor * l_total * deriv_factor;
        }
    }

    // Warp reduction and atomic add for grad_distance
    // Chain rule: grad_dist = grad_t * d(t)/d(dist) + grad_sh
    local_grad_t = warp_reduce_sum(local_grad_t);
    local_grad_sh = warp_reduce_sum(local_grad_sh);
    if ((tid % 32) == 0) {
        const float grad_dist = local_grad_t * binning_derivative<LOG_BINS>(dist, bin_params) + local_grad_sh;
        atomicAdd(&grad_distances[edge], grad_dist);
    }
}


/**
 * Backward kernel for m>0: compute grad_radial_table and grad_distance.
 * Each thread computes gradient for one weight element (either 'a' or 'b').
 * Includes solid harmonic scaling in gradient computation.
 *
 * Uses precomputed block parameters and l-value lookup tables for efficiency.
 * Scale factors and l_total values are precomputed in shared memory.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_table_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_distances,
    const int* __restrict__ block_params_flat,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    int mmax,
    DimInfo dims
) {
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= dims.num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;

    // Load precomputed block parameters for this m
    const BlockParams bp = load_block_params(block_params_flat, m);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int in_size = 2 * bp.n_in;
    const int out_size = 2 * bp.n_out;
    const int weights_per_pair = 2;  // 'a' and 'b' for each (o, i) pair
    const int num_weights = weights_per_pair * bp.n_out * bp.n_in;
    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;
    const int scale_size = bp.n_in * bp.n_out;

    // Shared memory layout: [features | grad_output | scale_factors | l_totals]
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + dims.Cin * in_size;
    float* scale_shared = grad_shared + dims.Cout * out_size;
    int* l_total_shared = reinterpret_cast<int*>(scale_shared + scale_size);

    // Precompute scale factors and l_totals for all (i, o) pairs
    // Optimization #8: avoid redundant ipowf calls in inner loop
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % bp.n_in;
        const int o = idx / bp.n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, m, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, m, o);
        const int l_total = l_in + l_out;
        l_total_shared[idx] = l_total;
        // l_total >= 2 for m>0 blocks (since l >= m >= 1 for both in and out)
        scale_shared[idx] = ipowf(sh_weight, l_total);
    }

    // Load features
    const int64_t feat_base = edge * dims.Cin * dims.Din + bp.in_off;
    for (int idx = tid; idx < dims.Cin * in_size; idx += THREADS) {
        const int ci = idx / in_size;
        const int local_idx = idx % in_size;
        feat_shared[idx] = static_cast<float>(features[feat_base + ci * dims.Din + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * dims.Cout * dims.Dout + bp.out_off;
    for (int idx = tid; idx < dims.Cout * out_size; idx += THREADS) {
        const int co = idx / out_size;
        const int local_idx = idx % out_size;
        grad_shared[idx] = static_cast<float>(grad_output[grad_base + co * dims.Dout + local_idx]);
    }
    __syncthreads();

    // Precompute derivative factor: sh_scale / (dist * (dist + sh_scale))
    // d/dr[(r/(r+s))^l] = (r/(r+s))^l * l * s / (r * (r+s))
    // = scale_factor * l_total * deriv_factor
    const float deriv_factor = sh_scale / (dist * (dist + sh_scale));

    float local_grad_t = 0.0f;
    float local_grad_sh = 0.0f;  // Gradient through solid harmonic scaling

    // Iterate over all weight elements: (co, ci, o, i, component)
    // Each weight pair (a, b) corresponds to one (output, input) connection
    for (int w_idx = tid; w_idx < dims.Cout * dims.Cin * num_weights; w_idx += THREADS) {
        // Decode flat index into (co, ci, o, i, is_b)
        const int co = w_idx / (dims.Cin * num_weights);
        const int ci = (w_idx / num_weights) % dims.Cin;
        const int w_local = w_idx % num_weights;
        const int pair_idx = w_local / 2;
        const int is_b = w_local % 2;  // 0 = 'a' (diagonal), 1 = 'b' (off-diagonal)
        const int o = pair_idx / bp.n_in;
        const int i = pair_idx % bp.n_in;

        // Use precomputed scale factor and l_total
        const int scale_idx = o * bp.n_in + i;
        const float scale_factor = scale_shared[scale_idx];
        const int l_total = l_total_shared[scale_idx];

        // Load feature and gradient
        const Complex f = load_complex(feat_shared + ci * in_size, i);
        const Complex go = load_complex(grad_shared + co * out_size, o);

        // Compute gradient for this weight component
        float grad_a, grad_b;
        compute_weight_gradient(f, go, grad_a, grad_b);
        // grad_w includes the scale factor since output = w * scale_factor * f
        const float grad_w = scale_factor * ((is_b == 0) ? grad_a : grad_b);
        const float unscaled_grad_w = (is_b == 0) ? grad_a : grad_b;

        // Accumulate to table gradient with interpolation
        const int table_idx = co * dims.Cin * dims.Wdim + ci * dims.Wdim + bp.w_off + w_local;
        const int addr_lo = interp.idx_lo * table_stride + table_idx;
        const int addr_hi = interp.idx_hi * table_stride + table_idx;

        atomicAdd(&grad_radial_table[addr_lo], interp.one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[addr_hi], interp.t * grad_w);

        // Gradient w.r.t. interpolation weight (also scaled)
        const float w_lo = static_cast<float>(__ldg(&radial_table[addr_lo]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[addr_hi]));
        const float interp_weight = interp.one_minus_t * w_lo + interp.t * w_hi;

        // Gradient through interpolation: d(output)/d(t) * d(t)/d(dist)
        local_grad_t += (w_hi - w_lo) * grad_w;

        // Gradient through solid harmonic scaling using precomputed values
        // d(scale)/d(dist) = scale_factor * l_total * deriv_factor
        // Note: l_total >= 2 for m>0 (since l >= m >= 1 for both in and out)
        local_grad_sh += interp_weight * unscaled_grad_w * scale_factor * l_total * deriv_factor;
    }

    // Warp reduction and atomic add for grad_distance
    // Chain rule: grad_dist = grad_t * d(t)/d(dist) + grad_sh
    local_grad_t = warp_reduce_sum(local_grad_t);
    local_grad_sh = warp_reduce_sum(local_grad_sh);
    if ((tid % 32) == 0) {
        const float grad_dist = local_grad_t * binning_derivative<LOG_BINS>(dist, bin_params) + local_grad_sh;
        atomicAdd(&grad_distances[edge], grad_dist);
    }
}


//------------------------------------------------------------------------------
// C++ Wrapper Functions
//------------------------------------------------------------------------------

/**
 * CPU helper: count how many l-values in lvals have l >= m.
 */
inline int count_l_geq_m_cpu(const int* lvals, int num_lvals, int m) {
    int count = 0;
    for (int i = 0; i < num_lvals; i++) {
        if (lvals[i] >= m) count++;
    }
    return count;
}

/**
 * CPU helper: precompute block parameters and l-value lookup tables.
 *
 * Optimization #1: Precompute BlockParams for all m values on CPU.
 * Optimization #2: Precompute l-value lookup tables for fast indexing.
 *
 * @param lvals_in Input l-values tensor (on CPU for reading)
 * @param lvals_out Output l-values tensor (on CPU for reading)
 * @param mmax Maximum m value
 * @param device Target device for output tensors
 * @return Tuple of (block_params_flat, l_lookup_in, l_lookup_out)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> precompute_block_info(
    torch::Tensor lvals_in,
    torch::Tensor lvals_out,
    int mmax,
    torch::Device device
) {
    const int num_lvals_in = static_cast<int>(lvals_in.size(0));
    const int num_lvals_out = static_cast<int>(lvals_out.size(0));

    // Access lvals on CPU
    auto lvals_in_cpu = lvals_in.cpu();
    auto lvals_out_cpu = lvals_out.cpu();
    const int* lvals_in_ptr = lvals_in_cpu.data_ptr<int>();
    const int* lvals_out_ptr = lvals_out_cpu.data_ptr<int>();

    // Allocate CPU tensors for precomputed data
    // BlockParams: (mmax+1) entries, each with BLOCK_PARAMS_STRIDE ints
    auto block_params_cpu = torch::zeros({mmax + 1, BLOCK_PARAMS_STRIDE}, torch::kInt32);
    int* bp_ptr = block_params_cpu.data_ptr<int>();

    // L-value lookup tables: (mmax+1) * MAX_LVALS for each of input and output
    auto l_lookup_in_cpu = torch::zeros({(mmax + 1) * MAX_LVALS}, torch::kInt32);
    auto l_lookup_out_cpu = torch::zeros({(mmax + 1) * MAX_LVALS}, torch::kInt32);
    int* ll_in_ptr = l_lookup_in_cpu.data_ptr<int>();
    int* ll_out_ptr = l_lookup_out_cpu.data_ptr<int>();

    // Compute block params and l-value lookups for each m
    int in_off = 0, out_off = 0, w_off = 0;

    for (int m = 0; m <= mmax; m++) {
        int n_in = count_l_geq_m_cpu(lvals_in_ptr, num_lvals_in, m);
        int n_out = count_l_geq_m_cpu(lvals_out_ptr, num_lvals_out, m);

        // Store BlockParams
        int* bp = bp_ptr + m * BLOCK_PARAMS_STRIDE;
        bp[0] = m;
        bp[1] = n_in;
        bp[2] = n_out;
        bp[3] = in_off;
        bp[4] = out_off;
        bp[5] = w_off;

        // Build l-value lookup tables for this m
        int* ll_in = ll_in_ptr + m * MAX_LVALS;
        int* ll_out = ll_out_ptr + m * MAX_LVALS;

        int idx_in = 0;
        for (int i = 0; i < num_lvals_in; i++) {
            if (lvals_in_ptr[i] >= m) {
                ll_in[idx_in++] = lvals_in_ptr[i];
            }
        }

        int idx_out = 0;
        for (int i = 0; i < num_lvals_out; i++) {
            if (lvals_out_ptr[i] >= m) {
                ll_out[idx_out++] = lvals_out_ptr[i];
            }
        }

        // Update offsets for next m
        if (n_in > 0 && n_out > 0) {
            int mult = (m == 0) ? 1 : 2;
            in_off += mult * n_in;
            out_off += mult * n_out;
            w_off += mult * n_out * n_in;
        }
    }

    // Move to target device
    return std::make_tuple(
        block_params_cpu.flatten().to(device),
        l_lookup_in_cpu.to(device),
        l_lookup_out_cpu.to(device)
    );
}

std::vector<torch::Tensor> block_diagonal_forward_cuda(
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor distances,
    float sh_scale,
    float bin_param1,
    float bin_param2,
    int num_bins,
    bool log_bins,
    torch::Tensor lvals_in,
    torch::Tensor lvals_out,
    int64_t Cout,
    int dim_out,
    int max_in_size
) {
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(distances);
    CHECK_INPUT(lvals_in);
    CHECK_INPUT(lvals_out);

    const int64_t B = features.size(0);
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int Cout_int = static_cast<int>(Cout);
    const int num_lvals_in = static_cast<int>(lvals_in.size(0));
    const int num_lvals_out = static_cast<int>(lvals_out.size(0));
    const int mmax = std::max(lvals_in.max().item<int>(), lvals_out.max().item<int>());

    TORCH_CHECK(mmax < MAX_LVALS,
        "Maximum l-value (", mmax, ") exceeds supported limit (", MAX_LVALS - 1, "). "
        "Recompile with larger MAX_LVALS to support higher angular momentum.");

    BinningParams bin_params = {bin_param1, bin_param2, num_bins};
    DimInfo dims = {B, Cin, Cout_int, Din, dim_out, Wdim};

    // Precompute block parameters and l-value lookup tables (Optimizations #1 and #2)
    auto [block_params_flat, l_lookup_in, l_lookup_out] = precompute_block_info(
        lvals_in, lvals_out, mmax, features.device());

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Shared memory size: features + scale_factors
    // Scale factors size: max n_in * n_out = num_lvals_in * num_lvals_out
    const size_t max_scale_size = num_lvals_in * num_lvals_out;
    const size_t shared_size_m0 = (Cin * num_lvals_in + max_scale_size) * sizeof(float);
    const size_t shared_size_mpos = (Cin * max_in_size + max_scale_size) * sizeof(float);

#if USE_TEXTURE_MEMORY
    // Create texture object for float32 radial table
    cudaTextureObject_t radial_tex = 0;
    const bool use_texture = (features.scalar_type() == torch::kFloat32);
    if (use_texture) {
        radial_tex = create_texture_object(
            radial_table.data_ptr<float>(),
            radial_table.numel()
        );
    }
#endif

    // Launch m=0 kernel with template dispatch for log_bins
    if (log_bins) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
#if USE_TEXTURE_MEMORY
            if constexpr (std::is_same_v<scalar_t, float>) {
                forward_m0_kernel<scalar_t, true, true><<<B, THREADS, shared_size_m0>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    dims,
                    radial_tex
                );
            } else {
                forward_m0_kernel<scalar_t, true, false><<<B, THREADS, shared_size_m0>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    dims,
                    0
                );
            }
#else
            forward_m0_kernel<scalar_t, true><<<B, THREADS, shared_size_m0>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                output.data_ptr<scalar_t>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
#endif
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
#if USE_TEXTURE_MEMORY
            if constexpr (std::is_same_v<scalar_t, float>) {
                forward_m0_kernel<scalar_t, false, true><<<B, THREADS, shared_size_m0>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    dims,
                    radial_tex
                );
            } else {
                forward_m0_kernel<scalar_t, false, false><<<B, THREADS, shared_size_m0>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    dims,
                    0
                );
            }
#else
            forward_m0_kernel<scalar_t, false><<<B, THREADS, shared_size_m0>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                output.data_ptr<scalar_t>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
#endif
        }));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Launch m>0 kernel if mmax > 0
    if (mmax > 0) {
        if (log_bins) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
#if USE_TEXTURE_MEMORY
                if constexpr (std::is_same_v<scalar_t, float>) {
                    forward_mpos_kernel<scalar_t, true, true><<<B * mmax, THREADS, shared_size_mpos>>>(
                        features.data_ptr<scalar_t>(),
                        radial_table.data_ptr<scalar_t>(),
                        distances.data_ptr<scalar_t>(),
                        sh_scale,
                        bin_params,
                        output.data_ptr<scalar_t>(),
                        block_params_flat.data_ptr<int>(),
                        l_lookup_in.data_ptr<int>(),
                        l_lookup_out.data_ptr<int>(),
                        mmax,
                        dims,
                        radial_tex
                    );
                } else {
                    forward_mpos_kernel<scalar_t, true, false><<<B * mmax, THREADS, shared_size_mpos>>>(
                        features.data_ptr<scalar_t>(),
                        radial_table.data_ptr<scalar_t>(),
                        distances.data_ptr<scalar_t>(),
                        sh_scale,
                        bin_params,
                        output.data_ptr<scalar_t>(),
                        block_params_flat.data_ptr<int>(),
                        l_lookup_in.data_ptr<int>(),
                        l_lookup_out.data_ptr<int>(),
                        mmax,
                        dims,
                        0
                    );
                }
#else
                forward_mpos_kernel<scalar_t, true><<<B * mmax, THREADS, shared_size_mpos>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
#endif
            }));
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
#if USE_TEXTURE_MEMORY
                if constexpr (std::is_same_v<scalar_t, float>) {
                    forward_mpos_kernel<scalar_t, false, true><<<B * mmax, THREADS, shared_size_mpos>>>(
                        features.data_ptr<scalar_t>(),
                        radial_table.data_ptr<scalar_t>(),
                        distances.data_ptr<scalar_t>(),
                        sh_scale,
                        bin_params,
                        output.data_ptr<scalar_t>(),
                        block_params_flat.data_ptr<int>(),
                        l_lookup_in.data_ptr<int>(),
                        l_lookup_out.data_ptr<int>(),
                        mmax,
                        dims,
                        radial_tex
                    );
                } else {
                    forward_mpos_kernel<scalar_t, false, false><<<B * mmax, THREADS, shared_size_mpos>>>(
                        features.data_ptr<scalar_t>(),
                        radial_table.data_ptr<scalar_t>(),
                        distances.data_ptr<scalar_t>(),
                        sh_scale,
                        bin_params,
                        output.data_ptr<scalar_t>(),
                        block_params_flat.data_ptr<int>(),
                        l_lookup_in.data_ptr<int>(),
                        l_lookup_out.data_ptr<int>(),
                        mmax,
                        dims,
                        0
                    );
                }
#else
                forward_mpos_kernel<scalar_t, false><<<B * mmax, THREADS, shared_size_mpos>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
#endif
            }));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

#if USE_TEXTURE_MEMORY
    // Clean up texture object
    destroy_texture_object(radial_tex);
#endif

    return {output};
}


std::vector<torch::Tensor> block_diagonal_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor distances,
    float sh_scale,
    float bin_param1,
    float bin_param2,
    int num_bins,
    bool log_bins,
    torch::Tensor lvals_in,
    torch::Tensor lvals_out,
    int dim_in,
    int max_in_size,
    int max_out_size
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(distances);
    CHECK_INPUT(lvals_in);
    CHECK_INPUT(lvals_out);

    const int64_t B = features.size(0);
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Cout = static_cast<int>(grad_output.size(1));
    const int Dout = static_cast<int>(grad_output.size(2));
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_lvals_in = static_cast<int>(lvals_in.size(0));
    const int num_lvals_out = static_cast<int>(lvals_out.size(0));
    const int mmax = std::max(lvals_in.max().item<int>(), lvals_out.max().item<int>());

    TORCH_CHECK(mmax < MAX_LVALS,
        "Maximum l-value (", mmax, ") exceeds supported limit (", MAX_LVALS - 1, "). "
        "Recompile with larger MAX_LVALS to support higher angular momentum.");

    BinningParams bin_params = {bin_param1, bin_param2, num_bins};
    DimInfo dims = {B, Cin, Cout, Din, Dout, Wdim};

    // Precompute block parameters and l-value lookup tables (Optimizations #1 and #2)
    auto [block_params_flat, l_lookup_in, l_lookup_out] = precompute_block_info(
        lvals_in, lvals_out, mmax, features.device());

    auto grad_features = torch::zeros({B, Cin, Din}, features.options());
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_distances = torch::zeros({B}, distances.options().dtype(torch::kFloat32));

    // Shared memory sizes: data + scale_factors + l_totals (for backward_table)
    // Scale factors size: max n_in * n_out = num_lvals_in * num_lvals_out
    const size_t max_scale_size = num_lvals_in * num_lvals_out;
    // l_totals need int storage, but we account for it in float units for simplicity
    const size_t l_total_size = (max_scale_size * sizeof(int) + sizeof(float) - 1) / sizeof(float);

    const size_t shared_size_feat_m0 = (Cout * num_lvals_out + max_scale_size) * sizeof(float);
    const size_t shared_size_table_m0 = (Cin * num_lvals_in + Cout * num_lvals_out + max_scale_size + l_total_size) * sizeof(float);
    const size_t shared_size_feat_mpos = (Cout * max_out_size + max_scale_size) * sizeof(float);
    const size_t shared_size_table_mpos = (Cin * max_in_size + Cout * max_out_size + max_scale_size + l_total_size) * sizeof(float);

    // Launch m=0 backward_features
    if (log_bins) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_m0", ([&] {
            backward_features_m0_kernel<scalar_t, true><<<B, THREADS, shared_size_feat_m0>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                grad_features.data_ptr<scalar_t>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_m0", ([&] {
            backward_features_m0_kernel<scalar_t, false><<<B, THREADS, shared_size_feat_m0>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                grad_features.data_ptr<scalar_t>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
        }));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Launch m=0 backward_table
    if (log_bins) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_m0", ([&] {
            backward_table_m0_kernel<scalar_t, true><<<B, THREADS, shared_size_table_m0>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                grad_radial_table.data_ptr<float>(),
                grad_distances.data_ptr<float>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_m0", ([&] {
            backward_table_m0_kernel<scalar_t, false><<<B, THREADS, shared_size_table_m0>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                grad_radial_table.data_ptr<float>(),
                grad_distances.data_ptr<float>(),
                block_params_flat.data_ptr<int>(),
                l_lookup_in.data_ptr<int>(),
                l_lookup_out.data_ptr<int>(),
                dims
            );
        }));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // m>0 kernels
    if (mmax > 0) {
        // backward_features m>0
        if (log_bins) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_mpos", ([&] {
                backward_features_mpos_kernel<scalar_t, true><<<B * mmax, THREADS, shared_size_feat_mpos>>>(
                    grad_output.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    grad_features.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
            }));
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_mpos", ([&] {
                backward_features_mpos_kernel<scalar_t, false><<<B * mmax, THREADS, shared_size_feat_mpos>>>(
                    grad_output.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    grad_features.data_ptr<scalar_t>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
            }));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // backward_table m>0
        if (log_bins) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_mpos", ([&] {
                backward_table_mpos_kernel<scalar_t, true><<<B * mmax, THREADS, shared_size_table_mpos>>>(
                    grad_output.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    grad_radial_table.data_ptr<float>(),
                    grad_distances.data_ptr<float>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
            }));
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_mpos", ([&] {
                backward_table_mpos_kernel<scalar_t, false><<<B * mmax, THREADS, shared_size_table_mpos>>>(
                    grad_output.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    grad_radial_table.data_ptr<float>(),
                    grad_distances.data_ptr<float>(),
                    block_params_flat.data_ptr<int>(),
                    l_lookup_in.data_ptr<int>(),
                    l_lookup_out.data_ptr<int>(),
                    mmax,
                    dims
                );
            }));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (radial_table.scalar_type() != torch::kFloat32) {
        grad_radial_table = grad_radial_table.to(radial_table.scalar_type());
    }
    if (distances.scalar_type() != torch::kFloat32) {
        grad_distances = grad_distances.to(distances.scalar_type());
    }

    return {grad_features, grad_radial_table, grad_distances};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda, "Block-diagonal forward (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda, "Block-diagonal backward (CUDA)");
}
