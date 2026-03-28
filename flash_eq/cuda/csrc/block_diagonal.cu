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

#include <ATen/cuda/Atomic.cuh>

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
template <typename T>
struct Complex {
    T re, im;
};

/**
 * Interpolation state for binned radial weights.
 */
template <typename T>
struct InterpState {
    int idx_lo;
    int idx_hi;
    T t;
    T one_minus_t;
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
template <bool LOG_BINS, typename T>
[[nodiscard]] __device__ __forceinline__ InterpState<T> compute_binning(T dist, BinningParams params) {
    // Use float to find bin index (integer result, float precision sufficient)
    float normalized;
    if constexpr (LOG_BINS) {
        normalized = (logf(fmaxf(static_cast<float>(dist), expf(params.param1))) - params.param1)
                     * params.param2 * params.num_bins;
    } else {
        normalized = (static_cast<float>(dist) - params.param1) * params.param2;
    }
    normalized = fminf(fmaxf(normalized, 0.0f), static_cast<float>(params.num_bins));

    InterpState<T> state;
    state.idx_lo = min(static_cast<int>(floorf(normalized)), params.num_bins - 1);
    state.idx_hi = min(state.idx_lo + 1, params.num_bins);

    if constexpr (LOG_BINS) {
        // Interpolate linearly in dist-space between bin edges (full precision)
        T inv_log_bin_width = static_cast<T>(params.param2 * params.num_bins);
        T log_edge_lo = static_cast<T>(params.param1) + static_cast<T>(state.idx_lo) / inv_log_bin_width;
        T edge_lo = exp(static_cast<double>(log_edge_lo));
        T edge_hi = exp(static_cast<double>(log_edge_lo + static_cast<T>(1) / inv_log_bin_width));
        T clamped_dist = dist > edge_lo ? dist : edge_lo;
        state.t = (clamped_dist - edge_lo) / (edge_hi - edge_lo);
        state.t = state.t < static_cast<T>(1) ? state.t : static_cast<T>(1);
    } else {
        state.t = static_cast<T>(normalized) - static_cast<T>(state.idx_lo);
    }
    state.one_minus_t = static_cast<T>(1) - state.t;
    return state;
}

/**
 * Compute derivative of interpolation weight w.r.t. distance.
 * Used in backward pass to convert grad_interp_weight to grad_distance.
 */
template <bool LOG_BINS, typename T>
[[nodiscard]] __device__ __forceinline__ T binning_derivative(T dist, BinningParams params) {
    if constexpr (LOG_BINS) {
        // dt/d(dist) = 1 / (edge_hi - edge_lo)
        float inv_log_bin_width = params.param2 * params.num_bins;
        float log_normalized = (logf(fmaxf(static_cast<float>(dist), expf(params.param1))) - params.param1)
                               * inv_log_bin_width;
        log_normalized = fminf(fmaxf(log_normalized, 0.0f), static_cast<float>(params.num_bins));
        int idx_lo = min(static_cast<int>(floorf(log_normalized)), params.num_bins - 1);
        T t_inv_lbw = static_cast<T>(inv_log_bin_width);
        T log_edge_lo = static_cast<T>(params.param1) + static_cast<T>(idx_lo) / t_inv_lbw;
        T edge_lo = exp(static_cast<double>(log_edge_lo));
        T edge_hi = exp(static_cast<double>(log_edge_lo + static_cast<T>(1) / t_inv_lbw));
        return static_cast<T>(1) / (edge_hi - edge_lo);
    } else {
        return static_cast<T>(params.param2);
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
template <typename T>
[[nodiscard]] __device__ __forceinline__ T ipow(T base, int exp) {
    T result = static_cast<T>(1);
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
    return ipow(weight, l);
}

/**
 * Compute derivative of solid harmonic scale factor w.r.t. distance.
 * d/dr[(r/(r+s))^l] = (r/(r+s))^l * l * s / (r * (r+s))
 * Returns 0.0 for l=0 since scaling is constant.
 */
[[nodiscard]] __device__ __forceinline__ float solid_harmonic_scale_derivative(float distance, float scale, int l) {
    if (l == 0) return 0.0f;
    float r_plus_s = distance + scale;
    float sh_scale = ipow(distance / r_plus_s, l);
    return sh_scale * l * scale / (distance * r_plus_s);
}

/**
 * Load a complex number from interleaved storage.
 * Storage format: [re_0, im_0, re_1, im_1, ...]
 */
template <typename T>
[[nodiscard]] __device__ __forceinline__ Complex<T> load_complex(const T* ptr, int idx) {
    return {ptr[2 * idx], ptr[2 * idx + 1]};
}

/**
 * Store a complex number to interleaved storage.
 */
template <typename T>
__device__ __forceinline__ void store_complex(T* ptr, int idx, Complex<T> c) {
    ptr[2 * idx] = c.re;
    ptr[2 * idx + 1] = c.im;
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
template <typename T>
[[nodiscard]] __device__ __forceinline__ T lerp_weight(
    const T* __restrict__ table,
    int idx_lo,
    int idx_hi,
    T t,
    T one_minus_t
) {
    return one_minus_t * static_cast<T>(__ldg(&table[idx_lo]))
         + t * static_cast<T>(__ldg(&table[idx_hi]));
}

// ============================================================================
// Shared kernel helpers
// ============================================================================

/**
 * Bundled edge state: distance, solid harmonic weight, and interpolation indices.
 */
template <typename T>
struct EdgeState {
    T dist;
    T sh_weight;
    InterpState<T> interp;
};

/**
 * Setup edge state from raw distance value.
 * Computes solid harmonic weight and binning indices.
 */
template <bool LOG_BINS, typename T>
[[nodiscard]] __device__ __forceinline__ EdgeState<T> setup_edge(
    const T* __restrict__ distances,
    int64_t edge,
    float sh_scale,
    BinningParams bin_params
) {
    EdgeState<T> s;
    s.dist = distances[edge];
    s.sh_weight = s.dist / (s.dist + static_cast<T>(sh_scale));
    s.interp = compute_binning<LOG_BINS>(s.dist, bin_params);
    return s;
}

/**
 * Precompute scale factors for all (i, o) pairs into shared memory.
 * Scale = ipow(sh_weight, l_in + l_out) for solid harmonic suppression.
 */
template <typename T>
__device__ __forceinline__ void precompute_scales(
    T* scale_shared,
    T sh_weight,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    int m,
    int n_in,
    int n_out,
    int tid
) {
    const int scale_size = n_in * n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % n_in;
        const int o = idx / n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, m, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, m, o);
        scale_shared[idx] = ipow(sh_weight, l_in + l_out);
    }
}

/**
 * Precompute scale factors AND l_totals for backward_table kernels.
 * Stores both scale factors and l_total values (needed for gradient computation).
 */
template <typename T>
__device__ __forceinline__ void precompute_scales_with_l(
    T* scale_shared,
    int* l_total_shared,
    T sh_weight,
    const int* __restrict__ l_lookup_in,
    const int* __restrict__ l_lookup_out,
    int m,
    int n_in,
    int n_out,
    int tid
) {
    const int scale_size = n_in * n_out;
    for (int idx = tid; idx < scale_size; idx += THREADS) {
        const int i = idx % n_in;
        const int o = idx / n_in;
        const int l_in = get_l_for_index_fast(l_lookup_in, m, i);
        const int l_out = get_l_for_index_fast(l_lookup_out, m, o);
        const int l_total = l_in + l_out;
        scale_shared[idx] = ipow(sh_weight, l_total);
        l_total_shared[idx] = l_total;
    }
}

/**
 * Compute weight table base indices for interpolation.
 */
__device__ __forceinline__ void compute_weight_bases(
    const InterpState<float>& interp,
    int table_stride,
    int co,
    int Cin,
    int Wdim,
    int w_off,
    int& w_base_lo,
    int& w_base_hi
) {
    w_base_lo = interp.idx_lo * table_stride + co * Cin * Wdim + w_off;
    w_base_hi = interp.idx_hi * table_stride + co * Cin * Wdim + w_off;
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
template <typename T>
__device__ __forceinline__ void complex_mul_add(T a, T b, Complex<T> f, Complex<T>& acc) {
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
template <typename T>
__device__ __forceinline__ void complex_mul_add_transpose(T a, T b, Complex<T> go, Complex<T>& grad) {
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
template <typename T>
__device__ __forceinline__ void compute_weight_gradient(Complex<T> f, Complex<T> go, T& grad_a, T& grad_b) {
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

[[nodiscard]] __device__ __forceinline__ double warp_reduce_sum(double val) {
    // Split double into two 32-bit ints for shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int lo = __double2loint(val);
        int hi = __double2hiint(val);
        lo = __shfl_down_sync(0xffffffff, lo, offset);
        hi = __shfl_down_sync(0xffffffff, hi, offset);
        val += __hiloint2double(hi, lo);
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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    // Shared memory layout: [features | scale_factors]
    extern __shared__ char shared_raw[];
    scalar_t* feat_shared = reinterpret_cast<scalar_t*>(shared_raw);
    scalar_t* scale_shared = feat_shared + dims.Cin * bp.n_in;

    // Precompute scale factors for all (i, o) pairs
    precompute_scales(scale_shared, es.sh_weight, l_lookup_in, l_lookup_out,
                      0, bp.n_in, bp.n_out, tid);

    // Load features into shared memory
    const int64_t feat_base = edge * dims.Cin * dims.Din;
    for (int i = tid; i < dims.Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = features[feat_base + ci * dims.Din + bp.in_off + local_idx];
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < dims.Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        scalar_t acc = 0;
        const int w_base_lo = es.interp.idx_lo * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;
        const int w_base_hi = es.interp.idx_hi * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;

        for (int ci = 0; ci < dims.Cin; ci++) {
            const scalar_t* f_ptr = feat_shared + ci * bp.n_in;
            const int w_ci_off = ci * dims.Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const int w_idx = w_ci_off + o * bp.n_in + i;
                scalar_t w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      es.interp.t, es.interp.one_minus_t);

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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    const int in_size = 2 * bp.n_in;  // Complex pairs

    // Shared memory layout: [features | scale_factors]
    extern __shared__ char shared_raw2[];
    scalar_t* feat_shared = reinterpret_cast<scalar_t*>(shared_raw2);
    scalar_t* scale_shared = feat_shared + dims.Cin * in_size;

    // Precompute scale factors for all (i, o) pairs
    precompute_scales(scale_shared, es.sh_weight, l_lookup_in, l_lookup_out,
                      m, bp.n_in, bp.n_out, tid);

    // Load features into shared memory
    const int64_t feat_base = edge * dims.Cin * dims.Din + bp.in_off;
    for (int i = tid; i < dims.Cin * in_size; i += THREADS) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = features[feat_base + ci * dims.Din + local_idx];
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute outputs: out[o] = sum over channels and inputs of rotation(a,b) * f[i]
    for (int out_idx = tid; out_idx < dims.Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        Complex<scalar_t> acc = {0, 0};
        const int w_base_lo = es.interp.idx_lo * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;
        const int w_base_hi = es.interp.idx_hi * table_stride + co * dims.Cin * dims.Wdim + bp.w_off;

        for (int ci = 0; ci < dims.Cin; ci++) {
            const scalar_t* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * dims.Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const auto f = load_complex(f_ptr, i);
                const int w_idx = w_ci_off + weight_idx_2x2(o, i, bp.n_in, 0);

                scalar_t a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      es.interp.t, es.interp.one_minus_t);
                scalar_t b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      es.interp.t, es.interp.one_minus_t);

                // Apply precomputed solid harmonic scaling
                const scalar_t scale_factor = scale_shared[o * bp.n_in + i];
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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    // Shared memory layout: [grad_output | scale_factors]
    extern __shared__ char shared_raw3[];
    scalar_t* grad_shared = reinterpret_cast<scalar_t*>(shared_raw3);
    scalar_t* scale_shared = grad_shared + dims.Cout * bp.n_out;

    // Precompute scale factors for all (i, o) pairs
    precompute_scales(scale_shared, es.sh_weight, l_lookup_in, l_lookup_out,
                      0, bp.n_in, bp.n_out, tid);

    // Load grad_output into shared memory
    const int64_t grad_base = edge * dims.Cout * dims.Dout;
    for (int i = tid; i < dims.Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = grad_output[grad_base + co * dims.Dout + bp.out_off + local_idx];
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < dims.Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        scalar_t grad = 0;
        const int w_base_lo = es.interp.idx_lo * table_stride + ci * dims.Wdim + bp.w_off;
        const int w_base_hi = es.interp.idx_hi * table_stride + ci * dims.Wdim + bp.w_off;

        for (int co = 0; co < dims.Cout; co++) {
            const scalar_t* go_ptr = grad_shared + co * bp.n_out;
            const int w_co_off = co * dims.Cin * dims.Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const int w_idx = w_co_off + o * bp.n_in + i;
                scalar_t w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      es.interp.t, es.interp.one_minus_t);

                // Apply precomputed solid harmonic scaling
                w *= scale_shared[o * bp.n_in + i];

                grad += w * go_ptr[o];
            }
        }

        grad_features[edge * dims.Cin * dims.Din + ci * dims.Din + bp.in_off + i] = grad;
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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    const int out_size = 2 * bp.n_out;

    // Shared memory layout: [grad_output | scale_factors]
    extern __shared__ char shared_raw4[];
    scalar_t* grad_shared = reinterpret_cast<scalar_t*>(shared_raw4);
    scalar_t* scale_shared = grad_shared + dims.Cout * out_size;

    // Precompute scale factors for all (i, o) pairs
    precompute_scales(scale_shared, es.sh_weight, l_lookup_in, l_lookup_out,
                      m, bp.n_in, bp.n_out, tid);

    // Load grad_output into shared memory
    const int64_t grad_base = edge * dims.Cout * dims.Dout + bp.out_off;
    for (int i = tid; i < dims.Cout * out_size; i += THREADS) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = grad_output[grad_base + co * dims.Dout + local_idx];
    }
    __syncthreads();

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;

    // Compute grad_features for each complex input
    for (int in_idx = tid; in_idx < dims.Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        Complex<scalar_t> grad_f = {0, 0};
        const int w_base_lo = es.interp.idx_lo * table_stride + ci * dims.Wdim + bp.w_off;
        const int w_base_hi = es.interp.idx_hi * table_stride + ci * dims.Wdim + bp.w_off;

        for (int co = 0; co < dims.Cout; co++) {
            const scalar_t* go_ptr = grad_shared + co * out_size;
            const int w_co_off = co * dims.Cin * dims.Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const auto go = load_complex(go_ptr, o);
                const int w_idx = w_co_off + weight_idx_2x2(o, i, bp.n_in, 0);

                scalar_t a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      es.interp.t, es.interp.one_minus_t);
                scalar_t b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      es.interp.t, es.interp.one_minus_t);

                // Apply precomputed solid harmonic scaling
                const scalar_t scale_factor = scale_shared[o * bp.n_in + i];
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
    scalar_t* __restrict__ grad_radial_table,
    scalar_t* __restrict__ grad_distances,
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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;
    const int w_block_size = bp.n_out * bp.n_in;
    const int scale_size = bp.n_in * bp.n_out;

    // Shared memory layout: [features | grad_output | scale_factors | l_totals]
    extern __shared__ char shared_raw5[];
    scalar_t* feat_shared = reinterpret_cast<scalar_t*>(shared_raw5);
    scalar_t* grad_shared = feat_shared + dims.Cin * bp.n_in;
    scalar_t* scale_shared = grad_shared + dims.Cout * bp.n_out;
    int* l_total_shared = reinterpret_cast<int*>(scale_shared + scale_size);

    // Precompute scale factors and l_totals for all (i, o) pairs
    precompute_scales_with_l(scale_shared, l_total_shared, es.sh_weight,
                             l_lookup_in, l_lookup_out, 0, bp.n_in, bp.n_out, tid);

    // Load features
    const int64_t feat_base = edge * dims.Cin * dims.Din;
    for (int i = tid; i < dims.Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = features[feat_base + ci * dims.Din + bp.in_off + local_idx];
    }

    // Load grad_output
    const int64_t grad_base = edge * dims.Cout * dims.Dout;
    for (int i = tid; i < dims.Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = grad_output[grad_base + co * dims.Dout + bp.out_off + local_idx];
    }
    __syncthreads();

    // Precompute derivative factor: sh_scale / (dist * (dist + sh_scale))
    const scalar_t deriv_factor = static_cast<scalar_t>(sh_scale) / (es.dist * (es.dist + static_cast<scalar_t>(sh_scale)));

    scalar_t local_grad_t = 0;
    scalar_t local_grad_sh = 0;

    // Compute weight gradients
    for (int w_idx = tid; w_idx < dims.Cout * dims.Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (dims.Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % dims.Cin;
        const int w_local = w_idx % w_block_size;
        const int o = w_local / bp.n_in;
        const int i = w_local % bp.n_in;

        const int scale_idx = o * bp.n_in + i;
        const scalar_t scale_factor = scale_shared[scale_idx];
        const int l_total = l_total_shared[scale_idx];

        const scalar_t f = feat_shared[ci * bp.n_in + i];
        const scalar_t go = grad_shared[co * bp.n_out + o];
        const scalar_t grad_w = scale_factor * f * go;

        const int table_idx = co * dims.Cin * dims.Wdim + ci * dims.Wdim + bp.w_off + w_local;
        const int addr_lo = es.interp.idx_lo * table_stride + table_idx;
        const int addr_hi = es.interp.idx_hi * table_stride + table_idx;

        gpuAtomicAdd(&grad_radial_table[addr_lo], es.interp.one_minus_t * grad_w);
        gpuAtomicAdd(&grad_radial_table[addr_hi], es.interp.t * grad_w);

        const scalar_t w_lo = static_cast<scalar_t>(__ldg(&radial_table[addr_lo]));
        const scalar_t w_hi = static_cast<scalar_t>(__ldg(&radial_table[addr_hi]));
        const scalar_t interp_weight = es.interp.one_minus_t * w_lo + es.interp.t * w_hi;

        local_grad_t += (w_hi - w_lo) * grad_w;

        if (l_total > 0) {
            local_grad_sh += interp_weight * f * go * scale_factor * static_cast<scalar_t>(l_total) * deriv_factor;
        }
    }

    // Warp reduction and atomic add for grad_distance
    local_grad_t = warp_reduce_sum(local_grad_t);
    local_grad_sh = warp_reduce_sum(local_grad_sh);
    if ((tid % 32) == 0) {
        const scalar_t grad_dist = local_grad_t * binning_derivative<LOG_BINS>(es.dist, bin_params) + local_grad_sh;
        gpuAtomicAdd(&grad_distances[edge], grad_dist);
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
    scalar_t* __restrict__ grad_radial_table,
    scalar_t* __restrict__ grad_distances,
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

    // Setup edge state: distance, solid harmonic weight, interpolation
    const auto es = setup_edge<LOG_BINS>(distances, edge, sh_scale, bin_params);

    const int in_size = 2 * bp.n_in;
    const int out_size = 2 * bp.n_out;
    const int weights_per_pair = 2;
    const int num_weights = weights_per_pair * bp.n_out * bp.n_in;
    const int table_stride = dims.Cout * dims.Cin * dims.Wdim;
    const int scale_size = bp.n_in * bp.n_out;

    // Shared memory layout: [features | grad_output | scale_factors | l_totals]
    extern __shared__ char shared_raw6[];
    scalar_t* feat_shared = reinterpret_cast<scalar_t*>(shared_raw6);
    scalar_t* grad_shared = feat_shared + dims.Cin * in_size;
    scalar_t* scale_shared = grad_shared + dims.Cout * out_size;
    int* l_total_shared = reinterpret_cast<int*>(scale_shared + scale_size);

    // Precompute scale factors and l_totals for all (i, o) pairs
    precompute_scales_with_l(scale_shared, l_total_shared, es.sh_weight,
                             l_lookup_in, l_lookup_out, m, bp.n_in, bp.n_out, tid);

    // Load features
    const int64_t feat_base = edge * dims.Cin * dims.Din + bp.in_off;
    for (int idx = tid; idx < dims.Cin * in_size; idx += THREADS) {
        const int ci = idx / in_size;
        const int local_idx = idx % in_size;
        feat_shared[idx] = features[feat_base + ci * dims.Din + local_idx];
    }

    // Load grad_output
    const int64_t grad_base = edge * dims.Cout * dims.Dout + bp.out_off;
    for (int idx = tid; idx < dims.Cout * out_size; idx += THREADS) {
        const int co = idx / out_size;
        const int local_idx = idx % out_size;
        grad_shared[idx] = grad_output[grad_base + co * dims.Dout + local_idx];
    }
    __syncthreads();

    const scalar_t deriv_factor = static_cast<scalar_t>(sh_scale) / (es.dist * (es.dist + static_cast<scalar_t>(sh_scale)));

    scalar_t local_grad_t = 0;
    scalar_t local_grad_sh = 0;

    for (int w_idx = tid; w_idx < dims.Cout * dims.Cin * num_weights; w_idx += THREADS) {
        const int co = w_idx / (dims.Cin * num_weights);
        const int ci = (w_idx / num_weights) % dims.Cin;
        const int w_local = w_idx % num_weights;
        const int pair_idx = w_local / 2;
        const int is_b = w_local % 2;
        const int o = pair_idx / bp.n_in;
        const int i = pair_idx % bp.n_in;

        const int scale_idx = o * bp.n_in + i;
        const scalar_t scale_factor = scale_shared[scale_idx];
        const int l_total = l_total_shared[scale_idx];

        const auto f = load_complex(feat_shared + ci * in_size, i);
        const auto go = load_complex(grad_shared + co * out_size, o);

        scalar_t grad_a, grad_b;
        compute_weight_gradient(f, go, grad_a, grad_b);
        const scalar_t grad_w = scale_factor * ((is_b == 0) ? grad_a : grad_b);
        const scalar_t unscaled_grad_w = (is_b == 0) ? grad_a : grad_b;

        const int table_idx = co * dims.Cin * dims.Wdim + ci * dims.Wdim + bp.w_off + w_local;
        const int addr_lo = es.interp.idx_lo * table_stride + table_idx;
        const int addr_hi = es.interp.idx_hi * table_stride + table_idx;

        gpuAtomicAdd(&grad_radial_table[addr_lo], es.interp.one_minus_t * grad_w);
        gpuAtomicAdd(&grad_radial_table[addr_hi], es.interp.t * grad_w);

        const scalar_t w_lo = static_cast<scalar_t>(__ldg(&radial_table[addr_lo]));
        const scalar_t w_hi = static_cast<scalar_t>(__ldg(&radial_table[addr_hi]));
        const scalar_t interp_weight = es.interp.one_minus_t * w_lo + es.interp.t * w_hi;

        local_grad_t += (w_hi - w_lo) * grad_w;

        local_grad_sh += interp_weight * unscaled_grad_w * scale_factor * static_cast<scalar_t>(l_total) * deriv_factor;
    }

    local_grad_t = warp_reduce_sum(local_grad_t);
    local_grad_sh = warp_reduce_sum(local_grad_sh);
    if ((tid % 32) == 0) {
        const scalar_t grad_dist = local_grad_t * binning_derivative<LOG_BINS>(es.dist, bin_params) + local_grad_sh;
        gpuAtomicAdd(&grad_distances[edge], grad_dist);
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
    const size_t fwd_elem_size = features.element_size();
    const size_t shared_size_m0 = (Cin * num_lvals_in + max_scale_size) * fwd_elem_size;
    const size_t shared_size_mpos = (Cin * max_in_size + max_scale_size) * fwd_elem_size;

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
                                           radial_table.options());
    auto grad_distances = torch::zeros({B}, distances.options());

    // Shared memory sizes: data + scale_factors + l_totals (for backward_table)
    const size_t max_scale_size = num_lvals_in * num_lvals_out;
    const size_t elem_size = features.element_size();
    const size_t l_total_bytes = max_scale_size * sizeof(int);

    const size_t shared_size_feat_m0 = (Cout * num_lvals_out + max_scale_size) * elem_size;
    const size_t shared_size_table_m0 = (Cin * num_lvals_in + Cout * num_lvals_out + max_scale_size) * elem_size + l_total_bytes;
    const size_t shared_size_feat_mpos = (Cout * max_out_size + max_scale_size) * elem_size;
    const size_t shared_size_table_mpos = (Cin * max_in_size + Cout * max_out_size + max_scale_size) * elem_size + l_total_bytes;

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
                grad_radial_table.data_ptr<scalar_t>(),
                grad_distances.data_ptr<scalar_t>(),
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
                grad_radial_table.data_ptr<scalar_t>(),
                grad_distances.data_ptr<scalar_t>(),
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
                    grad_radial_table.data_ptr<scalar_t>(),
                    grad_distances.data_ptr<scalar_t>(),
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
                    grad_radial_table.data_ptr<scalar_t>(),
                    grad_distances.data_ptr<scalar_t>(),
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

    return {grad_features, grad_radial_table, grad_distances};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda, "Block-diagonal forward (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda, "Block-diagonal backward (CUDA)");
}
