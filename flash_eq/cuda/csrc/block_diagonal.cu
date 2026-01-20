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

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
 * For linear: normalized = (dist - min_val) * inv_bin_width
 * For log: normalized = (log(dist) - log_min) * inv_log_range * num_bins
 */
struct BinningParams {
    float param1;      // min_val (linear) or log_min (log)
    float param2;      // inv_bin_width (linear) or inv_log_range (log)
    int num_bins;
};

/**
 * Compute binning from distance value.
 * Template parameter LOG_BINS selects linear (false) or logarithmic (true) spacing.
 */
template <bool LOG_BINS>
__device__ __forceinline__ InterpState compute_binning(float dist, BinningParams params) {
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
__device__ __forceinline__ float binning_derivative(float dist, BinningParams params) {
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
 */
struct BlockParams {
    int m;         // Magnetic quantum number
    int n_in;      // Number of input l-values contributing to this m
    int n_out;     // Number of output l-values contributing to this m
    int in_off;    // Offset into input feature vector
    int out_off;   // Offset into output feature vector
    int w_off;     // Offset into weight vector
};


//------------------------------------------------------------------------------
// Device Helper Functions
//------------------------------------------------------------------------------

/**
 * Get the l-value for the idx-th element within an m-block.
 * Within m-block, contributions come from l-values with l >= m, in order.
 * @param lvals Array of l-values in the representation
 * @param num_lvals Number of l-values
 * @param m The magnetic quantum number for this block
 * @param idx Index within the m-block (0-indexed)
 * @return The l-value corresponding to this index
 */
__device__ __forceinline__ int get_l_for_index(const int* lvals, int num_lvals, int m, int idx) {
    int count = 0;
    for (int i = 0; i < num_lvals; i++) {
        if (lvals[i] >= m) {
            if (count == idx) return lvals[i];
            count++;
        }
    }
    return 0;  // Should not reach here if idx is valid
}

/**
 * Integer power via repeated multiplication. Much faster than powf for small l.
 */
__device__ __forceinline__ float ipowf(float base, int exp) {
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
__device__ __forceinline__ float solid_harmonic_scale(float distance, float scale, int l) {
    if (l == 0) return 1.0f;
    float weight = distance / (distance + scale);
    return ipowf(weight, l);
}

/**
 * Compute derivative of solid harmonic scale factor w.r.t. distance.
 * d/dr[(r/(r+s))^l] = (r/(r+s))^l * l * s / (r * (r+s))
 * Returns 0.0 for l=0 since scaling is constant.
 */
__device__ __forceinline__ float solid_harmonic_scale_derivative(float distance, float scale, int l) {
    if (l == 0) return 0.0f;
    float r_plus_s = distance + scale;
    float sh_scale = ipowf(distance / r_plus_s, l);
    return sh_scale * l * scale / (distance * r_plus_s);
}

/**
 * Load a complex number from interleaved storage.
 * Storage format: [re_0, im_0, re_1, im_1, ...]
 */
__device__ __forceinline__ Complex load_complex(const float* ptr, int idx) {
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
 * Count how many l-values in lvals have l >= m.
 */
__device__ __forceinline__ int count_l_geq_m(const int* lvals, int num_lvals, int m) {
    int count = 0;
    for (int i = 0; i < num_lvals; i++) {
        if (lvals[i] >= m) count++;
    }
    return count;
}

/**
 * Compute block parameters for magnetic quantum number m from lvals tensors.
 *
 * Block layout in m-first basis:
 *   - m=0: n_in(0) real scalars
 *   - m>0: 2*n_in(m) interleaved real/imag pairs
 *
 * Weight layout:
 *   - m=0: n_out(0) * n_in(0) weights
 *   - m>0: 2 * n_out(m) * n_in(m) weights (a,b pairs for 2x2 rotation)
 */
__device__ __forceinline__ BlockParams compute_block_params(
    int m,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in,
    int num_lvals_out,
    int mmax
) {
    BlockParams bp;
    bp.m = m;
    bp.n_in = count_l_geq_m(lvals_in, num_lvals_in, m);
    bp.n_out = count_l_geq_m(lvals_out, num_lvals_out, m);

    // Compute offsets by summing sizes of all blocks with m' < m
    bp.in_off = 0;
    bp.out_off = 0;
    bp.w_off = 0;

    for (int mp = 0; mp < m; mp++) {
        int n_in_mp = count_l_geq_m(lvals_in, num_lvals_in, mp);
        int n_out_mp = count_l_geq_m(lvals_out, num_lvals_out, mp);

        if (n_in_mp > 0 && n_out_mp > 0) {
            int mult = (mp == 0) ? 1 : 2;
            bp.in_off += mult * n_in_mp;
            bp.out_off += mult * n_out_mp;
            bp.w_off += mult * n_out_mp * n_in_mp;
        }
    }

    return bp;
}


/**
 * Linearly interpolate a weight from the radial table.
 */
template <typename scalar_t>
__device__ __forceinline__ float lerp_weight(
    const scalar_t* __restrict__ table,
    int idx_lo,
    int idx_hi,
    float t,
    float one_minus_t
) {
    return one_minus_t * static_cast<float>(__ldg(&table[idx_lo]))
         + t * static_cast<float>(__ldg(&table[idx_hi]));
}

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
__device__ __forceinline__ int weight_idx_2x2(int o, int i, int n_in, int component) {
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
__device__ __forceinline__ float warp_reduce_sum(float val) {
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
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void forward_m0_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ output,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * Cin * Din;

    for (int i = tid; i < Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + bp.in_off + local_idx]);
    }
    __syncthreads();

    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        // Get output l-value for this index
        const int l_out = get_l_for_index(lvals_out, num_lvals_out, 0, o);

        float acc = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + co * Cin * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * Cin * Wdim + bp.w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * bp.n_in;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                // Get input l-value and compute total scaling exponent
                const int l_in = get_l_for_index(lvals_in, num_lvals_in, 0, i);
                const int l_total = l_in + l_out;

                const int w_idx = w_ci_off + o * bp.n_in + i;
                float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);

                // Apply solid harmonic scaling: w *= (r/(r+scale))^(l_in+l_out)
                if (l_total > 0) {
                    w *= ipowf(sh_weight, l_total);
                }

                acc += w * f_ptr[i];
            }
        }

        output[edge * Cout * Dout + co * Dout + bp.out_off + o] = static_cast<scalar_t>(acc);
    }
}


/**
 * Forward kernel for m>0 blocks (complex-like 2x2 structure).
 * One CUDA block per (edge, m) pair where m ranges from 1 to mmax.
 * Includes solid harmonic scaling: weights multiplied by (r/(r+scale))^(l_in+l_out)
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void forward_mpos_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ output,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    // blk indexes m values from 1 to mmax
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;  // m=1, 2, ..., mmax

    // Compute block parameters for this m
    const BlockParams bp = compute_block_params(m, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int in_size = 2 * bp.n_in;  // Complex pairs

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * Cin * Din + bp.in_off;

    for (int i = tid; i < Cin * in_size; i += THREADS) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + local_idx]);
    }
    __syncthreads();

    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs: out[o] = sum over channels and inputs of rotation(a,b) * f[i]
    for (int out_idx = tid; out_idx < Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        // Get output l-value for this index
        const int l_out = get_l_for_index(lvals_out, num_lvals_out, m, o);

        Complex acc = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + co * Cin * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * Cin * Wdim + bp.w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                // Get input l-value and compute total scaling exponent
                const int l_in = get_l_for_index(lvals_in, num_lvals_in, m, i);
                const int l_total = l_in + l_out;

                const Complex f = load_complex(f_ptr, i);
                const int w_idx = w_ci_off + weight_idx_2x2(o, i, bp.n_in, 0);

                float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);
                float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      interp.t, interp.one_minus_t);

                // Apply solid harmonic scaling: (a,b) *= (r/(r+scale))^(l_in+l_out)
                // l_total >= 2 for m>0 blocks (since l >= m >= 1 for both in and out)
                const float scale_factor = ipowf(sh_weight, l_total);
                a *= scale_factor;
                b *= scale_factor;

                complex_mul_add(a, b, f, acc);
            }
        }

        // Store output: output[edge, co, out_off + 2*o : out_off + 2*o + 2]
        scalar_t* out_ptr = &output[edge * Cout * Dout + co * Dout + bp.out_off];
        store_complex(out_ptr, o, acc);
    }
}


//------------------------------------------------------------------------------
// Backward Feature Kernels
//------------------------------------------------------------------------------

/**
 * Backward kernel for m=0: compute grad_features.
 * Includes solid harmonic scaling on weights.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_features_m0_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * Cout * Dout;

    for (int i = tid; i < Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + bp.out_off + local_idx]);
    }
    __syncthreads();

    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        // Get input l-value
        const int l_in = get_l_for_index(lvals_in, num_lvals_in, 0, i);

        float grad = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + ci * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * Wdim + bp.w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * bp.n_out;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                // Get output l-value and compute total scaling exponent
                const int l_out = get_l_for_index(lvals_out, num_lvals_out, 0, o);
                const int l_total = l_in + l_out;

                const int w_idx = w_co_off + o * bp.n_in + i;
                float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);

                // Apply solid harmonic scaling
                if (l_total > 0) {
                    w *= ipowf(sh_weight, l_total);
                }

                grad += w * go_ptr[o];
            }
        }

        grad_features[edge * Cin * Din + ci * Din + bp.in_off + i] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward kernel for m>0: compute grad_features.
 * Computes: grad_f = W^T @ grad_out (transpose of rotation).
 * Includes solid harmonic scaling on weights.
 */
template <typename scalar_t, bool LOG_BINS>
__global__ void backward_features_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const scalar_t* __restrict__ distances,
    float sh_scale,
    BinningParams bin_params,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;

    // Compute block parameters for this m
    const BlockParams bp = compute_block_params(m, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int out_size = 2 * bp.n_out;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * Cout * Dout + bp.out_off;

    for (int i = tid; i < Cout * out_size; i += THREADS) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + local_idx]);
    }
    __syncthreads();

    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features for each complex input
    for (int in_idx = tid; in_idx < Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        // Get input l-value
        const int l_in = get_l_for_index(lvals_in, num_lvals_in, m, i);

        Complex grad_f = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + ci * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * Wdim + bp.w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * out_size;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                // Get output l-value and compute total scaling exponent
                const int l_out = get_l_for_index(lvals_out, num_lvals_out, m, o);
                const int l_total = l_in + l_out;

                const Complex go = load_complex(go_ptr, o);
                const int w_idx = w_co_off + weight_idx_2x2(o, i, bp.n_in, 0);

                float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                      interp.t, interp.one_minus_t);
                float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                      interp.t, interp.one_minus_t);

                // Apply solid harmonic scaling
                const float scale_factor = ipowf(sh_weight, l_total);
                a *= scale_factor;
                b *= scale_factor;

                complex_mul_add_transpose(a, b, go, grad_f);
            }
        }

        // Store both real and imaginary gradients
        scalar_t* gf_ptr = &grad_features[edge * Cin * Din + ci * Din + bp.in_off];
        store_complex(gf_ptr, i, grad_f);
    }
}


//------------------------------------------------------------------------------
// Backward Table Kernels
//------------------------------------------------------------------------------

/**
 * Backward kernel for m=0: compute grad_radial_table and grad_distance.
 * Includes solid harmonic scaling in gradient computation.
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
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int table_stride = Cout * Cin * Wdim;
    const int w_block_size = bp.n_out * bp.n_in;

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * bp.n_in;

    // Load features
    const int64_t feat_base = edge * Cin * Din;
    for (int i = tid; i < Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + bp.in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * Cout * Dout;
    for (int i = tid; i < Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + bp.out_off + local_idx]);
    }
    __syncthreads();

    float local_grad_t = 0.0f;
    float local_grad_sh = 0.0f;  // Gradient through solid harmonic scaling

    // Compute weight gradients
    for (int w_idx = tid; w_idx < Cout * Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;
        const int o = w_local / bp.n_in;
        const int i = w_local % bp.n_in;

        // Get l-values and compute scaling factor
        const int l_in = get_l_for_index(lvals_in, num_lvals_in, 0, i);
        const int l_out = get_l_for_index(lvals_out, num_lvals_out, 0, o);
        const int l_total = l_in + l_out;
        const float scale_factor = (l_total > 0) ? ipowf(sh_weight, l_total) : 1.0f;

        const float f = feat_shared[ci * bp.n_in + i];
        const float go = grad_shared[co * bp.n_out + o];
        // grad_w includes the scale factor since output = w * scale_factor * f
        const float grad_w = scale_factor * f * go;

        const int table_idx = co * Cin * Wdim + ci * Wdim + bp.w_off + w_local;
        const int addr_lo = interp.idx_lo * table_stride + table_idx;
        const int addr_hi = interp.idx_hi * table_stride + table_idx;

        atomicAdd(&grad_radial_table[addr_lo], interp.one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[addr_hi], interp.t * grad_w);

        const float w_lo = static_cast<float>(__ldg(&radial_table[addr_lo]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[addr_hi]));
        const float interp_weight = interp.one_minus_t * w_lo + interp.t * w_hi;

        // Gradient through interpolation: d(output)/d(t) * d(t)/d(dist)
        local_grad_t += (w_hi - w_lo) * grad_w;

        // Gradient through solid harmonic scaling: d(output)/d(scale) * d(scale)/d(dist)
        // output = interp_weight * scale_factor * f, so d(output)/d(scale) = interp_weight * f * go
        if (l_total > 0) {
            local_grad_sh += interp_weight * f * go * solid_harmonic_scale_derivative(dist, sh_scale, l_total);
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
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim
) {
    const int blk = blockIdx.x % mmax;
    const int64_t edge = blockIdx.x / mmax;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;
    const int m = blk + 1;

    // Compute block parameters for this m
    const BlockParams bp = compute_block_params(m, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load distance for this edge and compute binning + solid harmonic scaling
    const float dist = static_cast<float>(distances[edge]);
    const float sh_weight = dist / (dist + sh_scale);
    const InterpState interp = compute_binning<LOG_BINS>(dist, bin_params);

    const int in_size = 2 * bp.n_in;
    const int out_size = 2 * bp.n_out;
    const int weights_per_pair = 2;  // 'a' and 'b' for each (o, i) pair
    const int num_weights = weights_per_pair * bp.n_out * bp.n_in;
    const int table_stride = Cout * Cin * Wdim;

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * in_size;

    // Load features
    const int64_t feat_base = edge * Cin * Din + bp.in_off;
    for (int idx = tid; idx < Cin * in_size; idx += THREADS) {
        const int ci = idx / in_size;
        const int local_idx = idx % in_size;
        feat_shared[idx] = static_cast<float>(features[feat_base + ci * Din + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * Cout * Dout + bp.out_off;
    for (int idx = tid; idx < Cout * out_size; idx += THREADS) {
        const int co = idx / out_size;
        const int local_idx = idx % out_size;
        grad_shared[idx] = static_cast<float>(grad_output[grad_base + co * Dout + local_idx]);
    }
    __syncthreads();

    float local_grad_t = 0.0f;
    float local_grad_sh = 0.0f;  // Gradient through solid harmonic scaling

    // Iterate over all weight elements: (co, ci, o, i, component)
    // Each weight pair (a, b) corresponds to one (output, input) connection
    for (int w_idx = tid; w_idx < Cout * Cin * num_weights; w_idx += THREADS) {
        // Decode flat index into (co, ci, o, i, is_b)
        const int co = w_idx / (Cin * num_weights);
        const int ci = (w_idx / num_weights) % Cin;
        const int w_local = w_idx % num_weights;
        const int pair_idx = w_local / 2;
        const int is_b = w_local % 2;  // 0 = 'a' (diagonal), 1 = 'b' (off-diagonal)
        const int o = pair_idx / bp.n_in;
        const int i = pair_idx % bp.n_in;

        // Get l-values and compute scaling factor
        const int l_in = get_l_for_index(lvals_in, num_lvals_in, m, i);
        const int l_out = get_l_for_index(lvals_out, num_lvals_out, m, o);
        const int l_total = l_in + l_out;
        const float scale_factor = ipowf(sh_weight, l_total);

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
        const int table_idx = co * Cin * Wdim + ci * Wdim + bp.w_off + w_local;
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

        // Gradient through solid harmonic scaling: d(output)/d(scale) * d(scale)/d(dist)
        // output = interp_weight * scale_factor * f, d(output)/d(scale) = interp_weight * f * go
        // Note: l_total >= 2 for m>0 (since l >= m >= 1 for both in and out)
        local_grad_sh += interp_weight * unscaled_grad_w * solid_harmonic_scale_derivative(dist, sh_scale, l_total);
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

    BinningParams bin_params = {bin_param1, bin_param2, num_bins};

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Shared memory size: conservative estimate using max_in_size
    const size_t shared_size_m0 = Cin * num_lvals_in * sizeof(float);
    const size_t shared_size_mpos = Cin * max_in_size * sizeof(float);

    // Launch m=0 kernel with template dispatch for log_bins
    if (log_bins) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
            forward_m0_kernel<scalar_t, true><<<B, THREADS, shared_size_m0>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                output.data_ptr<scalar_t>(),
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout_int, Din, dim_out, Wdim
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
            forward_m0_kernel<scalar_t, false><<<B, THREADS, shared_size_m0>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(),
                sh_scale,
                bin_params,
                output.data_ptr<scalar_t>(),
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout_int, Din, dim_out, Wdim
            );
        }));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Launch m>0 kernel if mmax > 0
    if (mmax > 0) {
        if (log_bins) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
                forward_mpos_kernel<scalar_t, true><<<B * mmax, THREADS, shared_size_mpos>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout_int, Din, dim_out, Wdim
                );
            }));
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
                forward_mpos_kernel<scalar_t, false><<<B * mmax, THREADS, shared_size_mpos>>>(
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    distances.data_ptr<scalar_t>(),
                    sh_scale,
                    bin_params,
                    output.data_ptr<scalar_t>(),
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout_int, Din, dim_out, Wdim
                );
            }));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

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

    BinningParams bin_params = {bin_param1, bin_param2, num_bins};

    auto grad_features = torch::zeros({B, Cin, Din}, features.options());
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_distances = torch::zeros({B}, distances.options().dtype(torch::kFloat32));

    // Shared memory sizes
    const size_t shared_size_feat_m0 = Cout * num_lvals_out * sizeof(float);
    const size_t shared_size_table_m0 = (Cin * num_lvals_in + Cout * num_lvals_out) * sizeof(float);
    const size_t shared_size_feat_mpos = Cout * max_out_size * sizeof(float);
    const size_t shared_size_table_mpos = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

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
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim
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
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim
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
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim
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
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim
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
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout, Din, Dout, Wdim
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
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout, Din, Dout, Wdim
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
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout, Din, Dout, Wdim
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
                    lvals_in.data_ptr<int>(),
                    lvals_out.data_ptr<int>(),
                    num_lvals_in, num_lvals_out, mmax,
                    B, Cin, Cout, Din, Dout, Wdim
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
