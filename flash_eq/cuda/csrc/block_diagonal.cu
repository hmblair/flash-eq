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
 * Load interpolation state for an edge.
 */
template <typename scalar_t>
__device__ __forceinline__ InterpState load_interp_state(
    int64_t edge,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    int num_bins
) {
    InterpState state;
    state.idx_lo = bin_lo[edge];
    state.idx_hi = min(state.idx_lo + 1, num_bins);
    state.t = static_cast<float>(interp_weight[edge]);
    state.one_minus_t = 1.0f - state.t;
    return state;
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
 */
template <typename scalar_t>
__global__ void forward_m0_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ output,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * Cin * Din;

    for (int i = tid; i < Cin * bp.n_in; i += THREADS) {
        const int ci = i / bp.n_in;
        const int local_idx = i % bp.n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + bp.in_off + local_idx]);
    }
    __syncthreads();

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);
    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        float acc = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + co * Cin * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * Cin * Wdim + bp.w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * bp.n_in;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const int w_idx = w_ci_off + o * bp.n_in + i;
                const float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                            interp.t, interp.one_minus_t);
                acc += w * f_ptr[i];
            }
        }

        output[edge * Cout * Dout + co * Dout + bp.out_off + o] = static_cast<scalar_t>(acc);
    }
}


/**
 * Forward kernel for m>0 blocks (complex-like 2x2 structure).
 * One CUDA block per (edge, m) pair where m ranges from 1 to mmax.
 */
template <typename scalar_t>
__global__ void forward_mpos_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ output,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
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

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);
    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs: out[o] = sum over channels and inputs of rotation(a,b) * f[i]
    for (int out_idx = tid; out_idx < Cout * bp.n_out; out_idx += THREADS) {
        const int co = out_idx / bp.n_out;
        const int o = out_idx % bp.n_out;

        Complex acc = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + co * Cin * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + co * Cin * Wdim + bp.w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < bp.n_in; i++) {
                const Complex f = load_complex(f_ptr, i);
                const int w_idx = w_ci_off + weight_idx_2x2(o, i, bp.n_in, 0);

                const float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                            interp.t, interp.one_minus_t);
                const float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                            interp.t, interp.one_minus_t);

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
 */
template <typename scalar_t>
__global__ void backward_features_m0_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * Cout * Dout;

    for (int i = tid; i < Cout * bp.n_out; i += THREADS) {
        const int co = i / bp.n_out;
        const int local_idx = i % bp.n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + bp.out_off + local_idx]);
    }
    __syncthreads();

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);
    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        float grad = 0.0f;
        const int w_base_lo = interp.idx_lo * table_stride + ci * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * Wdim + bp.w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * bp.n_out;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const int w_idx = w_co_off + o * bp.n_in + i;
                const float w = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                            interp.t, interp.one_minus_t);
                grad += w * go_ptr[o];
            }
        }

        grad_features[edge * Cin * Din + ci * Din + bp.in_off + i] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward kernel for m>0: compute grad_features.
 * Computes: grad_f = W^T @ grad_out (transpose of rotation).
 */
template <typename scalar_t>
__global__ void backward_features_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
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

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);
    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features for each complex input
    for (int in_idx = tid; in_idx < Cin * bp.n_in; in_idx += THREADS) {
        const int ci = in_idx / bp.n_in;
        const int i = in_idx % bp.n_in;

        Complex grad_f = {0.0f, 0.0f};
        const int w_base_lo = interp.idx_lo * table_stride + ci * Wdim + bp.w_off;
        const int w_base_hi = interp.idx_hi * table_stride + ci * Wdim + bp.w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * out_size;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < bp.n_out; o++) {
                const Complex go = load_complex(go_ptr, o);
                const int w_idx = w_co_off + weight_idx_2x2(o, i, bp.n_in, 0);

                const float a = lerp_weight(radial_table, w_base_lo + w_idx, w_base_hi + w_idx,
                                            interp.t, interp.one_minus_t);
                const float b = lerp_weight(radial_table, w_base_lo + w_idx + 1, w_base_hi + w_idx + 1,
                                            interp.t, interp.one_minus_t);

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
 * Backward kernel for m=0: compute grad_radial_table and grad_interp_weight.
 */
template <typename scalar_t>
__global__ void backward_table_m0_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_interp_weight,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Compute block parameters for m=0
    const BlockParams bp = compute_block_params(0, lvals_in, lvals_out,
                                                 num_lvals_in, num_lvals_out, mmax);
    if (bp.n_in == 0 || bp.n_out == 0) return;

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

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);

    float local_grad_t = 0.0f;

    // Compute weight gradients
    for (int w_idx = tid; w_idx < Cout * Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;
        const int o = w_local / bp.n_in;
        const int i = w_local % bp.n_in;

        const float f = feat_shared[ci * bp.n_in + i];
        const float go = grad_shared[co * bp.n_out + o];
        const float grad_w = f * go;

        const int table_idx = co * Cin * Wdim + ci * Wdim + bp.w_off + w_local;
        const int addr_lo = interp.idx_lo * table_stride + table_idx;
        const int addr_hi = interp.idx_hi * table_stride + table_idx;

        atomicAdd(&grad_radial_table[addr_lo], interp.one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[addr_hi], interp.t * grad_w);

        const float w_lo = static_cast<float>(__ldg(&radial_table[addr_lo]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[addr_hi]));
        local_grad_t += (w_hi - w_lo) * grad_w;
    }

    // Warp reduction and atomic add for grad_interp_weight
    local_grad_t = warp_reduce_sum(local_grad_t);
    if ((tid % 32) == 0) {
        atomicAdd(&grad_interp_weight[edge], local_grad_t);
    }
}


/**
 * Backward kernel for m>0: compute grad_radial_table and grad_interp_weight.
 * Each thread computes gradient for one weight element (either 'a' or 'b').
 */
template <typename scalar_t>
__global__ void backward_table_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_interp_weight,
    const int* __restrict__ lvals_in,
    const int* __restrict__ lvals_out,
    int num_lvals_in, int num_lvals_out, int mmax,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim, int num_bins
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

    // Interpolation state
    const InterpState interp = load_interp_state(edge, bin_lo, interp_weight, num_bins);

    float local_grad_t = 0.0f;

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

        // Load feature and gradient
        const Complex f = load_complex(feat_shared + ci * in_size, i);
        const Complex go = load_complex(grad_shared + co * out_size, o);

        // Compute gradient for this weight component
        float grad_a, grad_b;
        compute_weight_gradient(f, go, grad_a, grad_b);
        const float grad_w = (is_b == 0) ? grad_a : grad_b;

        // Accumulate to table gradient with interpolation
        const int table_idx = co * Cin * Wdim + ci * Wdim + bp.w_off + w_local;
        const int addr_lo = interp.idx_lo * table_stride + table_idx;
        const int addr_hi = interp.idx_hi * table_stride + table_idx;

        atomicAdd(&grad_radial_table[addr_lo], interp.one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[addr_hi], interp.t * grad_w);

        // Gradient w.r.t. interpolation weight
        const float w_lo = static_cast<float>(__ldg(&radial_table[addr_lo]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[addr_hi]));
        local_grad_t += (w_hi - w_lo) * grad_w;
    }

    // Warp reduction and atomic add for grad_interp_weight
    local_grad_t = warp_reduce_sum(local_grad_t);
    if ((tid % 32) == 0) {
        atomicAdd(&grad_interp_weight[edge], local_grad_t);
    }
}


//------------------------------------------------------------------------------
// C++ Wrapper Functions
//------------------------------------------------------------------------------

std::vector<torch::Tensor> block_diagonal_forward_cuda(
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_lo,
    torch::Tensor interp_weight,
    torch::Tensor lvals_in,
    torch::Tensor lvals_out,
    int64_t Cout,
    int dim_out,
    int num_bins,
    int max_in_size
) {
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(bin_lo);
    CHECK_INPUT(interp_weight);
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

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Shared memory size: conservative estimate using max_in_size
    const size_t shared_size_m0 = Cin * num_lvals_in * sizeof(float);
    const size_t shared_size_mpos = Cin * max_in_size * sizeof(float);

    // Launch m=0 kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
        forward_m0_kernel<scalar_t><<<B, THREADS, shared_size_m0>>>(
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            lvals_in.data_ptr<int>(),
            lvals_out.data_ptr<int>(),
            num_lvals_in, num_lvals_out, mmax,
            B, Cin, Cout_int, Din, dim_out, Wdim, num_bins
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Launch m>0 kernel if mmax > 0
    if (mmax > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
            forward_mpos_kernel<scalar_t><<<B * mmax, THREADS, shared_size_mpos>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout_int, Din, dim_out, Wdim, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {output};
}


std::vector<torch::Tensor> block_diagonal_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_lo,
    torch::Tensor interp_weight,
    torch::Tensor lvals_in,
    torch::Tensor lvals_out,
    int dim_in,
    int max_in_size,
    int max_out_size
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(bin_lo);
    CHECK_INPUT(interp_weight);
    CHECK_INPUT(lvals_in);
    CHECK_INPUT(lvals_out);

    const int64_t B = features.size(0);
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Cout = static_cast<int>(grad_output.size(1));
    const int Dout = static_cast<int>(grad_output.size(2));
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int num_bins = static_cast<int>(num_bins_plus_1 - 1);
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_lvals_in = static_cast<int>(lvals_in.size(0));
    const int num_lvals_out = static_cast<int>(lvals_out.size(0));
    const int mmax = std::max(lvals_in.max().item<int>(), lvals_out.max().item<int>());

    auto grad_features = torch::zeros({B, Cin, Din}, features.options());
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_interp_weight = torch::zeros({B}, interp_weight.options().dtype(torch::kFloat32));

    // Shared memory sizes
    const size_t shared_size_feat_m0 = Cout * num_lvals_out * sizeof(float);
    const size_t shared_size_table_m0 = (Cin * num_lvals_in + Cout * num_lvals_out) * sizeof(float);
    const size_t shared_size_feat_mpos = Cout * max_out_size * sizeof(float);
    const size_t shared_size_table_mpos = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

    // Launch m=0 backward_features
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_m0", ([&] {
        backward_features_m0_kernel<scalar_t><<<B, THREADS, shared_size_feat_m0>>>(
            grad_output.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
            grad_features.data_ptr<scalar_t>(),
            lvals_in.data_ptr<int>(),
            lvals_out.data_ptr<int>(),
            num_lvals_in, num_lvals_out, mmax,
            B, Cin, Cout, Din, Dout, Wdim, num_bins
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Launch m=0 backward_table
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_m0", ([&] {
        backward_table_m0_kernel<scalar_t><<<B, THREADS, shared_size_table_m0>>>(
            grad_output.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
            grad_radial_table.data_ptr<float>(),
            grad_interp_weight.data_ptr<float>(),
            lvals_in.data_ptr<int>(),
            lvals_out.data_ptr<int>(),
            num_lvals_in, num_lvals_out, mmax,
            B, Cin, Cout, Din, Dout, Wdim, num_bins
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // m>0 kernels
    if (mmax > 0) {
        // backward_features m>0
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_mpos", ([&] {
            backward_features_mpos_kernel<scalar_t><<<B * mmax, THREADS, shared_size_feat_mpos>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // backward_table m>0
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_mpos", ([&] {
            backward_table_mpos_kernel<scalar_t><<<B * mmax, THREADS, shared_size_table_mpos>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_radial_table.data_ptr<float>(),
                grad_interp_weight.data_ptr<float>(),
                lvals_in.data_ptr<int>(),
                lvals_out.data_ptr<int>(),
                num_lvals_in, num_lvals_out, mmax,
                B, Cin, Cout, Din, Dout, Wdim, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (radial_table.scalar_type() != torch::kFloat32) {
        grad_radial_table = grad_radial_table.to(radial_table.scalar_type());
    }
    if (interp_weight.scalar_type() != torch::kFloat32) {
        grad_interp_weight = grad_interp_weight.to(interp_weight.scalar_type());
    }

    return {grad_features, grad_radial_table, grad_interp_weight};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda, "Block-diagonal forward (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda, "Block-diagonal backward (CUDA)");
}
