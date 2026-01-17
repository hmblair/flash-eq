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
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int n_in, int n_out, int in_off, int out_off, int w_off, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * Cin * Din;

    for (int i = tid; i < Cin * n_in; i += THREADS) {
        const int ci = i / n_in;
        const int local_idx = i % n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < Cout * n_out; out_idx += THREADS) {
        const int co = out_idx / n_out;
        const int o = out_idx % n_out;

        float acc = 0.0f;
        const int w_base_lo = idx_lo * table_stride + co * Cin * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + co * Cin * Wdim + w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * n_in;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < n_in; i++) {
                const int w_idx = w_ci_off + o * n_in + i;
                const float w = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                acc += w * f_ptr[i];
            }
        }

        output[edge * Cout * Dout + co * Dout + out_off + o] = static_cast<scalar_t>(acc);
    }
}


/**
 * Forward kernel for m>0 blocks (complex-like 2x2 structure).
 * One CUDA block per (edge, m-block) pair.
 */
template <typename scalar_t>
__global__ void forward_mpos_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_data,  // (num_mpos_blocks, 5): [n_in, n_out, in_off, out_off, w_off]
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int num_mpos_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_mpos_blocks;
    const int64_t edge = blockIdx.x / num_mpos_blocks;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Unpack block parameters (no m needed, all are m>0)
    const int* blk_ptr = block_data + blk * 5;
    const int n_in = blk_ptr[0];
    const int n_out = blk_ptr[1];
    const int in_off = blk_ptr[2];
    const int out_off = blk_ptr[3];
    const int w_off = blk_ptr[4];
    const int in_size = 2 * n_in;

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * Cin * Din;

    for (int i = tid; i < Cin * in_size; i += THREADS) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs
    for (int out_idx = tid; out_idx < Cout * n_out; out_idx += THREADS) {
        const int co = out_idx / n_out;
        const int o = out_idx % n_out;

        float acc_re = 0.0f;
        float acc_im = 0.0f;
        const int w_base_lo = idx_lo * table_stride + co * Cin * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + co * Cin * Wdim + w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * Wdim;

            #pragma unroll 4
            for (int i = 0; i < n_in; i++) {
                const float f_re = f_ptr[2 * i];
                const float f_im = f_ptr[2 * i + 1];
                const int w_idx = w_ci_off + (o * n_in + i) * 2;

                const float a = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                const float b = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx + 1]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx + 1]));

                acc_re += a * f_re + b * f_im;
                acc_im += a * f_im - b * f_re;
            }
        }

        const int64_t out_base = edge * Cout * Dout + co * Dout + out_off;
        output[out_base + 2 * o] = static_cast<scalar_t>(acc_re);
        output[out_base + 2 * o + 1] = static_cast<scalar_t>(acc_im);
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
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int n_in, int n_out, int in_off, int out_off, int w_off, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * Cout * Dout;

    for (int i = tid; i < Cout * n_out; i += THREADS) {
        const int co = i / n_out;
        const int local_idx = i % n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < Cin * n_in; in_idx += THREADS) {
        const int ci = in_idx / n_in;
        const int i = in_idx % n_in;

        float grad = 0.0f;
        const int w_base_lo = idx_lo * table_stride + ci * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + ci * Wdim + w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * n_out;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < n_out; o++) {
                const int w_idx = w_co_off + o * n_in + i;
                const float w = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                grad += w * go_ptr[o];
            }
        }

        grad_features[edge * Cin * Din + ci * Din + in_off + i] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward kernel for m>0: compute grad_features.
 */
template <typename scalar_t>
__global__ void backward_features_mpos_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_data,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int num_mpos_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_mpos_blocks;
    const int64_t edge = blockIdx.x / num_mpos_blocks;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Unpack block parameters
    const int* blk_ptr = block_data + blk * 5;
    const int n_in = blk_ptr[0];
    const int n_out = blk_ptr[1];
    const int in_off = blk_ptr[2];
    const int out_off = blk_ptr[3];
    const int w_off = blk_ptr[4];
    const int out_size = 2 * n_out;
    const int in_size = 2 * n_in;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * Cout * Dout;

    for (int i = tid; i < Cout * out_size; i += THREADS) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Compute grad_features
    for (int in_idx = tid; in_idx < Cin * in_size; in_idx += THREADS) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;
        const bool is_real = (i_local % 2 == 0);
        const int i_idx = i_local / 2;

        float grad = 0.0f;
        const int w_base_lo = idx_lo * table_stride + ci * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + ci * Wdim + w_off;

        for (int co = 0; co < Cout; co++) {
            const float* go_ptr = grad_shared + co * out_size;
            const int w_co_off = co * Cin * Wdim;

            #pragma unroll 4
            for (int o = 0; o < n_out; o++) {
                const float go_re = go_ptr[2 * o];
                const float go_im = go_ptr[2 * o + 1];
                const int w_idx = w_co_off + (o * n_in + i_idx) * 2;

                const float a = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                const float b = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx + 1]))
                              + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx + 1]));

                if (is_real) {
                    grad += a * go_re - b * go_im;
                } else {
                    grad += b * go_re + a * go_im;
                }
            }
        }

        grad_features[edge * Cin * Din + ci * Din + in_off + i_local] = static_cast<scalar_t>(grad);
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
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int n_in, int n_out, int in_off, int out_off, int w_off, int num_bins
) {
    const int64_t edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;
    const int table_stride = Cout * Cin * Wdim;
    const int w_block_size = n_out * n_in;

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * n_in;

    // Load features
    const int64_t feat_base = edge * Cin * Din;
    for (int i = tid; i < Cin * n_in; i += THREADS) {
        const int ci = i / n_in;
        const int local_idx = i % n_in;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * Cout * Dout;
    for (int i = tid; i < Cout * n_out; i += THREADS) {
        const int co = i / n_out;
        const int local_idx = i % n_out;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;

    float local_grad_t = 0.0f;

    // Compute weight gradients
    for (int w_idx = tid; w_idx < Cout * Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;
        const int o = w_local / n_in;
        const int i = w_local % n_in;

        const float f = feat_shared[ci * n_in + i];
        const float go = grad_shared[co * n_out + o];
        const float grad_w = f * go;

        const int table_idx = co * Cin * Wdim + ci * Wdim + w_off + w_local;
        atomicAdd(&grad_radial_table[idx_lo * table_stride + table_idx], one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[idx_hi * table_stride + table_idx], t * grad_w);

        const float w_lo = static_cast<float>(__ldg(&radial_table[idx_lo * table_stride + table_idx]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[idx_hi * table_stride + table_idx]));
        local_grad_t += (w_hi - w_lo) * grad_w;
    }

    // Warp reduction for grad_interp_weight
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_t += __shfl_down_sync(0xffffffff, local_grad_t, offset);
    }
    if ((tid % 32) == 0) {
        atomicAdd(&grad_interp_weight[edge], local_grad_t);
    }
}


/**
 * Backward kernel for m>0: compute grad_radial_table and grad_interp_weight.
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
    const int* __restrict__ block_data,
    int64_t num_edges, int Cin, int Cout, int Din, int Dout, int Wdim,
    int num_mpos_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_mpos_blocks;
    const int64_t edge = blockIdx.x / num_mpos_blocks;
    if (edge >= num_edges) return;

    const int tid = threadIdx.x;

    // Unpack block parameters
    const int* blk_ptr = block_data + blk * 5;
    const int n_in = blk_ptr[0];
    const int n_out = blk_ptr[1];
    const int in_off = blk_ptr[2];
    const int out_off = blk_ptr[3];
    const int w_off = blk_ptr[4];
    const int in_size = 2 * n_in;
    const int out_size = 2 * n_out;
    const int w_block_size = 2 * n_out * n_in;
    const int table_stride = Cout * Cin * Wdim;

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * in_size;

    // Load features
    const int64_t feat_base = edge * Cin * Din;
    for (int i = tid; i < Cin * in_size; i += THREADS) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * Cout * Dout;
    for (int i = tid; i < Cout * out_size; i += THREADS) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Interpolation
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;

    float local_grad_t = 0.0f;

    // Compute weight gradients
    for (int w_idx = tid; w_idx < Cout * Cin * w_block_size; w_idx += THREADS) {
        const int co = w_idx / (Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const int temp = w_local / 2;
        const int ab = w_local % 2;
        const int o = temp / n_in;
        const int i = temp % n_in;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        const float f_re = f_ptr[2 * i];
        const float f_im = f_ptr[2 * i + 1];
        const float go_re = go_ptr[2 * o];
        const float go_im = go_ptr[2 * o + 1];

        const float grad_w = (ab == 0) ? (f_re * go_re + f_im * go_im)
                                       : (f_im * go_re - f_re * go_im);

        const int table_idx = co * Cin * Wdim + ci * Wdim + w_off + w_local;
        atomicAdd(&grad_radial_table[idx_lo * table_stride + table_idx], one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[idx_hi * table_stride + table_idx], t * grad_w);

        const float w_lo = static_cast<float>(__ldg(&radial_table[idx_lo * table_stride + table_idx]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[idx_hi * table_stride + table_idx]));
        local_grad_t += (w_hi - w_lo) * grad_w;
    }

    // Warp reduction for grad_interp_weight
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_t += __shfl_down_sync(0xffffffff, local_grad_t, offset);
    }
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
    torch::Tensor block_data,
    int64_t Cout,
    int dim_out,
    int num_bins,
    int max_in_size
) {
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(bin_lo);
    CHECK_INPUT(interp_weight);
    CHECK_INPUT(block_data);

    const int64_t B = features.size(0);
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_blocks = static_cast<int>(block_data.size(0));
    const int Cout_int = static_cast<int>(Cout);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Extract m=0 block (first row of block_data)
    auto block_data_cpu = block_data.cpu();
    const int* bd = block_data_cpu.data_ptr<int>();

    // m=0 block parameters (first block, m=0)
    const int m0_n_in = bd[1];
    const int m0_n_out = bd[2];
    const int m0_in_off = bd[3];
    const int m0_out_off = bd[4];
    const int m0_w_off = bd[5];

    // Launch m=0 kernel
    {
        const size_t shared_size = Cin * m0_n_in * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_m0", ([&] {
            forward_m0_kernel<scalar_t><<<B, THREADS, shared_size>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                B, Cin, Cout_int, Din, dim_out, Wdim,
                m0_n_in, m0_n_out, m0_in_off, m0_out_off, m0_w_off, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Launch m>0 kernel if there are m>0 blocks
    if (num_blocks > 1) {
        // Build m>0 block data (skip m column, just [n_in, n_out, in_off, out_off, w_off])
        auto mpos_block_data = torch::zeros({num_blocks - 1, 5}, torch::dtype(torch::kInt32).device(features.device()));
        auto mpos_bd = mpos_block_data.cpu();
        int* mpos_ptr = mpos_bd.data_ptr<int>();
        int max_mpos_in_size = 0;
        for (int i = 1; i < num_blocks; i++) {
            const int* src = bd + i * 6;
            int* dst = mpos_ptr + (i - 1) * 5;
            dst[0] = src[1];  // n_in
            dst[1] = src[2];  // n_out
            dst[2] = src[3];  // in_off
            dst[3] = src[4];  // out_off
            dst[4] = src[5];  // w_off
            max_mpos_in_size = std::max(max_mpos_in_size, 2 * src[1]);
        }
        mpos_block_data = mpos_bd.to(features.device());

        const int num_mpos = num_blocks - 1;
        const size_t shared_size = Cin * max_mpos_in_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_mpos", ([&] {
            forward_mpos_kernel<scalar_t><<<B * num_mpos, THREADS, shared_size>>>(
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                mpos_block_data.data_ptr<int>(),
                B, Cin, Cout_int, Din, dim_out, Wdim,
                num_mpos, num_bins
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
    torch::Tensor block_data,
    int dim_in,
    int max_in_size,
    int max_out_size
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(features);
    CHECK_INPUT(radial_table);
    CHECK_INPUT(bin_lo);
    CHECK_INPUT(interp_weight);
    CHECK_INPUT(block_data);

    const int64_t B = features.size(0);
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Cout = static_cast<int>(grad_output.size(1));
    const int Dout = static_cast<int>(grad_output.size(2));
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int num_bins = static_cast<int>(num_bins_plus_1 - 1);
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_blocks = static_cast<int>(block_data.size(0));

    auto grad_features = torch::zeros({B, Cin, Din}, features.options());
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_interp_weight = torch::zeros({B}, interp_weight.options().dtype(torch::kFloat32));

    // Extract block data
    auto block_data_cpu = block_data.cpu();
    const int* bd = block_data_cpu.data_ptr<int>();

    // m=0 block parameters
    const int m0_n_in = bd[1];
    const int m0_n_out = bd[2];
    const int m0_in_off = bd[3];
    const int m0_out_off = bd[4];
    const int m0_w_off = bd[5];

    // Launch m=0 backward_features
    {
        const size_t shared_size = Cout * m0_n_out * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_m0", ([&] {
            backward_features_m0_kernel<scalar_t><<<B, THREADS, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                B, Cin, Cout, Din, Dout, Wdim,
                m0_n_in, m0_n_out, m0_in_off, m0_out_off, m0_w_off, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Launch m=0 backward_table
    {
        const size_t shared_size = (Cin * m0_n_in + Cout * m0_n_out) * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_m0", ([&] {
            backward_table_m0_kernel<scalar_t><<<B, THREADS, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_radial_table.data_ptr<float>(),
                grad_interp_weight.data_ptr<float>(),
                B, Cin, Cout, Din, Dout, Wdim,
                m0_n_in, m0_n_out, m0_in_off, m0_out_off, m0_w_off, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // m>0 kernels
    if (num_blocks > 1) {
        auto mpos_block_data = torch::zeros({num_blocks - 1, 5}, torch::dtype(torch::kInt32).device(features.device()));
        auto mpos_bd = mpos_block_data.cpu();
        int* mpos_ptr = mpos_bd.data_ptr<int>();
        int max_mpos_in_size = 0;
        int max_mpos_out_size = 0;
        for (int i = 1; i < num_blocks; i++) {
            const int* src = bd + i * 6;
            int* dst = mpos_ptr + (i - 1) * 5;
            dst[0] = src[1];  // n_in
            dst[1] = src[2];  // n_out
            dst[2] = src[3];  // in_off
            dst[3] = src[4];  // out_off
            dst[4] = src[5];  // w_off
            max_mpos_in_size = std::max(max_mpos_in_size, 2 * src[1]);
            max_mpos_out_size = std::max(max_mpos_out_size, 2 * src[2]);
        }
        mpos_block_data = mpos_bd.to(features.device());

        const int num_mpos = num_blocks - 1;

        // backward_features m>0
        {
            const size_t shared_size = Cout * max_mpos_out_size * sizeof(float);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_mpos", ([&] {
                backward_features_mpos_kernel<scalar_t><<<B * num_mpos, THREADS, shared_size>>>(
                    grad_output.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    bin_lo.data_ptr<int>(),
                    interp_weight.data_ptr<scalar_t>(),
                    grad_features.data_ptr<scalar_t>(),
                    mpos_block_data.data_ptr<int>(),
                    B, Cin, Cout, Din, Dout, Wdim,
                    num_mpos, num_bins
                );
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // backward_table m>0
        {
            const size_t shared_size = (Cin * max_mpos_in_size + Cout * max_mpos_out_size) * sizeof(float);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table_mpos", ([&] {
                backward_table_mpos_kernel<scalar_t><<<B * num_mpos, THREADS, shared_size>>>(
                    grad_output.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    radial_table.data_ptr<scalar_t>(),
                    bin_lo.data_ptr<int>(),
                    interp_weight.data_ptr<scalar_t>(),
                    grad_radial_table.data_ptr<float>(),
                    grad_interp_weight.data_ptr<float>(),
                    mpos_block_data.data_ptr<int>(),
                    B, Cin, Cout, Din, Dout, Wdim,
                    num_mpos, num_bins
                );
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
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
