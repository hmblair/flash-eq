/**
 * @file block_diagonal_binned.cu
 * @brief Block-diagonal multiplication with binned radial weights.
 *
 * This kernel implements: out = Λ(r) @ f
 *
 * where:
 *   f: input features (num_edges, channels_in, dim_in) in m-first diagonal basis
 *   Λ(r): block-diagonal weights interpolated from radial table
 *   out: output features (num_edges, channels_out, dim_out) in m-first diagonal basis
 *
 * The kernel does NOT handle gathering or basis transforms (P^T, Q).
 * Those are done in PyTorch for simplicity and optimal matmul performance.
 *
 * Optimizations:
 *   - Binned weights: O(bins) memory instead of O(edges)
 *   - Bin-sorted edges: L2 cache locality for weight table
 *   - Linear interpolation between bin edges
 *
 * Block structure for SO(3):
 *   - m=0 blocks: n_in × n_out real scalars
 *   - m>0 blocks: 2×n_in × 2×n_out complex-type [a, b; -b, a]
 *
 * @author Hamish Blair
 * @see docs/theory.tex for mathematical details
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>


//------------------------------------------------------------------------------
// Input Validation Macros
//------------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


//------------------------------------------------------------------------------
// Forward Kernel
//------------------------------------------------------------------------------

/**
 * Forward pass: out[e] = Λ(r[e]) @ f[e]
 *
 * @param features      Input features (B, Cin, Din)
 * @param radial_table  Weight table at bin edges (K+1, Cout, Cin, Wdim)
 * @param bin_lo        Lower bin index per edge (B,)
 * @param interp_weight Interpolation weight t in [0,1] (B,)
 * @param output        Output features (B, Cout, Dout)
 * @param block_data    Block structure metadata (num_blocks, 6)
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_data,
    int64_t B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t edge = blockIdx.x / num_blocks;

    if (edge >= B) return;

    const int tid = threadIdx.x;

    // Unpack block parameters
    const int* blk_ptr = block_data + blk * 6;
    const int m = blk_ptr[0];
    const int n_in = blk_ptr[1];
    const int n_out = blk_ptr[2];
    const int in_off = blk_ptr[3];
    const int out_off = blk_ptr[4];
    const int w_off = blk_ptr[5];
    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = edge * static_cast<int64_t>(Cin) * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += 256) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Interpolation parameters
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Compute outputs
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += 256) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        const int w_base_lo = idx_lo * table_stride + co * Cin * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + co * Cin * Wdim + w_off;

        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const int w_ci_off = ci * Wdim;

            if (m == 0) {
                #pragma unroll 4
                for (int i = 0; i < n_in; i++) {
                    const int w_idx = w_ci_off + o_local * n_in + i;
                    const float w = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                                  + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                    acc += w * f_ptr[i];
                }
            } else {
                #pragma unroll 4
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];
                    const int w_idx = w_ci_off + (o_local * n_in + i) * 2;

                    const float a = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                                  + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                    const float bv = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx + 1]))
                                   + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx + 1]));

                    acc += a * f_re + bv * f_im;
                    acc_im += a * f_im - bv * f_re;
                }
            }
        }

        // Write output
        const int64_t out_base = edge * static_cast<int64_t>(Cout) * Dout + co * Dout + out_off;
        output[out_base + o_local] = static_cast<scalar_t>(acc);
        if (m > 0) {
            output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
        }
    }
}


//------------------------------------------------------------------------------
// Backward Kernels
//------------------------------------------------------------------------------

/**
 * Backward pass: compute grad_features.
 *
 * Direct output per edge (no scatter needed since input is edge features).
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_features_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_data,
    int64_t B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t edge = blockIdx.x / num_blocks;

    if (edge >= B) return;

    const int tid = threadIdx.x;

    // Unpack block parameters
    const int* blk_ptr = block_data + blk * 6;
    const int m = blk_ptr[0];
    const int n_in = blk_ptr[1];
    const int n_out = blk_ptr[2];
    const int in_off = blk_ptr[3];
    const int out_off = blk_ptr[4];
    const int w_off = blk_ptr[5];
    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Interpolation parameters
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = edge * static_cast<int64_t>(Cout) * Dout;
    const int total_grad_elems = Cout * out_size;

    for (int i = tid; i < total_grad_elems; i += 256) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Compute grad_features directly (no scatter)
    const int total_inputs = Cin * in_size;

    for (int in_idx = tid; in_idx < total_inputs; in_idx += 256) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;

        float grad = 0.0f;

        const int w_base_lo = idx_lo * table_stride + ci * Wdim + w_off;
        const int w_base_hi = idx_hi * table_stride + ci * Wdim + w_off;

        if (m == 0) {
            for (int co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const int w_co_off = co * Cin * Wdim;

                #pragma unroll 4
                for (int o = 0; o < n_out; o++) {
                    const int w_idx = w_co_off + o * n_in + i_local;
                    const float w = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                                  + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                    grad += w * go_ptr[o];
                }
            }
        } else {
            const bool is_real = (i_local < n_in);
            const int i_idx = is_real ? i_local : (i_local - n_in);

            for (int co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const int w_co_off = co * Cin * Wdim;

                #pragma unroll 4
                for (int o = 0; o < n_out; o++) {
                    const float go_re = go_ptr[o];
                    const float go_im = go_ptr[n_out + o];
                    const int w_idx = w_co_off + (o * n_in + i_idx) * 2;

                    const float a = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx]))
                                  + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx]));
                    const float bv = one_minus_t * static_cast<float>(__ldg(&radial_table[w_base_lo + w_idx + 1]))
                                   + t * static_cast<float>(__ldg(&radial_table[w_base_hi + w_idx + 1]));

                    if (is_real) {
                        grad += a * go_re - bv * go_im;
                    } else {
                        grad += bv * go_re + a * go_im;
                    }
                }
            }
        }

        // Write directly to grad_features (no scatter needed)
        const int64_t feat_idx = edge * static_cast<int64_t>(Cin) * Din + ci * Din + in_off + i_local;
        grad_features[feat_idx] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward pass: compute grad_radial_table and grad_interp_weight.
 *
 * Uses atomicAdd to scatter gradients to the lookup table (multiple edges share bins).
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_table_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_interp_weight,
    const int* __restrict__ block_data,
    int64_t B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t edge = blockIdx.x / num_blocks;

    if (edge >= B) return;

    const int tid = threadIdx.x;

    // Unpack block parameters
    const int* blk_ptr = block_data + blk * 6;
    const int m = blk_ptr[0];
    const int n_in = blk_ptr[1];
    const int n_out = blk_ptr[2];
    const int in_off = blk_ptr[3];
    const int out_off = blk_ptr[4];
    const int w_off = blk_ptr[5];
    const int in_size = (m == 0) ? n_in : 2 * n_in;
    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int w_block_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Interpolation parameters
    const int idx_lo = bin_lo[edge];
    const int idx_hi = min(idx_lo + 1, num_bins);
    const float t = static_cast<float>(interp_weight[edge]);
    const float one_minus_t = 1.0f - t;
    const int table_stride = Cout * Cin * Wdim;

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * in_size;

    // Load features
    const int64_t feat_base = edge * static_cast<int64_t>(Cin) * Din;
    for (int i = tid; i < Cin * in_size; i += 256) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = edge * static_cast<int64_t>(Cout) * Dout;
    for (int i = tid; i < Cout * out_size; i += 256) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Accumulate grad_interp_weight locally
    float local_grad_t = 0.0f;

    // Compute weight gradients
    const int total_weights = Cout * Cin * w_block_size;

    for (int w_idx = tid; w_idx < total_weights; w_idx += 256) {
        const int co = w_idx / (Cin * w_block_size);
        const int ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        float grad_w;

        if (m == 0) {
            const int o = w_local / n_in;
            const int i = w_local % n_in;
            grad_w = f_ptr[i] * go_ptr[o];
        } else {
            const int temp = w_local / 2;
            const int ab = w_local % 2;
            const int o = temp / n_in;
            const int i = temp % n_in;

            const float f_re = f_ptr[i];
            const float f_im = f_ptr[n_in + i];
            const float go_re = go_ptr[o];
            const float go_im = go_ptr[n_out + o];

            grad_w = (ab == 0) ? (f_re * go_re + f_im * go_im)
                               : (f_im * go_re - f_re * go_im);
        }

        // Scatter to grad_radial_table (atomicAdd since multiple edges share bins)
        const int table_idx = co * Cin * Wdim + ci * Wdim + w_off + w_local;
        atomicAdd(&grad_radial_table[idx_lo * table_stride + table_idx], one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[idx_hi * table_stride + table_idx], t * grad_w);

        // Accumulate grad_interp_weight
        const float w_lo = static_cast<float>(__ldg(&radial_table[idx_lo * table_stride + table_idx]));
        const float w_hi = static_cast<float>(__ldg(&radial_table[idx_hi * table_stride + table_idx]));
        local_grad_t += (w_hi - w_lo) * grad_w;
    }

    // Warp-level reduction for grad_interp_weight
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

    const int64_t B = features.size(0);  // num_edges
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_blocks = static_cast<int>(block_data.size(0));
    const int Cout_int = static_cast<int>(Cout);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;
    const size_t shared_size = static_cast<size_t>(Cin) * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward", ([&] {
        block_diagonal_forward_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_data.data_ptr<int>(),
            B, Cin, Cout_int, Din, dim_out, Wdim, num_blocks, num_bins
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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

    const int64_t B = features.size(0);  // num_edges
    const int Cin = static_cast<int>(features.size(1));
    const int Din = static_cast<int>(features.size(2));
    const int Cout = static_cast<int>(grad_output.size(1));
    const int Dout = static_cast<int>(grad_output.size(2));
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int num_bins = static_cast<int>(num_bins_plus_1 - 1);
    const int Wdim = static_cast<int>(radial_table.size(3));
    const int num_blocks = static_cast<int>(block_data.size(0));

    // Allocate outputs
    auto grad_features = torch::zeros({B, Cin, Din}, features.options());
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_interp_weight = torch::zeros({B}, interp_weight.options().dtype(torch::kFloat32));

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Features backward (direct output, no scatter)
    {
        const size_t shared_size = static_cast<size_t>(Cout) * max_out_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features", ([&] {
            block_diagonal_backward_features_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                block_data.data_ptr<int>(),
                B, Cin, Cout, Din, Dout, Wdim, num_blocks, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Table backward (scatter to bins)
    {
        const size_t shared_size = static_cast<size_t>(Cin) * max_in_size + static_cast<size_t>(Cout) * max_out_size;
        const size_t shared_bytes = shared_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table", ([&] {
            block_diagonal_backward_table_kernel<scalar_t><<<grid_size, threads, shared_bytes>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_radial_table.data_ptr<float>(),
                grad_interp_weight.data_ptr<float>(),
                block_data.data_ptr<int>(),
                B, Cin, Cout, Din, Dout, Wdim, num_blocks, num_bins
            );
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Convert back to original dtype if needed
    if (radial_table.scalar_type() != torch::kFloat32) {
        grad_radial_table = grad_radial_table.to(radial_table.scalar_type());
    }
    if (interp_weight.scalar_type() != torch::kFloat32) {
        grad_interp_weight = grad_interp_weight.to(interp_weight.scalar_type());
    }

    return {grad_features, grad_radial_table, grad_interp_weight};
}


//------------------------------------------------------------------------------
// Python Bindings
//------------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda,
          "Block-diagonal forward (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda,
          "Block-diagonal backward (CUDA)");
}
