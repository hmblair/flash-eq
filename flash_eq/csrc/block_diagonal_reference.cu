/**
 * @file block_diagonal_reference.cu
 * @brief Reference implementation of block-diagonal multiplication for SO(3)-equivariant layers.
 *
 * This file provides the standard (non-binned) block-diagonal kernels for:
 *   - Baseline performance comparisons
 *   - Cases where per-edge weights are needed
 *   - Validation of the binned approach
 *
 * For production use with radial MLPs, prefer block_diagonal_binned.cu which
 * provides 5-13x memory reduction and 1.2-1.8x speedup during training.
 *
 * Block structure for SO(3):
 *   - m=0 blocks: 1x1 real scalars
 *   - m>0 blocks: 2x2 complex-type matrices [a, b; -b, a]
 *
 * @author Hamish Blair
 * @see block_diagonal_binned.cu for production implementation
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


//------------------------------------------------------------------------------
// Forward Kernel
//------------------------------------------------------------------------------

/**
 * Forward pass with per-edge weights.
 *
 * Grid: B * num_m_blocks thread blocks
 * Each block: Caches features in shared memory, processes one m-block.
 *
 * @param features  Input features (B, Cin, Din)
 * @param weights   Per-edge weights (B, Cout, Cin, Wdim)
 * @param output    Output features (B, Cout, Dout)
 * @param block_*   Block structure metadata
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_v2_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];
    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Load features into shared memory
    extern __shared__ float feat_shared[];
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Compute outputs
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        const scalar_t* w_base = weights + b * Cout * Cin * Wdim + co * Cin * Wdim;

        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const scalar_t* w_ptr = w_base + ci * Wdim + w_off;

            if (m == 0) {
                for (int i = 0; i < n_in; i++) {
                    acc += static_cast<float>(w_ptr[o_local * n_in + i]) * f_ptr[i];
                }
            } else {
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];
                    const int w_idx = (o_local * n_in + i) * 2;
                    const float a = static_cast<float>(w_ptr[w_idx]);
                    const float bv = static_cast<float>(w_ptr[w_idx + 1]);

                    acc += a * f_re + bv * f_im;
                    acc_im += a * f_im - bv * f_re;
                }
            }
        }

        const int64_t out_base = b * Cout * Dout + co * Dout + out_off;
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
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_features_v2_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];
    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Load grad_output into shared memory
    extern __shared__ float grad_shared[];
    const int64_t grad_base = b * Cout * Dout;
    const int total_grad_elems = Cout * out_size;

    for (int i = tid; i < total_grad_elems; i += num_threads) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    const int total_inputs = Cin * in_size;

    for (int in_idx = tid; in_idx < total_inputs; in_idx += num_threads) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;

        float grad = 0.0f;

        if (m == 0) {
            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    grad += static_cast<float>(w_ptr[o * n_in + i_local]) * go_ptr[o];
                }
            }
        } else {
            bool is_real = (i_local < n_in);
            int i_idx = is_real ? i_local : (i_local - n_in);

            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    float go_re = go_ptr[o];
                    float go_im = go_ptr[n_out + o];
                    int w_idx = (o * n_in + i_idx) * 2;
                    float a = static_cast<float>(w_ptr[w_idx]);
                    float bv = static_cast<float>(w_ptr[w_idx + 1]);

                    if (is_real) {
                        grad += a * go_re - bv * go_im;
                    } else {
                        grad += bv * go_re + a * go_im;
                    }
                }
            }
        }

        const int64_t feat_idx = b * Cin * Din + ci * Din + in_off + i_local;
        grad_features[feat_idx] = static_cast<scalar_t>(grad);
    }
}


/**
 * Backward pass: compute grad_weights.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_weights_v2_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    scalar_t* __restrict__ grad_weights,
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];
    const int in_size = (m == 0) ? n_in : 2 * n_in;
    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int w_block_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Shared memory: features + grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * in_size;

    // Load features
    const int64_t feat_base = b * Cin * Din;
    for (int i = tid; i < Cin * in_size; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load grad_output
    const int64_t grad_base = b * Cout * Dout;
    for (int i = tid; i < Cout * out_size; i += num_threads) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    const int64_t total_weights = Cout * Cin * w_block_size;

    for (int64_t w_idx = tid; w_idx < total_weights; w_idx += num_threads) {
        const int64_t co = w_idx / (Cin * w_block_size);
        const int64_t ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        float grad = 0.0f;

        if (m == 0) {
            int o = w_local / n_in;
            int i = w_local % n_in;
            grad = f_ptr[i] * go_ptr[o];
        } else {
            int temp = w_local / 2;
            int ab = w_local % 2;
            int o = temp / n_in;
            int i = temp % n_in;

            float f_re = f_ptr[i];
            float f_im = f_ptr[n_in + i];
            float go_re = go_ptr[o];
            float go_im = go_ptr[n_out + o];

            if (ab == 0) {
                grad = f_re * go_re + f_im * go_im;
            } else {
                grad = f_im * go_re - f_re * go_im;
            }
        }

        const int64_t weight_idx = b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off + w_local;
        grad_weights[weight_idx] = static_cast<scalar_t>(grad);
    }
}


//------------------------------------------------------------------------------
// C++ Wrapper Functions
//------------------------------------------------------------------------------

std::vector<torch::Tensor> block_diagonal_forward_v2_cuda(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = weights.size(1);
    const int64_t Wdim = weights.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_v2", ([&] {
        block_diagonal_forward_v2_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_m.data_ptr<int>(),
            block_n_in.data_ptr<int>(),
            block_n_out.data_ptr<int>(),
            block_in_off.data_ptr<int>(),
            block_out_off.data_ptr<int>(),
            block_w_off.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks
        );
    }));

    return {output};
}


std::vector<torch::Tensor> block_diagonal_backward_v2_cuda(
    torch::Tensor grad_output,
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    torch::Tensor block_out_size,
    int dim_in
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = weights.size(1);
    const int64_t Wdim = weights.size(3);
    const int64_t Dout = grad_output.size(2);
    const int num_blocks = block_m.size(0);

    auto grad_features = torch::zeros_like(features);
    auto grad_weights = torch::zeros_like(weights);

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;
    const int max_in_size = block_in_size.max().item<int>();
    const int max_out_size = block_out_size.max().item<int>();

    // Features backward
    {
        const size_t shared_size = Cout * max_out_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features_v2", ([&] {
            block_diagonal_backward_features_v2_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                block_m.data_ptr<int>(),
                block_n_in.data_ptr<int>(),
                block_n_out.data_ptr<int>(),
                block_in_off.data_ptr<int>(),
                block_out_off.data_ptr<int>(),
                block_w_off.data_ptr<int>(),
                B, Cin, Cout, Din, Dout, Wdim, num_blocks
            );
        }));
    }

    // Weights backward
    {
        const size_t shared_size = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_weights_v2", ([&] {
            block_diagonal_backward_weights_v2_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                grad_weights.data_ptr<scalar_t>(),
                block_m.data_ptr<int>(),
                block_n_in.data_ptr<int>(),
                block_n_out.data_ptr<int>(),
                block_in_off.data_ptr<int>(),
                block_out_off.data_ptr<int>(),
                block_w_off.data_ptr<int>(),
                B, Cin, Cout, Din, Dout, Wdim, num_blocks
            );
        }));
    }

    return {grad_features, grad_weights};
}


//------------------------------------------------------------------------------
// Python Bindings
//------------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v2", &block_diagonal_forward_v2_cuda,
          "Block-diagonal forward with per-edge weights (CUDA)");
    m.def("backward_v2", &block_diagonal_backward_v2_cuda,
          "Block-diagonal backward with per-edge weights (CUDA)");
}
