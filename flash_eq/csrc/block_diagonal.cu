/*
 * Block-diagonal multiplication for SO(3)-equivariant layers.
 *
 * This kernel implements the Λ (Lambda) step in the low-rank equivariant
 * linear layer: output = Q @ Λ @ P^T @ features
 *
 * The block-diagonal structure has:
 *   - m=0 blocks: 1×1 real scalars
 *   - m>0 blocks: 2×2 complex-type matrices [a, b; -b, a]
 *
 * Optimized with parallel reduction over channels_in for better throughput.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 128


template <typename scalar_t>
__global__ void block_diagonal_forward_kernel(
    const scalar_t* __restrict__ features,  // (B, Cin, Din)
    const scalar_t* __restrict__ weights,   // (B, Cout, Cin, Wdim)
    scalar_t* __restrict__ output,          // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    const int* __restrict__ out_to_block,
    const int* __restrict__ out_to_local,
    int B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks
) {
    /*
     * Parallelization: Each thread block handles one output element (b, co, out_idx).
     * Threads within the block cooperatively reduce over channels_in.
     */
    extern __shared__ char shared_mem[];
    float* sdata = reinterpret_cast<float*>(shared_mem);

    const int out_idx = blockIdx.x % Dout;
    const int co = (blockIdx.x / Dout) % Cout;
    const int b = blockIdx.x / (Dout * Cout);

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int blk = out_to_block[out_idx];
    const int o_local = out_to_local[out_idx];

    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    // For m>0, threads with o_local >= n_out handle imag outputs (skip, computed by real threads)
    if (m > 0 && o_local >= n_out) return;

    // Accumulate in float for numerical stability
    float acc = 0.0f;
    float acc_im = 0.0f;

    // Each thread handles a subset of channels_in
    for (int ci = tid; ci < Cin; ci += blockDim.x) {
        const scalar_t* f_ptr = features + b * Cin * Din + ci * Din + in_off;
        const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

        if (m == 0) {
            // Real block: simple dot product
            for (int i = 0; i < n_in; i++) {
                acc += static_cast<float>(w_ptr[o_local * n_in + i]) *
                       static_cast<float>(f_ptr[i]);
            }
        } else {
            // Complex block: [a, b; -b, a] @ [f_re; f_im]
            for (int i = 0; i < n_in; i++) {
                float f_re = static_cast<float>(f_ptr[i]);
                float f_im = static_cast<float>(f_ptr[n_in + i]);

                int w_idx = (o_local * n_in + i) * 2;
                float a = static_cast<float>(w_ptr[w_idx]);
                float bv = static_cast<float>(w_ptr[w_idx + 1]);

                acc += a * f_re + bv * f_im;
                acc_im += a * f_im - bv * f_re;
            }
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (m > 0) {
            acc_im += __shfl_down_sync(0xffffffff, acc_im, offset);
        }
    }

    // Block-level reduction via shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    if (lane_id == 0) {
        sdata[warp_id] = acc;
        if (m > 0) {
            sdata[warp_id + num_warps] = acc_im;
        }
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        acc = (tid < num_warps) ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (m > 0) {
            acc_im = (tid < num_warps) ? sdata[tid + num_warps] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                acc_im += __shfl_down_sync(0xffffffff, acc_im, offset);
            }
        }

        if (tid == 0) {
            const int out_base = b * Cout * Dout + co * Dout + out_off;
            output[out_base + o_local] = static_cast<scalar_t>(acc);
            if (m > 0) {
                output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
            }
        }
    }
}


template <typename scalar_t>
__global__ void block_diagonal_backward_features_kernel(
    const scalar_t* __restrict__ grad_output,  // (B, Cout, Dout)
    const scalar_t* __restrict__ weights,      // (B, Cout, Cin, Wdim)
    scalar_t* __restrict__ grad_features,      // (B, Cin, Din)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * Cin * Din;
    if (idx >= total) return;

    const int b = idx / (Cin * Din);
    const int ci = (idx / Din) % Cin;
    const int i_global = idx % Din;

    // Find which block this input belongs to
    int blk = -1;
    int i_local = i_global;
    for (int k = 0; k < num_blocks; k++) {
        int m_val = block_m[k];
        int n_in_val = block_n_in[k];
        int block_size = (m_val == 0) ? n_in_val : 2 * n_in_val;
        if (i_local < block_size) {
            blk = k;
            break;
        }
        i_local -= block_size;
    }

    if (blk < 0) return;

    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    float grad = 0.0f;

    if (m == 0) {
        for (int co = 0; co < Cout; co++) {
            const scalar_t* go_ptr = grad_output + b * Cout * Dout + co * Dout + out_off;
            const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

            for (int o = 0; o < n_out; o++) {
                grad += static_cast<float>(w_ptr[o * n_in + i_local]) *
                        static_cast<float>(go_ptr[o]);
            }
        }
    } else {
        bool is_real_input = (i_local < n_in);
        int i_idx = is_real_input ? i_local : (i_local - n_in);

        for (int co = 0; co < Cout; co++) {
            const scalar_t* go_ptr = grad_output + b * Cout * Dout + co * Dout + out_off;
            const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

            for (int o = 0; o < n_out; o++) {
                float go_re = static_cast<float>(go_ptr[o]);
                float go_im = static_cast<float>(go_ptr[n_out + o]);

                int w_idx = (o * n_in + i_idx) * 2;
                float a = static_cast<float>(w_ptr[w_idx]);
                float bv = static_cast<float>(w_ptr[w_idx + 1]);

                if (is_real_input) {
                    grad += a * go_re - bv * go_im;
                } else {
                    grad += bv * go_re + a * go_im;
                }
            }
        }
    }

    grad_features[idx] = static_cast<scalar_t>(grad);
}


template <typename scalar_t>
__global__ void block_diagonal_backward_weights_kernel(
    const scalar_t* __restrict__ grad_output,  // (B, Cout, Dout)
    const scalar_t* __restrict__ features,     // (B, Cin, Din)
    scalar_t* __restrict__ grad_weights,       // (B, Cout, Cin, Wdim)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int B, int Cin, int Cout, int Din, int Dout, int Wdim, int num_blocks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * Cout * Cin * Wdim;
    if (idx >= total) return;

    const int b = idx / (Cout * Cin * Wdim);
    const int co = (idx / (Cin * Wdim)) % Cout;
    const int ci = (idx / Wdim) % Cin;
    const int w_idx = idx % Wdim;

    // Find which block this weight belongs to
    int blk = -1;
    int w_local = w_idx;
    for (int k = 0; k < num_blocks; k++) {
        int m_val = block_m[k];
        int n_in_val = block_n_in[k];
        int n_out_val = block_n_out[k];
        int block_size = (m_val == 0) ? (n_out_val * n_in_val) : (2 * n_out_val * n_in_val);
        if (w_local < block_size) {
            blk = k;
            break;
        }
        w_local -= block_size;
    }

    if (blk < 0) return;

    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];

    const scalar_t* f_ptr = features + b * Cin * Din + ci * Din + in_off;
    const scalar_t* go_ptr = grad_output + b * Cout * Dout + co * Dout + out_off;

    float grad = 0.0f;

    if (m == 0) {
        int o = w_local / n_in;
        int i = w_local % n_in;
        grad = static_cast<float>(f_ptr[i]) * static_cast<float>(go_ptr[o]);
    } else {
        int temp = w_local / 2;
        int ab = w_local % 2;
        int o = temp / n_in;
        int i = temp % n_in;

        float f_re = static_cast<float>(f_ptr[i]);
        float f_im = static_cast<float>(f_ptr[n_in + i]);
        float go_re = static_cast<float>(go_ptr[o]);
        float go_im = static_cast<float>(go_ptr[n_out + o]);

        if (ab == 0) {
            grad = f_re * go_re + f_im * go_im;
        } else {
            grad = f_im * go_re - f_re * go_im;
        }
    }

    grad_weights[idx] = static_cast<scalar_t>(grad);
}


// C++ interface

std::vector<torch::Tensor> block_diagonal_forward_cuda(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor out_to_block,
    torch::Tensor out_to_local,
    int dim_out
) {
    const int B = features.size(0);
    const int Cin = features.size(1);
    const int Din = features.size(2);
    const int Cout = weights.size(1);
    const int Wdim = weights.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    const int total_outputs = B * Cout * dim_out;
    const int threads = THREADS_PER_BLOCK;
    const int num_warps = threads / 32;
    const size_t shared_size = num_warps * 2 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward", ([&] {
        block_diagonal_forward_kernel<scalar_t><<<total_outputs, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_m.data_ptr<int>(),
            block_n_in.data_ptr<int>(),
            block_n_out.data_ptr<int>(),
            block_in_off.data_ptr<int>(),
            block_out_off.data_ptr<int>(),
            block_w_off.data_ptr<int>(),
            out_to_block.data_ptr<int>(),
            out_to_local.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks
        );
    }));

    return {output};
}


std::vector<torch::Tensor> block_diagonal_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    int dim_in
) {
    const int B = features.size(0);
    const int Cin = features.size(1);
    const int Din = features.size(2);
    const int Cout = weights.size(1);
    const int Wdim = weights.size(3);
    const int Dout = grad_output.size(2);
    const int num_blocks = block_m.size(0);

    auto grad_features = torch::zeros_like(features);
    auto grad_weights = torch::zeros_like(weights);

    const int threads = 256;

    // Features gradient
    {
        const int total = B * Cin * Din;
        const int blocks = (total + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_backward_features", ([&] {
            block_diagonal_backward_features_kernel<scalar_t><<<blocks, threads>>>(
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

    // Weights gradient
    {
        const int total = B * Cout * Cin * Wdim;
        const int blocks = (total + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_backward_weights", ([&] {
            block_diagonal_backward_weights_kernel<scalar_t><<<blocks, threads>>>(
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda, "Block diagonal forward (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda, "Block diagonal backward (CUDA)");
}
