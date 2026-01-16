/**
 * @file block_diagonal_binned.cu
 * @brief Memory-efficient block-diagonal multiplication with binned radial weights.
 *
 * This kernel implements the Lambda step in SO(3)-equivariant linear layers using
 * binned weight interpolation for memory efficiency. Instead of storing per-edge
 * weights (B, Cout, Cin, Wdim), weights are precomputed at K bin edges and
 * linearly interpolated at runtime.
 *
 * Memory reduction: O(B * Cout * Cin * Wdim) -> O(K * Cout * Cin * Wdim)
 * Typical speedup: 1.5-2x faster than standard approach during training
 *
 * Block structure for SO(3):
 *   - m=0 blocks: 1x1 real scalars
 *   - m>0 blocks: 2x2 complex-type matrices [a, b; -b, a]
 *
 * @author Hamish Blair
 * @see docs/theory.tex for mathematical details
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


//------------------------------------------------------------------------------
// Forward Kernel
//------------------------------------------------------------------------------

/**
 * Forward pass with linear interpolation between bin edges.
 *
 * Grid: B * num_m_blocks thread blocks
 * Each block: Loads features into shared memory, interpolates weights from
 *             two adjacent bins, and computes block-diagonal output.
 *
 * @param features      Input features (B, Cin, Din)
 * @param radial_table  Weight table at bin edges (K+1, Cout, Cin, Wdim)
 * @param bin_lo        Lower bin index per edge (B,)
 * @param interp_weight Interpolation weight t in [0,1] (B,)
 * @param output        Output features (B, Cout, Dout)
 * @param block_*       Block structure metadata
 * @param num_bins      Number of bins (for computing bin_hi = min(bin_lo + 1, num_bins))
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_binned_interp_kernel(
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_data,  // (num_blocks, 6): [m, n_in, n_out, in_off, out_off, w_off]
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Unpack block parameters from packed tensor
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
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Interpolation parameters
    const int idx_lo = bin_lo[b];
    const int idx_hi = min(idx_lo + 1, num_bins);  // Compute bin_hi on the fly
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;
    const int64_t table_stride = Cout * Cin * Wdim;

    // Compute outputs
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        const scalar_t* w_base_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim;
        const scalar_t* w_base_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim;

        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const scalar_t* w_ptr_lo = w_base_lo + ci * Wdim + w_off;
            const scalar_t* w_ptr_hi = w_base_hi + ci * Wdim + w_off;

            if (m == 0) {
                // Real block: dot product with interpolated weights
                for (int i = 0; i < n_in; i++) {
                    const float w_lo = static_cast<float>(w_ptr_lo[o_local * n_in + i]);
                    const float w_hi = static_cast<float>(w_ptr_hi[o_local * n_in + i]);
                    acc += (one_minus_t * w_lo + t * w_hi) * f_ptr[i];
                }
            } else {
                // Complex block: [a, b; -b, a] @ [f_re; f_im]
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];
                    const int w_idx = (o_local * n_in + i) * 2;

                    const float a = one_minus_t * static_cast<float>(w_ptr_lo[w_idx])
                                  + t * static_cast<float>(w_ptr_hi[w_idx]);
                    const float bv = one_minus_t * static_cast<float>(w_ptr_lo[w_idx + 1])
                                   + t * static_cast<float>(w_ptr_hi[w_idx + 1]);

                    acc += a * f_re + bv * f_im;
                    acc_im += a * f_im - bv * f_re;
                }
            }
        }

        // Write output
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
 *
 * Uses on-the-fly weight interpolation to avoid materializing the full
 * (B, Cout, Cin, Wdim) weights tensor.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_binned_interp_features_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    scalar_t* __restrict__ grad_features,
    const int* __restrict__ block_data,  // (num_blocks, 6): [m, n_in, n_out, in_off, out_off, w_off]
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Unpack block parameters from packed tensor
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
    const int idx_lo = bin_lo[b];
    const int idx_hi = min(idx_lo + 1, num_bins);  // Compute bin_hi on the fly
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;
    const int64_t table_stride = Cout * Cin * Wdim;

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

    // Compute grad_features
    const int total_inputs = Cin * in_size;

    for (int in_idx = tid; in_idx < total_inputs; in_idx += num_threads) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;

        float grad = 0.0f;

        if (m == 0) {
            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim + ci * Wdim + w_off;
                const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    const float w = one_minus_t * static_cast<float>(w_ptr_lo[o * n_in + i_local])
                                  + t * static_cast<float>(w_ptr_hi[o * n_in + i_local]);
                    grad += w * go_ptr[o];
                }
            }
        } else {
            bool is_real = (i_local < n_in);
            int i_idx = is_real ? i_local : (i_local - n_in);

            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim + ci * Wdim + w_off;
                const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    float go_re = go_ptr[o];
                    float go_im = go_ptr[n_out + o];
                    int w_idx = (o * n_in + i_idx) * 2;

                    float a = one_minus_t * static_cast<float>(w_ptr_lo[w_idx])
                            + t * static_cast<float>(w_ptr_hi[w_idx]);
                    float bv = one_minus_t * static_cast<float>(w_ptr_lo[w_idx + 1])
                             + t * static_cast<float>(w_ptr_hi[w_idx + 1]);

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
 * Backward pass: compute grad_radial_table and grad_interp_weight.
 *
 * Uses atomic operations to scatter gradients to the lookup table.
 * Also computes grad_interp_weight for force computation via distances.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_binned_interp_table_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ features,
    const scalar_t* __restrict__ radial_table,
    const int* __restrict__ bin_lo,
    const scalar_t* __restrict__ interp_weight,
    float* __restrict__ grad_radial_table,
    float* __restrict__ grad_interp_weight,
    const int* __restrict__ block_data,  // (num_blocks, 6): [m, n_in, n_out, in_off, out_off, w_off]
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks, int num_bins
) {
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Unpack block parameters from packed tensor
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
    const int idx_lo = bin_lo[b];
    const int idx_hi = min(idx_lo + 1, num_bins);  // Compute bin_hi on the fly
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;
    const int64_t table_stride = Cout * Cin * Wdim;

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

    // Accumulate grad_interp_weight locally
    float local_grad_t = 0.0f;

    // Compute weight gradients
    const int64_t total_weights = Cout * Cin * w_block_size;

    for (int64_t w_idx = tid; w_idx < total_weights; w_idx += num_threads) {
        const int64_t co = w_idx / (Cin * w_block_size);
        const int64_t ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        float grad_w = 0.0f;

        if (m == 0) {
            int o = w_local / n_in;
            int i = w_local % n_in;
            grad_w = f_ptr[i] * go_ptr[o];
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
                grad_w = f_re * go_re + f_im * go_im;
            } else {
                grad_w = f_im * go_re - f_re * go_im;
            }
        }

        // Scatter to grad_radial_table
        const int64_t table_idx = co * Cin * Wdim + ci * Wdim + w_off + w_local;
        atomicAdd(&grad_radial_table[idx_lo * table_stride + table_idx], one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[idx_hi * table_stride + table_idx], t * grad_w);

        // Accumulate grad_interp_weight
        const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + table_idx;
        const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + table_idx;
        float w_diff = static_cast<float>(*w_ptr_hi) - static_cast<float>(*w_ptr_lo);
        local_grad_t += w_diff * grad_w;
    }

    // Warp-level reduction for grad_interp_weight
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_t += __shfl_down_sync(0xffffffff, local_grad_t, offset);
    }
    if ((tid % 32) == 0) {
        atomicAdd(&grad_interp_weight[b], local_grad_t);
    }
}


//------------------------------------------------------------------------------
// C++ Wrapper Functions
//------------------------------------------------------------------------------

std::vector<torch::Tensor> block_diagonal_forward_binned_interp_cuda(
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_lo,
    torch::Tensor interp_weight,
    torch::Tensor block_data,  // (num_blocks, 6): [m, n_in, n_out, in_off, out_off, w_off]
    int64_t Cout,
    int dim_out,
    int num_bins,
    int max_in_size
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Wdim = radial_table.size(3);
    const int num_blocks = block_data.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "forward_binned_interp", ([&] {
        block_diagonal_forward_binned_interp_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_data.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks, num_bins
        );
    }));

    return {output};
}


std::vector<torch::Tensor> block_diagonal_backward_binned_interp_cuda(
    torch::Tensor grad_output,
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_lo,
    torch::Tensor interp_weight,
    torch::Tensor block_data,  // (num_blocks, 6): [m, n_in, n_out, in_off, out_off, w_off]
    int dim_in,
    int max_in_size,
    int max_out_size
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = grad_output.size(1);
    const int64_t Dout = grad_output.size(2);
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int num_bins = num_bins_plus_1 - 1;
    const int64_t Wdim = radial_table.size(3);
    const int num_blocks = block_data.size(0);

    auto grad_features = torch::zeros_like(features);
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_interp_weight = torch::zeros({B}, interp_weight.options().dtype(torch::kFloat32));

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Features backward
    {
        const size_t shared_size = Cout * max_out_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_features", ([&] {
            block_diagonal_backward_binned_interp_features_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                block_data.data_ptr<int>(),
                B, Cin, Cout, Din, Dout, Wdim, num_blocks, num_bins
            );
        }));
    }

    // Table backward
    {
        const size_t shared_size = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_table", ([&] {
            block_diagonal_backward_binned_interp_table_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
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
    m.def("forward_binned_interp", &block_diagonal_forward_binned_interp_cuda,
          "Block-diagonal forward with binned interpolated weights (CUDA)");
    m.def("backward_binned_interp", &block_diagonal_backward_binned_interp_cuda,
          "Block-diagonal backward with binned interpolated weights (CUDA)");
}
