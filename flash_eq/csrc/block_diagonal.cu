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
 * Uses int64_t for all index calculations to avoid overflow with large tensors.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 128


/*
 * Forward kernel: Each thread block handles one output element (b, co, out_idx).
 * Threads within the block cooperatively reduce over channels_in.
 */
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
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    extern __shared__ char shared_mem[];
    float* sdata = reinterpret_cast<float*>(shared_mem);

    const int64_t global_idx = blockIdx.x;
    const int64_t out_idx = global_idx % Dout;
    const int64_t co = (global_idx / Dout) % Cout;
    const int64_t b = global_idx / (Dout * Cout);

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
    for (int64_t ci = tid; ci < Cin; ci += blockDim.x) {
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
            const int64_t out_base = b * Cout * Dout + co * Dout + out_off;
            output[out_base + o_local] = static_cast<scalar_t>(acc);
            if (m > 0) {
                output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
            }
        }
    }
}


/*
 * V2 Forward kernel: Process by m-block instead of by output element.
 *
 * Grid: B × num_m_blocks thread blocks
 * Each block: Loads features for that m-block into shared memory, then
 *             threads compute (cout, out_local) pairs in parallel.
 *
 * This reduces thread block count from B×Cout×Dout to B×num_m_blocks,
 * and improves data reuse by caching features in shared memory.
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_v2_kernel(
    const scalar_t* __restrict__ features,  // (B, Cin, Din)
    const scalar_t* __restrict__ weights,   // (B, Cout, Cin, Wdim)
    scalar_t* __restrict__ output,          // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    // Each block handles one (batch, m-block) pair
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get m-block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Shared memory layout: features for all Cin channels
    // feat_shared[ci * in_size + local_idx]
    extern __shared__ float feat_shared[];

    // Cooperatively load features into shared memory
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }
    __syncthreads();

    // Each thread computes a subset of (cout, out_local) pairs
    // Total outputs for this m-block: Cout × n_out (each produces real, and imag if m>0)
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Weight base for (b, co)
        const scalar_t* w_base = weights + b * Cout * Cin * Wdim + co * Cin * Wdim;

        // Sum over all input channels (using cached features)
        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const scalar_t* w_ptr = w_base + ci * Wdim + w_off;

            if (m == 0) {
                // Real block: dot product
                for (int i = 0; i < n_in; i++) {
                    acc += static_cast<float>(w_ptr[o_local * n_in + i]) * f_ptr[i];
                }
            } else {
                // Complex block: [a, b; -b, a] @ [f_re; f_im]
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

        // Write output
        const int64_t out_base = b * Cout * Dout + co * Dout + out_off;
        output[out_base + o_local] = static_cast<scalar_t>(acc);
        if (m > 0) {
            output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
        }
    }
}


/*
 * Binned Forward kernel: Uses lookup table for radial weights.
 *
 * Instead of storing weights per edge (B, Cout, Cin, Wdim), we store:
 *   - radial_table: (num_bins, Wdim) - weights per distance bin
 *   - bin_indices: (B,) - which bin each edge belongs to
 *
 * The weights are SHARED across all (cout, cin) pairs for a given edge.
 * This reduces memory from O(B * Cout * Cin * Wdim) to O(num_bins * Wdim).
 *
 * Grid: B × num_m_blocks thread blocks (same as V2)
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_binned_kernel(
    const scalar_t* __restrict__ features,      // (B, Cin, Din)
    const scalar_t* __restrict__ radial_table,  // (num_bins, Wdim)
    const int* __restrict__ bin_indices,        // (B,)
    scalar_t* __restrict__ output,              // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    // Each block handles one (batch, m-block) pair
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get m-block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    const int in_size = (m == 0) ? n_in : 2 * n_in;
    const int w_block_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Shared memory layout:
    // [0, Cin * in_size): features
    // [Cin * in_size, Cin * in_size + w_block_size): weights for this m-block
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* weight_shared = shared + Cin * in_size;

    // Cooperatively load features into shared memory
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load weights from radial table (same for all channels!)
    const int bin_idx = bin_indices[b];
    const scalar_t* w_ptr = radial_table + bin_idx * Wdim + w_off;

    for (int i = tid; i < w_block_size; i += num_threads) {
        weight_shared[i] = static_cast<float>(w_ptr[i]);
    }
    __syncthreads();

    // Each thread computes a subset of (cout, cin, out_local) tuples
    // With shared weights, we sum over cin for each (cout, out_local) pair
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Sum over all input channels (using cached features and shared weights)
        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;

            if (m == 0) {
                // Real block: dot product with shared weights
                for (int i = 0; i < n_in; i++) {
                    acc += weight_shared[o_local * n_in + i] * f_ptr[i];
                }
            } else {
                // Complex block: [a, b; -b, a] @ [f_re; f_im]
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];

                    const int w_idx = (o_local * n_in + i) * 2;
                    const float a = weight_shared[w_idx];
                    const float bv = weight_shared[w_idx + 1];

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


/*
 * Binned Forward kernel with linear interpolation between bins.
 *
 * Uses two adjacent bin entries and interpolates:
 *   weight = (1 - t) * table[bin_lo] + t * table[bin_hi]
 *
 * This provides smoother gradients for training.
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_binned_interp_kernel(
    const scalar_t* __restrict__ features,      // (B, Cin, Din)
    const scalar_t* __restrict__ radial_table,  // (num_bins + 1, Wdim)
    const int* __restrict__ bin_lo,             // (B,)
    const int* __restrict__ bin_hi,             // (B,)
    const scalar_t* __restrict__ interp_weight, // (B,)
    scalar_t* __restrict__ output,              // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    // Each block handles one (batch, m-block) pair
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get m-block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    const int in_size = (m == 0) ? n_in : 2 * n_in;
    const int w_block_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Shared memory layout:
    // [0, Cin * in_size): features
    // [Cin * in_size, Cin * in_size + w_block_size): interpolated weights
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* weight_shared = shared + Cin * in_size;

    // Cooperatively load features
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Load and interpolate weights
    const int idx_lo = bin_lo[b];
    const int idx_hi = bin_hi[b];
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;

    const scalar_t* w_lo = radial_table + idx_lo * Wdim + w_off;
    const scalar_t* w_hi = radial_table + idx_hi * Wdim + w_off;

    for (int i = tid; i < w_block_size; i += num_threads) {
        weight_shared[i] = one_minus_t * static_cast<float>(w_lo[i]) +
                           t * static_cast<float>(w_hi[i]);
    }
    __syncthreads();

    // Compute outputs (same as non-interpolated version)
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;

            if (m == 0) {
                for (int i = 0; i < n_in; i++) {
                    acc += weight_shared[o_local * n_in + i] * f_ptr[i];
                }
            } else {
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];

                    const int w_idx = (o_local * n_in + i) * 2;
                    const float a = weight_shared[w_idx];
                    const float bv = weight_shared[w_idx + 1];

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


/*
 * V2 Backward Features kernel: Process by m-block instead of by input element.
 *
 * Grid: B × num_m_blocks thread blocks
 * Each block: Loads grad_output for that m-block into shared memory, then
 *             threads compute (cin, in_local) pairs in parallel.
 *
 * This reduces thread block count and improves data reuse.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_features_v2_kernel(
    const scalar_t* __restrict__ grad_output,  // (B, Cout, Dout)
    const scalar_t* __restrict__ weights,      // (B, Cout, Cin, Wdim)
    scalar_t* __restrict__ grad_features,      // (B, Cin, Din)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    // Each block handles one (batch, m-block) pair
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get m-block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int in_size = (m == 0) ? n_in : 2 * n_in;

    // Shared memory: grad_output for all Cout channels
    // grad_shared[co * out_size + local_idx]
    extern __shared__ float grad_shared[];

    // Cooperatively load grad_output into shared memory
    const int64_t grad_base = b * Cout * Dout;
    const int total_grad_elems = Cout * out_size;

    for (int i = tid; i < total_grad_elems; i += num_threads) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Each thread computes a subset of (cin, in_local) pairs
    // Total inputs for this m-block: Cin × in_size
    const int total_inputs = Cin * in_size;

    for (int in_idx = tid; in_idx < total_inputs; in_idx += num_threads) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;

        float grad = 0.0f;

        if (m == 0) {
            // Real block: grad_f[i] = sum_o W[o,i] * grad_out[o]
            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    grad += static_cast<float>(w_ptr[o * n_in + i_local]) * go_ptr[o];
                }
            }
        } else {
            // Complex block: transpose of [a, b; -b, a] is [a, -b; b, a]
            // grad_f_re = a*grad_out_re - b*grad_out_im
            // grad_f_im = b*grad_out_re + a*grad_out_im
            bool is_real_input = (i_local < n_in);
            int i_idx = is_real_input ? i_local : (i_local - n_in);

            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr = weights + b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    float go_re = go_ptr[o];
                    float go_im = go_ptr[n_out + o];

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

        // Write gradient
        const int64_t feat_idx = b * Cin * Din + ci * Din + in_off + i_local;
        grad_features[feat_idx] = static_cast<scalar_t>(grad);
    }
}


/*
 * V2 Backward Weights kernel: Process by m-block instead of by weight element.
 *
 * Grid: B × num_m_blocks thread blocks
 * Each block: Loads features and grad_output for that m-block into shared memory,
 *             then threads compute (cout, cin, w_local) tuples in parallel.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_weights_v2_kernel(
    const scalar_t* __restrict__ grad_output,  // (B, Cout, Dout)
    const scalar_t* __restrict__ features,     // (B, Cin, Din)
    scalar_t* __restrict__ grad_weights,       // (B, Cout, Cin, Wdim)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    // Each block handles one (batch, m-block) pair
    const int blk = blockIdx.x % num_blocks;
    const int64_t b = blockIdx.x / num_blocks;

    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get m-block parameters
    const int m = block_m[blk];
    const int n_in = block_n_in[blk];
    const int n_out = block_n_out[blk];
    const int in_off = block_in_off[blk];
    const int out_off = block_out_off[blk];
    const int w_off = block_w_off[blk];

    const int in_size = (m == 0) ? n_in : 2 * n_in;
    const int out_size = (m == 0) ? n_out : 2 * n_out;
    const int w_block_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Shared memory layout:
    // [0, Cin * in_size): features
    // [Cin * in_size, Cin * in_size + Cout * out_size): grad_output
    extern __shared__ float shared[];
    float* feat_shared = shared;
    float* grad_shared = shared + Cin * in_size;

    // Cooperatively load features into shared memory
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // Cooperatively load grad_output into shared memory
    const int64_t grad_base = b * Cout * Dout;
    const int total_grad_elems = Cout * out_size;

    for (int i = tid; i < total_grad_elems; i += num_threads) {
        const int co = i / out_size;
        const int local_idx = i % out_size;
        grad_shared[i] = static_cast<float>(grad_output[grad_base + co * Dout + out_off + local_idx]);
    }
    __syncthreads();

    // Each thread computes a subset of (cout, cin, w_local) tuples
    // Total weights for this m-block: Cout × Cin × w_block_size
    const int64_t total_weights = Cout * Cin * w_block_size;

    for (int64_t w_idx = tid; w_idx < total_weights; w_idx += num_threads) {
        const int64_t co = w_idx / (Cin * w_block_size);
        const int64_t ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        float grad = 0.0f;

        if (m == 0) {
            // Real block: grad_W[o,i] = f[i] * grad_out[o]
            int o = w_local / n_in;
            int i = w_local % n_in;
            grad = f_ptr[i] * go_ptr[o];
        } else {
            // Complex block: grad for (a, b) in [a, b; -b, a]
            // grad_a = f_re * grad_out_re + f_im * grad_out_im
            // grad_b = f_im * grad_out_re - f_re * grad_out_im
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

        // Write gradient
        const int64_t weight_idx = b * Cout * Cin * Wdim + co * Cin * Wdim + ci * Wdim + w_off + w_local;
        grad_weights[weight_idx] = static_cast<scalar_t>(grad);
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
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = B * Cin * Din;
    if (idx >= total) return;

    const int64_t b = idx / (Cin * Din);
    const int64_t ci = (idx / Din) % Cin;
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
        for (int64_t co = 0; co < Cout; co++) {
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

        for (int64_t co = 0; co < Cout; co++) {
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
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim, int num_blocks
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = B * Cout * Cin * Wdim;
    if (idx >= total) return;

    const int64_t CoutCinWdim = Cout * Cin * Wdim;
    const int64_t CinWdim = Cin * Wdim;
    const int64_t b = idx / CoutCinWdim;
    const int64_t co = (idx / CinWdim) % Cout;
    const int64_t ci = (idx / Wdim) % Cin;
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
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = weights.size(1);
    const int64_t Wdim = weights.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    const int64_t total_outputs = B * Cout * dim_out;
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
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = weights.size(1);
    const int64_t Wdim = weights.size(3);
    const int64_t Dout = grad_output.size(2);
    const int num_blocks = block_m.size(0);

    auto grad_features = torch::zeros_like(features);
    auto grad_weights = torch::zeros_like(weights);

    const int threads = 256;

    // Features gradient
    {
        const int64_t total = B * Cin * Din;
        const int64_t blocks = (total + threads - 1) / threads;

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
        const int64_t total = B * Cout * Cin * Wdim;
        const int64_t blocks = (total + threads - 1) / threads;

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
    torch::Tensor block_in_size,  // precomputed: n_in for m=0, 2*n_in for m>0
    torch::Tensor block_out_size, // precomputed: n_out for m=0, 2*n_out for m>0
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

    // Compute max sizes for shared memory
    const int max_in_size = block_in_size.max().item<int>();
    const int max_out_size = block_out_size.max().item<int>();

    // Features backward: shared memory for Cout × max_out_size floats
    {
        const size_t shared_size = Cout * max_out_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_backward_features_v2", ([&] {
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

    // Weights backward: shared memory for Cin × max_in_size + Cout × max_out_size floats
    {
        const size_t shared_size = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_backward_weights_v2", ([&] {
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


std::vector<torch::Tensor> block_diagonal_forward_v2_cuda(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,  // precomputed: n_in for m=0, 2*n_in for m>0
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = weights.size(1);
    const int64_t Wdim = weights.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size floats
    // We compute max_in_size from block_in_size tensor
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_v2", ([&] {
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


std::vector<torch::Tensor> block_diagonal_forward_binned_cuda(
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_indices,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    torch::Tensor block_w_size,  // weight block size for each m-block
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Wdim = radial_table.size(1);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size + max_w_size floats
    const int max_in_size = block_in_size.max().item<int>();
    const int max_w_size = block_w_size.max().item<int>();
    const size_t shared_size = (Cin * max_in_size + max_w_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_binned", ([&] {
        block_diagonal_forward_binned_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_indices.data_ptr<int>(),
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


std::vector<torch::Tensor> block_diagonal_forward_binned_interp_cuda(
    torch::Tensor features,
    torch::Tensor radial_table,
    torch::Tensor bin_lo,
    torch::Tensor bin_hi,
    torch::Tensor interp_weight,
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    torch::Tensor block_w_size,
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Wdim = radial_table.size(1);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size + max_w_size floats
    const int max_in_size = block_in_size.max().item<int>();
    const int max_w_size = block_w_size.max().item<int>();
    const size_t shared_size = (Cin * max_in_size + max_w_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_binned_interp", ([&] {
        block_diagonal_forward_binned_interp_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            radial_table.data_ptr<scalar_t>(),
            bin_lo.data_ptr<int>(),
            bin_hi.data_ptr<int>(),
            interp_weight.data_ptr<scalar_t>(),
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_diagonal_forward_cuda, "Block diagonal forward (CUDA)");
    m.def("forward_v2", &block_diagonal_forward_v2_cuda, "Block diagonal forward V2 - m-block parallel (CUDA)");
    m.def("forward_binned", &block_diagonal_forward_binned_cuda, "Block diagonal forward with binned weights (CUDA)");
    m.def("forward_binned_interp", &block_diagonal_forward_binned_interp_cuda, "Block diagonal forward with interpolated binned weights (CUDA)");
    m.def("backward", &block_diagonal_backward_cuda, "Block diagonal backward (CUDA)");
    m.def("backward_v2", &block_diagonal_backward_v2_cuda, "Block diagonal backward V2 - m-block parallel (CUDA)");
}
