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
 *   - radial_table: (num_bins, Cout, Cin, Wdim) - weights per distance bin
 *   - bin_indices: (B,) - which bin each edge belongs to
 *
 * This reduces memory from O(B * Cout * Cin * Wdim) to O(num_bins * Cout * Cin * Wdim).
 * Memory reduction factor is batch_size / num_bins (e.g., 50x for batch=5000, bins=100).
 *
 * Identical to V2 kernel, but indexes weights by bin_indices[b] instead of b.
 *
 * Grid: B × num_m_blocks thread blocks
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_binned_kernel(
    const scalar_t* __restrict__ features,      // (B, Cin, Din)
    const scalar_t* __restrict__ radial_table,  // (num_bins, Cout, Cin, Wdim)
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

    // Shared memory layout: features for all Cin channels
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
    const int total_outputs = Cout * n_out;

    // Get bin index for this batch element - only difference from V2!
    const int bin_idx = bin_indices[b];

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Weight base: radial_table[bin_idx, co, :, :]
        const scalar_t* w_base = radial_table + bin_idx * Cout * Cin * Wdim + co * Cin * Wdim;

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
 * Binned Forward kernel with linear interpolation between bins.
 *
 * Uses two adjacent bin entries and interpolates:
 *   weight = (1 - t) * table[bin_lo] + t * table[bin_hi]
 *
 * Table shape: (num_bins + 1, Cout, Cin, Wdim) - evaluated at bin edges for interpolation.
 * This provides smoother gradients for training.
 *
 * Identical to V2 kernel structure, but interpolates weights from two bin entries.
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_binned_interp_kernel(
    const scalar_t* __restrict__ features,      // (B, Cin, Din)
    const scalar_t* __restrict__ radial_table,  // (num_bins + 1, Cout, Cin, Wdim)
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

    // Shared memory layout: features for all Cin channels
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

    // Interpolation parameters for this batch element
    const int idx_lo = bin_lo[b];
    const int idx_hi = bin_hi[b];
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;

    // Table stride: Cout * Cin * Wdim elements per bin entry
    const int64_t table_stride = Cout * Cin * Wdim;

    // Each thread computes a subset of (cout, out_local) pairs
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Weight bases for low and high bin entries: table[idx, co, :, :]
        const scalar_t* w_base_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim;
        const scalar_t* w_base_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim;

        // Sum over all input channels (using cached features)
        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;
            const scalar_t* w_ptr_lo = w_base_lo + ci * Wdim + w_off;
            const scalar_t* w_ptr_hi = w_base_hi + ci * Wdim + w_off;

            if (m == 0) {
                // Real block: dot product with interpolated weights
                for (int i = 0; i < n_in; i++) {
                    const float w_lo = static_cast<float>(w_ptr_lo[o_local * n_in + i]);
                    const float w_hi = static_cast<float>(w_ptr_hi[o_local * n_in + i]);
                    const float w = one_minus_t * w_lo + t * w_hi;
                    acc += w * f_ptr[i];
                }
            } else {
                // Complex block: [a, b; -b, a] @ [f_re; f_im] with interpolated weights
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];

                    const int w_idx = (o_local * n_in + i) * 2;
                    const float a_lo = static_cast<float>(w_ptr_lo[w_idx]);
                    const float b_lo = static_cast<float>(w_ptr_lo[w_idx + 1]);
                    const float a_hi = static_cast<float>(w_ptr_hi[w_idx]);
                    const float b_hi = static_cast<float>(w_ptr_hi[w_idx + 1]);

                    const float a = one_minus_t * a_lo + t * a_hi;
                    const float bv = one_minus_t * b_lo + t * b_hi;

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
 * Broadcast Forward kernel: Batch-independent weights broadcast across batch.
 *
 * Instead of per-batch weights (B, Cout, Cin, Wdim), this kernel accepts
 * batch-independent weights (Cout, Cin, Wdim) and broadcasts them.
 *
 * This enables memory-efficient computation where the same weights are
 * applied to all batch elements - useful for the reformulated fused approach.
 *
 * Grid: B × num_m_blocks thread blocks
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_broadcast_kernel(
    const scalar_t* __restrict__ features,  // (B, Cin, Din)
    const scalar_t* __restrict__ weights,   // (Cout, Cin, Wdim) - NO batch dimension!
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

    // Shared memory: features for all Cin channels
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
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Weight base for co - NO batch indexing!
        const scalar_t* w_base = weights + co * Cin * Wdim;

        // Sum over all input channels
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

        // Write output
        const int64_t out_base = b * Cout * Dout + co * Dout + out_off;
        output[out_base + o_local] = static_cast<scalar_t>(acc);
        if (m > 0) {
            output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
        }
    }
}


/*
 * Batched Broadcast Forward kernel: Process multiple weight slices at once.
 *
 * Instead of H separate kernel launches for W3[:,:,0,:], W3[:,:,1,:], etc.,
 * this kernel processes all H slices in one launch and outputs (B, H, Cout, Dout).
 *
 * Then the caller can do: output = einsum('bh, bhcd -> bcd', hidden2, batched_output)
 *
 * Grid: B × num_m_blocks × H thread blocks (3D grid)
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_broadcast_batched_kernel(
    const scalar_t* __restrict__ features,  // (B, Cin, Din)
    const scalar_t* __restrict__ weights,   // (H, Cout, Cin, Wdim) - H weight slices
    scalar_t* __restrict__ output,          // (B, H, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim,
    int num_blocks, int H
) {
    // 3D grid: (blk, b, h)
    const int blk = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int h = blockIdx.z;

    if (b >= B || h >= H) return;

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

    // Shared memory: features for all Cin channels
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
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Weight base for (h, co) - note H is first dimension
        const scalar_t* w_base = weights + h * Cout * Cin * Wdim + co * Cin * Wdim;

        // Sum over all input channels
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

        // Write output: (B, H, Cout, Dout)
        const int64_t out_base = b * H * Cout * Dout + h * Cout * Dout + co * Dout + out_off;
        output[out_base + o_local] = static_cast<scalar_t>(acc);
        if (m > 0) {
            output[out_base + n_out + o_local] = static_cast<scalar_t>(acc_im);
        }
    }
}


/*
 * Optimized Fused Final Layer + Block-Diagonal kernel.
 *
 * Computes: output = block_diagonal(features, hidden2 @ W3 + b3)
 * without materializing the (B, Cout, Cin, Wdim) weight tensor.
 *
 * Key optimizations:
 *   1. W3 is transposed to (Cout, Cin, Wdim, H) for contiguous hidden dim access
 *   2. hidden2 cached in shared memory (reused across all outputs)
 *   3. Features cached in shared memory
 *   4. Each thread independently computes assigned outputs (no serialization)
 *
 * Grid: B × num_m_blocks thread blocks
 */
template <typename scalar_t>
__global__ void block_diagonal_fused_final_kernel(
    const scalar_t* __restrict__ features,  // (B, Cin, Din)
    const scalar_t* __restrict__ hidden2,   // (B, H) - precomputed MLP hidden layer
    const scalar_t* __restrict__ W3,        // (Cout, Cin, Wdim, H) - TRANSPOSED for contiguous H access
    const scalar_t* __restrict__ b3,        // (Cout, Cin, Wdim)
    scalar_t* __restrict__ output,          // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim,
    int num_blocks, int H
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

    // Shared memory layout:
    // [0, H): hidden2 for this batch element
    // [H, H + Cin*in_size): features for this m-block
    extern __shared__ float smem[];
    float* hidden2_shared = smem;
    float* feat_shared = smem + H;

    // Cooperatively load hidden2 into shared memory
    for (int i = tid; i < H; i += num_threads) {
        hidden2_shared[i] = static_cast<float>(hidden2[b * H + i]);
    }

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
    const int total_outputs = Cout * n_out;

    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;

        // Sum over all input channels
        for (int ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;

            if (m == 0) {
                // Real block: need n_in weights
                for (int i = 0; i < n_in; i++) {
                    const int w_idx = w_off + o_local * n_in + i;

                    // Compute weight = dot(hidden2, W3[co, ci, w_idx, :]) + b3[co, ci, w_idx]
                    // W3 layout: (Cout, Cin, Wdim, H) - H is contiguous
                    const scalar_t* W3_ptr = W3 + ((co * Cin + ci) * Wdim + w_idx) * H;
                    float weight = static_cast<float>(b3[(co * Cin + ci) * Wdim + w_idx]);

                    // Dot product with hidden2 (contiguous access to W3)
                    for (int h = 0; h < H; h++) {
                        weight += hidden2_shared[h] * static_cast<float>(W3_ptr[h]);
                    }

                    acc += weight * f_ptr[i];
                }
            } else {
                // Complex block: need 2*n_in weights (a, b pairs)
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];
                    const int w_idx_a = w_off + (o_local * n_in + i) * 2;
                    const int w_idx_b = w_idx_a + 1;

                    // Compute weight_a and weight_b
                    const scalar_t* W3_ptr_a = W3 + ((co * Cin + ci) * Wdim + w_idx_a) * H;
                    const scalar_t* W3_ptr_b = W3 + ((co * Cin + ci) * Wdim + w_idx_b) * H;
                    float a = static_cast<float>(b3[(co * Cin + ci) * Wdim + w_idx_a]);
                    float bv = static_cast<float>(b3[(co * Cin + ci) * Wdim + w_idx_b]);

                    for (int h = 0; h < H; h++) {
                        a += hidden2_shared[h] * static_cast<float>(W3_ptr_a[h]);
                        bv += hidden2_shared[h] * static_cast<float>(W3_ptr_b[h]);
                    }

                    // Complex multiplication: [a, b; -b, a] @ [f_re; f_im]
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
 * Fused MLP + Block-Diagonal kernel (V2 - Cooperative Weight Computation).
 *
 * Instead of precomputing weights = MLP(edge_features) and storing them in a
 * large (B, Cout, Cin, Wdim) tensor, we compute the MLP on-the-fly inside the
 * kernel and immediately use the weights for block-diagonal multiplication.
 *
 * This eliminates the O(B * Cout * Cin * Wdim) intermediate tensor, reducing
 * memory usage dramatically (e.g., from 36GB to ~1GB for typical configs).
 *
 * Key optimization: For each (co, ci) pair, ALL threads cooperatively compute
 * the weights into shared memory, then ALL threads use those weights. This
 * provides parallelism and weight reuse across threads.
 *
 * MLP structure:
 *   hidden1 = silu(edge_feat * w1 + b1)     // (hidden_dim,)
 *   hidden2 = silu(hidden1 @ W2 + b2)       // (hidden_dim,)
 *   For each (co, ci):
 *     weights[co,ci,w_off:w_off+block_w_size] = hidden2 @ W3[co,ci,:,w_off:...] + b3[...]
 *
 * Grid: B × num_m_blocks thread blocks
 */
template <typename scalar_t>
__global__ void block_diagonal_forward_fused_mlp_kernel(
    const scalar_t* __restrict__ features,      // (B, Cin, Din)
    const scalar_t* __restrict__ edge_features, // (B,) - single distance value per edge
    // MLP parameters (all float for simplicity)
    const float* __restrict__ mlp_w1,           // (hidden_dim,) - first layer weights
    const float* __restrict__ mlp_b1,           // (hidden_dim,) - first layer bias
    const float* __restrict__ mlp_W2,           // (hidden_dim, hidden_dim) - second layer weights
    const float* __restrict__ mlp_b2,           // (hidden_dim,) - second layer bias
    const float* __restrict__ mlp_W3,           // (Cout, Cin, hidden_dim, Wdim) - third layer weights
    const float* __restrict__ mlp_b3,           // (Cout, Cin, Wdim) - third layer bias
    scalar_t* __restrict__ output,              // (B, Cout, Dout)
    const int* __restrict__ block_m,
    const int* __restrict__ block_n_in,
    const int* __restrict__ block_n_out,
    const int* __restrict__ block_in_off,
    const int* __restrict__ block_out_off,
    const int* __restrict__ block_w_off,
    int64_t B, int64_t Cin, int64_t Cout, int64_t Din, int64_t Dout, int64_t Wdim,
    int num_blocks, int hidden_dim
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
    const int block_w_size = (m == 0) ? (n_out * n_in) : (2 * n_out * n_in);

    // Shared memory layout:
    // [0, Cin*in_size): features
    // [Cin*in_size, Cin*in_size + hidden_dim): hidden2
    // [Cin*in_size + hidden_dim, Cin*in_size + hidden_dim + block_w_size): weights for current (co, ci)
    extern __shared__ float smem[];
    float* feat_shared = smem;
    float* hidden2_shared = smem + Cin * in_size;
    float* weights_shared = hidden2_shared + hidden_dim;

    // Cooperatively load features into shared memory
    const int64_t feat_base = b * Cin * Din;
    const int total_feat_elems = Cin * in_size;

    for (int i = tid; i < total_feat_elems; i += num_threads) {
        const int ci = i / in_size;
        const int local_idx = i % in_size;
        feat_shared[i] = static_cast<float>(features[feat_base + ci * Din + in_off + local_idx]);
    }

    // --- Compute MLP hidden layers cooperatively ---
    // Load edge feature for this batch element
    const float edge_feat = static_cast<float>(edge_features[b]);

    // Compute hidden1 = silu(edge_feat * w1 + b1) and store temporarily
    for (int h = tid; h < hidden_dim; h += num_threads) {
        float h1 = edge_feat * mlp_w1[h] + mlp_b1[h];
        h1 = h1 / (1.0f + expf(-h1));  // SiLU activation
        hidden2_shared[h] = h1;  // Store hidden1 temporarily
    }
    __syncthreads();

    // Compute hidden2 = silu(hidden1 @ W2 + b2)
    // Need temp storage since we're overwriting hidden2_shared
    // Use weights_shared temporarily (it's not in use yet)
    for (int h = tid; h < hidden_dim; h += num_threads) {
        float h2 = mlp_b2[h];
        for (int j = 0; j < hidden_dim; j++) {
            h2 += hidden2_shared[j] * mlp_W2[j * hidden_dim + h];
        }
        h2 = h2 / (1.0f + expf(-h2));  // SiLU activation
        weights_shared[h] = h2;  // Store in temp location
    }
    __syncthreads();

    // Copy back to hidden2_shared
    for (int h = tid; h < hidden_dim; h += num_threads) {
        hidden2_shared[h] = weights_shared[h];
    }
    __syncthreads();

    // --- Main loop: each thread handles assigned (co, o_local) outputs ---
    // Total outputs for this m-block: Cout × n_out (real) or Cout × 2*n_out (complex real/imag separate)
    // For m>0, we only output real indices; imaginary is computed alongside
    const int total_outputs = Cout * n_out;

    // Each thread handles a subset of output indices and accumulates in registers
    for (int out_idx = tid; out_idx < total_outputs; out_idx += num_threads) {
        const int co = out_idx / n_out;
        const int o_local = out_idx % n_out;

        float acc = 0.0f;
        float acc_im = 0.0f;  // Only used for m > 0

        // Accumulate over all input channels
        for (int64_t ci = 0; ci < Cin; ci++) {
            const float* f_ptr = feat_shared + ci * in_size;

            // Compute the weights this thread needs for (co, ci, o_local)
            // weights[w] = hidden2 @ W3[co, ci, :, w_off+w] + b3[co, ci, w_off+w]
            const float* W3_co_ci = mlp_W3 + (co * Cin + ci) * hidden_dim * Wdim;
            const float* b3_co_ci = mlp_b3 + (co * Cin + ci) * Wdim;

            if (m == 0) {
                // Real block: need n_in weights for this o_local
                for (int i = 0; i < n_in; i++) {
                    const int w_idx = w_off + o_local * n_in + i;
                    float weight_val = b3_co_ci[w_idx];
                    for (int h = 0; h < hidden_dim; h++) {
                        weight_val += hidden2_shared[h] * W3_co_ci[h * Wdim + w_idx];
                    }
                    acc += weight_val * f_ptr[i];
                }
            } else {
                // Complex block: need 2*n_in weights (a,b pairs) for this o_local
                for (int i = 0; i < n_in; i++) {
                    const float f_re = f_ptr[i];
                    const float f_im = f_ptr[n_in + i];
                    const int w_base_idx = w_off + (o_local * n_in + i) * 2;

                    // Compute a and b weights
                    float a = b3_co_ci[w_base_idx];
                    float bv = b3_co_ci[w_base_idx + 1];
                    for (int h = 0; h < hidden_dim; h++) {
                        a += hidden2_shared[h] * W3_co_ci[h * Wdim + w_base_idx];
                        bv += hidden2_shared[h] * W3_co_ci[h * Wdim + w_base_idx + 1];
                    }

                    // Complex multiplication: [a, b; -b, a] @ [f_re; f_im]
                    acc += a * f_re + bv * f_im;
                    acc_im += a * f_im - bv * f_re;
                }
            }
        }

        // Write final accumulated output
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
    torch::Tensor block_w_size,
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    // radial_table shape: (num_bins, Cout, Cin, Wdim)
    const int64_t Wdim = radial_table.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size floats (same as V2)
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

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
    // radial_table shape: (num_bins + 1, Cout, Cin, Wdim)
    const int64_t Wdim = radial_table.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size floats (same as V2)
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

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


std::vector<torch::Tensor> block_diagonal_forward_broadcast_cuda(
    torch::Tensor features,           // (B, Cin, Din)
    torch::Tensor weights,            // (Cout, Cin, Wdim) - NO batch dimension
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Wdim = weights.size(2);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size floats
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_broadcast", ([&] {
        block_diagonal_forward_broadcast_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
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


std::vector<torch::Tensor> block_diagonal_forward_broadcast_batched_cuda(
    torch::Tensor features,           // (B, Cin, Din)
    torch::Tensor weights,            // (H, Cout, Cin, Wdim) - H weight slices
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t H = weights.size(0);
    const int64_t Wdim = weights.size(3);
    const int num_blocks = block_m.size(0);

    // Output: (B, H, Cout, Dout)
    auto output = torch::zeros({B, H, Cout, dim_out}, features.options());

    // 3D grid: (num_m_blocks, B, H)
    dim3 grid(num_blocks, B, H);
    const int threads = 256;

    // Shared memory: Cin × max_in_size floats
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_broadcast_batched", ([&] {
        block_diagonal_forward_broadcast_batched_kernel<scalar_t><<<grid, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_m.data_ptr<int>(),
            block_n_in.data_ptr<int>(),
            block_n_out.data_ptr<int>(),
            block_in_off.data_ptr<int>(),
            block_out_off.data_ptr<int>(),
            block_w_off.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks, H
        );
    }));

    return {output};
}


std::vector<torch::Tensor> block_diagonal_fused_final_cuda(
    torch::Tensor features,           // (B, Cin, Din)
    torch::Tensor hidden2,            // (B, H) - precomputed MLP hidden layer
    torch::Tensor W3,                 // (Cout, Cin, Wdim, H) - TRANSPOSED
    torch::Tensor b3,                 // (Cout, Cin, Wdim)
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    int64_t Cout,
    int dim_out
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Wdim = W3.size(2);
    const int64_t H = W3.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: H floats for hidden2 + Cin × max_in_size floats for features
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = (H + Cin * max_in_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_fused_final", ([&] {
        block_diagonal_fused_final_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            hidden2.data_ptr<scalar_t>(),
            W3.data_ptr<scalar_t>(),
            b3.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_m.data_ptr<int>(),
            block_n_in.data_ptr<int>(),
            block_n_out.data_ptr<int>(),
            block_in_off.data_ptr<int>(),
            block_out_off.data_ptr<int>(),
            block_w_off.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks, H
        );
    }));

    return {output};
}


std::vector<torch::Tensor> block_diagonal_forward_fused_mlp_cuda(
    torch::Tensor features,           // (B, Cin, Din)
    torch::Tensor edge_features,      // (B,) - distance per edge
    // MLP parameters
    torch::Tensor mlp_w1,             // (hidden_dim,)
    torch::Tensor mlp_b1,             // (hidden_dim,)
    torch::Tensor mlp_W2,             // (hidden_dim, hidden_dim)
    torch::Tensor mlp_b2,             // (hidden_dim,)
    torch::Tensor mlp_W3,             // (Cout, Cin, hidden_dim, Wdim)
    torch::Tensor mlp_b3,             // (Cout, Cin, Wdim)
    // Block metadata
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
    const int64_t hidden_dim = mlp_w1.size(0);
    const int64_t Wdim = mlp_W3.size(3);
    const int num_blocks = block_m.size(0);

    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Grid: B × num_m_blocks
    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Shared memory: Cin × max_in_size + hidden_dim + max(max_block_w_size, hidden_dim) floats
    // Note: weights_shared is also used as temp storage for hidden_dim during MLP computation
    const int max_in_size = block_in_size.max().item<int>();
    const int max_w_size = block_w_size.max().item<int>();
    const int temp_size = (max_w_size > hidden_dim) ? max_w_size : hidden_dim;
    const size_t shared_size = (Cin * max_in_size + hidden_dim + temp_size) * sizeof(float);

    // Ensure MLP weights are float and contiguous
    auto w1 = mlp_w1.to(torch::kFloat32).contiguous();
    auto b1 = mlp_b1.to(torch::kFloat32).contiguous();
    auto W2 = mlp_W2.to(torch::kFloat32).contiguous();
    auto b2 = mlp_b2.to(torch::kFloat32).contiguous();
    auto W3 = mlp_W3.to(torch::kFloat32).contiguous();
    auto b3 = mlp_b3.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_forward_fused_mlp", ([&] {
        block_diagonal_forward_fused_mlp_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
            features.data_ptr<scalar_t>(),
            edge_features.data_ptr<scalar_t>(),
            w1.data_ptr<float>(),
            b1.data_ptr<float>(),
            W2.data_ptr<float>(),
            b2.data_ptr<float>(),
            W3.data_ptr<float>(),
            b3.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            block_m.data_ptr<int>(),
            block_n_in.data_ptr<int>(),
            block_n_out.data_ptr<int>(),
            block_in_off.data_ptr<int>(),
            block_out_off.data_ptr<int>(),
            block_w_off.data_ptr<int>(),
            B, Cin, Cout, Din, dim_out, Wdim, num_blocks, hidden_dim
        );
    }));

    return {output};
}


/*
 * Chunked matmul + block-diagonal in C++.
 *
 * Exactly replicates the Python version but without Python loop overhead:
 *
 *   for co in range(0, Cout, chunk_size):
 *       weights_chunk = einsum('bh,chw->bcw', hidden2, W3[co:co+chunk]) + b3[co:co+chunk]
 *       output[:, co:co+chunk] = block_diagonal_v2(features, weights_chunk)
 */
std::vector<torch::Tensor> block_diagonal_chunked_matmul_cuda(
    torch::Tensor features,      // (B, Cin, Din)
    torch::Tensor hidden2,       // (B, H)
    torch::Tensor W3,            // (Cout, Cin, H, Wdim)
    torch::Tensor b3,            // (Cout, Cin, Wdim)
    torch::Tensor block_m,
    torch::Tensor block_n_in,
    torch::Tensor block_n_out,
    torch::Tensor block_in_off,
    torch::Tensor block_out_off,
    torch::Tensor block_w_off,
    torch::Tensor block_in_size,
    int dim_out,
    int chunk_size
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = W3.size(0);
    const int64_t H = W3.size(2);
    const int64_t Wdim = W3.size(3);
    const int num_blocks = block_m.size(0);

    // Ensure contiguous (matches Python)
    features = features.contiguous();

    // Allocate output
    auto output = torch::zeros({B, Cout, dim_out}, features.options());

    // Kernel config
    const int threads = 256;
    const int max_in_size = block_in_size.max().item<int>();
    const size_t shared_size = Cin * max_in_size * sizeof(float);

    // Process chunks (mirrors Python loop exactly)
    for (int64_t co_start = 0; co_start < Cout; co_start += chunk_size) {
        const int64_t co_end = std::min(co_start + (int64_t)chunk_size, Cout);
        const int64_t chunk_cout = co_end - co_start;

        // W3_flat = W3[co_start:co_end].reshape(chunk_cout * Cin, H, Wdim)
        auto W3_flat = W3.slice(0, co_start, co_end).reshape({chunk_cout * Cin, H, Wdim});
        auto b3_flat = b3.slice(0, co_start, co_end).reshape({chunk_cout * Cin, Wdim});

        // weights_flat = einsum('bh,chw->bcw', hidden2, W3_flat) + b3_flat
        auto weights_flat = torch::einsum("bh,chw->bcw", {hidden2, W3_flat}) + b3_flat;

        // weights_chunk = weights_flat.view(B, chunk_cout, Cin, Wdim).contiguous()
        auto weights_chunk = weights_flat.view({B, chunk_cout, Cin, Wdim}).contiguous();

        // output_chunk = forward_v2(features, weights_chunk, ...)
        auto output_chunk = torch::zeros({B, chunk_cout, dim_out}, features.options());
        const int64_t grid_size = B * num_blocks;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "block_diagonal_chunked", ([&] {
            block_diagonal_forward_v2_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                features.data_ptr<scalar_t>(),
                weights_chunk.data_ptr<scalar_t>(),
                output_chunk.data_ptr<scalar_t>(),
                block_m.data_ptr<int>(),
                block_n_in.data_ptr<int>(),
                block_n_out.data_ptr<int>(),
                block_in_off.data_ptr<int>(),
                block_out_off.data_ptr<int>(),
                block_w_off.data_ptr<int>(),
                B, Cin, chunk_cout, Din, dim_out, Wdim, num_blocks
            );
        }));

        // output[:, co_start:co_end] = output_chunk
        output.slice(1, co_start, co_end).copy_(output_chunk);
    }

    return {output};
}


/*
 * Fused backward kernel for binned interpolated weights: grad_features
 *
 * Computes grad_features using on-the-fly weight interpolation from radial_table.
 * Avoids materializing the full (B, Cout, Cin, Wdim) weights tensor.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_binned_interp_features_kernel(
    const scalar_t* __restrict__ grad_output,   // (B, Cout, Dout)
    const scalar_t* __restrict__ radial_table,  // (num_bins+1, Cout, Cin, Wdim)
    const int* __restrict__ bin_lo,             // (B,)
    const int* __restrict__ bin_hi,             // (B,)
    const scalar_t* __restrict__ interp_weight, // (B,)
    scalar_t* __restrict__ grad_features,       // (B, Cin, Din)
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

    // Interpolation parameters
    const int idx_lo = bin_lo[b];
    const int idx_hi = bin_hi[b];
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;

    // Table stride
    const int64_t table_stride = Cout * Cin * Wdim;

    // Shared memory: grad_output for all Cout channels
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
    const int total_inputs = Cin * in_size;

    for (int in_idx = tid; in_idx < total_inputs; in_idx += num_threads) {
        const int ci = in_idx / in_size;
        const int i_local = in_idx % in_size;

        float grad = 0.0f;

        if (m == 0) {
            // Real block: grad_f[i] = sum_o W[o,i] * grad_out[o]
            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim + ci * Wdim + w_off;
                const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    const float w_lo = static_cast<float>(w_ptr_lo[o * n_in + i_local]);
                    const float w_hi = static_cast<float>(w_ptr_hi[o * n_in + i_local]);
                    const float w = one_minus_t * w_lo + t * w_hi;
                    grad += w * go_ptr[o];
                }
            }
        } else {
            // Complex block
            bool is_real_input = (i_local < n_in);
            int i_idx = is_real_input ? i_local : (i_local - n_in);

            for (int64_t co = 0; co < Cout; co++) {
                const float* go_ptr = grad_shared + co * out_size;
                const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + co * Cin * Wdim + ci * Wdim + w_off;
                const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + co * Cin * Wdim + ci * Wdim + w_off;

                for (int o = 0; o < n_out; o++) {
                    float go_re = go_ptr[o];
                    float go_im = go_ptr[n_out + o];

                    int w_idx = (o * n_in + i_idx) * 2;
                    float a_lo = static_cast<float>(w_ptr_lo[w_idx]);
                    float b_lo = static_cast<float>(w_ptr_lo[w_idx + 1]);
                    float a_hi = static_cast<float>(w_ptr_hi[w_idx]);
                    float b_hi = static_cast<float>(w_ptr_hi[w_idx + 1]);

                    float a = one_minus_t * a_lo + t * a_hi;
                    float bv = one_minus_t * b_lo + t * b_hi;

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
 * Fused backward kernel for binned interpolated weights: grad_radial_table
 *
 * Computes grad_weights and atomically scatters to grad_radial_table.
 * Also computes grad_interp_weight for force computation.
 */
template <typename scalar_t>
__global__ void block_diagonal_backward_binned_interp_table_kernel(
    const scalar_t* __restrict__ grad_output,      // (B, Cout, Dout)
    const scalar_t* __restrict__ features,         // (B, Cin, Din)
    const scalar_t* __restrict__ radial_table,     // (num_bins+1, Cout, Cin, Wdim)
    const int* __restrict__ bin_lo,                // (B,)
    const int* __restrict__ bin_hi,                // (B,)
    const scalar_t* __restrict__ interp_weight,    // (B,)
    float* __restrict__ grad_radial_table,         // (num_bins+1, Cout, Cin, Wdim) - use float for atomics
    float* __restrict__ grad_interp_weight,        // (B,) - use float for atomics
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

    // Interpolation parameters
    const int idx_lo = bin_lo[b];
    const int idx_hi = bin_hi[b];
    const float t = static_cast<float>(interp_weight[b]);
    const float one_minus_t = 1.0f - t;

    // Table stride
    const int64_t table_stride = Cout * Cin * Wdim;

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

    // Accumulate grad_interp_weight contribution in thread-local variable
    float local_grad_t = 0.0f;

    // Each thread computes a subset of (cout, cin, w_local) tuples
    const int64_t total_weights = Cout * Cin * w_block_size;

    for (int64_t w_idx = tid; w_idx < total_weights; w_idx += num_threads) {
        const int64_t co = w_idx / (Cin * w_block_size);
        const int64_t ci = (w_idx / w_block_size) % Cin;
        const int w_local = w_idx % w_block_size;

        const float* f_ptr = feat_shared + ci * in_size;
        const float* go_ptr = grad_shared + co * out_size;

        float grad_w = 0.0f;

        if (m == 0) {
            // Real block: grad_W[o,i] = f[i] * grad_out[o]
            int o = w_local / n_in;
            int i = w_local % n_in;
            grad_w = f_ptr[i] * go_ptr[o];
        } else {
            // Complex block
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

        // Atomically scatter to grad_radial_table
        // grad_table[lo] += (1-t) * grad_w
        // grad_table[hi] += t * grad_w
        const int64_t table_idx = co * Cin * Wdim + ci * Wdim + w_off + w_local;
        atomicAdd(&grad_radial_table[idx_lo * table_stride + table_idx], one_minus_t * grad_w);
        atomicAdd(&grad_radial_table[idx_hi * table_stride + table_idx], t * grad_w);

        // Accumulate grad_interp_weight: grad_t += (w_hi - w_lo) * grad_w
        const scalar_t* w_ptr_lo = radial_table + idx_lo * table_stride + table_idx;
        const scalar_t* w_ptr_hi = radial_table + idx_hi * table_stride + table_idx;
        float w_diff = static_cast<float>(*w_ptr_hi) - static_cast<float>(*w_ptr_lo);
        local_grad_t += w_diff * grad_w;
    }

    // Reduce local_grad_t across threads using warp shuffle + atomicAdd
    // First, warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_t += __shfl_down_sync(0xffffffff, local_grad_t, offset);
    }

    // Thread 0 of each warp does atomicAdd
    if ((tid % 32) == 0) {
        atomicAdd(&grad_interp_weight[b], local_grad_t);
    }
}


std::vector<torch::Tensor> block_diagonal_backward_binned_interp_cuda(
    torch::Tensor grad_output,
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
    torch::Tensor block_out_size,
    int dim_in
) {
    const int64_t B = features.size(0);
    const int64_t Cin = features.size(1);
    const int64_t Din = features.size(2);
    const int64_t Cout = grad_output.size(1);
    const int64_t Dout = grad_output.size(2);
    const int64_t num_bins_plus_1 = radial_table.size(0);
    const int64_t Wdim = radial_table.size(3);
    const int num_blocks = block_m.size(0);

    auto grad_features = torch::zeros_like(features);
    // Use float32 for grad_radial_table and grad_interp_weight for atomic operations
    auto grad_radial_table = torch::zeros({num_bins_plus_1, Cout, Cin, Wdim},
                                           radial_table.options().dtype(torch::kFloat32));
    auto grad_interp_weight = torch::zeros({B}, interp_weight.options().dtype(torch::kFloat32));

    const int64_t grid_size = B * num_blocks;
    const int threads = 256;

    // Compute max sizes for shared memory
    const int max_in_size = block_in_size.max().item<int>();
    const int max_out_size = block_out_size.max().item<int>();

    // Features backward: shared memory for Cout × max_out_size floats
    {
        const size_t shared_size = Cout * max_out_size * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_binned_interp_features", ([&] {
            block_diagonal_backward_binned_interp_features_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                bin_hi.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
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

    // Table backward: shared memory for Cin × max_in_size + Cout × max_out_size floats
    {
        const size_t shared_size = (Cin * max_in_size + Cout * max_out_size) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "backward_binned_interp_table", ([&] {
            block_diagonal_backward_binned_interp_table_kernel<scalar_t><<<grid_size, threads, shared_size>>>(
                grad_output.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                radial_table.data_ptr<scalar_t>(),
                bin_lo.data_ptr<int>(),
                bin_hi.data_ptr<int>(),
                interp_weight.data_ptr<scalar_t>(),
                grad_radial_table.data_ptr<float>(),
                grad_interp_weight.data_ptr<float>(),
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

    // Convert back to original dtype if needed
    if (radial_table.scalar_type() != torch::kFloat32) {
        grad_radial_table = grad_radial_table.to(radial_table.scalar_type());
    }
    if (interp_weight.scalar_type() != torch::kFloat32) {
        grad_interp_weight = grad_interp_weight.to(interp_weight.scalar_type());
    }

    return {grad_features, grad_radial_table, grad_interp_weight};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v2", &block_diagonal_forward_v2_cuda, "Block diagonal forward V2 - m-block parallel (CUDA)");
    m.def("forward_binned", &block_diagonal_forward_binned_cuda, "Block diagonal forward with binned weights (CUDA)");
    m.def("forward_binned_interp", &block_diagonal_forward_binned_interp_cuda, "Block diagonal forward with interpolated binned weights (CUDA)");
    m.def("forward_chunked_matmul", &block_diagonal_chunked_matmul_cuda, "Block diagonal with chunked matmul (CUDA)");
    m.def("backward_v2", &block_diagonal_backward_v2_cuda, "Block diagonal backward V2 - m-block parallel (CUDA)");
    m.def("backward_binned_interp", &block_diagonal_backward_binned_interp_cuda, "Block diagonal backward with interpolated binned weights (CUDA)");
}
