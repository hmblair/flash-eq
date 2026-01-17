"""Canonical reference implementation for SO(3)-equivariant layers.

This module provides a single source of truth for the equivariant computation.
All other implementations (CUDA kernel, optimized PyTorch) should match this.

The computation follows docs/theory.tex exactly:
    out = Q @ Λ(r) @ P^T @ f

where:
    P = D(g_x) with columns permuted to m-first order
    Q = D(g_x) with columns permuted to m-first order
    Λ = block-diagonal weights with structure:
        - m=0: scalar λ₀
        - m>0: 2x2 block [[a, b], [-b, a]]
    g_x = rotation taking e_z to x̂ (unit direction)
"""

import torch
from typing import Tuple

from .representations import Repr
from .basis import WignerDBasis


def expand_weights(
    compact_weights: torch.Tensor,
    repr_in: Repr,
    repr_out: Repr,
) -> torch.Tensor:
    """Expand compact block-diagonal weights to dense matrix in standard SH order.

    The compact weights are in m-first order matching the kernel format:
    - For m=0: n_in * n_out scalars (coupling all l_in to all l_out at m=0)
    - For m>0: 2 * n_in * n_out parameters (a, b pairs for each coupling)

    Within each m-block, the ordering is:
    - Outer loop: output l (ascending)
    - Inner loop: input l (ascending)

    Args:
        compact_weights: (weight_dim,) compact representation
        repr_in: input representation
        repr_out: output representation

    Returns:
        W: (dim_out, dim_in) dense weight matrix in standard SH order
    """
    lvals_in = repr_in.lvals
    lvals_out = repr_out.lvals

    device = compact_weights.device
    dtype = compact_weights.dtype
    dim_in = repr_in.dim()
    dim_out = repr_out.dim()
    lmax = max(repr_in.lmax(), repr_out.lmax())

    W = torch.zeros(dim_out, dim_in, device=device, dtype=dtype)

    # Compute cumulative dimensions for indexing into standard SH order
    cumdims_in = repr_in.cumdims()
    cumdims_out = repr_out.cumdims()

    # Helper to get position of m within l-block in standard SH order
    # Standard order: m = -l, -l+1, ..., l, so position of m is l + m
    def std_pos_in(l_idx: int, m: int) -> int:
        l = lvals_in[l_idx]
        return cumdims_in[l_idx] + l + m

    def std_pos_out(l_idx: int, m: int) -> int:
        l = lvals_out[l_idx]
        return cumdims_out[l_idx] + l + m

    w_idx = 0

    for m in range(lmax + 1):
        # Find which l's have this m value
        in_l_indices = [i for i, l in enumerate(lvals_in) if l >= m]
        out_l_indices = [i for i, l in enumerate(lvals_out) if l >= m]

        if not in_l_indices or not out_l_indices:
            continue

        if m == 0:
            # m=0: scalar coupling between all (l_out, l_in) pairs
            for out_idx in out_l_indices:
                for in_idx in in_l_indices:
                    pos_out = std_pos_out(out_idx, 0)
                    pos_in = std_pos_in(in_idx, 0)
                    W[pos_out, pos_in] = compact_weights[w_idx]
                    w_idx += 1
        else:
            # m>0: 2x2 block [[a, b], [-b, a]] for each (l_out, l_in) pair
            # The block couples (m, -m) positions
            for out_idx in out_l_indices:
                for in_idx in in_l_indices:
                    a = compact_weights[w_idx]
                    b = compact_weights[w_idx + 1]
                    w_idx += 2

                    # Positions for +m and -m in standard order
                    pos_out_plus = std_pos_out(out_idx, m)
                    pos_out_minus = std_pos_out(out_idx, -m)
                    pos_in_plus = std_pos_in(in_idx, m)
                    pos_in_minus = std_pos_in(in_idx, -m)

                    # 2x2 block structure: [[a, b], [-b, a]]
                    # Maps (in_plus, in_minus) -> (out_plus, out_minus)
                    W[pos_out_plus, pos_in_plus] = a
                    W[pos_out_plus, pos_in_minus] = b
                    W[pos_out_minus, pos_in_plus] = -b
                    W[pos_out_minus, pos_in_minus] = a

    return W


def reference_layer(
    node_features: torch.Tensor,
    src_indices: torch.Tensor,
    directions: torch.Tensor,
    compact_weights: torch.Tensor,
    repr_in: Repr,
    repr_out: Repr = None,
) -> torch.Tensor:
    """Canonical reference implementation of SO(3)-equivariant layer.

    Computes: out = Q @ W @ P^T @ f

    This is the simplest possible implementation:
    1. Get P, Q from WignerDBasis
    2. Build dense W matrix from compact weights
    3. Do the matmul

    Args:
        node_features: (num_nodes, cin, dim_in) in standard SH basis
        src_indices: (num_edges,) source node for each edge
        directions: (num_edges, 3) edge direction vectors
        compact_weights: (cout, cin, weight_dim) block-diagonal weights
        repr_in: input representation
        repr_out: output representation (defaults to repr_in)

    Returns:
        output: (num_edges, cout, dim_out) in standard SH basis
    """
    if repr_out is None:
        repr_out = repr_in

    cout, cin, _ = compact_weights.shape
    device = node_features.device
    dtype = node_features.dtype

    # 1. Get P, Q from WignerDBasis
    basis = WignerDBasis(repr_in, repr_out)
    P, Q = basis(directions)
    P = P.to(dtype)
    Q = Q.to(dtype)

    # 2. Build W matrices for all (cout, cin) pairs
    W = torch.stack([
        torch.stack([expand_weights(compact_weights[o, c], repr_in, repr_out) for c in range(cin)])
        for o in range(cout)
    ])  # (cout, cin, dim_out, dim_in)

    # 3. Gather features and compute Q @ W @ P^T @ f
    f = node_features[src_indices]  # (num_edges, cin, dim_in)

    # P^T @ f: f @ P computes f_diag[e, c, i] = sum_j f[e, c, j] * P[e, j, i]
    f_diag = f @ P  # (num_edges, cin, dim_in)

    # W @ f_diag: out[e, o, i] = sum_c sum_j W[o, c, i, j] * f_diag[e, c, j]
    Wf = torch.einsum('ocij,ecj->eoi', W, f_diag)  # (num_edges, cout, dim_out)

    # Q @ Wf: out[e, o, i] = sum_j Q[e, i, j] * Wf[e, o, j]
    output = Wf @ Q.mT  # (num_edges, cout, dim_out)

    return output


def equivariance_test(
    node_features: torch.Tensor,
    src_indices: torch.Tensor,
    directions: torch.Tensor,
    compact_weights: torch.Tensor,
    repr_in: Repr,
    repr_out: Repr,
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Test equivariance of the reference implementation.

    Verifies: D_out @ layer(f, d) = layer(D_in @ f, R @ d)

    where D_in/D_out are Wigner-D matrices for input/output representations,
    and R is the 3x3 rotation matrix for directions.

    Args:
        node_features: (num_nodes, cin, dim_in)
        src_indices: (num_edges,)
        directions: (num_edges, 3)
        compact_weights: (cout, cin, weight_dim)
        repr_in: input representation
        repr_out: output representation
        axis: (..., 3) rotation axis (normalized)
        angle: (...) rotation angle in radians

    Returns:
        output_then_rotate: D_out @ layer(f, d)
        rotate_then_output: layer(D_in @ f, R @ d)
        relative_error: ||difference|| / ||expected||
    """
    device = node_features.device
    dtype = node_features.dtype

    # Wigner-D for input and output representations
    D_in = repr_in.rot(axis, angle).to(dtype).to(device)
    D_out = repr_out.rot(axis, angle).to(dtype).to(device)

    # 3x3 rotation matrix for directions (l=1 Wigner-D with cartesian=True)
    R = Repr(lvals=[1]).rot(axis, angle, cartesian=True).to(dtype).to(device)

    # Method 1: compute output, then rotate
    output = reference_layer(node_features, src_indices, directions, compact_weights, repr_in, repr_out)
    output_then_rotate = torch.einsum('ij,ecj->eci', D_out, output)

    # Method 2: rotate inputs, then compute output
    rotated_features = torch.einsum('ij,ncj->nci', D_in, node_features)
    rotated_directions = directions @ R.T
    rotate_then_output = reference_layer(
        rotated_features, src_indices, rotated_directions, compact_weights, repr_in, repr_out
    )

    # Compute error
    diff = output_then_rotate - rotate_then_output
    relative_error = diff.norm() / output_then_rotate.norm()

    return output_then_rotate, rotate_then_output, relative_error.item()
