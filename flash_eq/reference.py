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
from typing import List, Tuple

from .representations import Repr, ProductRepr
from .basis import _build_m_order_permutation


def get_weight_dim(lvals_in: List[int], lvals_out: List[int] = None) -> int:
    """Compute weight dimension for given lvals.

    For each m from 0 to lmax:
    - n_in(m) = number of input l's with l >= m
    - n_out(m) = number of output l's with l >= m
    - m=0 contributes n_in * n_out scalar parameters
    - m>0 contributes 2 * n_in * n_out parameters (a, b for each coupling)

    Args:
        lvals_in: list of input l values
        lvals_out: list of output l values (defaults to lvals_in)

    Returns:
        Total number of weight parameters
    """
    if lvals_out is None:
        lvals_out = lvals_in

    lmax = max(max(lvals_in), max(lvals_out))
    weight_dim = 0

    for m in range(lmax + 1):
        n_in = sum(1 for l in lvals_in if l >= m)
        n_out = sum(1 for l in lvals_out if l >= m)
        if n_in > 0 and n_out > 0:
            mult = 1 if m == 0 else 2
            weight_dim += mult * n_in * n_out

    return weight_dim


def reference_expand_weights(
    compact_weights: torch.Tensor,
    lvals_in: List[int],
    lvals_out: List[int] = None,
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
        lvals_in: list of input l values
        lvals_out: list of output l values (defaults to lvals_in)

    Returns:
        W: (dim_out, dim_in) dense weight matrix in standard SH order
    """
    if lvals_out is None:
        lvals_out = lvals_in

    device = compact_weights.device
    dtype = compact_weights.dtype
    dim_in = sum(2 * l + 1 for l in lvals_in)
    dim_out = sum(2 * l + 1 for l in lvals_out)
    lmax = max(max(lvals_in), max(lvals_out))

    W = torch.zeros(dim_out, dim_in, device=device, dtype=dtype)

    # Compute cumulative dimensions for indexing into standard SH order
    cumdims_in = [0]
    for l in lvals_in:
        cumdims_in.append(cumdims_in[-1] + 2 * l + 1)
    cumdims_out = [0]
    for l in lvals_out:
        cumdims_out.append(cumdims_out[-1] + 2 * l + 1)

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
    lvals: List[int],
) -> torch.Tensor:
    """Canonical reference implementation of SO(3)-equivariant layer.

    Computes: out = Q @ W @ P^T @ f

    This is the simplest possible implementation:
    1. Get P, Q from WignerDBasis (includes m-first permutation)
    2. Build one W matrix in m-first order
    3. Do the matmul

    Args:
        node_features: (num_nodes, cin, dim) in standard SH basis
        src_indices: (num_edges,) source node for each edge
        directions: (num_edges, 3) edge direction vectors
        compact_weights: (cout, cin, weight_dim) block-diagonal weights
        lvals: list of angular momentum values

    Returns:
        output: (num_edges, cout, dim) in standard SH basis
    """
    from .basis import WignerDBasis

    cout, cin, _ = compact_weights.shape
    dim = sum(2 * l + 1 for l in lvals)
    device = node_features.device
    dtype = node_features.dtype

    # 1. Get P, Q from WignerDBasis
    repr_in = Repr(lvals=lvals, mult=1)
    repr_out = Repr(lvals=lvals, mult=1)
    basis = WignerDBasis(repr_in, repr_out)
    P, Q = basis(directions)  # (num_edges, dim, dim)
    P = P.to(dtype)
    Q = Q.to(dtype)

    # 2. Build W matrices for all (cout, cin) pairs
    W = torch.zeros(cout, cin, dim, dim, device=device, dtype=dtype)
    for o in range(cout):
        for c in range(cin):
            W[o, c] = reference_expand_weights(compact_weights[o, c], lvals, lvals)

    # 3. Gather features and compute Q @ W @ P^T @ f
    f = node_features[src_indices]  # (num_edges, cin, dim)

    # P^T @ f: (num_edges, dim, dim) @ (num_edges, cin, dim).T -> need einsum
    # f_diag[e, c, i] = sum_j P[e, j, i] * f[e, c, j] = (P^T @ f)[e, c, i]
    f_diag = f @ P  # (num_edges, cin, dim)

    # W @ f_diag: (cout, cin, dim, dim) @ (num_edges, cin, dim)
    # out[e, o, i] = sum_c sum_j W[o, c, i, j] * f_diag[e, c, j]
    Wf = torch.einsum('ocij,ecj->eoi', W, f_diag)  # (num_edges, cout, dim)

    # Q @ Wf: (num_edges, dim, dim) @ (num_edges, cout, dim)
    # out[e, o, i] = sum_j Q[e, i, j] * Wf[e, o, j]
    output = Wf @ Q.mT  # (num_edges, cout, dim)

    return output


def reference_equivariance_test(
    node_features: torch.Tensor,
    src_indices: torch.Tensor,
    directions: torch.Tensor,
    compact_weights: torch.Tensor,
    lvals: List[int],
    rotation_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Test equivariance of the reference implementation.

    Verifies: layer(R @ f, R @ d) = R @ layer(f, d)

    where R is applied via Wigner-D to features and via 3x3 matrix to directions.

    Args:
        node_features: (num_nodes, cin, dim)
        src_indices: (num_edges,)
        directions: (num_edges, 3)
        compact_weights: (cout, cin, weight_dim)
        lvals: list of l values
        rotation_matrix: (3, 3) rotation matrix

    Returns:
        output_then_rotate: D @ layer(f, d)
        rotate_then_output: layer(D @ f, R @ d)
        relative_error: ||difference|| / ||expected||
    """
    repr = Repr(lvals=lvals, mult=1)

    # Compute rotation axis and angle from matrix
    from scipy.spatial.transform import Rotation
    R_scipy = Rotation.from_matrix(rotation_matrix.cpu().numpy())
    rotvec = R_scipy.as_rotvec()
    angle = torch.tensor(float(torch.norm(torch.tensor(rotvec))))
    axis = torch.tensor(rotvec) / (angle + 1e-12)

    # Wigner-D for this rotation (axis is in Cartesian coordinates)
    D = repr.rot(axis, angle).to(node_features.dtype).to(node_features.device)

    # Method 1: compute output, then rotate
    output = reference_layer(node_features, src_indices, directions, compact_weights, lvals)
    output_then_rotate = torch.einsum('ij,ecj->eci', D, output)

    # Method 2: rotate inputs, then compute output
    rotated_features = torch.einsum('ij,ncj->nci', D, node_features)
    # Use the original 3x3 rotation matrix for directions (Cartesian coordinates)
    rotated_directions = directions @ rotation_matrix.T
    rotate_then_output = reference_layer(
        rotated_features, src_indices, rotated_directions, compact_weights, lvals
    )

    # Compute error
    diff = output_then_rotate - rotate_then_output
    relative_error = diff.norm() / output_then_rotate.norm()

    return output_then_rotate, rotate_then_output, relative_error.item()
