"""
PatchedConvSE3: drop-in replacement for NVIDIA's ConvSE3.

Uses flash-eq's block-diagonal CUDA kernel with precomputed weight
tables instead of per-edge CG basis tensors, achieving significant
memory savings.

Author: Hamish M. Blair <hmblair@stanford.edu>
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from flash_eq import ProductRepr, Repr
from flash_eq.cuda.block_diagonal import block_diagonal_cuda


class PatchedConvSE3(nn.Module):
    """Drop-in replacement for NVIDIA ``ConvSE3`` using flash-eq's CUDA kernel.

    Same ``forward(node_feats, edge_feats, graph, basis)`` signature as
    the original. The ``basis`` dict is repurposed to carry Wigner-D
    matrices and edge distances instead of CG basis tensors::

        basis = {
            '_P': Tensor,       # (E, dim, dim) input Wigner-D matrix
            '_Q': Tensor,       # (E, dim, dim) output Wigner-D matrix
            '_distances': Tensor,  # (E,) edge distances
        }

    This is set by the patched ``SE3Transformer.forward``.
    """

    def __init__(
        self,
        original_conv: nn.Module,
        table: Tensor,
        product_repr: ProductRepr,
        in_repr: Repr,
        out_repr: Repr,
        num_bins: int,
        min_dist: float,
        max_dist: float,
    ) -> None:
        super().__init__()

        # Converted weight table
        self.register_buffer("table", table)
        self.product_repr = product_repr
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self._inv_bin_width = num_bins / (max_dist - min_dist)

        # Copy attributes from original ConvSE3
        self.pool = original_conv.pool
        self.fiber_in = original_conv.fiber_in
        self.fiber_out = original_conv.fiber_out
        self.self_interaction = original_conv.self_interaction

        # Copy self-interaction weights
        if self.self_interaction and hasattr(original_conv, "to_kernel_self"):
            self.to_kernel_self = original_conv.to_kernel_self

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_feats: Dict[str, Tensor],
        graph: object,
        basis: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        P = basis["_P"]
        Q = basis["_Q"]
        distances = basis["_distances"]
        src, dst = graph.edges()

        # Gather source node features and concatenate edge features
        features_list = []
        for d in self.fiber_in.degrees:
            f = node_feats[str(d)][src]
            if d > 0 and str(d) in edge_feats:
                f = torch.cat([f, edge_feats[str(d)]], dim=1)
            features_list.append(f)
        features = torch.cat(features_list, dim=-1)  # (E, C_in, dim_in)

        # Flash-eq kernel: P^T @ features -> block_diagonal -> Q
        f_diag = torch.bmm(features, P)
        out_diag = block_diagonal_cuda(
            f_diag,
            self.table,
            distances,
            self.product_repr,
            bin_param1=self.min_dist,
            bin_param2=self._inv_bin_width,
            num_bins=self.num_bins,
            sh_scale=0.0,
        )
        output = torch.bmm(out_diag, Q.mT)  # (E, C_out, dim_out)

        # Split into per-degree dict
        out: Dict[str, Tensor] = {}
        offset = 0
        for d in self.fiber_out.degrees:
            dim_d = 2 * d + 1
            out[str(d)] = output[..., offset : offset + dim_d]
            offset += dim_d

        # Self-interaction (unchanged from original)
        if self.self_interaction:
            for d in self.fiber_out.degrees:
                if str(d) in self.to_kernel_self:
                    dst_features = node_feats[str(d)][dst]
                    kernel_self = self.to_kernel_self[str(d)]
                    out[str(d)] = out[str(d)] + kernel_self @ dst_features

        # Pooling (unchanged — uses DGL)
        if self.pool:
            import dgl

            for d in self.fiber_out.degrees:
                out[str(d)] = dgl.ops.copy_e_sum(graph, out[str(d)])

        return out
