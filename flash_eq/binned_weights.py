"""
Binned radial weights for memory-efficient SO(3)-equivariant layers.

Instead of storing radial weights per edge (batch, cout, cin, weight_dim),
we store weights per distance bin (num_bins, weight_dim) and look them up
by edge length. This reduces memory by 50,000-200,000x for typical configs.

Key components:
- RadialBinning: Manages bin edges and computes bin indices/interpolation weights
- BinnedRadialEmbedding: nn.Module that creates and caches the lookup table
- block_diagonal_binned_*: CUDA kernels that use binned weights directly

Example:
    # Setup
    binning = RadialBinning(num_bins=100, max_dist=10.0, device='cuda')
    radial_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, weight_dim))

    # Create table (once, or when radial_mlp updates)
    with torch.no_grad():
        radial_table = binning.create_table(radial_mlp)

    # Forward pass
    bin_data = binning.compute_bins(edge_lengths)
    output = block_diagonal_binned_interp_cuda(
        features, radial_table, bin_data.lo, bin_data.hi, bin_data.weight,
        channels_out, metadata
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass
class BinData:
    """Container for bin indices and interpolation weights."""
    lo: torch.Tensor      # (batch,) lower bin indices, int32
    hi: torch.Tensor      # (batch,) upper bin indices, int32
    weight: torch.Tensor  # (batch,) interpolation weight in [0, 1]

    def to(self, device: torch.device) -> BinData:
        return BinData(
            lo=self.lo.to(device),
            hi=self.hi.to(device),
            weight=self.weight.to(device),
        )


class RadialBinning:
    """
    Manages distance binning for radial weight lookup.

    Handles bin edge creation, index computation, and interpolation weights.
    Immutable after creation - create a new instance to change bin parameters.

    Args:
        num_bins: Number of distance bins
        min_dist: Minimum distance (default 0.0)
        max_dist: Maximum distance (default 10.0 Angstroms)
        device: Target device for bin edges

    Example:
        binning = RadialBinning(num_bins=100, max_dist=10.0, device='cuda')
        bin_data = binning.compute_bins(edge_lengths)
    """

    __slots__ = ('num_bins', 'min_dist', 'max_dist', '_bin_edges', '_bin_width')

    def __init__(
        self,
        num_bins: int = 100,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        if max_dist <= min_dist:
            raise ValueError(f"max_dist ({max_dist}) must be > min_dist ({min_dist})")

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self._bin_edges = torch.linspace(min_dist, max_dist, num_bins + 1, device=device)
        self._bin_width = (max_dist - min_dist) / num_bins

    @property
    def bin_edges(self) -> torch.Tensor:
        """(num_bins + 1,) tensor of bin boundaries."""
        return self._bin_edges

    @property
    def bin_centers(self) -> torch.Tensor:
        """(num_bins,) tensor of bin centers."""
        return (self._bin_edges[:-1] + self._bin_edges[1:]) / 2

    @property
    def table_size(self) -> int:
        """Number of entries needed in lookup table for interpolation."""
        return self.num_bins + 1

    def to(self, device: torch.device) -> RadialBinning:
        """Move bin edges to specified device."""
        if self._bin_edges.device == device:
            return self
        new = RadialBinning.__new__(RadialBinning)
        new.num_bins = self.num_bins
        new.min_dist = self.min_dist
        new.max_dist = self.max_dist
        new._bin_edges = self._bin_edges.to(device)
        new._bin_width = self._bin_width
        return new

    def compute_bins(self, distances: torch.Tensor) -> BinData:
        """
        Compute bin indices and interpolation weights for given distances.

        Args:
            distances: (...,) tensor of distances

        Returns:
            BinData with lo, hi indices and interpolation weights
        """
        # Ensure bin_edges are on the same device
        if distances.device != self._bin_edges.device:
            bin_edges = self._bin_edges.to(distances.device)
        else:
            bin_edges = self._bin_edges

        # Compute lower bin index using searchsorted with right=True
        # This ensures values exactly at bin edges go to the higher bin
        # e.g., distance=5.0 with edges [0,1,...,10] -> bin 5, not bin 4
        bin_lo = torch.searchsorted(bin_edges[1:-1], distances.contiguous(), right=True)
        bin_lo = bin_lo.clamp(0, self.num_bins - 1)
        bin_hi = (bin_lo + 1).clamp(max=self.num_bins)

        # Compute interpolation weight: (d - edge_lo) / bin_width
        edge_lo = bin_edges[bin_lo]
        edge_hi = bin_edges[bin_hi]
        bin_width = (edge_hi - edge_lo).clamp(min=1e-8)
        interp_weight = ((distances - edge_lo) / bin_width).clamp(0.0, 1.0)

        return BinData(
            lo=bin_lo.int(),
            hi=bin_hi.int(),
            weight=interp_weight,
        )

    def compute_indices(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute bin indices only (for nearest-neighbor lookup).

        Args:
            distances: (...,) tensor of distances

        Returns:
            (...,) tensor of bin indices (int32)
        """
        if distances.device != self._bin_edges.device:
            bin_edges = self._bin_edges.to(distances.device)
        else:
            bin_edges = self._bin_edges

        indices = torch.searchsorted(bin_edges[1:-1], distances.contiguous(), right=True)
        return indices.clamp(0, self.num_bins - 1).int()

    @torch.no_grad()
    def create_table(
        self,
        radial_fn: Callable[[torch.Tensor], torch.Tensor],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Create lookup table by evaluating radial function at bin edges.

        Args:
            radial_fn: Function mapping (N,) distances -> (N, cout, cin, weight_dim) weights
            dtype: Output dtype (default: infer from radial_fn output)

        Returns:
            (num_bins + 1, cout, cin, weight_dim) lookup table for interpolation
        """
        table = radial_fn(self._bin_edges)
        if dtype is not None and table.dtype != dtype:
            table = table.to(dtype)
        return table.contiguous()


class BinnedRadialEmbedding(nn.Module):
    """
    Neural network module that produces binned radial weights.

    Wraps a radial MLP and manages the lookup table. The table is updated
    automatically when parameters change (checked via hash).

    Args:
        radial_net: nn.Module mapping (N, 1) -> (N, cout, cin, weight_dim)
        cout: Number of output channels
        cin: Number of input channels
        weight_dim: Weight dimension per channel pair
        num_bins: Number of distance bins
        min_dist: Minimum distance
        max_dist: Maximum distance

    Example:
        # radial_net outputs (N, cout, cin, weight_dim) for N distances
        radial_net = MyRadialNetwork(cout=64, cin=64, weight_dim=441)
        embedding = BinnedRadialEmbedding(radial_net, cout=64, cin=64,
                                          weight_dim=441, num_bins=100)

        # In forward pass:
        radial_table = embedding.get_table()  # (num_bins+1, cout, cin, weight_dim)
        bin_data = embedding.binning.compute_bins(edge_lengths)
    """

    def __init__(
        self,
        radial_net: nn.Module,
        cout: int,
        cin: int,
        weight_dim: int,
        num_bins: int = 100,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
    ):
        super().__init__()
        self.radial_net = radial_net
        self.cout = cout
        self.cin = cin
        self.weight_dim = weight_dim
        self.binning = RadialBinning(num_bins, min_dist, max_dist)

        # Cache for lookup table
        self.register_buffer('_cached_table', None, persistent=False)
        self._param_hash: Optional[int] = None

    def _compute_param_hash(self) -> int:
        """Compute hash of parameters to detect changes."""
        h = 0
        for p in self.radial_net.parameters():
            h ^= hash(p.data_ptr()) ^ hash(tuple(p.shape))
        return h

    def _radial_fn(self, distances: torch.Tensor) -> torch.Tensor:
        """Wrapper to call radial_net with proper input shape."""
        return self.radial_net(distances.unsqueeze(-1))

    @torch.no_grad()
    def update_table(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Recompute the lookup table from current radial_net parameters.

        Args:
            device: Device for the table (default: same as radial_net)

        Returns:
            (num_bins + 1, cout, cin, weight_dim) lookup table
        """
        if device is None:
            device = next(self.radial_net.parameters()).device

        binning = self.binning.to(device)
        dtype = next(self.radial_net.parameters()).dtype

        self._cached_table = binning.create_table(self._radial_fn, dtype=dtype)
        self._param_hash = self._compute_param_hash()

        return self._cached_table

    def get_table(self, force_update: bool = False) -> torch.Tensor:
        """
        Get the lookup table, updating if parameters changed.

        Args:
            force_update: If True, always recompute table

        Returns:
            (num_bins + 1, cout, cin, weight_dim) lookup table
        """
        current_hash = self._compute_param_hash()

        if force_update or self._cached_table is None or current_hash != self._param_hash:
            return self.update_table()

        return self._cached_table

    def forward(self, distances: torch.Tensor) -> tuple[torch.Tensor, BinData]:
        """
        Compute lookup table and bin data for given distances.

        Args:
            distances: (batch,) tensor of edge lengths

        Returns:
            radial_table: (num_bins + 1, cout, cin, weight_dim) lookup table
            bin_data: BinData with indices and interpolation weights
        """
        radial_table = self.get_table()
        binning = self.binning.to(distances.device)
        bin_data = binning.compute_bins(distances)
        return radial_table, bin_data


def interpolate_weights(
    table: torch.Tensor,
    bin_data: BinData,
) -> torch.Tensor:
    """
    Interpolate weights from lookup table (pure Python, for reference/testing).

    For production, use block_diagonal_binned_interp_cuda which fuses this
    with the block-diagonal multiplication.

    Args:
        table: (num_bins + 1, cout, cin, weight_dim) lookup table
        bin_data: BinData with lo, hi, weight

    Returns:
        (batch, cout, cin, weight_dim) interpolated weights
    """
    w_lo = table[bin_data.lo]  # (batch, cout, cin, weight_dim)
    w_hi = table[bin_data.hi]  # (batch, cout, cin, weight_dim)
    # Reshape weight for broadcasting: (batch,) -> (batch, 1, 1, 1)
    t = bin_data.weight.view(-1, 1, 1, 1)
    return torch.lerp(w_lo, w_hi, t)


# Legacy API compatibility
def create_bin_edges(
    min_dist: float = 0.0,
    max_dist: float = 10.0,
    num_bins: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create bin edges. Prefer RadialBinning class for new code."""
    return torch.linspace(min_dist, max_dist, num_bins + 1, device=device)


def compute_bin_interpolation(
    edge_lengths: torch.Tensor,
    bin_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute bin indices and interpolation weights. Prefer RadialBinning for new code."""
    num_bins = len(bin_edges) - 1
    bin_lo = torch.searchsorted(bin_edges[1:-1], edge_lengths.contiguous(), right=True)
    bin_lo = bin_lo.clamp(0, num_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, num_bins)

    edge_lo = bin_edges[bin_lo]
    edge_hi = bin_edges[bin_hi]
    bin_width = (edge_hi - edge_lo).clamp(min=1e-8)
    interp_weight = ((edge_lengths - edge_lo) / bin_width).clamp(0, 1)

    return bin_lo, bin_hi, interp_weight


def compute_bin_indices(
    edge_lengths: torch.Tensor,
    bin_edges: torch.Tensor,
    clamp: bool = True,
) -> torch.Tensor:
    """Compute bin indices. Prefer RadialBinning for new code."""
    indices = torch.searchsorted(bin_edges[1:-1], edge_lengths.contiguous(), right=True)
    if clamp:
        num_bins = len(bin_edges) - 1
        indices = indices.clamp(0, num_bins - 1)
    return indices
