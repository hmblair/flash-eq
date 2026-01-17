"""
SE3-Transformer baseline implementation.

This implements the standard TFN/SE3-Transformer approach with two dense matmuls:
    output = radial_weights @ (features @ basis)

Memory complexity: O(E * C^2 * F) for weights + O(E * D * F * D) for basis
where E=edges, C=channels, F=freq_sum, D=feature_dim

This serves as a reference for comparing against flash-eq's optimized approach.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def compute_freq_sum(lmax: int) -> int:
    """
    Compute the frequency sum for SE3-T basis.

    For each (l_in, l_out) pair, we have min(l_in, l_out) + 1 frequencies.
    """
    lvals = list(range(lmax + 1))
    return sum(min(l1, l2) + 1 for l1 in lvals for l2 in lvals)


def compute_dim(lmax: int) -> int:
    """Compute feature dimension: sum of (2l+1) for l in [0, lmax]."""
    return sum(2 * l + 1 for l in range(lmax + 1))


class SE3TBaseline(nn.Module):
    """
    SE3-Transformer style equivariant layer.

    This uses the standard two-matmul approach:
        1. tmp = features @ basis  (apply spherical harmonic basis)
        2. output = radial_weights @ tmp  (apply radial-dependent weights)

    The basis matrices encode the Clebsch-Gordan coefficients / spherical harmonics.
    The radial weights are computed per-edge by an MLP.

    Args:
        lmax: Maximum angular momentum
        channels_in: Number of input channels
        channels_out: Number of output channels
        radial_hidden: Hidden dimension for radial MLP

    Input shapes:
        features: (num_edges, channels_in, dim) - edge features
        basis: (num_edges, dim, freq_sum * dim) - pre-computed basis matrices
        distances: (num_edges,) - edge distances for radial MLP

    Output shape:
        (num_edges, channels_out, dim)
    """

    def __init__(
        self,
        lmax: int,
        channels_in: int,
        channels_out: int,
        radial_hidden: int = 64,
    ):
        super().__init__()
        self.lmax = lmax
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dim = compute_dim(lmax)
        self.freq_sum = compute_freq_sum(lmax)

        # Radial MLP: distance -> per-edge weights
        # Output shape per edge: (channels_out, channels_in * freq_sum)
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, channels_out * channels_in * self.freq_sum),
        )

    def forward(
        self,
        features: torch.Tensor,
        basis: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply SE3-T style equivariant transformation.

        Args:
            features: (num_edges, channels_in, dim) - input features per edge
            basis: (num_edges, dim, freq_sum * dim) - basis matrices per edge
            distances: (num_edges,) - edge distances

        Returns:
            output: (num_edges, channels_out, dim)
        """
        num_edges = features.shape[0]

        # Compute per-edge radial weights from MLP
        # Shape: (num_edges, channels_out, channels_in * freq_sum)
        radial_weights = self.radial_mlp(distances.unsqueeze(-1))
        radial_weights = radial_weights.view(num_edges, self.channels_out, self.channels_in * self.freq_sum)

        # First matmul: apply basis
        # features: (num_edges, channels_in, dim)
        # basis: (num_edges, dim, freq_sum * dim)
        # tmp: (num_edges, channels_in, freq_sum * dim)
        tmp = features @ basis

        # Reshape for second matmul
        # tmp: (num_edges, channels_in * freq_sum, dim)
        tmp = tmp.view(num_edges, self.channels_in * self.freq_sum, self.dim)

        # Second matmul: apply radial weights
        # radial_weights: (num_edges, channels_out, channels_in * freq_sum)
        # tmp: (num_edges, channels_in * freq_sum, dim)
        # output: (num_edges, channels_out, dim)
        output = radial_weights @ tmp

        return output

    def extra_repr(self) -> str:
        return (
            f"lmax={self.lmax}, channels_in={self.channels_in}, "
            f"channels_out={self.channels_out}, dim={self.dim}, freq_sum={self.freq_sum}"
        )


def create_random_basis(num_edges: int, lmax: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create random basis matrices for testing.

    In a real SE3-T implementation, these would be computed from spherical harmonics
    and Clebsch-Gordan coefficients based on edge directions.

    Args:
        num_edges: Number of edges
        lmax: Maximum angular momentum
        device: Target device
        dtype: Target dtype

    Returns:
        basis: (num_edges, dim, freq_sum * dim)
    """
    dim = compute_dim(lmax)
    freq_sum = compute_freq_sum(lmax)
    return torch.randn(num_edges, dim, freq_sum * dim, device=device, dtype=dtype)
