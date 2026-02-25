"""
EquiformerV2 / eSCN baseline implementation.

Implements the SO(2) convolution approach from EquiformerV2 (Liao et al., ICLR 2024):
    1. Rotate features to edge-aligned frame using Wigner-D matrices
    2. Apply SO(2) convolution: per-m linear maps with radial modulation
    3. Rotate back to global frame

Memory complexity:
    - Wigner-D matrices: O(E * D^2) where D = (L+1)^2
    - Rotated features: O(E * C * D)
    - SO(2) weights (from radial MLP): O(E * sum_m (L-|m|+1) * C)

This serves as a reference for comparing against flash-eq's optimized approach.

Reference: https://arxiv.org/abs/2306.12059
Official code: https://github.com/atomicarchitects/equiformer_v2
"""

import torch
import torch.nn as nn


def compute_dim(lmax: int) -> int:
    """Compute feature dimension: (lmax+1)^2."""
    return (lmax + 1) ** 2


def compute_m_sizes(lmax: int) -> list[int]:
    """Number of l-values contributing to each m-order.

    For m=0: l can be 0, 1, ..., lmax -> lmax+1 values
    For |m|>0: l can be |m|, |m|+1, ..., lmax -> lmax-|m|+1 values
    """
    return [lmax - m + 1 for m in range(lmax + 1)]


def compute_radial_dim(lmax: int, channels: int) -> int:
    """Total radial output dimension: sum over all m of (num_l_for_m * channels)."""
    return sum(sz * channels for sz in compute_m_sizes(lmax))


class EquiformerV2Baseline(nn.Module):
    """
    EquiformerV2 / eSCN style equivariant layer.

    This uses the SO(2) convolution approach:
        1. Rotate features into edge-aligned frame via Wigner-D bmm
        2. Reindex from l-primary to m-primary order
        3. For each m-order, apply radial-modulated linear map
        4. Reindex back to l-primary order
        5. Rotate back via inverse Wigner-D bmm

    The Wigner-D matrices are stored dense (block-diagonal structure not exploited),
    matching the official implementation.

    Args:
        lmax: Maximum angular momentum
        channels_in: Number of input channels
        channels_out: Number of output channels
        radial_hidden: Hidden dimension for radial MLP

    Input shapes:
        features: (num_edges, channels_in, dim) where dim = (lmax+1)^2
        wigner: (num_edges, dim, dim) - Wigner-D rotation matrices
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
        self.m_sizes = compute_m_sizes(lmax)

        # Per-m linear layers (SO(2) convolution weights)
        # m=0: real-valued linear
        self.fc_m0 = nn.Linear(
            self.m_sizes[0] * channels_in,
            self.m_sizes[0] * channels_out,
            bias=False,
        )

        # m>0: complex-valued linear (2x output for real/imag parts)
        self.fc_m = nn.ModuleList()
        for m in range(1, lmax + 1):
            n_l = self.m_sizes[m]
            self.fc_m.append(
                nn.Linear(n_l * channels_in, 2 * n_l * channels_out, bias=False)
            )

        # Radial MLP: distance -> per-edge modulation weights
        # Output: one weight per (m, l, channel_in) for element-wise modulation
        radial_out_dim = compute_radial_dim(lmax, channels_in)
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, radial_out_dim),
        )

        # Precompute l-primary to m-primary reindexing
        self._build_reindex()

    def _build_reindex(self):
        """Build permutation indices for l-primary <-> m-primary reordering."""
        # l-primary order: (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), ...
        # m-primary order: group by |m|, with m=0 first, then |m|=1 (real, imag), etc.
        l_to_m = []
        # m=0: indices where m=0, i.e., l^2 + l for each l
        for l in range(self.lmax + 1):
            l_to_m.append(l * l + l)
        # |m| > 0: for each m, collect (l, +m) and (l, -m) blocks
        for m in range(1, self.lmax + 1):
            # +m components (real part in EquiformerV2 convention)
            for l in range(m, self.lmax + 1):
                l_to_m.append(l * l + l + m)
            # -m components (imaginary part)
            for l in range(m, self.lmax + 1):
                l_to_m.append(l * l + l - m)

        self.register_buffer("l_to_m", torch.tensor(l_to_m, dtype=torch.long))
        # Inverse permutation
        m_to_l = torch.empty_like(self.l_to_m)
        m_to_l[self.l_to_m] = torch.arange(len(l_to_m))
        self.register_buffer("m_to_l", m_to_l)

    def forward(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply EquiformerV2 SO(2) convolution.

        Args:
            features: (num_edges, channels_in, dim)
            wigner: (num_edges, dim, dim) - Wigner-D rotation matrices
            distances: (num_edges,) - edge distances

        Returns:
            output: (num_edges, channels_out, dim)
        """
        E = features.shape[0]

        # Step 1: Rotate to edge-aligned frame
        # features: (E, C_in, D) -> transpose to (E, D, C_in) for bmm
        # wigner: (E, D, D) @ (E, D, C_in) -> (E, D, C_in)
        x = torch.bmm(wigner, features.transpose(1, 2))

        # Step 2: Reindex l-primary -> m-primary
        x = x[:, self.l_to_m, :]  # (E, D, C_in)

        # Step 3: Compute radial modulation weights
        rad_weights = self.radial_mlp(distances.unsqueeze(-1))  # (E, radial_dim)

        # Step 4: SO(2) convolution per m-order
        out_parts = []
        coeff_offset = 0
        rad_offset = 0

        # m=0: real-valued
        n_l = self.m_sizes[0]
        x_m0 = x[:, coeff_offset:coeff_offset + n_l, :]  # (E, n_l, C_in)
        x_m0 = x_m0.reshape(E, n_l * self.channels_in)

        # Radial modulation
        rad_m0 = rad_weights[:, rad_offset:rad_offset + n_l * self.channels_in]
        x_m0 = x_m0 * rad_m0

        # Linear
        x_m0 = self.fc_m0(x_m0)  # (E, n_l * C_out)
        out_parts.append(x_m0.view(E, n_l, self.channels_out))

        coeff_offset += n_l
        rad_offset += n_l * self.channels_in

        # m>0: complex-valued
        for m in range(1, self.lmax + 1):
            n_l = self.m_sizes[m]
            rad_dim = n_l * self.channels_in

            # Extract real (+m) and imaginary (-m) blocks
            x_real = x[:, coeff_offset:coeff_offset + n_l, :]  # (E, n_l, C_in)
            x_imag = x[:, coeff_offset + n_l:coeff_offset + 2 * n_l, :]

            x_pair = torch.stack([
                x_real.reshape(E, rad_dim),
                x_imag.reshape(E, rad_dim),
            ], dim=1)  # (E, 2, n_l * C_in)

            # Radial modulation (shared for real/imag)
            rad_m = rad_weights[:, rad_offset:rad_offset + rad_dim]
            x_pair = x_pair * rad_m.unsqueeze(1)

            # Shared linear on both real and imaginary
            x_pair = self.fc_m[m - 1](x_pair)  # (E, 2, 2 * n_l * C_out)

            # Complex multiplication: split output into "real weight" and "imag weight" halves
            half = n_l * self.channels_out
            x_wr = x_pair[:, :, :half]      # real weight results
            x_wi = x_pair[:, :, half:]       # imag weight results

            # Re(out) = Wr*Xr - Wi*Xi, Im(out) = Wr*Xi + Wi*Xr
            out_real = x_wr[:, 0:1, :] - x_wi[:, 1:2, :]  # (E, 1, half)
            out_imag = x_wr[:, 1:2, :] + x_wi[:, 0:1, :]  # (E, 1, half)
            x_m = torch.cat([out_real, out_imag], dim=1)  # (E, 2, half)
            out_parts.append(x_m.view(E, 2 * n_l, self.channels_out))

            coeff_offset += 2 * n_l
            rad_offset += rad_dim

        # Step 5: Concatenate and reindex m-primary -> l-primary
        x = torch.cat(out_parts, dim=1)  # (E, D, C_out)
        x = x[:, self.m_to_l, :]

        # Step 6: Rotate back to global frame
        # wigner^T: (E, D, D)^T @ (E, D, C_out) -> (E, D, C_out)
        wigner_inv = wigner.transpose(1, 2)
        x = torch.bmm(wigner_inv, x)

        # (E, D, C_out) -> (E, C_out, D)
        return x.transpose(1, 2)

    def extra_repr(self) -> str:
        return (
            f"lmax={self.lmax}, channels_in={self.channels_in}, "
            f"channels_out={self.channels_out}, dim={self.dim}, "
            f"m_sizes={self.m_sizes}"
        )


def create_random_wigner(
    num_edges: int,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create random Wigner-D rotation matrices for benchmarking.

    Generates random block-diagonal orthogonal matrices matching the structure
    of real Wigner-D matrices. Each block is (2l+1) x (2l+1) orthogonal.

    In a real implementation, these would be computed from edge directions.

    Args:
        num_edges: Number of edges
        lmax: Maximum angular momentum
        device: Target device
        dtype: Target dtype

    Returns:
        wigner: (num_edges, dim, dim) where dim = (lmax+1)^2
    """
    dim = compute_dim(lmax)
    wigner = torch.zeros(num_edges, dim, dim, device=device, dtype=dtype)

    offset = 0
    for l in range(lmax + 1):
        size = 2 * l + 1
        # Random orthogonal block via QR decomposition
        random_mat = torch.randn(num_edges, size, size, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(random_mat)
        wigner[:, offset:offset + size, offset:offset + size] = q
        offset += size

    return wigner
