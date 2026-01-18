"""Utility functions for flash-eq."""
from __future__ import annotations

import torch


def get_epsilon(dtype: torch.dtype) -> float:
    """Get epsilon value appropriate for the given dtype.

    FP16 has machine epsilon ~9.77e-4, so we use 1e-5 for safety.
    FP32 has machine epsilon ~1.19e-7, so we use 1e-8.
    """
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 1e-5
    return 1e-8
