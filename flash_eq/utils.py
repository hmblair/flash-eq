"""Utility functions for flash-eq."""
from __future__ import annotations

import torch
import torch.nn as nn


def init_linear_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for Linear layers.

    Intended for use with module.apply(init_linear_weights).
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_epsilon(dtype: torch.dtype) -> float:
    """Get epsilon value appropriate for the given dtype.

    FP16 has machine epsilon ~9.77e-4, so we use 1e-5 for safety.
    FP32 has machine epsilon ~1.19e-7, so we use 1e-8.
    """
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 1e-5
    return 1e-8
