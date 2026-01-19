"""Lebedev quadrature tables.

Loads precomputed Lebedev quadrature rules from compressed numpy archive.
Source: https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule

To regenerate the data file:
    python scripts/generate_lebedev_npz.py

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).parent / "data" / "lebedev.npz"
_data: dict | None = None


def _load_data() -> dict:
    """Load Lebedev data from npz file (cached)."""
    global _data
    if _data is None:
        _data = dict(np.load(_DATA_PATH))
    return _data


def get_available_precisions() -> list[int]:
    """Return list of available Lebedev precision values."""
    data = _load_data()
    return data['precisions'].tolist()


def get_rule_info(precision: int) -> tuple[int, int]:
    """Return (n_points, max_l) for a given precision."""
    data = _load_data()
    idx = data['precisions'].tolist().index(precision)
    return int(data['n_points'][idx]), int(data['max_l'][idx])


@lru_cache(maxsize=8)
def get_lebedev_rule(precision: int) -> tuple[np.ndarray, np.ndarray]:
    """Get Lebedev quadrature points and weights for a given precision.

    Args:
        precision: Lebedev precision (17, 23, 29, ..., 131).

    Returns:
        points: (n_points, 3) array of unit vectors
        weights: (n_points,) array of quadrature weights (sum to 1.0)
    """
    data = _load_data()

    if precision not in data['precisions']:
        available = data['precisions'].tolist()
        raise ValueError(f"Unknown precision {precision}. Available: {available}")

    points = data[f'points_{precision}']
    weights = data[f'weights_{precision}']

    return points, weights


# Legacy compatibility: LEBEDEV_RULES dict interface
class _LebedevRulesDict:
    """Dict-like interface for backward compatibility."""

    def __getitem__(self, precision: int):
        data = _load_data()
        idx = data['precisions'].tolist().index(precision)
        n_points = int(data['n_points'][idx])
        max_l = int(data['max_l'][idx])
        points = data[f'points_{precision}']
        weights = data[f'weights_{precision}']
        # Return (n_points, max_l, list of (x, y, z, w) tuples)
        point_list = [(p[0], p[1], p[2], w) for p, w in zip(points, weights)]
        return (n_points, max_l, point_list)

    def __contains__(self, precision: int) -> bool:
        data = _load_data()
        return precision in data['precisions']

    def keys(self):
        return get_available_precisions()


LEBEDEV_RULES = _LebedevRulesDict()
