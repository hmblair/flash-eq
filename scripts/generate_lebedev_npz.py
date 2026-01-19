#!/usr/bin/env python3
"""Generate Lebedev quadrature tables as npz file.

Downloads verified Lebedev quadrature rules and saves as compressed numpy archive.

Source: https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/

Usage:
    python scripts/generate_lebedev_npz.py

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

import urllib.request
import math
import numpy as np
from pathlib import Path

BASE_URL = "https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule"

# (precision, n_points, max_l for exact integration)
RULES = [
    (17, 110, 8),
    (23, 194, 11),
    (29, 302, 14),
    (35, 434, 17),
    (41, 590, 20),
    (47, 770, 23),
    (53, 974, 26),
    (59, 1202, 29),
    (65, 1454, 32),
    (71, 1730, 35),
    (77, 2030, 38),
    (83, 2354, 41),
    (89, 2702, 44),
    (95, 3074, 47),
    (101, 3470, 50),
    (107, 3890, 53),
    (113, 4334, 56),
    (119, 4802, 59),
    (125, 5294, 62),
    (131, 5810, 65),
]


def download_rule(precision: int) -> list[tuple[float, float, float]]:
    """Download Lebedev rule and return (theta_deg, phi_deg, weight) tuples."""
    url = f"{BASE_URL}/lebedev_{precision:03d}.txt"

    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')

    points = []
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3:
            theta_deg = float(parts[0])
            phi_deg = float(parts[1])
            weight = float(parts[2])
            points.append((theta_deg, phi_deg, weight))

    return points


def convert_to_cartesian(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    """Convert Burkardt's (theta, phi) in degrees to (x, y, z) unit vector."""
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return (x, y, z)


def main():
    output_path = Path(__file__).parent.parent / "flash_eq" / "data" / "lebedev.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}

    # Store metadata
    precisions = [r[0] for r in RULES]
    n_points_list = [r[1] for r in RULES]
    max_l_list = [r[2] for r in RULES]

    data['precisions'] = np.array(precisions, dtype=np.int32)
    data['n_points'] = np.array(n_points_list, dtype=np.int32)
    data['max_l'] = np.array(max_l_list, dtype=np.int32)

    for precision, expected_points, max_l in RULES:
        print(f"Downloading precision {precision}...")

        raw_points = download_rule(precision)

        if len(raw_points) != expected_points:
            print(f"  WARNING: Expected {expected_points} points, got {len(raw_points)}")

        # Convert to arrays
        points = np.zeros((len(raw_points), 3), dtype=np.float64)
        weights = np.zeros(len(raw_points), dtype=np.float64)

        for i, (theta_deg, phi_deg, weight) in enumerate(raw_points):
            x, y, z = convert_to_cartesian(theta_deg, phi_deg)
            points[i] = [x, y, z]
            weights[i] = weight

        print(f"  {len(raw_points)} points, weights sum = {weights.sum():.10f}")

        data[f'points_{precision}'] = points
        data[f'weights_{precision}'] = weights

    # Save compressed
    np.savez_compressed(output_path, **data)
    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
