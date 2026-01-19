#!/usr/bin/env python3
"""Test S2Activation equivariance error at high L values."""

import torch
import sys
sys.path.insert(0, "/home/groups/rhiju/hmblair/flash-eq")

from flash_eq import Repr, S2Activation
from flash_eq.representations import WignerD


def test_equivariance(l_max: int, precision: int = 47, n_samples: int = 50, mult: int = 16):
    torch.manual_seed(42)

    repr_obj = Repr(lvals=list(range(l_max + 1)), mult=mult)
    model = S2Activation(repr_obj, precision=precision)
    model.eval()

    wigner = WignerD(repr_obj)
    x = torch.randn(n_samples, mult, repr_obj.dim())

    axis = torch.randn(1, 3)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(1) * 2 * 3.14159

    D = wigner.rot(axis, angle, cartesian=True).squeeze(0)

    with torch.no_grad():
        y1 = model(x)
        y1_rot = y1 @ D.T

        x_rot = x @ D.T
        y2 = model(x_rot)

        err = (y1_rot - y2).norm() / y2.norm()

    return err.item()


if __name__ == "__main__":
    from flash_eq.spherical import LEBEDEV_RULES

    print("Available precisions:", sorted(LEBEDEV_RULES.keys()))
    print()

    # Test multiple precisions for each L
    l_values = [6, 8, 10, 12, 14]
    precisions = [47, 59, 71, 83, 95, 107, 131]

    print("Equivariance Error (%) by L_max and Precision")
    print("=" * 80)

    header = f"{'L_max':<8}"
    for p in precisions:
        n_pts = LEBEDEV_RULES[p][0]
        header += f"p={p} ({n_pts})"[:14].ljust(14)
    print(header)
    print("-" * 80)

    for l_max in l_values:
        row = f"{l_max:<8}"
        for prec in precisions:
            try:
                err = test_equivariance(l_max, precision=prec)
                row += f"{err*100:.2f}%".ljust(14)
            except Exception as e:
                row += "ERR".ljust(14)
        print(row)
