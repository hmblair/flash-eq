"""SO(3) representation theory for equivariant neural networks.

This subpackage implements the real irreducible representations of SO(3),
the 3D rotation group. These are the mathematical foundation for
building rotation-equivariant neural networks.

Types:
    Irrep: Single irreducible representation
    ProductIrrep: Tensor product decomposition of two irreps
    Repr: Collection of irreps into a unified representation
    ProductRepr: Tensor product of two representations

Wigner matrices:
    WignerD: Computes Wigner D-matrices for a representation
    WignerDBasis: Computes basis matrices from direction vectors

Author: Hamish M. Blair <hmblair@stanford.edu>
"""

from .types import Irrep, ProductIrrep, Repr, ProductRepr
from .wigner import WignerD, WignerDBasis

__all__ = [
    # Types
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    # Wigner matrices
    "WignerD",
    "WignerDBasis",
]
