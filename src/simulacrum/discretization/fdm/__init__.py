"""
Finite differences on uniform Cartesian grids.

    stencils    coefficient sets, and what each one is actually accurate to
    assembler   the Laplacian and its right-hand side, 1-D and 2-D

Two things here are easy to get wrong and are documented where they bite.

A multi-dimensional central Laplacian is a **sum of 1-D operators**, one per
direction with its own spacing. Assembling it as a single 2-D stencil divided by
one number is correct on a square grid and silently wrong on any other -- the
scheme then converges to a different equation, at rate zero.

The compact 9-point stencil is **second** order as written. It reaches fourth
order only when the right-hand side is corrected to ``f + (h^2/12) lap f``;
without that it costs nine points for the accuracy of five.
"""

from simulacrum.discretization.fdm.assembler import FDMAssembler
from simulacrum.discretization.fdm.stencils import (
    is_consistent,
    laplacian_1d,
    laplacian_2d_5pt,
    laplacian_2d_9pt,
    mehrstellen_rhs_stencil,
)

__all__ = [
    "FDMAssembler",
    "laplacian_1d",
    "laplacian_2d_5pt",
    "laplacian_2d_9pt",
    "mehrstellen_rhs_stencil",
    "is_consistent",
]
