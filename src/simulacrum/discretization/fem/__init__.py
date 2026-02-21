"""
Finite Element Method discretization.

Modules
-------
elements    Reference element shape functions and quadrature
quadrature  Gauss quadrature rules
assembler   Global matrix / vector assembly
"""

from simulacrum.discretization.fem.assembler import FEMAssembler

__all__ = ["FEMAssembler"]
