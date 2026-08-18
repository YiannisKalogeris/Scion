"""
Finite Element Method discretization.

Modules
-------
elements    Reference element shape functions and quadrature
quadrature  Gauss quadrature rules
assembler   Global matrix / vector assembly
"""

from scion.discretization.fem.assembler import FEMAssembler
from .mass import assemble_mass_matrix

__all__ = ["FEMAssembler", "assemble_mass_matrix"]
