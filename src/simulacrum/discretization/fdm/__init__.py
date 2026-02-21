"""
Finite Difference Method discretization.

Modules (planned)
-----------------
stencils    Finite-difference stencil definitions (central, upwind, compact)
assembler   Structured-grid assembly using stencil operators
"""

from simulacrum.discretization.fdm.assembler import FDMAssembler

__all__ = ["FDMAssembler"]
