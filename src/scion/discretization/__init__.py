"""
Discretization schemes: FEM, FDM, FVM, SPH.

All schemes inherit from ``DiscretizationBase`` and implement the same
``assemble()`` interface so they can be swapped transparently.
"""

from scion.discretization.base import DiscretizationBase

__all__ = ["DiscretizationBase"]
