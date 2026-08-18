"""
Abstract base for discretization schemes.

Every scheme (FEM, FDM, FVM, SPH) must implement ``assemble`` which
produces a ``LinearSystem`` (or contributes to a nonlinear residual).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from scion.core.domain import Domain


@dataclass
class LinearSystem:
    """Container for the assembled K u = f system."""
    matrix: sparse.spmatrix
    rhs: np.ndarray


class DiscretizationBase(ABC):
    """Interface every discretization scheme must satisfy."""

    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def assemble(self) -> LinearSystem:
        """Assemble the global system of equations."""
        ...

    @abstractmethod
    def recover_fields(self, solution: np.ndarray) -> dict[str, np.ndarray]:
        """Post-process the solution vector into physical fields."""
        ...
