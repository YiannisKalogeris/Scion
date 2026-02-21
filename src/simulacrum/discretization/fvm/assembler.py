"""
Cell-centred Finite Volume assembler (scaffold).

The FVM assembler will operate on a dual/cell-centred mesh and assemble
flux contributions across cell faces.
"""

from __future__ import annotations

import numpy as np

from simulacrum.core.domain import Domain
from simulacrum.discretization.base import DiscretizationBase, LinearSystem


class FVMAssembler(DiscretizationBase):
    """Finite Volume Method assembler (to be implemented).

    Parameters
    ----------
    domain : Domain
    flux_scheme : str
        Name of the numerical flux function (e.g. ``"central"``, ``"upwind"``).
    """

    def __init__(self, domain: Domain, flux_scheme: str = "central"):
        super().__init__(domain)
        self.flux_scheme = flux_scheme

    def assemble(self) -> LinearSystem:
        raise NotImplementedError("FVM assembly not yet implemented")

    def recover_fields(self, solution: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError("FVM field recovery not yet implemented")
