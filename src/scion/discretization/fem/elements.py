"""
Reference element shape functions and their derivatives.

Each element type is a callable: ``shape_functions(xi) -> (N, dN_dxi)``
where *xi* lives in the reference element.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ReferenceElement(ABC):
    """Base class for reference elements."""

    n_nodes: int
    dim: int  # parametric dimension

    @abstractmethod
    def evaluate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (N, dN/dxi) at parametric coordinate(s) *xi*.

        Parameters
        ----------
        xi : (dim,) parametric coordinate

        Returns
        -------
        N     : (n_nodes,)       shape-function values
        dN_dxi: (n_nodes, dim)   shape-function gradients in parametric space
        """
        ...


# =====================================================================
# 1-D elements
# =====================================================================

class Line2(ReferenceElement):
    """2-node linear line element, xi in [-1, 1]."""
    n_nodes = 2
    dim = 1

    def evaluate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = float(xi[0])
        N = np.array([0.5 * (1 - x), 0.5 * (1 + x)])
        dN = np.array([[-0.5], [0.5]])
        return N, dN


# =====================================================================
# 2-D elements
# =====================================================================

class Tri3(ReferenceElement):
    """3-node linear triangle, (xi, eta) in standard simplex."""
    n_nodes = 3
    dim = 2

    def evaluate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, t = float(xi[0]), float(xi[1])
        N = np.array([1 - s - t, s, t])
        dN = np.array([
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        return N, dN


class Quad4(ReferenceElement):
    """4-node bilinear quad, (xi, eta) in [-1,1]^2."""
    n_nodes = 4
    dim = 2

    def evaluate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, t = float(xi[0]), float(xi[1])
        N = 0.25 * np.array([
            (1 - s) * (1 - t),
            (1 + s) * (1 - t),
            (1 + s) * (1 + t),
            (1 - s) * (1 + t),
        ])
        dN = 0.25 * np.array([
            [-(1 - t), -(1 - s)],
            [(1 - t), -(1 + s)],
            [(1 + t), (1 + s)],
            [-(1 + t), (1 - s)],
        ])
        return N, dN


# =====================================================================
# 3-D elements
# =====================================================================

class Hex8(ReferenceElement):
    """8-node trilinear hexahedron, (xi, eta, zeta) in [-1,1]^3."""
    n_nodes = 8
    dim = 3

    _corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)

    def evaluate(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, t, u = float(xi[0]), float(xi[1]), float(xi[2])
        c = self._corners
        N = 0.125 * np.array([
            (1 + c[i, 0] * s) * (1 + c[i, 1] * t) * (1 + c[i, 2] * u)
            for i in range(8)
        ])
        dN = 0.125 * np.array([
            [
                c[i, 0] * (1 + c[i, 1] * t) * (1 + c[i, 2] * u),
                c[i, 1] * (1 + c[i, 0] * s) * (1 + c[i, 2] * u),
                c[i, 2] * (1 + c[i, 0] * s) * (1 + c[i, 1] * t),
            ]
            for i in range(8)
        ])
        return N, dN


# =====================================================================
# Registry
# =====================================================================

ELEMENT_REGISTRY: dict[str, type[ReferenceElement]] = {
    "line2": Line2,
    "tri3": Tri3,
    "quad4": Quad4,
    "hex8": Hex8,
}


def get_element(etype: str) -> ReferenceElement:
    """Instantiate a reference element by name."""
    if etype not in ELEMENT_REGISTRY:
        raise NotImplementedError(f"Element type '{etype}' not implemented yet")
    return ELEMENT_REGISTRY[etype]()
