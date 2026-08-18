r"""
The load domain, and why only its vertices matter.

A shakedown problem is not posed at a load, it is posed on a *set* of loads. The
structure must survive any history that stays inside the set, in any order,
forever. The set is normally a box: ``P`` basic load cases, each free to vary
independently between a lower and an upper multiplier.

    domain = LoadDomain.box(lower=[0.0, -1.0], upper=[1.0, 1.0])

Vertices are enough
-------------------
Melan's condition is that one residual field keeps the total stress admissible
for *every* load in the domain. The elastic stress depends linearly on the
multipliers and the yield function is convex, so the constraint over the whole
box is implied by the constraint at its extreme points. Checking the vertices is
not an approximation -- it is exactly equivalent, and it turns an infinite family
of constraints into ``2^P`` of them.

That exponent is the cost of shakedown analysis, and it is worth seeing rather
than discovering. Four load cases is 16 vertices; ten is 1024, each carrying a
full stress field at every quadrature point. :attr:`LoadDomain.n_vertices`
reports it and the constructor refuses a domain large enough to exhaust memory
silently.

Degenerate directions
---------------------
A case with equal bounds is held fixed and contributes no vertex pair, so a
domain with every bound fixed has exactly one vertex. That is limit analysis:
a single load, no variation, and the shakedown factor collapses onto the plastic
collapse factor. :meth:`LoadDomain.fixed` builds it, and it is the cheapest
external check available on a shakedown implementation, because plastic collapse
has closed-form answers that shakedown does not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LoadDomain"]


class LoadDomain:
    """A box of load multipliers, and its vertices.

    Parameters
    ----------
    lower, upper : per-case multiplier bounds, length ``P``.
    names : optional labels for the basic load cases, used only in reporting.
    max_vertices : refuse to build a domain with more vertices than this. The
        default of 4096 is roughly where the vertex stress fields stop fitting
        comfortably in memory for a mesh of any size.
    """

    def __init__(self, lower, upper, names: list[str] | None = None,
                 max_vertices: int = 4096):
        self.lower = np.atleast_1d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(upper, dtype=float))
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower {self.lower.shape} and upper {self.upper.shape} must "
                f"have the same length")
        if np.any(self.upper < self.lower):
            bad = int(np.argmin(self.upper - self.lower))
            raise ValueError(
                f"load case {bad} has upper {self.upper[bad]:g} below lower "
                f"{self.lower[bad]:g}")

        self.names = (list(names) if names is not None
                      else [f"case {i}" for i in range(self.n_cases)])
        if len(self.names) != self.n_cases:
            raise ValueError(f"got {len(self.names)} names for "
                             f"{self.n_cases} load cases")

        self._varying = np.flatnonzero(self.upper > self.lower)
        n = 1 << int(self._varying.size)
        if n > max_vertices:
            raise ValueError(
                f"{self._varying.size} varying load cases give {n} vertices, "
                f"over the limit of {max_vertices}. Every vertex carries a "
                f"stress field at every quadrature point, so this is a memory "
                f"decision, not a tolerance: reduce the number of independent "
                f"cases, or raise max_vertices deliberately.")

    # ------------------------------------------------------------------
    @classmethod
    def box(cls, lower, upper, **kw) -> "LoadDomain":
        """The general case: each multiplier free between its bounds."""
        return cls(lower, upper, **kw)

    @classmethod
    def fixed(cls, multipliers=1.0, **kw) -> "LoadDomain":
        """A single load with no variation -- limit analysis.

        The shakedown factor of a one-vertex domain *is* the plastic collapse
        factor, which is what makes this the calibration case.
        """
        m = np.atleast_1d(np.asarray(multipliers, dtype=float))
        return cls(m, m, **kw)

    @classmethod
    def pulsating(cls, n_cases: int = 1, **kw) -> "LoadDomain":
        """Each case free between zero and one: loads that switch on and off
        independently. The usual first domain to try."""
        return cls(np.zeros(n_cases), np.ones(n_cases), **kw)

    @classmethod
    def alternating(cls, n_cases: int = 1, **kw) -> "LoadDomain":
        """Each case free between minus one and one, fully reversed."""
        return cls(-np.ones(n_cases), np.ones(n_cases), **kw)

    # ------------------------------------------------------------------
    @property
    def n_cases(self) -> int:
        return int(self.lower.size)

    @property
    def n_varying(self) -> int:
        return int(self._varying.size)

    @property
    def n_vertices(self) -> int:
        return 1 << self.n_varying

    @property
    def vertices(self) -> np.ndarray:
        """``(n_vertices, n_cases)`` of multipliers.

        Fixed cases take their common value in every vertex; varying cases take
        every combination of their two bounds.
        """
        out = np.tile(self.lower, (self.n_vertices, 1))
        for bit, case in enumerate(self._varying):
            take_upper = (np.arange(self.n_vertices) >> bit) & 1
            out[take_upper.astype(bool), case] = self.upper[case]
        return out

    # ------------------------------------------------------------------
    def contains(self, multipliers) -> np.ndarray:
        """Whether each row of ``multipliers`` lies in the domain."""
        m = np.atleast_2d(np.asarray(multipliers, dtype=float))
        return np.all((m >= self.lower - 1e-12)
                      & (m <= self.upper + 1e-12), axis=1)

    def describe(self) -> str:
        rows = [f"    {n:<16} [{lo:g}, {hi:g}]"
                + ("   fixed" if hi == lo else "")
                for n, lo, hi in zip(self.names, self.lower, self.upper)]
        return "\n".join([repr(self), *rows])

    def __len__(self) -> int:
        return self.n_vertices

    def __repr__(self) -> str:
        kind = ("limit analysis" if self.n_varying == 0
                else f"{self.n_varying} varying")
        return (f"LoadDomain({self.n_cases} cases, {kind}, "
                f"{self.n_vertices} vertices)")
