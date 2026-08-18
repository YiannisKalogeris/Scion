r"""
The fictitious elastic response, one field per basic load case.

Every shakedown method is built on ``sigma^E``: the stress the structure would
carry if it stayed elastic forever, however far past yield that is. It is
fictitious by construction and that is the point -- Melan's theorem adds a
residual field to it to bring it back inside the yield surface, and the size of
the correction is what the shakedown factor measures.

Because the elastic response is linear, the whole load domain costs ``P`` linear
solves and no more: the stress at any point of the domain is the multiplier
combination of the basic fields. That is the reason direct methods are cheap
compared with running the cyclic history to a steady state.

Mandel, and all six components
------------------------------
Fields are stored as ``(n_cases, n_points, 6)`` in Mandel form, including in two
dimensions. The out-of-plane component is not optional: in plane strain
``sigma_33 = nu (sigma_11 + sigma_22)`` is nonzero, it enters ``J2``, and a
yield check that omits it is checking a different criterion. Here it is exact
rather than reconstructed, because the field really is elastic -- unlike the
elastoplastic case, where the same relation is wrong and the value has to come
from ``C : (eps - eps_p)``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import linalg as spla

from ..materials.linear_elastic import isotropic_stiffness_mandel
from ..materials.notation import mandel_to_voigt_tangent, voigt_strain_to_mandel

__all__ = ["ElasticEnvelope", "solve_elastic"]


def solve_elastic(tables, C_mandel: np.ndarray, free_dofs, forces,
                  prescribed: dict | None = None) -> list[np.ndarray]:
    """Solve one linear elastic problem per load case, sharing a factorisation.

    ``forces`` is a list of nodal force vectors, one per basic load case.
    ``prescribed`` maps a degree of freedom to a fixed value and is applied to
    every case alike -- a nonzero prescribed displacement is itself a load case,
    so pass it as one rather than trying to vary it here.
    """
    n_dof = tables.vector_dofs.n_total
    nv = tables.n_voigt
    C_v = mandel_to_voigt_tangent(np.asarray(C_mandel)[None], tables.dim)[0]
    C = np.broadcast_to(C_v, (tables.n_points, nv, nv))
    K, _ = tables.stiffness_and_force(np.zeros((tables.n_points, nv)), C, n_dof)

    free = np.asarray(free_dofs, dtype=np.int64)
    pres = np.array(sorted(prescribed or {}), dtype=np.int64)
    vals = np.array([prescribed[int(d)] for d in pres]) if pres.size \
        else np.zeros(0)

    K = K.tocsr()
    solve = spla.factorized(K[free][:, free].tocsc())

    out = []
    for f in forces:
        u = np.zeros(n_dof)
        if pres.size:
            u[pres] = vals
        rhs = np.asarray(f, dtype=float)[free]
        if pres.size:
            rhs = rhs - K[free][:, pres] @ u[pres]
        u[free] = solve(rhs)
        out.append(u)
    return out


class ElasticEnvelope:
    """Elastic stress fields for a set of basic load cases.

    Parameters
    ----------
    tables : the quadrature tables of the mesh.
    C_mandel : ``(6, 6)`` elastic stiffness in Mandel form.
    """

    def __init__(self, tables, C_mandel: np.ndarray):
        self.tables = tables
        self.C = np.asarray(C_mandel, dtype=float)
        self.dim = tables.dim
        self._cases: list[np.ndarray] = []
        self._displacements: list[np.ndarray] = []

    # ------------------------------------------------------------------
    @classmethod
    def from_elastic_constants(cls, tables, E: float, nu: float
                               ) -> "ElasticEnvelope":
        return cls(tables, isotropic_stiffness_mandel(E, nu))

    def add_case(self, u: np.ndarray) -> "ElasticEnvelope":
        """Add a basic load case from its nodal displacement solution."""
        eps = voigt_strain_to_mandel(self.tables.strains_voigt(u), self.dim)
        self._cases.append(eps @ self.C.T)
        self._displacements.append(np.asarray(u, dtype=float))
        return self

    def add_cases(self, displacements) -> "ElasticEnvelope":
        for u in displacements:
            self.add_case(u)
        return self

    # ------------------------------------------------------------------
    @property
    def n_cases(self) -> int:
        return len(self._cases)

    @property
    def stresses(self) -> np.ndarray:
        """``(n_cases, n_points, 6)`` Mandel."""
        if not self._cases:
            raise ValueError("no load cases have been added")
        return np.stack(self._cases)

    def stress_at(self, multipliers) -> np.ndarray:
        """The elastic stress at one point of the load domain, ``(n_points, 6)``."""
        m = np.atleast_1d(np.asarray(multipliers, dtype=float))
        if m.size != self.n_cases:
            raise ValueError(f"got {m.size} multipliers for "
                             f"{self.n_cases} load cases")
        return np.einsum("c,cpi->pi", m, self.stresses)

    def vertex_stresses(self, domain) -> np.ndarray:
        """``(n_vertices, n_points, 6)``, the fields the theorems act on.

        This is the array that sets the memory cost of a shakedown analysis,
        and it is materialised in full deliberately: every vertex is visited on
        every objective evaluation, and recomputing the combination each time
        costs more than storing it.
        """
        if domain.n_cases != self.n_cases:
            raise ValueError(
                f"the load domain has {domain.n_cases} cases but the envelope "
                f"holds {self.n_cases}; they must describe the same loads")
        return np.einsum("vc,cpi->vpi", domain.vertices, self.stresses)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"ElasticEnvelope({self.n_cases} cases, "
                f"{self.tables.n_points} points, dim={self.dim})")
