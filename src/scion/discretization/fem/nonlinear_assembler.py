r"""
Assembly for path-dependent materials.

``FEMAssembler`` builds a fixed stiffness from a tangent that depends on
nothing. That is right for elasticity and impossible for plasticity, where the
tangent depends on the current strain *and* on the committed history, and where
the internal force is not ``K u``.

What this assembler provides instead is the pair a Newton solver needs,

    f_int(u, state)     and     K_t = d f_int / d u,

together with the updated state. The constitutive work is delegated entirely to
an :class:`~scion.materials.plasticity.model.InelasticMaterial`; nothing
here knows what a yield surface is.

Precomputation
--------------
For small strains the mesh does not move, so ``B``, ``det J`` and the quadrature
weights are computed once in the constructor. That is most of the cost of an
element loop, and a plasticity run is thousands of them.

State discipline
----------------
:meth:`assemble` takes the **committed** state and never mutates it, returning
the trial state alongside the operators. The caller commits only once the load
step has converged. Advancing the plastic strain on a Newton iterate that is
later discarded corrupts the history irrecoverably -- the same failure mode as
writing a phase-field history field from a rejected iterate, and the reason
:class:`~scion.discretization.fem.quadrature_state.QuadratureState` keeps
trial and committed copies.

Solver symmetry
---------------
:attr:`symmetric` forwards the material's own flag. Non-associative plastic
flow gives an unsymmetric tangent; storing it in a symmetric format discards
half of it, and a Cholesky factorisation of it will either fail or return
nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from scion.core.dof import DOFMap
from scion.discretization.fem.assembler import FEMAssembler
from scion.discretization.fem.elements import get_element
from scion.discretization.fem.quadrature import get_quadrature

__all__ = ["ElastoplasticAssembler", "AssemblyResult"]


@dataclass
class AssemblyResult:
    """Operators and diagnostics from one assembly."""

    K: sparse.csr_matrix
    f_int: np.ndarray
    state: object                 # trial state, not yet committed
    sigma: np.ndarray             # (n_points, n_voigt)
    plastic: np.ndarray           # (n_points,) bool
    apex: np.ndarray              # (n_points,) bool
    iterations: int
    converged: bool

    def __repr__(self) -> str:
        n = len(self.plastic)
        return (f"AssemblyResult(points={n}, "
                f"plastic={int(self.plastic.sum())} "
                f"({100 * self.plastic.mean():.1f}%), "
                f"apex={int(self.apex.sum())}, "
                f"return_map_iters={self.iterations}, "
                f"converged={self.converged})")


class ElastoplasticAssembler:
    """Internal force and consistent tangent for an inelastic material.

    Parameters
    ----------
    domain : the usual ``Domain``; a single element block is assumed, since the
        quadrature-point numbering that indexes the state must be unambiguous.
    material : an ``InelasticMaterial``.
    thickness : out-of-plane thickness in 2-D.
    quad_order : points per direction; the default rule is used when omitted.
        Worth raising for nearly incompressible plastic flow, where a low-order
        rule locks.
    """

    def __init__(self, domain, material, thickness: float = 1.0,
                 quad_order: int | None = None):
        self.domain = domain
        self.material = material
        self.thickness = float(thickness)
        self.dim = domain.coordinates.shape[1]
        self.n_voigt = 3 if self.dim == 2 else 6

        blocks = domain.mesh.element_blocks
        if len(blocks) != 1:
            raise NotImplementedError(
                "a single element block is assumed so that quadrature points "
                "are numbered unambiguously; the state array is indexed by "
                "that numbering")
        self.block = blocks[0]
        self.dof_map = DOFMap(domain.n_nodes, self.dim)

        self._precompute(quad_order)

    # ------------------------------------------------------------------
    def _precompute(self, quad_order):
        ref = get_element(self.block.etype)
        qpts, qwts = (get_quadrature(self.block.etype, order=quad_order)
                      if quad_order else get_quadrature(self.block.etype))
        self.n_qp = len(qwts)

        coords_all = self.domain.coordinates
        conn = self.block.connectivity
        self.conn = conn
        self.n_elements, self.npe = conn.shape
        self.n_points = self.n_elements * self.n_qp

        B = np.empty((self.n_elements, self.n_qp, self.n_voigt,
                      self.npe * self.dim))
        w = np.empty((self.n_elements, self.n_qp))

        for e in range(self.n_elements):
            xe = coords_all[conn[e]]
            for q, (qp, qw) in enumerate(zip(qpts, qwts)):
                _, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ xe
                # same convention as FEMAssembler; the transpose matters for
                # any element whose Jacobian is not symmetric
                dN_dx = dN_dxi @ np.linalg.inv(J).T
                B[e, q] = FEMAssembler._strain_displacement(dN_dx, self.dim)
                w[e, q] = abs(np.linalg.det(J)) * qw * self.thickness

        self.B = B
        self.w = w
        self.edofs = np.array([self.dof_map.element_dofs(c) for c in conn])

        nd = self.edofs.shape[1]
        self._rows = np.repeat(self.edofs, nd, axis=1).ravel()
        self._cols = np.tile(self.edofs, (1, nd)).ravel()

    # ------------------------------------------------------------------
    @property
    def symmetric(self) -> bool:
        """Whether the tangent may be treated as symmetric."""
        return getattr(self.material, "symmetric_tangent", True)

    @property
    def n_dof(self) -> int:
        return self.dof_map.n_total

    def initial_state(self):
        return self.material.initial_state(self.n_points)

    # ------------------------------------------------------------------
    def strains(self, u: np.ndarray) -> np.ndarray:
        """Voigt strains at every quadrature point, ``(n_points, n_voigt)``."""
        ue = np.asarray(u, dtype=float)[self.edofs]
        eps = np.einsum("eqab,eb->eqa", self.B, ue)
        return eps.reshape(self.n_points, self.n_voigt)

    # ------------------------------------------------------------------
    def assemble(self, u: np.ndarray, state, **kw) -> AssemblyResult:
        """Internal force, consistent tangent and trial state at ``u``.

        ``state`` is the committed state and is not modified.
        """
        eps = self.strains(u)
        sigma, C, res = self.material.update(eps, state, dim=self.dim, **kw)

        nv = self.n_voigt
        C = C.reshape(self.n_elements, self.n_qp, nv, nv)
        s = sigma.reshape(self.n_elements, self.n_qp, nv)

        ke = np.einsum("eqab,eqbc,eqcd,eq->ead",
                       np.swapaxes(self.B, 2, 3), C, self.B, self.w)
        fe = np.einsum("eqab,eqa,eq->eb", self.B, s, self.w)

        K = sparse.csr_matrix((ke.ravel(), (self._rows, self._cols)),
                              shape=(self.n_dof, self.n_dof))
        f_int = np.zeros(self.n_dof)
        np.add.at(f_int, self.edofs.ravel(), fe.ravel())

        return AssemblyResult(K, f_int, res.state, sigma, res.plastic,
                              res.apex, res.iterations, res.converged)

    # ------------------------------------------------------------------
    def residual_and_tangent(self, state, f_ext=None):
        """Callables ``(residual, tangent)`` suitable for a Newton solver and
        for :func:`~scion.solvers.tangent_check.check_tangent`.

        Both close over a *fixed* committed ``state``, which is what makes the
        residual a pure function of ``u`` -- the property the finite-difference
        check requires and the one an in-place state update would destroy.
        """
        f_ext = np.zeros(self.n_dof) if f_ext is None else np.asarray(f_ext)

        def residual(u):
            return self.assemble(u, state).f_int - f_ext

        def tangent(u):
            return self.assemble(u, state).K

        return residual, tangent

    def __repr__(self) -> str:
        return (f"ElastoplasticAssembler({self.n_elements} {self.block.etype}, "
                f"{self.n_qp} qp/element, {self.n_points} points, "
                f"{self.n_dof} dof, symmetric={self.symmetric})")


class StatefulOperators:
    """Adapter between an :class:`ElastoplasticAssembler` and a Newton driver.

    The drivers in :mod:`scion.solvers.nonlinear` take
    ``internal_force_fn(u)`` and ``tangent_fn(u)`` -- pure functions of the
    displacement. A path-dependent material has no such functions: the answer
    also depends on the committed history. This holds that history, so the two
    callables become pure again for the duration of a load step.

    Two details it exists to get right.

    **Commit once per accepted step.** :meth:`commit` promotes the trial state,
    and it must be called only when a step has converged -- wire it to the
    driver's ``on_step_converged`` hook. Committing inside the assembly would
    let a Newton iterate that is later backtracked over, or a step that is
    rejected outright, advance the plastic strain permanently.

    **Assemble once per iterate.** The drivers call ``f_int(u)`` and ``K_T(u)``
    separately at the same ``u``, and a line search calls ``f_int`` again at
    trial points. Without the cache here each Newton iteration would run the
    return map two or three times over.
    """

    def __init__(self, assembler: ElastoplasticAssembler, state=None):
        self.assembler = assembler
        self.state = state if state is not None else assembler.initial_state()
        self._key = None
        self._result = None
        self.n_assemblies = 0

    # ------------------------------------------------------------------
    def _evaluate(self, u: np.ndarray) -> AssemblyResult:
        key = np.asarray(u, dtype=float).tobytes()
        if key != self._key:
            self._result = self.assembler.assemble(u, self.state)
            self._key = key
            self.n_assemblies += 1
        return self._result

    def internal_force(self, u: np.ndarray) -> np.ndarray:
        return self._evaluate(u).f_int

    def tangent(self, u: np.ndarray):
        return self._evaluate(u).K

    # ------------------------------------------------------------------
    def commit(self, u: np.ndarray | None = None) -> None:
        """Accept the trial state. Wire this to ``on_step_converged``."""
        if u is not None:
            self._evaluate(u)
        if self._result is not None:
            self.state = self._result.state
            self._key = None                  # the cache is now stale
            self._result = None

    @property
    def last(self) -> AssemblyResult | None:
        """The most recent assembly, for diagnostics."""
        return self._result
