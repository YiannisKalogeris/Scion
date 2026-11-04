r"""
Phase-field brittle fracture: the concrete two-field driver.

Assembles and solves

    momentum:     INT sigma(eps, d) : eps(v) dx = external work
    phase field:  INT [ c_g grad d . grad w + r(d, H) w ] dx = 0

with ``sigma``, ``r`` and ``c_g`` supplied by
:class:`~scion.materials.fracture.PhaseFieldModel`, and the alternation
driven by :class:`~scion.solvers.StaggeredSolver`.

Why this does its own element loop
----------------------------------
``FEMAssembler`` builds a fixed stiffness from a material tangent that depends
on nothing. Here both tangents depend on the *other* field evaluated at the
quadrature points -- ``C`` on ``d``, and the phase-field mass term on ``H``,
which in turn depends on ``eps``. So the loop has to interleave.

The tables it interleaves over -- shape functions, gradients, weights, B
matrices -- and the scalar-field operators built from them are shared with
gradient damage and live in
:class:`~scion.discretization.fem.tables.QuadratureTables`. What is left
here is only the physics: which equations, and in what order.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from scion.discretization.fem.quadrature_state import QuadratureState
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.fracture import PhaseFieldModel
from scion.solvers.bound_constrained import solve_bounded

from .two_field import Field, TwoFieldProblem

__all__ = ["PhaseFieldFractureProblem", "solve_phase_field_system",
           "isolated_unknowns"]


def isolated_unknowns(A) -> np.ndarray:
    """Mask of unknowns appearing in one equation only, their own.

    A row holding a single stored entry with a nonzero diagonal holds *the*
    diagonal, so the first test identifies one equation in one unknown. The
    second is not redundant: a row can hold only its diagonal while its
    *column* still has entries, and splitting such an unknown off would leave
    it in the free block as well. Neither test alone is safe -- ``nnz == 1``
    without the diagonal check divides by a structural zero.

    Read off the matrix rather than from the caller's unsupported-node set on
    purpose: `PhaseFieldMPM.refresh` moves that set as the body deforms, and
    a mask derived from the operator in hand cannot fall out of step with it.
    """
    A = A.tocsr()
    lone = (np.diff(A.indptr) == 1) & (A.diagonal() != 0.0)
    return lone & (np.bincount(A.indices, minlength=A.shape[0]) == 1)


def solve_phase_field_system(K, R, solver=None):
    """Solve the linearised phase-field system ``K delta = R``.

    ``solver`` is ``None`` for SciPy's sparse direct solve, which is the
    default and the path every published number here was produced with, or a
    :class:`~scion.solvers.linear.LinearSolverBase`. The one to reach for is
    ``KrylovSolver(method="pcg")``: this operator is
    ``K_grad + mass(c alpha'' + g'' H)`` with both terms strictly positive, so
    unlike the micropolar tangent it is scalar, SPD and diagonally dominant on
    the scale of ``l``, and Jacobi-PCG converges in tens of iterations at any
    mesh. **AMG is not the answer here** despite the operator being elliptic:
    measured on a real torsion phase-field matrix its hierarchy stops
    contracting exactly as on the 6-DOF tangent ("2552 unknowns from 2552"),
    after 43 s of setup. IC(0) takes 20 iterations and loses on setup, at
    Python speed. Jacobi wins by having no setup at all.

    Why the direct solve has to go, measured on the torsion path with the
    matrix lifted out of a real run:

        a/dx   coupled dof   spsolve   Jacobi-PCG   iters    gain
           6         2,755     0.073        0.004      46      19x
           8         6,225     0.652        0.008      55      85x
          12        19,129     8.053        0.033      63     248x

    ``spsolve`` is ``N^2.46`` -- the 3-D fill wall already recorded for the
    micropolar tangent, reached here through a 24-nonzeros-per-row operator
    because MPM stencils overlap -- against ``N^1.17`` for PCG, whose
    iteration count is governed by ``l/dx`` rather than by the specimen size.
    So this is the rare lever whose value *grows* with resolution.

    **Not bit-identical to the direct solve**, which is why it is opt-in. One
    coupled load step of the notched micropolar plate at ``tol = 1e-10``: the
    damage maximum, the history field and ``||F||`` all *exactly* equal,
    ``d_sum`` 1.6e-14, the reaction 1.1e-12. The increment is what moves, and
    the stagger loop re-converges it.

    Structurally isolated unknowns -- a unit diagonal and nothing else in
    their row *and* column, which is what `PhaseFieldMPM._laplacian` leaves on
    a node with no particle support -- are solved directly and kept out of the
    Krylov space. They are 74% of the rows at ``a/dx = 8``. They cost
    `spsolve` nothing, since SuperLU eliminates them for free (1.800 s against
    1.834 with them removed by hand), and double PCG's matvec for no change in
    its iteration count.
    """
    if solver is None:
        return spla.spsolve(K.tocsc(), R)
    A = K.tocsr()
    iso = isolated_unknowns(A)
    if not iso.any():
        return solver.solve(A, R)
    delta = np.zeros(A.shape[0])
    # one equation in one unknown, so this is exact rather than a tolerance
    delta[iso] = R[iso] / A.diagonal()[iso]
    free = np.flatnonzero(~iso)
    delta[free] = solver.solve(A[free][:, free].tocsr(), R[free])
    return delta


def _solve_constrained(K, f, free, prescribed, values):
    """Solve ``K x = f`` with ``x[prescribed] = values``, returning x and the
    reaction at the prescribed DOFs."""
    n = K.shape[0]
    x = np.zeros(n)
    if len(prescribed):
        x[prescribed] = values
    K = K.tocsr()
    rhs = f[free] - (K[free][:, prescribed] @ x[prescribed]
                     if len(prescribed) else 0.0)
    if len(free):
        x[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    reaction = (K @ x - f)[prescribed] if len(prescribed) else np.zeros(0)
    return x, reaction


class PhaseFieldFractureProblem(TwoFieldProblem):
    """Quasi-static phase-field fracture under displacement control.

    Parameters
    ----------
    domain     : Domain -- mesh, coordinates, element blocks
    model      : PhaseFieldModel -- the four composed axes
    thickness  : out-of-plane thickness for 2-D (plane strain)
    strategy, stagger_tol, max_stagger, verbose : see TwoFieldProblem

    The displacement field is ``self.u`` and the phase field ``self.d``; both
    are :class:`Field` objects, so boundary conditions are set through
    ``problem.u.prescribe(nodes, component, value)``. Load stepping scales the
    prescribed displacements by the load factor, so the values set at setup are
    the values at factor 1.
    """

    field_names = ("displacement", "phase field")

    def __init__(self, domain, model: PhaseFieldModel, thickness: float = 1.0,
                 strategy: str = "staggered", stagger_tol: float = 10.0,
                 max_stagger: int = 100, damage_tol: float = 1.0e-4,
                 verbose: bool = False):
        super().__init__(domain, strategy, stagger_tol, max_stagger,
                         damage_tol, verbose)
        self.model = model
        self.thickness = float(thickness)
        self.dim = domain.coordinates.shape[1]
        #: Linear solver for the phase-field system. ``None`` is SciPy's
        #: direct solve; see `solve_phase_field_system` for what to set it to
        #: and what it costs in the last digits.
        self.damage_solver = None

        n_nodes = domain.n_nodes
        self.u = Field("u", n_nodes, self.dim)
        self.d = Field("d", n_nodes, 1)
        # The damage at the last *committed* load step. The irreversibility
        # bound is measured against this and not against the previous
        # staggered sweep -- the same reason ``H`` is taken against
        # ``state.committed("H")``: a sweep that is later discarded must not
        # leave a permanent mark. Clamping against the running field makes the
        # converged answer depend on how the stagger loop reached it.
        self._d_committed = np.zeros(n_nodes)
        self._u_base: dict[int, float] = {}

        self._build_quadrature_tables()
        self.state = QuadratureState(self.n_elements, self.n_qp)
        self.state.add("H")

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    def _build_quadrature_tables(self) -> None:
        """Build the shared tables and mirror the names this class uses.

        The tables themselves are :class:`QuadratureTables`; the aliases below
        exist so the element loops read the same as they always did.
        """
        self.tables = QuadratureTables(self.domain, self.thickness)
        t = self.tables
        self.conn = t.conn
        self.n_elements, self.npe, self.n_qp = t.n_elements, t.npe, t.n_qp
        self.N, self.dN, self.w, self.B = t.N, t.dN, t.w, t.B
        self.u_edofs = t.u_edofs
        self.d_edofs = t.scalar_edofs
        self.element_size = t.element_size

    # ------------------------------------------------------------------
    def check_mesh(self) -> None:
        """Warn when the mesh cannot resolve the regularization length."""
        self.model.check_mesh_size(self.element_size)

    # ------------------------------------------------------------------
    # Quadrature-point quantities
    # ------------------------------------------------------------------
    def strains(self) -> np.ndarray:
        """``(n_points, dim, dim)`` strain tensors from the current u."""
        return self.tables.strain_tensors(self.u.values)

    def damage_at_points(self) -> np.ndarray:
        return self.tables.interpolate(self.d.values)

    def damage_gradient(self) -> np.ndarray:
        return self.tables.gradient(self.d.values)

    # ------------------------------------------------------------------
    # Momentum equation
    # ------------------------------------------------------------------
    def solve_first(self) -> int:
        """Solve for u at fixed d.

        One linear solve for the isotropic and hybrid couplings, Newton for the
        anisotropic ones. That difference is the entire computational argument
        for the hybrid formulation.
        """
        return self._solve_u_linear() if self.model.linear_momentum \
            else self._solve_u_newton()

    def _assemble_u(self):
        eps = self.strains()
        d = self.damage_at_points()
        d = self.model.apply_compression_constraint(d, eps)
        sig_v, C_v = self.model.stress_tangent_voigt(eps, d)
        return self.tables.stiffness_and_force(sig_v, C_v, self.u.n_dof)

    def _solve_u_linear(self) -> int:
        K, _ = self._assemble_u()
        x, reac = _solve_constrained(K, np.zeros(self.u.n_dof),
                                     self.u.free_dofs, self.u.prescribed_dofs,
                                     self.u.prescribed_values)
        self.u.values = x
        self._reaction = reac
        return 1

    def _solve_u_newton(self, tol: float = 1e-9, max_iter: int = 20) -> int:
        self.u.apply_prescribed()
        free, pres = self.u.free_dofs, self.u.prescribed_dofs
        for it in range(1, max_iter + 1):
            K, f_int = self._assemble_u()
            R = -f_int
            R[pres] = 0.0
            norm = np.linalg.norm(R[free])
            if it == 1:
                ref = max(norm, 1.0)
            if norm / ref < tol and it > 1:
                self._reaction = f_int[pres]
                return it
            K = K.tocsr()
            du = np.zeros(self.u.n_dof)
            du[free] = spla.spsolve(K[free][:, free].tocsc(), R[free])
            self.u.values = self.u.values + du
        self._reaction = f_int[pres]
        return max_iter

    # ------------------------------------------------------------------
    # Phase-field equation
    # ------------------------------------------------------------------
    def solve_second(self) -> int:
        """Solve for d at fixed u, after updating the history field.

        ``H`` is taken against the *committed* value, not the running one, so a
        staggered iterate that is later discarded cannot leave a permanent mark
        on the history.
        """
        eps = self.strains()
        self.state["H"] = self.model.update_history(
            self.state.committed("H"), eps)
        H = self.state["H"]

        c_g = self.model.gradient_coefficient()
        K_grad = self._laplacian(c_g)

        d = self.d.values.copy()
        max_iter = 1 if self.model.linear_phase_field else 20
        for it in range(1, max_iter + 1):
            d_qp = np.einsum("eqa,ea->eq", self.N,
                             d[self.d_edofs]).ravel()
            r = self.model.reaction(d_qp, H)
            dr = self.model.dreaction(d_qp, H)

            K = K_grad + self._mass(dr)
            R = -(K_grad @ d) - self._load(r)

            lo = -d if self.model.needs_bound_constraint else None
            if lo is not None:
                res = solve_bounded(K, R, lo=lo, hi=1.0 - d, x0=np.zeros_like(d))
                delta = res.x
            else:
                delta = solve_phase_field_system(K, R, self.damage_solver)

            d = d + delta
            if np.linalg.norm(delta) <= 1e-10 * max(1.0, np.linalg.norm(d)):
                break

        # irreversibility: damage never decreases, so cracks cannot heal.
        # Against the committed step, for the reason given on ``_d_committed``.
        d = np.clip(np.maximum(d, self._d_committed), 0.0, 1.0)
        # second convergence test: a damage field that has stopped moving means
        # the cycle is done, whatever shape the energy decay happens to have
        self._damage_change = float(np.abs(d - self.d.values).max())
        self.d.values = d
        return it


    # ------------------------------------------------------------------
    def _laplacian(self, coeff: float):
        return self.tables.laplacian(coeff, self.d.n_dof)

    def _mass(self, c_qp: np.ndarray):
        return self.tables.mass(c_qp, self.d.n_dof)

    def _load(self, c_qp: np.ndarray) -> np.ndarray:
        return self.tables.load(c_qp, self.d.n_dof)

    # ------------------------------------------------------------------
    def field_change(self) -> float:
        """``max |d_k - d_k-1|`` over the last staggered iteration."""
        return getattr(self, "_damage_change", float("inf"))

    # ------------------------------------------------------------------
    # Energy, stepping, reporting
    # ------------------------------------------------------------------
    def total_energy(self) -> float:
        eps = self.strains()
        d_qp = self.damage_at_points()
        grad_d = self.damage_gradient()
        w = self.w.ravel()
        elastic = float((self.model.elastic_energy_density(eps, d_qp) * w).sum())
        fracture = float(
            (self.model.fracture_energy_density(d_qp, grad_d) * w).sum())
        self._energies = {"elastic": elastic, "fracture": fracture}
        return elastic + fracture

    def surface_energy(self) -> float:
        """Regularized crack surface energy. Divided by Gc it is a crack
        length, which is the cheapest global check on a computed crack."""
        d_qp = self.damage_at_points()
        return float((self.model.fracture_energy_density(
            d_qp, self.damage_gradient()) * self.w.ravel()).sum())

    # ------------------------------------------------------------------
    def set_displacement_control(self, nodes, component: int, value=1.0):
        """Prescribe the loaded DOFs and remember them as the unit load."""
        self.u.prescribe(nodes, component, value)
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        vals = np.broadcast_to(np.asarray(value, dtype=float), nodes.shape)
        for n, v in zip(nodes, vals):
            self._u_base[self.u.dof_map.dof(int(n), component)] = float(v)
        self._control_dofs = np.array(sorted(self._u_base), dtype=np.int64)

    def begin_step(self, step: int, factor: float, dt: float) -> None:
        self.u.scale_prescribed(factor, self._u_base)
        self.u.apply_prescribed()

    def end_step(self, result) -> None:
        self.state.commit()
        self._d_committed = self.d.values.copy()

    def reject_step(self, step: int) -> None:
        # ``d`` has to be rolled back with the history. Restoring only the
        # quadrature state leaves the damage grown by the failed attempt in
        # place, and since damage is monotone it can never be undone -- the
        # retry then starts from a more broken body at a smaller increment.
        self.state.restore()
        self.d.values = self._d_committed.copy()

    def report(self) -> dict:
        reac = getattr(self, "_reaction", np.zeros(0))
        pres = self.u.prescribed_dofs
        total = 0.0
        if len(reac) and hasattr(self, "_control_dofs"):
            sel = np.isin(pres, self._control_dofs)
            total = float(reac[sel].sum())
        return {
            "reactions": {"control": total},
            "energies": dict(getattr(self, "_energies", {}),
                             surface=self.surface_energy()),
        }
