r"""
Phase-field brittle fracture: the concrete two-field driver.

Assembles and solves

    momentum:     INT sigma(eps, d) : eps(v) dx = external work
    phase field:  INT [ c_g grad d . grad w + r(d, H) w ] dx = 0

with ``sigma``, ``r`` and ``c_g`` supplied by
:class:`~simulacrum.materials.fracture.PhaseFieldModel`, and the alternation
driven by :class:`~simulacrum.solvers.StaggeredSolver`.

Why this does its own element loop
----------------------------------
``FEMAssembler`` builds a fixed stiffness from a material tangent that depends
on nothing. Here both tangents depend on the *other* field evaluated at the
quadrature points -- ``C`` on ``d``, and the phase-field mass term on ``H``,
which in turn depends on ``eps``. So the loop has to interleave, and it reuses
what it can: ``get_element``, ``get_quadrature``,
``FEMAssembler._strain_displacement`` and ``DOFMap`` are all shared, and the
Jacobian convention is deliberately identical (``dN_dxi @ inv(J).T``).

Quadrature ordering
-------------------
Points are numbered element-major over the concatenation of all element blocks,
then quadrature point within the element. :class:`QuadratureState` uses the same
order, so ``H`` lines up with the strains without any bookkeeping. Blocks with
different element types are allowed as long as they share a quadrature count;
mixed counts would break the flat indexing and are rejected up front.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from simulacrum.core.dof import DOFMap
from simulacrum.discretization.fem.assembler import FEMAssembler
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature
from simulacrum.discretization.fem.quadrature_state import QuadratureState
from simulacrum.materials.fracture import PhaseFieldModel
from simulacrum.solvers.bound_constrained import solve_bounded

from .two_field import Field, TwoFieldProblem

__all__ = ["PhaseFieldFractureProblem"]


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

        n_nodes = domain.n_nodes
        self.u = Field("u", n_nodes, self.dim)
        self.d = Field("d", n_nodes, 1)
        self._u_base: dict[int, float] = {}

        self._build_quadrature_tables()
        self.state = QuadratureState(self.n_elements, self.n_qp)
        self.state.add("H")

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    def _build_quadrature_tables(self) -> None:
        """Shape functions, gradients and weights at every quadrature point.

        The mesh does not move, so these are computed once. That is most of the
        cost of an element loop, and phase-field runs take thousands of them.
        """
        coords = self.domain.coordinates
        blocks = self.domain.mesh.element_blocks

        n_qp = {len(get_quadrature(b.etype)[1]) for b in blocks}
        if len(n_qp) != 1:
            raise NotImplementedError(
                "all element blocks must share a quadrature count; got "
                f"{sorted(n_qp)}"
            )
        self.n_qp = n_qp.pop()

        conn, N_all, dN_all, w_all = [], [], [], []
        for block in blocks:
            ref = get_element(block.etype)
            qpts, qwts = get_quadrature(block.etype)
            for e in range(block.connectivity.shape[0]):
                enodes = block.connectivity[e]
                xe = coords[enodes]
                conn.append(enodes)
                for qp, qw in zip(qpts, qwts):
                    N, dN_dxi = ref.evaluate(qp)
                    J = dN_dxi.T @ xe
                    # same convention as FEMAssembler: the transpose matters
                    # for any element whose Jacobian is not symmetric
                    dN_dx = dN_dxi @ np.linalg.inv(J).T
                    N_all.append(N)
                    dN_all.append(dN_dx)
                    w_all.append(abs(np.linalg.det(J)) * qw * self.thickness)

        self.conn = np.array(conn, dtype=np.int64)          # (n_el, npe)
        self.n_elements, self.npe = self.conn.shape
        self.N = np.array(N_all).reshape(self.n_elements, self.n_qp, self.npe)
        self.dN = np.array(dN_all).reshape(self.n_elements, self.n_qp,
                                           self.npe, self.dim)
        self.w = np.array(w_all).reshape(self.n_elements, self.n_qp)

        self.u_edofs = np.array([self.u.dof_map.element_dofs(c)
                                 for c in self.conn])
        self.d_edofs = self.conn                             # scalar field

        self.B = np.array([[FEMAssembler._strain_displacement(
            self.dN[e, q], self.dim) for q in range(self.n_qp)]
            for e in range(self.n_elements)])

        h = np.sqrt(self.w.sum(axis=1) / self.thickness).min()
        self.element_size = float(h)

    # ------------------------------------------------------------------
    def check_mesh(self) -> None:
        """Warn when the mesh cannot resolve the regularization length."""
        self.model.check_mesh_size(self.element_size)

    # ------------------------------------------------------------------
    # Quadrature-point quantities
    # ------------------------------------------------------------------
    def strains(self) -> np.ndarray:
        """``(n_points, dim, dim)`` strain tensors from the current u."""
        ue = self.u.values[self.u_edofs]                      # (n_el, npe*dim)
        ev = np.einsum("eqab,eb->eqa", self.B, ue)            # Voigt
        n = self.n_elements * self.n_qp
        ev = ev.reshape(n, -1)
        eps = np.zeros((n, self.dim, self.dim))
        if self.dim == 2:
            eps[:, 0, 0], eps[:, 1, 1] = ev[:, 0], ev[:, 1]
            eps[:, 0, 1] = eps[:, 1, 0] = 0.5 * ev[:, 2]
        else:
            eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2] = ev[:, 0], ev[:, 1], ev[:, 2]
            eps[:, 1, 2] = eps[:, 2, 1] = 0.5 * ev[:, 3]
            eps[:, 0, 2] = eps[:, 2, 0] = 0.5 * ev[:, 4]
            eps[:, 0, 1] = eps[:, 1, 0] = 0.5 * ev[:, 5]
        return eps

    def damage_at_points(self) -> np.ndarray:
        de = self.d.values[self.d_edofs]                      # (n_el, npe)
        return np.einsum("eqa,ea->eq", self.N, de).ravel()

    def damage_gradient(self) -> np.ndarray:
        de = self.d.values[self.d_edofs]
        return np.einsum("eqai,ea->eqi", self.dN, de).reshape(-1, self.dim)

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

        nv = sig_v.shape[1]
        C_v = C_v.reshape(self.n_elements, self.n_qp, nv, nv)
        sig_v = sig_v.reshape(self.n_elements, self.n_qp, nv)

        ke = np.einsum("eqab,eqbc,eqcd,eq->ead", np.swapaxes(self.B, 2, 3),
                       C_v, self.B, self.w)
        fe = np.einsum("eqab,eqa,eq->eb", self.B, sig_v, self.w)

        n_dof = self.u.n_dof
        rows = np.repeat(self.u_edofs, self.u_edofs.shape[1], axis=1).ravel()
        cols = np.tile(self.u_edofs, (1, self.u_edofs.shape[1])).ravel()
        K = sparse.csr_matrix((ke.ravel(), (rows, cols)),
                              shape=(n_dof, n_dof))
        f_int = np.zeros(n_dof)
        np.add.at(f_int, self.u_edofs.ravel(), fe.ravel())
        return K, f_int

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
                delta = spla.spsolve(K.tocsc(), R)

            d = d + delta
            if np.linalg.norm(delta) <= 1e-10 * max(1.0, np.linalg.norm(d)):
                break

        # irreversibility: damage never decreases, so cracks cannot heal
        d = np.clip(np.maximum(d, self.d.values), 0.0, 1.0)
        # second convergence test: a damage field that has stopped moving means
        # the cycle is done, whatever shape the energy decay happens to have
        self._damage_change = float(np.abs(d - self.d.values).max())
        self.d.values = d
        return it


    # ------------------------------------------------------------------
    def _laplacian(self, coeff: float):
        ke = coeff * np.einsum("eqai,eqbi,eq->eab", self.dN, self.dN, self.w)
        return self._scatter(ke)

    def _mass(self, c_qp: np.ndarray):
        c = c_qp.reshape(self.n_elements, self.n_qp)
        ke = np.einsum("eqa,eqb,eq,eq->eab", self.N, self.N, c, self.w)
        return self._scatter(ke)

    def _load(self, c_qp: np.ndarray) -> np.ndarray:
        c = c_qp.reshape(self.n_elements, self.n_qp)
        fe = np.einsum("eqa,eq,eq->ea", self.N, c, self.w)
        out = np.zeros(self.d.n_dof)
        np.add.at(out, self.conn.ravel(), fe.ravel())
        return out

    def _scatter(self, ke: np.ndarray):
        rows = np.repeat(self.conn, self.npe, axis=1).ravel()
        cols = np.tile(self.conn, (1, self.npe)).ravel()
        return sparse.csr_matrix((ke.ravel(), (rows, cols)),
                                 shape=(self.d.n_dof, self.d.n_dof))

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

    def reject_step(self, step: int) -> None:
        self.state.restore()

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
