r"""
Implicit gradient damage: the two-field driver.

Assembles and solves

    momentum:   INT sigma(eps, D(kappa)) : eps(v) dx = external work
    nonlocal:   INT [ ebar w + ell^2 grad(ebar) . grad(w) ] dx = INT eps_eq w dx

with the damage supplied by
:class:`~scion.materials.damage.model.GradientDamageModel` and the
alternation driven by :class:`~scion.solvers.StaggeredSolver`.

The second equation is the implicit gradient of Peerlings et al. (1996): a
Helmholtz filter that smears the local equivalent strain over a neighbourhood of
size ``ell``. Damage is then driven by the *filtered* field,
``kappa = max over history of ebar``, and the localisation band acquires a width
the material owns rather than one the mesh imposes.

Why this is cheaper than it looks
---------------------------------
Both equations are linear.

At fixed ``ebar`` the damage at every point is a fixed number, so the stress is
``(1 - D) C_0 : eps`` and momentum is one linear symmetric solve -- the rank-one
tangent term that makes local damage unsymmetric simply does not arise, because
at fixed ``ebar`` the damage does not respond to the local strain. Newton is
needed only with an energy split, which is nonlinear in the strain for its own
reasons; :attr:`GradientDamageModel.linear_momentum` reports which case applies.

At fixed ``u`` the Helmholtz equation is linear in ``ebar`` with a constant
operator, so ``M + ell^2 K`` is assembled and factorised **once** for the whole
analysis and every subsequent solve is a back-substitution. Only the right-hand
side changes.

So the regularised problem costs less per iteration than the local one it fixes,
which is not the usual trade.

No boundary condition on ebar
-----------------------------
The natural condition ``grad(ebar) . n = 0`` comes for free from the weak form,
and it is the standard choice. It is not innocent: it makes the filter reflect
at the boundary rather than average across it, so a point on a free surface sees
its own neighbourhood counted twice and damages slightly early. That is a known
artefact of implicit gradient damage, visible as damage initiating at a surface
before an interior point at equal strain, and it is the reason the operator is
never given a Dirichlet condition instead -- ``ebar = 0`` on a boundary would be
far worse, forcing the filtered strain to vanish where the load is applied.

Convergence of the staggered cycle
----------------------------------
The energy criterion in :class:`~scion.solvers.StaggeredSolver` measures the
shape of an energy decay, which presumes the scheme is minimising something.
This model is not variational -- it is a phenomenological evolution law, not the
stationarity condition of a functional -- so the energy is monitored for
reporting and the cycle is really decided by the field-change test on the damage
field. ``max |D_k - D_k-1|`` is dimensionless and bounded in ``[0, 1]``, so its
tolerance transfers between problems in a way that neither an energy nor a
strain-valued tolerance does.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import linalg as spla

from scion.discretization.fem.quadrature_state import QuadratureState
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.damage.model import DamageState, GradientDamageModel

from .two_field import Field, TwoFieldProblem

__all__ = ["GradientDamageProblem"]


class GradientDamageProblem(TwoFieldProblem):
    """Quasi-static implicit gradient damage under displacement control.

    Parameters
    ----------
    domain    : mesh, coordinates, element blocks.
    model     : a :class:`GradientDamageModel`.
    thickness : out-of-plane thickness in 2-D.
    strategy, stagger_tol, max_stagger, verbose : see
        :class:`~scion.multiphysics.two_field.TwoFieldProblem`.
    damage_tol : tolerance on ``max |dD|`` between staggered iterations.

    The displacement field is ``self.u`` and the nonlocal equivalent strain
    ``self.e``. Boundary conditions go on ``u`` through
    :meth:`set_displacement_control`; ``e`` needs none.
    """

    field_names = ("displacement", "nonlocal equivalent strain")

    def __init__(self, domain, model: GradientDamageModel,
                 thickness: float = 1.0, strategy: str = "staggered",
                 stagger_tol: float = 10.0, max_stagger: int = 100,
                 damage_tol: float = 1.0e-4, verbose: bool = False):
        super().__init__(domain, strategy, stagger_tol, max_stagger,
                         damage_tol, verbose)
        self.model = model
        self.thickness = float(thickness)
        self.dim = domain.coordinates.shape[1]

        self.tables = QuadratureTables(domain, thickness)
        n_nodes = domain.n_nodes
        self.u = Field("u", n_nodes, self.dim)
        self.e = Field("ebar", n_nodes, 1)
        self._u_base: dict[int, float] = {}

        self.state = QuadratureState(self.tables.n_elements, self.tables.n_qp)
        self.state.add("kappa", init=model.evolution.kappa_0)

        self._damage = np.zeros(self.tables.n_points)
        self._damage_change = float("inf")
        self._reaction = np.zeros(0)
        self._filter = None
        self.external_work = 0.0
        self._last = (0.0, 0.0)          # (control displacement, force)

    # ------------------------------------------------------------------
    def check_mesh(self) -> None:
        """Refuse a mesh too coarse to resolve the internal length."""
        self.model.check_mesh_size(self.tables.element_size)

    @property
    def element_size(self) -> float:
        return self.tables.element_size

    # ------------------------------------------------------------------
    # Field one: momentum at fixed ebar
    # ------------------------------------------------------------------
    def _committed_state(self) -> DamageState:
        """The damage history as of the last accepted step.

        Taken against the *committed* array, never the running one: a staggered
        iterate that is later discarded must not be able to advance ``kappa``,
        for exactly the reason a phase-field history field must not.
        """
        return DamageState(self.state.committed("kappa").copy())

    def _constitutive(self):
        """Stress, tangent and the updated history at the current ``(u, ebar)``."""
        eps = self.tables.strains_voigt(self.u.values)
        ebar = self.tables.interpolate(self.e.values)
        sigma, C, res = self.model.update(eps, self._committed_state(),
                                          dim=self.dim, driver=ebar)
        return sigma, C, res

    def solve_first(self) -> int:
        return (self._solve_u_linear() if self.model.linear_momentum
                else self._solve_u_newton())

    def _solve_u_linear(self) -> int:
        """One linear solve. No iteration and no dependence on the trial ``u``.

        With no split and the damage driven by ``ebar``, neither ``D`` nor the
        tangent responds to the local strain at all, so the constitutive call
        below is evaluated once at whatever ``u`` happens to be current and its
        tangent is exact for every other ``u``. Only the stress would change,
        and the stress is not needed: the equation is ``K u = 0`` with the
        prescribed displacements imposed.
        """
        _, C, res = self._constitutive()
        self._record(res)

        K, _ = self.tables.stiffness_and_force(
            np.zeros((self.tables.n_points, self.tables.n_voigt)), C,
            self.u.n_dof)
        self.u.values, self._reaction = _solve_constrained(
            K, np.zeros(self.u.n_dof), self.u.free_dofs,
            self.u.prescribed_dofs, self.u.prescribed_values)
        return 1

    def _solve_u_newton(self, tol: float = 1.0e-9, max_iter: int = 20) -> int:
        self.u.apply_prescribed()
        free, pres = self.u.free_dofs, self.u.prescribed_dofs
        ref = 1.0
        for it in range(1, max_iter + 1):
            sigma, C, res = self._constitutive()
            K, f_int = self.tables.stiffness_and_force(sigma, C, self.u.n_dof)
            R = -f_int
            R[pres] = 0.0
            norm = float(np.linalg.norm(R[free]))
            if it == 1:
                ref = max(norm, 1.0)
            if it > 1 and norm / ref < tol:
                self._record(res)
                self._reaction = f_int[pres]
                return it
            du = np.zeros(self.u.n_dof)
            du[free] = spla.spsolve(K.tocsr()[free][:, free].tocsc(), R[free])
            self.u.values = self.u.values + du
        self._record(res)
        self._reaction = f_int[pres]
        return max_iter

    def _record(self, res) -> None:
        self.state["kappa"] = res.state.kappa
        self._damage_change = float(np.abs(res.damage - self._damage).max())
        self._damage = res.damage

    # ------------------------------------------------------------------
    # Field two: the Helmholtz filter at fixed u
    # ------------------------------------------------------------------
    def solve_second(self) -> int:
        """One linear solve of ``(M + ell^2 K) ebar = M eps_eq``.

        The operator does not depend on either field, so it is built and
        factorised once and reused for the whole analysis.
        """
        if self._filter is None:
            n = self.e.n_dof
            A = (self.tables.mass(1.0, n)
                 + self.model.gradient_coefficient()
                 * self.tables.laplacian(1.0, n))
            self._filter = spla.factorized(A.tocsc())

        eps = self.tables.strains_voigt(self.u.values)
        eps_eq = self.model.equivalent_strain(eps, dim=self.dim)
        rhs = self.tables.load(eps_eq, self.e.n_dof)
        self.e.values = self._filter(rhs)
        return 1

    # ------------------------------------------------------------------
    def field_change(self) -> float:
        """``max |D_k - D_k-1|``, the test the cycle actually stops on."""
        return self._damage_change

    def total_energy(self) -> float:
        """Stored elastic energy.

        Reported and monitored, but not a functional this scheme minimises --
        the model is not variational. See the module docstring.
        """
        eps = self.tables.strains_voigt(self.u.values)
        psi = self.model.elastic_energy(eps, self._damage, dim=self.dim)
        energy = self.tables.integrate(psi)
        self._energies = {"elastic": energy}
        return energy

    # ------------------------------------------------------------------
    # Quantities of interest
    # ------------------------------------------------------------------
    @property
    def damage(self) -> np.ndarray:
        """Damage at the quadrature points, ``(n_points,)``."""
        return self._damage

    def damage_by_element(self) -> np.ndarray:
        """Element-averaged damage, the natural VTK cell field."""
        return self._damage.reshape(self.tables.n_elements,
                                    self.tables.n_qp).mean(axis=1)

    def control_force(self) -> float:
        pres = self.u.prescribed_dofs
        if not len(self._reaction) or not hasattr(self, "_control_dofs"):
            return 0.0
        return float(self._reaction[np.isin(pres, self._control_dofs)].sum())

    def control_displacement(self) -> float:
        if not hasattr(self, "_control_dofs") or not len(self._control_dofs):
            return 0.0
        return float(np.abs(self.u.values[self._control_dofs]).max())

    def dissipated_energy(self) -> float:
        """External work minus stored energy.

        Computed from the global balance rather than by integrating ``psi dD``
        over the history, because the balance is exact for whatever the solver
        actually did while the incremental integral inherits the step size. This
        is the number that should converge under mesh refinement and does not,
        for a local model -- which is the whole point of the length scale.
        """
        return self.external_work - self.total_energy()

    # ------------------------------------------------------------------
    # Stepping
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
        # trapezoidal external work, accumulated over accepted steps only
        d, f = self.control_displacement(), self.control_force()
        d0, f0 = self._last
        self.external_work += 0.5 * (f + f0) * (d - d0)
        self._last = (d, f)

    def reject_step(self, step: int) -> None:
        self.state.restore()
        self._damage = self.model.evolution.D(self.state.committed("kappa"))

    def report(self) -> dict:
        return {
            "reactions": {"control": self.control_force()},
            "energies": dict(getattr(self, "_energies", {}),
                             work=self.external_work,
                             dissipated=self.dissipated_energy()),
        }

    def __repr__(self) -> str:
        return (f"GradientDamageProblem({self.tables!r}, "
                f"ell={self.model.ell:g}, "
                f"h/ell={self.element_size / self.model.ell:.2f})")


# ---------------------------------------------------------------------------

def _solve_constrained(K, f, free, prescribed, values):
    """Solve ``K x = f`` with ``x[prescribed] = values``; return x and reaction."""
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
