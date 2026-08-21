r"""
Implicit (quasi-static) MPM: Newton-Raphson on the grid, load-stepped.

Follows the framework of O'Hare, Gourgiotis, Coombs & Augarde, *An implicit
Material Point Method for micropolar solids undergoing large deformations*,
CMAME 419 (2024) 116668 (CC BY), reduced to the classical continuum -- their
formulation with the microrotation ``Q = I``. The micropolar degrees of freedom
are the intended next layer and this module is deliberately shaped so they can
be added as a second field rather than by rewriting the solve.

Why implicit at all
-------------------
``ExplicitMPMStepper`` integrates momentum and never satisfies equilibrium; it
does not need to, because inertia carries the residual. Everything downstream of
*equilibrium* therefore has no explicit counterpart. In particular a
configurational (material) force is only path-independent when ``Div P = 0``, so
every material force computed on an explicit run is contaminated by the residual
-- and the recorded failure of ``Pi_h`` descent in
``projects/configurational_mpm`` is the same statement: the Rayleigh-Ritz bound
needs ``u`` to be the *solution* on the current configuration, which requires
re-solving equilibrium. This stepper is what supplies that.

The parameterisation, and why there is no geometric stiffness term
------------------------------------------------------------------
The reference for a load step is the configuration at the *start* of that step,
so within Newton the problem is total-Lagrangian and only the commit is
"updated". With ``F_0`` the deformation gradient and ``g_I = grad_x N_I`` both
frozen at step start, the unknown is the nodal increment ``du_H`` and

    dF = I + sum_H du_H (x) g_H ,      F = dF F_0 ,      b_I = F_0^T g_I

(``b_I`` is just ``Grad_X N_I``, constant over the step). The internal force and
its exact derivative are then

    R_Ii  = sum_p V_p^0 P_iB(F) b_IB
    K_IiJj = sum_p V_p^0 b_IB A_iBjT b_JT ,     A_iJkL = dP_iJ/dF_kL

and that is the *whole* tangent. It is worth being explicit about this, because
every textbook updated-Lagrangian derivation carries a separate initial-stress
("geometric") term and here there is none. It has not been dropped: pushing the
derivative through the current-configuration form produces ``+ grad N_I . P``
from differentiating the stress and ``- grad N_I . P`` from differentiating the
volume and the spatial gradient, and the two cancel identically (the paper's
Eqs. A.26-A.30 is the same cancellation written in Biot-like stress). Adding a
geometric term on top of ``A`` is therefore a real error, and one that leaves
Newton converging -- linearly.

The consequence for materials is that the implicit path needs ``dP/dF``, the
*first* elasticity tensor, and not ``dS/dE``. See
``materials/hyperelastic_batched.py``.

Scope
-----
Hyperelastic only. A path-dependent material would need the *algorithmic*
tangent of its return map, not the continuum one, and substituting the second
for the first is the classic silent error -- so ``state=`` is refused here
rather than approximated. Quasi-static only: there is no mass matrix, which is
the point (a shear band is a quasi-static localisation and explicit MPM has to
mass-scale to reach it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from scion.core import small_linalg as sla
from scion.core.dof import DOFMap
from scion.solvers.linear import DirectSolver

from .background_grid import CartesianGrid
from .particles import ParticleSet

__all__ = ["GridDirichlet", "LoadStepResult", "ImplicitMPMStepper",
           "InvertedParticle"]


class InvertedParticle(RuntimeError):
    """A trial state drove ``det F`` non-positive. Cut the step."""


@dataclass
class GridDirichlet:
    """Prescribed displacement on the grid nodes selected by ``predicate``.

    Parameters
    ----------
    predicate : callable ``(n_nodes, dim) coordinates -> (n_nodes,) bool``
    value : either a ``(dim,)`` constant or a callable
        ``(m, dim) coordinates -> (m, dim)`` giving a prescribed *field*, which
        is what a patch test or a manufactured solution needs. Entries that are
        ``nan`` leave that component free, which is how a roller is written --
        ``[0.0, np.nan]`` fixes x and lets y move. A callable may return ``nan``
        per entry for the same effect, but every selected node must then agree
        on which components are free, since the free set is a property of the
        constraint and not of the node.

    The value is a *total* displacement at load factor 1, not a rate, so a run
    of ``n`` load steps applies ``value / n`` per step. Handing it a per-step
    increment is the obvious misreading and gives an answer that silently
    depends on the step count.

    Trap: ``predicate`` is re-evaluated against the grid node coordinates at
    every load step, which is what it must do -- the body moves through a fixed
    grid, so the set of nodes on its boundary changes. Capturing a node *index*
    list once at step zero instead looks equivalent and is not: the body reaches
    new nodes, those nodes are then free and carry almost no stiffness, and the
    solve produces a large spurious displacement there rather than an error.
    """

    predicate: object
    value: object

    def dofs(self, grid: CartesianGrid,
             n_components: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(dof_indices, prescribed_totals)`` for this constraint.

        ``n_components`` defaults to ``grid.dim`` -- displacement only. A
        micropolar solve passes 6, and the value is then read as three
        translations followed by three spin components.
        """
        nc = grid.dim if n_components is None else int(n_components)
        coords = grid.node_coordinates()
        sel = np.flatnonzero(np.asarray(self.predicate(coords), dtype=bool))
        if callable(self.value):
            val = np.atleast_2d(np.asarray(self.value(coords[sel]),
                                           dtype=float))
            if val.shape != (sel.size, nc):
                raise ValueError(
                    f"GridDirichlet value callable returned {val.shape}, "
                    f"expected {(sel.size, nc)}")
        else:
            v = np.asarray(self.value, dtype=float).ravel()
            if v.size != nc:
                raise ValueError(
                    f"GridDirichlet value has {v.size} components, but the "
                    f"solve has {nc} per node")
            val = np.tile(v, (sel.size, 1))

        free = np.isnan(val)
        if sel.size and free.any(axis=0).any() and not (
                free == free[0]).all():
            raise ValueError(
                "GridDirichlet: the nan pattern differs between selected "
                "nodes, so the constrained components are not well defined")
        comps = np.flatnonzero(~free[0]) if sel.size else np.arange(0)
        idx = (sel[:, None] * nc + comps[None, :]).ravel()
        return idx, val[:, comps].ravel()


@dataclass
class LoadStepResult:
    """What one load step did. ``residuals`` is the instrument, not decoration.

    A correct consistent tangent gives asymptotically quadratic convergence, so
    the residual *sequence* is the only thing that distinguishes a right tangent
    from a wrong one that still converges. Keep it and look at it.
    """

    load_factor: float
    converged: bool
    iterations: int
    residuals: list = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        """Observed Newton order: ~2 for a consistent tangent, ~1 for one that
        is merely close. ``nan`` if the sequence is too short to say.

        Defined as the **largest** ratio ``log(c/b)/log(b/a)`` over consecutive
        triples, with the first residual dropped. Both parts of that are needed,
        and each was put there by a sequence that fooled the obvious version:

        * the first residual is 1.0 by construction (it *is* the normalising
          reference), so the first ratio is an artefact -- it read 2.68 on a
          sequence whose true rate was 2.08;
        * the last residual is limited by machine precision rather than by the
          tangent, so a fixed final window drags a clean quadratic sequence down
          to 0.85. Taking the maximum discards that tail automatically, and it
          is the right summary anyway because the order is a *limit*: early
          triples are pre-asymptotic and understate it. A linear tangent gives
          ~1 for every triple, so nothing is hidden by the max.
        """
        r = self.residuals[1:]
        if len(r) < 3:
            return float("nan")
        rates = [np.log(c / b) / np.log(b / a)
                 for a, b, c in zip(r, r[1:], r[2:])
                 if a > b > c > 0.0]
        return float(max(rates)) if rates else float("nan")


class ImplicitMPMStepper:
    """Quasi-static implicit MPM.

    Parameters
    ----------
    grid : CartesianGrid
    material : batched hyperelastic model exposing
        ``pk1_and_tangent(F) -> (P, J, A)``
    dirichlet : sequence of GridDirichlet
    body_force : (dim,) acceleration, ramped with the load factor
    particle_force : (n, dim) point forces applied at the particles, ramped with
        the load factor. This is how the reference applies a tip load, and it is
        a pseudo-Neumann condition rather than a true traction -- MPM has no
        surface to integrate over.
    tol : relative force-residual tolerance
    energy_tol : relative energy-residual tolerance; both must be met
    max_iter : Newton iterations per load step
    ghost : ghost-stabilisation penalty, or None. A fraction of an elastic
        modulus (the reference uses ``lam/100``); see ``ghost.py``. Costs
        nothing in accuracy -- the penalty annihilates affine fields -- and is
        what makes a badly cut boundary cell solvable.
    support_tol : a node enters the free set when its particle support
        ``sum_p V_p^0 N_Ip`` exceeds this fraction of the largest. This matters
        far more implicitly than explicitly. A node the body has only just
        reached carries a stiffness proportional to that support, so at 1e-12 it
        contributes a row that is numerically zero and the reduced matrix is
        singular; the explicit stepper's equivalent threshold guards a division
        by nodal mass, which degrades gracefully, and can afford to be tiny.
        1e-6 is loose enough to keep every node that carries load and tight
        enough to drop the ones that do not.

    Attributes
    ----------
    history : list[LoadStepResult]
    """

    def __init__(self, grid: CartesianGrid, material, dirichlet=(),
                 body_force=None, particle_force=None, tol: float = 1e-9,
                 energy_tol: float = 1e-16, max_iter: int = 20,
                 linear_solver=None, support_tol: float = 1e-6,
                 state=None, ghost: float | None = None):
        if state is not None:
            raise NotImplementedError(
                "implicit MPM here is hyperelastic only. A path-dependent "
                "material needs the algorithmic tangent of its return map; the "
                "continuum tangent converges linearly and looks like a hard "
                "problem rather than a bug, which is exactly the failure this "
                "library tries not to ship.")
        if not hasattr(material, "pk1_and_tangent"):
            raise TypeError(
                f"{type(material).__name__} has no pk1_and_tangent(F); the "
                f"implicit tangent needs dP/dF")

        self.grid = grid
        self.material = material
        self.dirichlet = list(dirichlet)
        self.body_force = (None if body_force is None
                           else np.asarray(body_force, dtype=float))
        self.particle_force = (None if particle_force is None
                               else np.asarray(particle_force, dtype=float))
        self.tol = float(tol)
        self.energy_tol = float(energy_tol)
        self.max_iter = int(max_iter)
        self.linear_solver = linear_solver or DirectSolver()
        self.support_tol = float(support_tol)
        self.ghost = None if ghost is None else float(ghost)
        self.history: list[LoadStepResult] = []

        self._n_dof = grid.n_nodes * grid.dim
        self._st = None      # stencil, frozen for the current load step

    # ------------------------------------------------------------------
    # per-load-step setup
    # ------------------------------------------------------------------
    def prepare(self, particles: ParticleSet) -> None:
        """Freeze the stencil, reference gradients and free set for one step.

        Called by :meth:`step`; exposed because ``residual``/``tangent`` are
        only defined against a prepared state, which is what makes them the pure
        functions ``check_tangent`` requires.
        """
        grid = self.grid
        st = grid.build_stencil(particles.x,
                                lp=getattr(particles, "lp", None))
        self._st = st
        self._F0 = particles.F.copy()
        self._vol0 = particles.vol0.copy()
        #: b_I = F_0^T grad_x N_I = Grad_X N_I, (n, k, dim)
        self._b = np.einsum("prb,pkr->pkb", self._F0, st.grad_w)

        # free set: nodes with real particle support. Anything below the
        # threshold has no stiffness and would make K_ff singular.
        support = np.zeros(grid.n_nodes)
        np.add.at(support, st.node_id, self._vol0[:, None] * st.w)
        if self.ghost is None:
            active_node = support > self.support_tol * support.max()
        else:
            # stabilised: keep every node of every populated cell. The penalty
            # is what makes the badly cut ones solvable, so thresholding them
            # away here would remove the thing being stabilised.
            from .ghost import stabilised_nodes
            active_node = stabilised_nodes(
                grid, particles.x,
                lp=getattr(particles, "lp", None)
                if grid.basis.requires_domain else None)
            # stabilisation replaces the support *threshold*, not the
            # requirement of support at all. A node with literally zero weight
            # has no material equation, and it gets no ghost one either if its
            # cell happens to be isolated -- which GIMP produces at a thin tip,
            # where a domain overlaps a cell no particle centre is in.
            active_node &= support > 0.0
        active = np.repeat(active_node, grid.dim)

        dofmap = DOFMap(grid.n_nodes, grid.dim)
        self._prescribed = np.zeros(self._n_dof)
        for bc in self.dirichlet:
            idx, val = bc.dofs(grid)
            if idx.size == 0:
                # Almost always an inequality written the wrong side of a node
                # line. The consequence is not an error: the body is left
                # unrestrained, and with no external load the residual is
                # already zero, so Newton "converges" in one iteration and
                # returns F = I. A wrong answer that looks like a fast solve.
                raise ValueError(
                    "a GridDirichlet predicate selected no grid nodes; the "
                    "body would be left unrestrained and the solve would "
                    "return the undeformed state without complaining")
            self._prescribed[idx] = val
            for i in idx:
                dofmap.constrain(int(i) // grid.dim, int(i) % grid.dim)
        constrained = np.zeros(self._n_dof, dtype=bool)
        constrained[dofmap.constrained_dofs] = True

        self._free = np.flatnonzero(active & ~constrained)
        self._constrained = np.flatnonzero(active & constrained)
        self._dofmap = dofmap
        self._ghost = None
        if self.ghost:
            from .ghost import ghost_penalty
            self._ghost = self.ghost * ghost_penalty(
                grid, particles.x,
                lp=getattr(particles, 'lp', None)
                if grid.basis.requires_domain else None)
        if self.dirichlet and self._constrained.size == 0:
            raise ValueError(
                "every constrained node lies outside the particles' support, "
                "so nothing restrains the body. The predicate selects real "
                "grid nodes but not ones the body reaches.")

    # ------------------------------------------------------------------
    def _deformation(self, du: np.ndarray, check: bool = True) -> np.ndarray:
        """``F(du)`` for the frozen step. ``du`` is flat, node-major.

        Raises on ``det F <= 0``. A Newton iterate that inverts a particle is
        ordinary -- it is a trial state, not a physical one -- but the neo-
        Hookean ``log J`` turns it into a nan that propagates through the whole
        remaining iteration and reports as "did not converge", which reads as a
        hard problem rather than a step that was simply too large. The caller
        that can act on it is ``run``, which cuts the increment.
        """
        st, grid = self._st, self.grid
        d = grid.dim
        du_node = du.reshape(grid.n_nodes, d)
        dF = np.einsum("pka,pkb->pab", du_node[st.node_id], st.grad_w)
        dF += np.eye(d)
        F = np.einsum("pab,pbc->pac", dF, self._F0)
        if check:
            J = sla.det(F)
            n_bad = int((J <= 0.0).sum())
            if n_bad:
                raise InvertedParticle(
                    f"{n_bad} of {F.shape[0]} particles inverted in the trial "
                    f"state (min det F = {J.min():.3e}); the load increment is "
                    f"too large")
        return F

    def external_force(self, particles: ParticleSet,
                       load_factor: float) -> np.ndarray:
        """Nodal external force at ``load_factor``, flat."""
        st, grid = self._st, self.grid
        f = np.zeros((grid.n_nodes, grid.dim))
        if self.body_force is not None:
            np.add.at(f, st.node_id,
                      (particles.mass[:, None] * st.w)[:, :, None]
                      * self.body_force)
        if self.particle_force is not None:
            np.add.at(f, st.node_id,
                      st.w[:, :, None] * self.particle_force[:, None, :])
        return load_factor * f.ravel()

    def internal_force(self, du: np.ndarray) -> np.ndarray:
        """``sum_p V_p^0 P b_I``, flat. The residual's only nonlinear part."""
        st, grid = self._st, self.grid
        P, _ = self.material.pk1(self._deformation(du))
        f = np.zeros((grid.n_nodes, grid.dim))
        np.add.at(f, st.node_id,
                  np.einsum("p,piB,pkB->pki", self._vol0, P, self._b))
        out = f.ravel()
        # the penalty must appear in the residual as well as the tangent, or
        # the two describe different problems and Newton converges linearly
        if self._ghost is not None:
            out = out + self._ghost @ du
        return out

    def tangent(self, du: np.ndarray) -> sparse.csr_matrix:
        """Consistent tangent ``dR/d(du)``, sparse ``(n_dof, n_dof)``."""
        st, grid = self._st, self.grid
        d = grid.dim
        _, _, A = self.material.pk1_and_tangent(self._deformation(du))

        # (n, k, d, k, d): b_IB A_iBjT b_JT, weighted by the particle volume
        ke = np.einsum("p,pKB,piBjT,pLT->pKiLj", self._vol0, self._b, A,
                       self._b)

        dofs = (st.node_id[:, :, None] * d
                + np.arange(d)[None, None, :]).reshape(st.node_id.shape[0], -1)
        n_loc = dofs.shape[1]
        rows = np.repeat(dofs, n_loc, axis=1).ravel()
        cols = np.tile(dofs, (1, n_loc)).ravel()
        K = sparse.coo_matrix(
            (ke.reshape(-1), (rows, cols)),
            shape=(self._n_dof, self._n_dof)).tocsr()
        return K if self._ghost is None else K + self._ghost

    # ------------------------------------------------------------------
    def step(self, particles: ParticleSet, load_factor: float,
             d_load: float) -> LoadStepResult:
        """One load step to ``load_factor``, prescribing ``d_load`` of the
        Dirichlet totals. Commits on convergence and leaves the particles
        untouched otherwise."""
        self.prepare(particles)
        free, con = self._free, self._constrained

        du = np.zeros(self._n_dof)
        du[con] = d_load * self._prescribed[con]

        f_ext = self.external_force(particles, load_factor)
        result = LoadStepResult(load_factor, False, 0)
        # Normalise by the external load, falling back to the internal one: a
        # displacement-controlled run has f_ext = 0 and dividing by its norm is
        # a zero-division dressed up as a convergence test.
        ref = float(np.linalg.norm(f_ext[free]))
        if ref <= 0.0:
            ref = max(float(np.linalg.norm(self.internal_force(du)[free])),
                      1.0)
        energy0 = None
        energy_ok = False

        for it in range(self.max_iter):
            R = self.internal_force(du) - f_ext
            norm = float(np.linalg.norm(R[free])) / ref
            result.residuals.append(norm)

            # Both criteria, and the energy one lags by an iteration because it
            # is built from the increment the previous solve produced. Testing
            # force alone stops too early on a stiff step; testing energy alone
            # stops on a small increment that has not reduced the residual.
            if it > 0 and norm < self.tol and energy_ok:
                result.converged = True
                break

            K = self.tangent(du)
            Kff = K[free][:, free]
            # An unsupported node produces a numerically zero row rather than a
            # badly scaled one, and spsolve answers that with a warning and a
            # vector of nan. Name it instead: the cause is always the same, a
            # node the body has just reached that no constraint covers.
            diag = Kff.diagonal()
            if diag.max() <= 0.0 or diag.min() <= 1e-12 * diag.max():
                bad = free[diag <= 1e-12 * max(diag.max(), 0.0)]
                raise RuntimeError(
                    f"the reduced tangent is singular at load factor "
                    f"{load_factor:.4g}: {bad.size} of {free.size} free DOFs "
                    f"have no stiffness (grid nodes "
                    f"{sorted(set(bad // self.grid.dim))[:8]}...). The body has "
                    f"reached grid nodes that carry no support and no Dirichlet "
                    f"condition; raise support_tol or extend the constrained "
                    f"region.")
            ddu = self.linear_solver.solve(Kff, -R[free])

            energy = abs(float(R[free] @ ddu))
            if energy0 is None:
                energy0 = max(energy, 1e-300)
            energy_ok = energy / energy0 < self.energy_tol

            du[free] += ddu
            result.iterations = it + 1

        if not result.converged:
            raise RuntimeError(
                f"implicit MPM load step at lambda = {load_factor:.4g} did not "
                f"converge in {self.max_iter} iterations; residuals "
                f"{['%.2e' % r for r in result.residuals]}")

        self._commit(particles, du)
        self.history.append(result)
        return result

    def _commit(self, particles: ParticleSet, du: np.ndarray) -> None:
        """Accept the step: advance F, then positions."""
        st, grid = self._st, self.grid
        particles.F = self._deformation(du)
        du_node = du.reshape(grid.n_nodes, grid.dim)
        particles.x = particles.x + np.einsum(
            "pk,pka->pa", st.w, du_node[st.node_id])
        if grid.any_periodic:
            grid.wrap_positions(particles.x)
        if grid.basis.requires_domain:
            particles.update_domain("stretch")

    # ------------------------------------------------------------------
    def run(self, particles: ParticleSet, n_steps: int = 10, callback=None,
            min_step: float = 1.0e-4) -> ParticleSet:
        """Ramp the load from 0 to 1, cutting the increment where it fails.

        Starts from ``n_steps`` equal increments and halves on a failed step,
        which covers both causes of failure -- an inverted trial state and a
        Newton that simply will not converge. It does **not** recover a
        snap-back: the load factor only ever increases, so a path that needs the
        load to fall is out of reach here for the same reason it is in
        ``TwoFieldProblem.run``, and would need arc-length control.
        """
        d_load = 1.0 / n_steps
        lam = 0.0
        s = 0
        while lam < 1.0 - 1.0e-12:
            inc = min(d_load, 1.0 - lam)
            try:
                res = self.step(particles, lam + inc, inc)
            except (InvertedParticle, RuntimeError) as exc:
                d_load *= 0.5
                if d_load < min_step:
                    raise RuntimeError(
                        f"load stepping stalled at lambda = {lam:.4g} with an "
                        f"increment below {min_step:g}: {exc}") from exc
                continue
            lam += inc
            s += 1
            if callback is not None:
                callback(s, res, particles)
        return particles
