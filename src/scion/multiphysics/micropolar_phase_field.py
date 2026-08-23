r"""Phase-field brittle fracture in a micropolar (Cosserat) continuum, on MPM.

Why this is not another adapter
-------------------------------
:class:`~scion.multiphysics.phase_field_mpm.PhaseFieldMPMProblem` works because
the displacement solve goes through ``QuadratureTables``. The micropolar solve
does not: :class:`~scion.discretization.mpm.micropolar.MicropolarMPMStepper` is
a Newton solver with its own residual and tangent over six degrees of freedom
per node. So the coupling here is *staggered against that stepper* -- solve
``(u, phi)`` with the micropolar Newton at frozen ``d``, then solve ``d`` on the
same grid -- rather than a substitution of tables.

What degrades, and why it is not a choice of convenience
--------------------------------------------------------
The Cosserat strain ``E = Q^T F - I`` is **not symmetric**, and every standard
tension/compression split presupposes symmetry. That looks like an obstacle and
is not, because the isotropic micropolar energy decouples *exactly* -- verified
to 2.6e-16 relative, for every coupling number:

    W(E, Gamma) = W_iso(sym E; lam, mu) + mu a |skew E|^2 + W_curv(Gamma)

with no cross terms. So the split applies to ``sym(E)`` alone, using the
existing `materials/splits.py` untouched, and the other two terms degrade
unconditionally:

    W = g(d) [ W_iso+(S) + mu a |A|^2 + W_curv ] + W_iso-(S)

The unconditional part is physical rather than expedient. ``skew(E)`` is the
**relative rotation between the microstructure and the continuum**: its energy
is quadratic and sign-indifferent and cannot represent material
interpenetration, so there is no compressive branch to protect from healing.
The same holds for the wryness -- bending of the microstructure has no
compression analogue. Leaving the couple term undegraded would also model a
surface that carries no traction while still transmitting moment, which is not
a crack.

Including ``W_curv`` in the driving force is what puts rotation *gradients*
into damage evolution, and those are largest exactly where the internal length
lives. That is the whole reason to reach for phase field here: the
``J``-integral route measured the micropolar contribution at 2%, because ``J``
is an energy integral fixed by the far field.

The regime to watch
-------------------
Phase field regularises the crack over ``l_pf``, so the rotation gradient at the
tip is capped by ``l_pf`` rather than by the true singularity. If
``l_pf >> l_b`` the curvature share of the driving force will be negligible and
micropolarity will contribute nothing. The regime with content is
``l_pf <~ l_b``; :meth:`driving_shares` reports the split so this can be
measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from ..discretization.mpm.tables import MPMQuadratureTables
from ..solvers.bound_constrained import solve_bounded

__all__ = ["MicropolarPhaseFieldMPM"]


class MicropolarPhaseFieldMPM:
    """Staggered micropolar / phase-field solve on a shared MPM grid.

    Parameters
    ----------
    stepper : a prepared :class:`MicropolarMPMStepper`. Its ``degradation``
        attribute is driven by this class and should not be set by the caller.
    particles : the particle set the stepper was prepared with.
    model : :class:`~scion.materials.fracture.PhaseFieldModel`, whose ``E`` and
        ``nu`` must match the micropolar ``lam``/``mu`` -- the decomposition
        above is only exact if the split sees the same isotropic constants.
    """

    def __init__(self, stepper, particles, model, max_stagger: int = 50,
                 damage_tol: float = 1.0e-4):
        self.stepper = stepper
        self.particles = particles
        self.model = model
        self.max_stagger = int(max_stagger)
        self.damage_tol = float(damage_tol)

        self.tables = MPMQuadratureTables(stepper.grid, particles)
        self.n_nodes = int(stepper.grid.n_nodes)
        self.d = np.zeros(self.n_nodes)
        self.H = np.zeros(particles.n)
        self._supported = self.tables.supported_nodes()
        self._unsupported = np.flatnonzero(~self._supported)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _strain_parts(self):
        """``(sym E, skew E, Gamma)`` at the particles, from the current state."""
        c = self.stepper.constitutive(self.particles)
        Q, F = c["Q"], c["F"]
        E = np.einsum("pji,pjk->pik", Q, F) - np.eye(3)
        return (0.5 * (E + E.transpose(0, 2, 1)),
                0.5 * (E - E.transpose(0, 2, 1)),
                c["Ge"])

    def driving_energy(self) -> np.ndarray:
        """The undegraded active energy, ``(n_particles,)``.

        Three terms, matching the exact decomposition: the split's positive
        part of the symmetric strain, the relative-rotation penalty, and the
        curvature energy. Only the first is split; see the module docstring.
        """
        S, A, Gam = self._strain_parts()
        el = self.stepper.elastic
        psi_sym = self.model.driving_energy(S)
        psi_skew = el.mu * el.a * np.einsum("pij,pij->p", A, A)
        _, couple = el.stresses(np.zeros_like(S), Gam)
        psi_curv = 0.5 * np.einsum("pij,pij->p", couple, Gam)
        return psi_sym + psi_skew + psi_curv

    def driving_shares(self) -> dict:
        """The three contributions as fractions of the total.

        The diagnostic the regime question turns on: if ``curvature`` stays
        sub-percent the internal length is not influencing damage evolution,
        whatever else the model does.

        Note both micropolar terms are **identically zero in a homogeneous
        deformation** -- no relative rotation, no wryness -- so they enter only
        through *gradients*. A uniform tension test cannot exercise them at all,
        which is not a defect of the model but a statement about where the
        internal length can possibly matter.
        """
        S, A, Gam = self._strain_parts()
        el = self.stepper.elastic
        psi_sym = self.model.driving_energy(S)
        psi_skew = el.mu * el.a * np.einsum("pij,pij->p", A, A)
        _, couple = el.stresses(np.zeros_like(S), Gam)
        psi_curv = 0.5 * np.einsum("pij,pij->p", couple, Gam)
        tot = psi_sym + psi_skew + psi_curv
        # Energy-weighted, not a mean of per-particle ratios: unloaded
        # particles contribute 0/0 and would otherwise dilute every share
        # towards the fraction of particles carrying any energy at all.
        Z = max(float(tot.sum()), 1e-300)
        peak = int(np.argmax(tot))
        return {"symmetric": float(psi_sym.sum() / Z),
                "relative_rotation": float(psi_skew.sum() / Z),
                "curvature": float(psi_curv.sum() / Z),
                "curvature_at_peak": float(
                    psi_curv[peak] / max(tot[peak], 1e-300))}

    # ------------------------------------------------------------------
    def _damage_at_particles(self) -> np.ndarray:
        return self.tables.interpolate(self.d)

    def _apply_degradation(self) -> None:
        self.stepper.degradation = self.model.degradation.g(
            np.clip(self._damage_at_particles(), 0.0, 1.0))

    # ------------------------------------------------------------------
    def solve_damage(self) -> int:
        """One phase-field solve at frozen kinematics.

        Mirrors ``PhaseFieldFractureProblem.solve_second``, with two MPM-only
        differences: the driving force is the micropolar energy above, and an
        identity block is placed on unsupported nodes. The latter is exact
        rather than a regularisation -- those rows, columns and right-hand
        sides all vanish, so the nodes solve to zero, which is what a node
        carrying no material should hold.
        """
        self.H = np.maximum(self.H, self.driving_energy())
        K_grad = self.tables.laplacian(self.model.gradient_coefficient(),
                                       self.n_nodes).tolil()
        if self._unsupported.size:
            K_grad[self._unsupported, self._unsupported] = 1.0
        K_grad = K_grad.tocsr()

        d = self.d.copy()
        max_iter = 1 if self.model.linear_phase_field else 20
        for it in range(1, max_iter + 1):
            d_qp = self.tables.interpolate(d)
            r = self.model.reaction(d_qp, self.H)
            dr = self.model.dreaction(d_qp, self.H)
            K = K_grad + self.tables.mass(dr, self.n_nodes)
            R = -(K_grad @ d) - self.tables.load(r, self.n_nodes)
            if self.model.needs_bound_constraint:
                res = solve_bounded(K, R, lo=-d, hi=1.0 - d,
                                    x0=np.zeros_like(d))
                delta = res.x
            else:
                delta = spla.spsolve(K.tocsc(), R)
            d = d + delta
            if np.linalg.norm(delta) <= 1e-10 * max(1.0, np.linalg.norm(d)):
                break
        # irreversibility: damage never decreases, so cracks cannot heal
        d = np.clip(np.maximum(d, self.d), 0.0, 1.0)
        self._damage_change = float(np.abs(d - self.d).max())
        self.d = d
        return it

    # ------------------------------------------------------------------
    def solve_step(self, load_factor: float, d_load: float) -> dict:
        """One load step: staggered until the damage field stops moving.

        Each stagger runs the **full micropolar Newton solve** at the current
        damage, then re-solves the damage at the resulting kinematics.
        ``apply_increment`` is not enough and the failure is silent: it applies
        a prescribed increment without solving equilibrium, so the microrotation
        is never solved for at all, ``Gamma`` stays identically zero, and both
        micropolar contributions to the driving force read as exact zeros
        rather than small numbers.

        The state is rolled back before each stagger so every sweep sees the
        same load rather than an accumulating one.
        """
        base = self.stepper.snapshot(self.particles)
        info = {"stagger": 0, "damage_change": np.inf, "converged": False,
                "equilibrium": False, "newton": 0}
        for k in range(1, self.max_stagger + 1):
            self.stepper.restore(self.particles, base)
            self._apply_degradation()
            res = self.stepper.step(self.particles, load_factor, d_load)
            self.solve_damage()
            info.update(stagger=k, damage_change=self._damage_change,
                        equilibrium=bool(res.converged),
                        newton=len(getattr(res, "residuals", []) or []))
            if self._damage_change <= self.damage_tol:
                info["converged"] = True
                break
        info["max_damage"] = float(self.d.max())
        self.history.append(info)
        return info

    def __repr__(self) -> str:
        return (f"MicropolarPhaseFieldMPM({self.particles.n} particles, "
                f"{self.n_nodes} nodes, max d = {self.d.max():.3f})")
