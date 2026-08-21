r"""
Implicit micropolar (Cosserat) MPM: displacement *and* microrotation.

Implements O'Hare, Gourgiotis, Coombs & Augarde, *An implicit Material Point
Method for micropolar solids undergoing large deformations*, CMAME 419 (2024)
116668 (CC BY). Six degrees of freedom per grid node: three translations and
three spin components. ``implicit.py`` is the same solve with the microrotation
removed, and is the thing to read first.

Three-dimensional only
----------------------
``SO(3)`` is the point: the microrotation is a genuine rotation with a moving
axis, and the whole quaternion machinery exists because that cannot be
parameterised additively. A plane-strain Cosserat continuum (one microrotation
about ``z``) is a real and useful reduction with 3 DOFs per node, but it is a
*different* discretisation, not this one with two components zeroed -- so it is
obtained here by constraining the out-of-plane DOFs, which is what the
reference's cantilever example does.

Why the unknown is the *iterative* increment
---------------------------------------------
This is the one structural difference from ``ImplicitMPMStepper`` and it is
forced. The rotation tangent

    dQ_mt / dw_j = e_mjn Q_nt                                  (Eq. B.6)

is the Gateaux derivative *at the current* ``Q``, so it is exact only at
``dw = 0``. Newton therefore re-bases the state at every iteration: the solution
vector is the increment from the previous iterate, and ``F``, ``Q``, ``Gamma``
and the spatial gradients are all advanced as soon as it is solved for. A
total-increment parameterisation would need the differential of the exponential
map inside the tangent -- the reference's footnote 10 declines it for exactly
that reason.

Two consequences worth stating, because they are easy to trip over:

* **The residual is not a pure function of a vector**, so ``check_tangent``
  cannot be pointed straight at it. :meth:`residual_at` and :meth:`tangent_at`
  snapshot and restore the state around each evaluation, which makes them pure
  functions of the iterative increment and puts the finite-difference check
  back within reach. That check is the whole verification of this module.
* **A failed load step must restore the state**, not merely discard a vector.
  :meth:`step` snapshots on entry and rolls back before re-raising.

What is *not* here
------------------
The rotation field is interpolated with the same Lagrange shape functions as the
displacement, which is not rigorous -- rotations do not live on a linear
manifold, so their nodal values do not concatenate by addition, and iterating
embeds that error in ``Q``. This is the reference's own choice (Section 3.2.2),
it makes the scheme mildly path-dependent, and it is why rotation and
couple-stress converge more slowly than displacement and stress. Sansour and
Wagner's fix stores history at *nodes*, which MPM cannot do.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from scion.core import small_linalg as sla
from scion.core.dof import DOFMap
from scion.core.rotation import (
    quat_from_rotvec,
    quat_identity,
    quat_multiply,
    quat_to_matrix,
    rotvec_from_quat,
    spin_map,
)
from scion.solvers.linear import DirectSolver

from .background_grid import CartesianGrid
from scion.materials.micropolar_plasticity import NonFiniteTrialState

from .implicit import InvertedParticle, LoadStepResult
from .particles import ParticleSet

__all__ = ["MicropolarMPMStepper", "attach_micropolar_state"]

#: Levi-Civita, built once
_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0

#: cap on local-matrix entries held at once during assembly. The per-particle
#: block is ``(6k)^2``, which at k = 27 is 26244 numbers per particle -- a
#: 30000-particle problem would want 6 GB in one array.
_CHUNK_ENTRIES = 4_000_000


def attach_micropolar_state(particles: ParticleSet,
                            plastic: bool = False) -> ParticleSet:
    """Give ``particles`` an identity microrotation and zero wryness.

    Stored on the particle set rather than on the stepper because it is
    particle state: it must survive a stepper being rebuilt, and it must move
    with the particles if they are ever resampled.

    ``plastic=True`` adds the *committed* inelastic history -- the plastic
    deformation gradient, plastic wryness and hardening variable. These are the
    quantities the trial state is measured from, and unlike ``F``, ``Q`` and
    ``Gamma`` they advance **once per converged load step**, never per Newton
    iteration. Keeping that distinction is the whole of the trial/committed
    discipline: an iterate that is later discarded must not touch them.
    """
    particles.quat = quat_identity(particles.n, dtype=particles.dtype)
    particles.wryness = np.zeros((particles.n, 3, 3), dtype=particles.dtype)
    if plastic:
        particles.F_plastic = sla.identity(particles.n, 3,
                                           dtype=particles.dtype)
        particles.wryness_plastic = np.zeros((particles.n, 3, 3),
                                             dtype=particles.dtype)
        particles.hardening = np.zeros(particles.n, dtype=particles.dtype)
    return particles


class MicropolarMPMStepper:
    """Quasi-static implicit micropolar MPM, load-stepped with cut-back.

    Parameters
    ----------
    grid : CartesianGrid, ``dim == 3``
    material : :class:`~scion.materials.micropolar.MicropolarHooke`
    dirichlet : sequence of :class:`~.implicit.GridDirichlet`, whose ``value``
        has **six** components -- three translations then three rotations --
        with ``nan`` marking a free one.
    particle_force, particle_couple : ``(n, 3)`` loads applied at the particles
        per unit *reference* volume times that volume, i.e. already integrated.
        Ramped with the load factor. MPM has no surface to integrate a traction
        over, so this is how the reference applies a tip load or an end moment.
    tol, energy_tol, max_iter, support_tol : as for
        :class:`~.implicit.ImplicitMPMStepper`.
    """

    n_components = 6

    def __init__(self, grid: CartesianGrid, material, dirichlet=(),
                 particle_force=None, particle_couple=None,
                 tol: float = 1.0e-9, energy_tol: float = 1.0e-16,
                 max_iter: int = 20, linear_solver=None,
                 support_tol: float = 1.0e-6, ghost=None):
        if grid.dim != 3:
            raise NotImplementedError(
                f"micropolar MPM is 3-D only, got a {grid.dim}-D grid. The "
                f"microrotation is an element of SO(3); a plane-strain Cosserat "
                f"continuum has one rotation DOF and is a different "
                f"discretisation -- constrain the out-of-plane DOFs instead.")
        self.grid = grid
        self.material = material
        # An elastoplastic material *contains* its elastic law, so the two are
        # told apart by that rather than by a flag: the tangent needs the
        # elastic moduli either way.
        self.plasticity = material if hasattr(material, "elastic") else None
        self.elastic = (material if self.plasticity is None
                        else material.elastic)
        self.dirichlet = list(dirichlet)
        self.particle_force = (None if particle_force is None
                               else np.asarray(particle_force, dtype=float))
        self.particle_couple = (None if particle_couple is None
                                else np.asarray(particle_couple, dtype=float))
        self.tol = float(tol)
        self.energy_tol = float(energy_tol)
        self.max_iter = int(max_iter)
        self.linear_solver = linear_solver or DirectSolver()
        self.support_tol = float(support_tol)
        self._ghost = None
        self._ghost_state = None
        # one penalty per field: the *same* J is added to K^pu and to K^qphi
        # with different coefficients, since the two have different units.
        # A scalar sets the displacement penalty and scales the rotation one by
        # the curvature-to-stiffness ratio, which is what keeps them comparable.
        if ghost is None:
            self.ghost = None
        elif np.isscalar(ghost):
            self.ghost = (float(ghost),
                          float(ghost) * self.elastic.gam / max(self.elastic.mu, 1e-300))
        else:
            self.ghost = (float(ghost[0]), float(ghost[1]))
        self.history: list[LoadStepResult] = []

        self._n_dof = grid.n_nodes * self.n_components
        self._D = self.elastic.D
        self._Dc = self.elastic.D_curvature
        self._st = None

    # ------------------------------------------------------------------
    # per-load-step setup
    # ------------------------------------------------------------------
    def prepare(self, particles: ParticleSet) -> None:
        """Build the stencil and the free set for one load step.

        The node list and the shape-function *values* are frozen here; the
        spatial gradients are not -- they are pushed forward by ``dF^-1`` at
        every iteration (reference Algorithm 1, step 2b), which is what keeps
        them the gradients of the same functions in the *current* configuration.
        """
        if not hasattr(particles, "quat"):
            raise AttributeError(
                "particles carry no microrotation; call "
                "attach_micropolar_state(particles) first")
        grid = self.grid
        st = grid.build_stencil(particles.x,
                                lp=getattr(particles, "lp", None))
        self._st = st
        self._grad = st.grad_w.copy()          # advanced every iteration
        self._vol0 = particles.vol0

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
        active = np.repeat(active_node, self.n_components)

        dofmap = DOFMap(grid.n_nodes, self.n_components)
        self._prescribed = np.zeros(self._n_dof)
        for bc in self.dirichlet:
            idx, val = bc.dofs(grid, self.n_components)
            if idx.size == 0:
                raise ValueError(
                    "a GridDirichlet predicate selected no grid nodes; the "
                    "body would be left unrestrained and the solve would "
                    "return the undeformed state without complaining")
            self._prescribed[idx] = val
            for i in idx:
                dofmap.constrain(int(i) // self.n_components,
                                 int(i) % self.n_components)
        constrained = np.zeros(self._n_dof, dtype=bool)
        constrained[dofmap.constrained_dofs] = True

        self._free = np.flatnonzero(active & ~constrained)
        self._constrained = np.flatnonzero(active & constrained)
        self._ghost = None
        self._ghost_state = np.zeros(self._n_dof)
        if self.ghost is not None:
            from .ghost import ghost_penalty
            # J lives on a 3-component field; place one copy on the
            # translation DOFs of each node and one on the rotation DOFs
            J = ghost_penalty(
                grid, particles.x, n_components=3,
                lp=getattr(particles, 'lp', None)
                if grid.basis.requires_domain else None).tocoo()
            node, comp = J.row // 3, J.row % 3
            node_c, comp_c = J.col // 3, J.col % 3
            rows = np.concatenate([node * 6 + comp, node * 6 + comp + 3])
            cols = np.concatenate([node_c * 6 + comp_c, node_c * 6 + comp_c + 3])
            data = np.concatenate([self.ghost[0] * J.data,
                                   self.ghost[1] * J.data])
            self._ghost = sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(self._n_dof, self._n_dof)).tocsr()
        if self.dirichlet and self._constrained.size == 0:
            raise ValueError(
                "every constrained node lies outside the particles' support, "
                "so nothing restrains the body")

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def snapshot(self, particles: ParticleSet) -> tuple:
        return (particles.x.copy(), particles.F.copy(), particles.quat.copy(),
                particles.wryness.copy(), self._grad.copy(),
                self._ghost_state.copy())

    def restore(self, particles: ParticleSet, snap: tuple) -> None:
        (particles.x, particles.F, particles.quat, particles.wryness,
         self._grad, self._ghost_state) = (a.copy() for a in snap)

    # ------------------------------------------------------------------
    def apply_increment(self, particles: ParticleSet, dof: np.ndarray) -> None:
        """Advance the state by one *iterative* increment (Algorithm 1).

        ``dof`` is flat and node-major, six per node. Everything here is
        multiplicative or incremental in the way the reference specifies, and
        the two places where a plausible-looking additive shortcut is wrong are
        marked.
        """
        st = self._st
        nodal = dof.reshape(self.grid.n_nodes, self.n_components)
        du = nodal[:, :3][st.node_id]           # (n, k, 3)
        dw = nodal[:, 3:][st.node_id]

        du_p = np.einsum("pk,pka->pa", st.w, du)
        dw_p = np.einsum("pk,pka->pa", st.w, dw)
        grad_du = np.einsum("pka,pkb->pab", du, self._grad)
        grad_dw = np.einsum("pka,pkb->pab", dw, self._grad)

        dF = np.eye(3) + grad_du
        detdF = sla.det(dF)
        if (detdF <= 0.0).any():
            raise InvertedParticle(
                f"{int((detdF <= 0).sum())} of {particles.n} particles "
                f"inverted in the trial state (min det dF = {detdF.min():.3e})")

        particles.F = np.einsum("pab,pbc->pac", dF, particles.F)
        particles.x = particles.x + du_p
        # gradients follow the incremental motion, so they stay the gradients of
        # the *same* shape functions in the new configuration
        self._grad = np.einsum("pkt,ptr->pkr", self._grad, sla.inv(dF, detdF))

        # rotation composes, it does not add. phi + dw is a first-order
        # approximation that stays plausible and drifts.
        particles.quat = quat_multiply(quat_from_rotvec(dw_p), particles.quat)
        Q = quat_to_matrix(particles.quat)

        # wryness accumulates through the exponential map's differential; H = I
        # is correct to first order and makes the answer depend on the number of
        # load steps
        particles.wryness = particles.wryness + np.einsum(
            "pjg,pjk,pkn,pnt->pgt", Q, spin_map(dw_p), grad_dw, particles.F)

        # the ghost penalty acts on the *increment over the load step*, because
        # J is rebuilt each step from the current particle positions. Tracking
        # it here rather than at the call sites keeps snapshot/restore honest.
        if self._ghost is not None:
            self._ghost_state = self._ghost_state + dof

    # ------------------------------------------------------------------
    # kinematics -> stress
    # ------------------------------------------------------------------
    def _trial(self, particles: ParticleSet) -> tuple:
        """``((F^e)^tr, (Gamma^e)^tr, chi_committed)`` for the current iterate.

        Measured from the state *committed at the last converged load step*,
        never from the previous Newton iterate: ``(F^e)^tr = F (F^p)^-1`` and
        ``(Gamma^e)^tr = Gamma - Gamma^p``. Taking them from the previous
        iterate instead would integrate the flow rule along the Newton path
        rather than along the load path -- an error that shrinks as the solve
        converges and so never announces itself, while leaving the answer
        dependent on how the iteration got there.
        """
        Fp = particles.F_plastic
        Fe_tr = np.einsum("nij,njk->nik", particles.F,
                          sla.inv(Fp, sla.det(Fp)))
        return Fe_tr, particles.wryness - particles.wryness_plastic, \
            particles.hardening

    def constitutive(self, particles: ParticleSet) -> dict:
        """``tau``, ``nu`` and everything the tangent needs, elastic or not.

        The single seam between the two paths. Both return the same keys, so
        the force and tangent assembly below never branches -- what differs is
        only ``dUe`` and ``dGe``, the sensitivities of the *converged* elastic
        stretch and wryness to the return map's inputs.
        """
        Q = quat_to_matrix(particles.quat)
        F = particles.F
        if self.plasticity is None:
            Ue = np.einsum("nji,njk->nik", Q, F)
            Ge = particles.wryness
            state = None
        else:
            Fe_tr, Ge_tr, chi = self._trial(particles)
            self._Fe_tr, self._Ge_tr, self._chi = Fe_tr, Ge_tr, chi
            state = self.plasticity.return_map(Q, F, Fe_tr, Ge_tr, chi)
            Ue, Ge = state["Ue"], state["Gam_e"]

        B = np.einsum("abgp,ngp->nab", self._D, Ue - np.eye(3))
        S = np.einsum("abgp,ngp->nab", self._Dc, Ge)
        Fe = np.einsum("nia,nab->nib", Q, Ue)
        tau = np.einsum("nia,nab,njb->nij", Q, B, Fe)
        nu = np.einsum("nia,nab,njb->nij", Q, S, F)
        return dict(Q=Q, F=F, Ue=Ue, Ge=Ge, B=B, S=S, Fe=Fe, tau=tau, nu=nu,
                    state=state)

    def _kinematics(self, particles: ParticleSet) -> tuple:
        """``(Q, F, J_sigma, J_mu)`` at the current state.

        ``J_sigma = Q B F^T`` is ``J`` times the Cauchy stress, and carrying it
        that way rather than dividing by ``J`` is what keeps every volume
        weight below equal to the *reference* volume.
        """
        Q = quat_to_matrix(particles.quat)
        F = particles.F
        E = np.einsum("pji,pjk->pik", Q, F) - np.eye(3)
        B, S = self.material.stresses(E, particles.wryness)
        J_sigma = np.einsum("pia,pab,prb->pir", Q, B, F)
        J_mu = np.einsum("pia,pab,prb->pir", Q, S, F)
        return Q, F, J_sigma, J_mu

    def internal_force(self, particles: ParticleSet) -> np.ndarray:
        """Nodal internal force *and* couple, flat, six per node."""
        st, grid = self._st, self.grid
        c = self.constitutive(particles)
        # tau is J*sigma and nu is J*mu by definition, so every weight below is
        # the *reference* volume and nothing divides by J
        J_sigma, J_mu = c["tau"], c["nu"]
        V = self._vol0

        # s[p, k, i] is the force integrand, c[p, k, i] the couple's
        s = np.einsum("pir,pkr->pki", J_sigma, self._grad)
        c = np.einsum("pir,pkr->pki", J_mu, self._grad)
        # the antisymmetric part of the stress is what the couple balances --
        # it is identically zero in a classical continuum, which is why the
        # rotation DOFs there would have no equation of their own
        axial = np.einsum("irn,prn->pi", _EPS, J_sigma)

        out = np.zeros((grid.n_nodes, self.n_components))
        np.add.at(out[:, :3], st.node_id, V[:, None, None] * s)
        np.add.at(out[:, 3:], st.node_id,
                  V[:, None, None] * (c + st.w[:, :, None] * axial[:, None, :]))
        flat = out.ravel()
        # the penalty is a function of the *accumulated* state, so it needs the
        # total displacement and rotation from the start of the load step, not
        # the iterative increment -- see _ghost_state
        if self._ghost is not None:
            flat = flat + self._ghost @ self._ghost_state
        return flat

    def external_force(self, particles: ParticleSet,
                       load_factor: float) -> np.ndarray:
        st, grid = self._st, self.grid
        out = np.zeros((grid.n_nodes, self.n_components))
        if self.particle_force is not None:
            np.add.at(out[:, :3], st.node_id,
                      st.w[:, :, None] * self.particle_force[:, None, :])
        if self.particle_couple is not None:
            np.add.at(out[:, 3:], st.node_id,
                      st.w[:, :, None] * self.particle_couple[:, None, :])
        return load_factor * out.ravel()

    # ------------------------------------------------------------------
    def tangent(self, particles: ParticleSet) -> sparse.csr_matrix:
        """The four consistent tangent blocks, assembled (reference Eqs. 30-33
        and Appendix A).

        Unsymmetric in general, and that is not a bug: a rotation-parameterised
        tangent is symmetric only at equilibrium. Do not hand this to PCG.
        """
        st = self._st
        nc = self.n_components
        c = self.constitutive(particles)

        n, k = st.node_id.shape
        per_particle = (nc * k) ** 2
        chunk = max(1, int(_CHUNK_ENTRIES // max(per_particle, 1)))

        K = sparse.csr_matrix((self._n_dof, self._n_dof))
        for lo in range(0, n, chunk):
            sl = slice(lo, min(lo + chunk, n))
            K = K + self._tangent_chunk_general(sl, c, particles)
        # must match the penalty term in internal_force exactly; adding it to
        # one and not the other makes the residual and the tangent describe
        # different problems, and Newton then diverges on the first iterate
        return K if self._ghost is None else K + self._ghost

    def _tangent_chunk(self, sl, st, Q, F, J_sigma, J_mu, D, Dc):
        nc = self.n_components
        V = self._vol0[sl]
        g = self._grad[sl]
        w = st.w[sl]
        Qs, Fs = Q[sl], F[sl]
        Jsig, Jmu = J_sigma[sl], J_mu[sl]

        # h = Grad_X N, the reference gradient
        h = np.einsum("prb,pkr->pkb", Fs, g)
        # qh[p, k, i, a, b] = Q_ia h_kb, the object every D contraction eats
        qh = np.einsum("pia,pkb->pkiab", Qs, h)
        A = np.einsum("pkiab,abgt->pkigt", qh, D)
        Ac = np.einsum("pkiab,abgt->pkigt", qh, Dc)
        # Z[p, g, t, j] = e_nmj Q_ng F_mt
        Z = np.einsum("nmj,png,pmt->pgtj", _EPS, Qs, Fs)

        s = np.einsum("pir,pkr->pki", Jsig, g)
        c = np.einsum("pir,pkr->pki", Jmu, g)

        Kpu = np.einsum("pkigt,pljgt->pkilj", A, qh)
        Kpw = (np.einsum("pkigt,pgtj,pl->pkilj", A, Z, w)
               + np.einsum("nij,pkn,pl->pkilj", _EPS, s, w))
        Kmu = (np.einsum("pk,pgti,pljgt->pkilj", w, Z, A)
               + np.einsum("pk,irj,plr->pkilj", w, _EPS, s))
        Kmw = (np.einsum("pk,pl,pigt,pgtj->pkilj", w, w,
                         np.einsum("pabi,abgt->pigt", Z, D), Z)
               + np.einsum("pkigt,pljgt->pkilj", Ac, qh)
               + np.einsum("nij,pkn,pl->pkilj", _EPS, c, w)
               # the e:sigma:e term: the only place a transposed Levi-Civita
               # index is invisible in the assembled matrix, since it is
               # antisymmetric in both slots
               + np.einsum("pk,pl,imn,smj,psn->pkilj", w, w, _EPS, _EPS, Jsig))

        block = np.zeros(Kpu.shape[:2] + (nc,) + Kpu.shape[3:4] + (nc,))
        block[:, :, :3, :, :3] = Kpu
        block[:, :, :3, :, 3:] = Kpw
        block[:, :, 3:, :, :3] = Kmu
        block[:, :, 3:, :, 3:] = Kmw
        block *= V[:, None, None, None, None]

        dofs = (st.node_id[sl][:, :, None] * nc
                + np.arange(nc)[None, None, :]).reshape(V.size, -1)
        n_loc = dofs.shape[1]
        rows = np.repeat(dofs, n_loc, axis=1).ravel()
        cols = np.tile(dofs, (1, n_loc)).ravel()
        return sparse.coo_matrix(
            (block.reshape(-1), (rows, cols)),
            shape=(self._n_dof, self._n_dof)).tocsr()

    # ------------------------------------------------------------------
    def _sensitivities(self, particles, c: dict, sl) -> dict:
        """``d(U^e)/d(nodal)`` and ``d(Gamma^e)/d(nodal)`` for one chunk.

        The only place plasticity enters the tangent. Elastically
        ``U^e = Q^T F`` and ``Gamma^e = Gamma`` differentiate directly.
        Plastically both are *converged outputs* of the return map, so the chain
        runs through ``P = -C^-1 T`` over the four inputs that move with the
        nodal degrees of freedom:

            F, (F^e)^tr   <- du            Q, (Gamma^e)^tr   <- dw

        ``chi_old`` is frozen within a load step and contributes nothing, which
        is the formal statement of trial-versus-committed.

        Returned in a common shape ``(n, k, j, ., .)`` -- particle, stencil
        slot, driving component, then the two tensor indices -- so the assembly
        below never branches on which path produced them.
        """
        st = self._st
        g, w = self._grad[sl], st.w[sl]
        Q, F = c["Q"][sl], c["F"][sl]
        h = np.einsum("prb,pkr->pkb", F, g)          # Grad_X N, (n, k, 3)

        # dQ_mt/dw_j = e_mjc Q_ct  (Eq. B.6), carried onto node k by N_k
        dQdw = np.einsum("mjc,pct,pk->pkjmt", _EPS, Q, w)

        if self.plasticity is None:
            return dict(
                h=h, dQdw=dQdw,
                dUe_du=np.einsum("pjy,pkb->pkjyb", Q, h),
                dUe_dw=np.einsum("pkjmt,pmb->pkjtb", dQdw, F),
                dGe_du=np.zeros((Q.shape[0], w.shape[1], 3, 3, 3)),
                dGe_dw=np.einsum("pjy,pkb->pkjyb", Q, h))

        state = c["state"]
        Fe_tr = self._Fe_tr[sl]
        plastic = state["plastic"][sl]

        # The *elastic* sensitivities, for every point. U^e = Q^T (F^e)^tr and
        # Gamma^e = Gamma^tr differentiate directly, and note ``he`` rather than
        # ``h``: with a committed F^p the elastic response is measured from the
        # plastic state, not from the reference one.
        Fp = particles.F_plastic[sl]
        he = np.einsum("pkg,pgt->pkt", h, sla.inv(Fp, sla.det(Fp)))
        out = dict(dUe_du=np.einsum("pjy,pkb->pkjyb", Q, he),
                   dUe_dw=np.einsum("pkjmt,pmb->pkjtb", dQdw, Fe_tr),
                   dGe_du=np.zeros((Q.shape[0], w.shape[1], 3, 3, 3)),
                   dGe_dw=np.einsum("pjy,pkb->pkjyb", Q, h))

        # ``P = -C^-1 T`` is solved **only where the point is plastic**. At an
        # elastic point C is not merely the wrong operator, it is *singular*
        # whenever ``H = 0``: the yield-function row loses both its ``chi`` and
        # its ``dgamma`` entry, so the 20x20 system has no inverse at all. A
        # perfectly plastic material -- which is what a shear-band study uses --
        # therefore fails outright rather than returning a wrong tangent.
        if plastic.any():
            idx = np.flatnonzero(plastic)
            # ``k`` may be per-particle -- a seeded imperfection is exactly
            # that -- so it has to be sliced to the same subset. Missed until
            # now because the shear layer happened to have *every* particle
            # plastic inside a single chunk, which made the full-length array
            # match the subset by coincidence.
            k_all = np.broadcast_to(
                np.asarray(self.plasticity.k, dtype=float),
                (particles.n,))[sl][idx]

            # Apex points get their own linearisation. ``P = -C^-1 T`` is not
            # merely inaccurate for them, it is the wrong problem: ``C`` carries
            # the smooth surface's yield row, and an apex point's constraint is
            # the tip, not ``f = 0`` along the cone.
            apex = state.get(
                "apex", np.zeros(particles.n, dtype=bool))[sl][idx]
            P = np.zeros((len(idx), self.plasticity.n_unknowns,
                          self.plasticity.n_inputs))
            sm = ~apex
            if sm.any():
                j = idx[sm]
                _, _, _, _, work = self.plasticity.residuals(
                    Q[j], F[j], Fe_tr[j], self._Ge_tr[sl][j],
                    self._chi[sl][j], state["Ue"][sl][j],
                    state["Gam_e"][sl][j], state["chi"][sl][j],
                    state["dgamma"][sl][j], k_all[sm])
                P[sm] = self.plasticity.algorithmic_jacobian(
                    Q[j], F[j], Fe_tr[j], state["Ue"][sl][j],
                    state["Gam_e"][sl][j], state["dgamma"][sl][j], work)
            if apex.any():
                j = idx[apex]
                P[apex] = self.plasticity.apex_jacobian(
                    Fe_tr[j], state["Ue"][sl][j])

            n = P.shape[0]
            isl = self.plasticity.input_slices
            rows = dict(Ue=slice(0, 9), Ge=slice(9, 18))
            blk = {(r, i): P[:, rows[r], isl[i]].reshape(n, 3, 3, 3, 3)
                   for r in rows for i in ("F", "Q", "Fe_tr", "Gam_tr")}

            for key, r in (("dUe_du", "Ue"), ("dGe_du", "Ge")):
                out[key][idx] = (
                    np.einsum("pybjt,pkt->pkjyb", blk[(r, "F")], h[idx])
                    + np.einsum("pybjt,pkt->pkjyb", blk[(r, "Fe_tr")], he[idx]))
            for key, r in (("dUe_dw", "Ue"), ("dGe_dw", "Ge")):
                out[key][idx] = (
                    np.einsum("pybmt,pkjmt->pkjyb", blk[(r, "Q")], dQdw[idx])
                    + np.einsum("pybgq,pjg,pkq->pkjyb", blk[(r, "Gam_tr")],
                                Q[idx], h[idx]))

        out["h"], out["dQdw"] = h, dQdw
        return out

    # ------------------------------------------------------------------
    def _tangent_chunk_general(self, sl, c: dict, particles) -> sparse.csr_matrix:
        """One chunk of the tangent, elastic or elastoplastic.

        Derived directly from the internal force rather than transcribed from
        the reference's ``a1..a7``. Writing

            p_int_Ii = sum_p V^0 tau_ir g_Ir
            m_int_Ii = sum_p V^0 (nu_ir g_Ir + N_I e_irn tau_rn)

        and differentiating leaves only three things to know: how ``tau`` and
        ``nu`` respond to the nodal degrees of freedom, and how ``g_I`` does.
        The last is the term that is easy to forget --

            dg_Ir/du_Jj = -g_Ij g_Jr

        because the spatial gradient is pushed forward by ``dF^-1`` -- and it is
        what supplies the initial-stress ("geometric") contribution that
        otherwise has to be added by hand.
        """
        nc = self.n_components
        st = self._st
        V, g, w = self._vol0[sl], self._grad[sl], st.w[sl]
        Q, F, B, S = c["Q"][sl], c["F"][sl], c["B"][sl], c["S"][sl]
        Fe, tau, nu = c["Fe"][sl], c["tau"][sl], c["nu"][sl]
        D, Dc = self._D, self._Dc
        eye = np.eye(3)

        d = self._sensitivities(particles, c, sl)
        h, dQdw = d["h"], d["dQdw"]

        # how tau and nu respond to their own arguments
        TU = (np.einsum("pia,abyc,pjb->pijyc", Q, D, Fe)
              + np.einsum("pia,pac,pjy->pijyc", Q, B, Q))
        NU = np.einsum("pia,abgd,pjb->pijgd", Q, Dc, F)
        QtT = np.einsum("pia,pij->paj", Q, tau)
        TQm = np.einsum("pij,pjb->pib", tau, Q)
        Qtn = np.einsum("pia,pij->paj", Q, nu)
        QS = np.einsum("pia,pab->pib", Q, S)

        # dtau / d(nodal), dnu / d(nodal); index order (p, L, j, i, r)
        dtau_du = np.einsum("pirYC,pLjYC->pLjir", TU, d["dUe_du"])
        dtau_dw = (np.einsum("pirYC,pLjYC->pLjir", TU, d["dUe_dw"])
                   + np.einsum("im,pLjmt,ptr->pLjir", eye, dQdw, QtT)
                   + np.einsum("rm,pLjmt,pit->pLjir", eye, dQdw, TQm))
        dnu_du = (np.einsum("pirGD,pLjGD->pLjir", NU, d["dGe_du"])
                  + np.einsum("rj,pit,pLt->pLjir", eye, QS, h))
        dnu_dw = (np.einsum("pirGD,pLjGD->pLjir", NU, d["dGe_dw"])
                  + np.einsum("im,pLjmt,ptr->pLjir", eye, dQdw, Qtn))

        Kpu = (np.einsum("pLjir,pKr->pKiLj", dtau_du, g)
               - np.einsum("pir,pKj,pLr->pKiLj", tau, g, g))
        Kpw = np.einsum("pLjir,pKr->pKiLj", dtau_dw, g)
        Kmu = (np.einsum("pLjir,pKr->pKiLj", dnu_du, g)
               - np.einsum("pir,pKj,pLr->pKiLj", nu, g, g)
               + np.einsum("pK,irn,pLjrn->pKiLj", w, _EPS, dtau_du))
        Kmw = (np.einsum("pLjir,pKr->pKiLj", dnu_dw, g)
               + np.einsum("pK,irn,pLjrn->pKiLj", w, _EPS, dtau_dw))

        block = np.zeros(Kpu.shape[:2] + (nc,) + Kpu.shape[3:4] + (nc,))
        block[:, :, :3, :, :3] = Kpu
        block[:, :, :3, :, 3:] = Kpw
        block[:, :, 3:, :, :3] = Kmu
        block[:, :, 3:, :, 3:] = Kmw
        block *= V[:, None, None, None, None]

        dofs = (st.node_id[sl][:, :, None] * nc
                + np.arange(nc)[None, None, :]).reshape(V.size, -1)
        n_loc = dofs.shape[1]
        rows = np.repeat(dofs, n_loc, axis=1).ravel()
        cols = np.tile(dofs, (1, n_loc)).ravel()
        return sparse.coo_matrix(
            (block.reshape(-1), (rows, cols)),
            shape=(self._n_dof, self._n_dof)).tocsr()

    # ------------------------------------------------------------------
    # pure-function wrappers, for the finite-difference tangent check
    # ------------------------------------------------------------------
    def residual_at(self, particles: ParticleSet, dof: np.ndarray) -> np.ndarray:
        """Internal force after applying ``dof``, with the state restored.

        The state advance is undone before returning, which is what makes this
        a pure function of ``dof`` and therefore differentiable by
        ``check_tangent``. Without the restore the differences pick up the
        accumulated state as well as the derivative, and the check passes on a
        wrong tangent for a step size small enough.
        """
        snap = self.snapshot(particles)
        try:
            self.apply_increment(particles, dof)
            return self.internal_force(particles)
        finally:
            self.restore(particles, snap)

    def tangent_at(self, particles: ParticleSet, dof: np.ndarray):
        snap = self.snapshot(particles)
        try:
            self.apply_increment(particles, dof)
            return self.tangent(particles)
        finally:
            self.restore(particles, snap)

    # ------------------------------------------------------------------
    def step(self, particles: ParticleSet, load_factor: float,
             d_load: float) -> LoadStepResult:
        """One load step. Rolls the state back if it does not converge."""
        self.prepare(particles)          # also zeroes the ghost accumulator
        free, con = self._free, self._constrained
        snap = self.snapshot(particles)
        result = LoadStepResult(load_factor, False, 0)

        try:
            if con.size:
                pres = np.zeros(self._n_dof)
                pres[con] = d_load * self._prescribed[con]
                self.apply_increment(particles, pres)

            f_ext = self.external_force(particles, load_factor)
            # Normalise by the external load. A displacement-controlled step
            # has none, and the meaningful scale is then the *reaction* -- the
            # internal force over the constrained DOFs as well as the free
            # ones. Falling back to the free-DOF residual alone is wrong twice:
            # at equilibrium it is near zero, and clamping it at an absolute
            # 1.0 assumes the problem is measured in newtons. A soil sample at
            # 100 kPa over a few square millimetres carries ~1e-3 N, so the
            # clamp made every residual read as 1e-13, the energy test could
            # never fire against an equally tiny first energy, and the step cut
            # back forever while already converged.
            ref = float(np.linalg.norm(f_ext[free]))
            if ref <= 0.0:
                act = np.concatenate([free, con]) if con.size else free
                ref = float(np.linalg.norm(self.internal_force(particles)[act]))
            if ref <= 0.0:
                ref = 1.0
            energy0 = None
            energy_ok = False

            for it in range(self.max_iter):
                R = self.internal_force(particles) - f_ext
                norm = float(np.linalg.norm(R[free])) / ref
                result.residuals.append(norm)
                # ``it == 0`` counts: a step whose prescribed increment leaves
                # the body in equilibrium has nothing to do, and demanding an
                # iteration there means demanding the energy criterion, which
                # cannot be met when the first energy is itself at round-off.
                if norm < self.tol and (energy_ok or it == 0):
                    result.converged = True
                    break

                K = self.tangent(particles)[free][:, free]
                self._check_conditioning(K, free, load_factor)
                ddu = self.linear_solver.solve(K, -R[free])

                energy = abs(float(R[free] @ ddu))
                if energy0 is None:
                    energy0 = max(energy, 1e-300)
                energy_ok = energy / energy0 < self.energy_tol

                full = np.zeros(self._n_dof)
                full[free] = ddu
                self.apply_increment(particles, full)
                result.iterations = it + 1
        except Exception:
            self.restore(particles, snap)
            raise

        if not result.converged:
            self.restore(particles, snap)
            raise RuntimeError(
                f"micropolar load step at lambda = {load_factor:.4g} did not "
                f"converge in {self.max_iter} iterations; residuals "
                f"{['%.2e' % r for r in result.residuals]}")
        self.commit_plastic(particles)
        self.history.append(result)
        return result

    def _check_conditioning(self, K, free: np.ndarray,
                            load_factor: float) -> None:
        """Reject a numerically empty row, comparing like with like.

        Translation stiffness has units of force/length and rotation stiffness
        force*length, so they differ by the square of a length and a single
        ``min(diag)/max(diag)`` test over all six components is dimensionally
        meaningless -- it flags a perfectly good fine-mesh problem as singular
        purely because ``h`` got smaller. Each block is therefore compared
        against its own maximum.
        """
        diag = np.abs(K.diagonal())
        comp = free % self.n_components
        for lo, hi, what in ((0, 3, "translation"), (3, 6, "rotation")):
            sel = (comp >= lo) & (comp < hi)
            if not sel.any():
                continue
            d = diag[sel]
            if d.max() <= 0.0 or d.min() <= 1e-12 * d.max():
                nodes = sorted(set(free[sel][d <= 1e-12 * d.max()]
                                   // self.n_components))
                raise RuntimeError(
                    f"the reduced tangent has no {what} stiffness on "
                    f"{len(nodes)} grid nodes at load factor "
                    f"{load_factor:.4g} (nodes {nodes[:8]}...). Either the "
                    f"body has reached nodes that carry no support and no "
                    f"constraint, or -- for the rotation block -- the "
                    f"microrotation has no equation at all: its stiffness at "
                    f"the undeformed state is exactly 4 mu a, so a = 0 with "
                    f"no couple stress leaves it undetermined.")

    def run(self, particles: ParticleSet, n_steps: int = 10, callback=None,
            min_step: float = 1.0e-4, remesh=None) -> ParticleSet:
        """Ramp the load from 0 to 1, halving the increment where a step fails.

        ``remesh(load_factor) -> CartesianGrid`` replaces the background grid
        before each step. That is how a *moving mesh* is done: a prescribed
        displacement can only be applied exactly on a grid line, so a boundary
        that moves needs the grid to follow it, or the condition lands inside
        the material. The grid must keep the same ``shape`` -- only ``dx`` and
        ``origin`` may change -- because the number of degrees of freedom is
        fixed at construction.
        """
        d_load = 1.0 / n_steps
        lam, s = 0.0, 0
        while lam < 1.0 - 1.0e-12:
            inc = min(d_load, 1.0 - lam)
            if remesh is not None:
                grid = remesh(lam)
                if grid.n_nodes != self.grid.n_nodes:
                    raise ValueError(
                        f"remesh changed the node count from "
                        f"{self.grid.n_nodes} to {grid.n_nodes}; only dx and "
                        f"origin may move, since the DOF count is fixed")
                self.grid = grid
            try:
                res = self.step(particles, lam + inc, inc)
            except (InvertedParticle, NonFiniteTrialState,
                    RuntimeError) as exc:
                d_load *= 0.5
                if d_load < min_step:
                    raise RuntimeError(
                        f"load stepping stalled at lambda = {lam:.4g}: "
                        f"{exc}") from exc
                continue
            lam += inc
            s += 1
            if callback is not None:
                callback(s, res, particles)
        return particles

    # ------------------------------------------------------------------
    def commit_plastic(self, particles: ParticleSet) -> None:
        """Advance the inelastic history, **once per converged load step**.

        This is the trial/committed boundary for a path-dependent material and
        the reason it is a separate method rather than a line inside the Newton
        loop. A Newton iterate that is later discarded, or a load step that is
        cut back, must leave ``F^p``, ``Gamma^p`` and ``chi`` exactly as they
        were -- otherwise the plastic history is corrupted permanently and the
        answer silently depends on how the iteration got there.
        """
        if self.plasticity is None:
            return
        c = self.constitutive(particles)
        Fe = c["Fe"]
        particles.F_plastic = np.einsum(
            "nij,njk->nik", sla.inv(Fe, sla.det(Fe)), particles.F)
        particles.wryness_plastic = particles.wryness - c["Ge"]
        particles.hardening = c["state"]["chi"]

    @staticmethod
    def microrotation_vector(particles: ParticleSet) -> np.ndarray:
        """``(n, 3)`` rotation vectors, for output only -- see
        ``core/rotation.py``. Meaningless at a whole revolution, where the axis
        is genuinely indeterminate."""
        return rotvec_from_quat(particles.quat)
