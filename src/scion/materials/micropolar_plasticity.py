r"""
Micropolar Drucker-Prager plasticity: the local return map.

Implements the constitutive model of O'Hare, Coombs, Gourgiotis & Augarde,
CMAME 449 (2026) 118516, Sections 2.7.2-2.7.3 and 3.2-3.3. This is the piece
that makes a micropolar solid *localise*, and the reason the whole micropolar
layer was built.

What is and is not split
------------------------
``F = F^e F^p`` as usual, but the **microrotation carries no plastic part**.
That is a deliberate break with de Borst and Steinmann: a rotation is rigid and
therefore non-dissipative, and dissipation comes from slip between grains, not
from their turning. Only the microrotation *gradient* splits, and additively:

    Gamma = Gamma^e + Gamma^p

which is legitimate because the wryness is assembled from three vectors defined
wholly in the material frame. Dropping the ``Q`` split is what keeps the local
problem to four residuals instead of a tangle, and it is the single biggest
simplification in the model.

The yield surface sees the couple stress
-----------------------------------------
Yield is Drucker-Prager on a *generalised* invariant that mixes stress and
couple-stress,

    q = sqrt(3/2 (s : s + nu : nu / l_p^2)) ,   s = sym(tau) - p I ,
    f = q + alpha_DP p - (k + Lambda) ,        g = q + beta_DP p - Lambda ,

with ``l_p`` a plastic length carried purely for dimensional consistency -- it
is what lets a couple-stress be added to a stress. Setting
``alpha_DP = beta_DP = 0`` recovers von Mises with ``k`` the yield stress.

**Softening does not need a softening law.** A non-associated flow rule
(``psi < phi``, so ``beta_DP < alpha_DP``) produces structural softening on its
own, with ``H = 0``. Worth contrasting with ``materials/damage/``, where every
law softens and the resulting loss of ellipticity is the whole problem: here
the constitutive law never loses ellipticity and the regularisation is
micropolar.

The local problem
-----------------
Four residuals in ``{U^e, Gamma^e, chi, dgamma}`` -- 20 scalar unknowns in 3-D:

    r_U   = U^e - Q^T exp(-dgamma dg/dtau) (F^e)^tr
    r_Gam = Gamma^e + dgamma Q^T (dg/dnu) F - (Gamma^e)^tr
    r_chi = chi - chi_old - dgamma
    f     = 0

The stretch flow is integrated with an **exponential map**, which is what keeps
the update exactly plastically incompressible when it should be, and is why the
derivative of a matrix exponential appears in the Jacobian.
"""

from __future__ import annotations

import numpy as np

from .micropolar import MicropolarHooke

__all__ = ["MicropolarDruckerPrager", "NonFiniteTrialState",
           "dexp_symmetric"]


class NonFiniteTrialState(RuntimeError):
    """A trial iterate overflowed. Cut the load step."""

#: eigenvalues closer than this use the divided-difference limit
_DEGENERATE = 1.0e-6


def dexp_symmetric(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(exp(X), dexp(X)/dX)`` for a stack of **symmetric** matrices.

    ``(n, 3, 3) -> ((n, 3, 3), (n, 3, 3, 3, 3))``.

    In the eigenbasis of ``X`` the derivative of any isotropic tensor function
    is diagonal in pairs:

        dexp/dX_ijkl = sum_ab  g_ab  n_a_i n_b_j n_a_k n_b_l

    with ``g_ab = (e^la - e^lb)/(la - lb)`` off the diagonal and ``e^la`` on it.
    The off-diagonal expression is ``0/0`` at a repeated eigenvalue and repeated
    eigenvalues are the *normal* case here, not a corner one -- the argument is
    ``-dgamma dg/dtau``, which is a deviator plus a multiple of the identity,
    and any axisymmetric stress state gives two equal eigenvalues. A naive
    implementation returns ``nan`` on the first yielding particle of a uniaxial
    test. The limit is ``e^la``, approached through a series so the transition
    is smooth rather than a switch.
    """
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        # A trial iterate can drive dgamma or the stress to inf, and the
        # overflow then arrives here as "Eigenvalues did not converge" from
        # LAPACK -- a crash that says nothing about its cause and that load-step
        # cut-back cannot catch, because it is not an exception the stepper
        # knows. Named instead, so the step is simply cut.
        raise NonFiniteTrialState(
            f"the plastic flow direction is not finite on "
            f"{int((~np.isfinite(X)).any(axis=(1, 2)).sum())} points; the trial "
            f"state has overflowed and the load step must be cut")
    lam, vec = np.linalg.eigh(X)                     # (n, 3), (n, 3, 3)
    exp_lam = np.exp(lam)
    expX = np.einsum("nia,na,nja->nij", vec, exp_lam, vec)

    d = lam[:, :, None] - lam[:, None, :]            # (n, 3, 3)
    mean = 0.5 * (exp_lam[:, :, None] + exp_lam[:, None, :])
    near = np.abs(d) < _DEGENERATE
    safe = np.where(near, 1.0, d)
    # (e^a - e^b)/(a - b); at a == b the limit is e^a, and the leading
    # correction is the d^2/24 term, which keeps the two branches continuous
    g = np.where(near,
                 mean * (1.0 + d ** 2 / 24.0),
                 (exp_lam[:, :, None] - exp_lam[:, None, :]) / safe)

    dexp = np.einsum("nab,nia,njb,nka,nlb->nijkl", g, vec, vec, vec, vec)
    # Symmetrised in the *argument* pair. The raw spectral form obeys
    # ``E_ijkl = E_jilk`` but not ``E_ijkl = E_ijlk``, so contracting it with a
    # symmetric argument silently picks the symmetric part anyway -- making
    # that explicit means the stored tensor can be finite-differenced entry by
    # entry, and it changes nothing for any argument actually used here (every
    # one is symmetric, being a derivative of the symmetric ``s``).
    return expX, 0.5 * (dexp + dexp.transpose(0, 1, 2, 4, 3))


class MicropolarDruckerPrager:
    """Generalised Drucker-Prager return map for a micropolar solid.

    Parameters
    ----------
    elastic : :class:`~scion.materials.micropolar.MicropolarHooke`
    cohesion : ``c``. With ``friction = 0`` this is the von Mises yield stress
        through ``k = 6 c cos(phi)/(3 - sin(phi))``, which at ``phi = 0`` is
        ``2 c``.
    friction, dilation : ``phi`` and ``psi`` in **degrees**. ``psi < phi`` is
        non-associated flow and is what produces structural softening without a
        softening law; ``psi = phi`` is associated and does not localise.
    hardening : ``H``, the slope of ``Lambda = H chi``. Zero is
        perfectly plastic, negative is true material softening (which this model
        does not need -- see the module docstring).
    plastic_length : ``l_p``, the length that makes ``nu : nu`` commensurate
        with ``s : s``. Defaults to the elastic bending length, following the
        reference's ad hoc ``l_b = l_p``; it is a *separate* physical parameter
        and the equality is a convenience, not a result.

    Known limit: the apex
    ---------------------
    A trial state whose pressure exceeds ``k / alpha_DP`` sits beyond the tip of
    the Drucker-Prager cone, where the smooth surface has **no admissible
    return** -- the flow direction points away from every point of the surface.
    Measured on random trial states at 119 times the yield limit: exactly the
    one point past the apex fails and all 39 others converge. The return map
    names the apex in its error rather than reporting generic non-convergence,
    because the remedy is a smaller load step or an apex return, not more
    iterations. This is the same refusal the classical Drucker-Prager in
    ``materials/plasticity/`` already carries.

    Away from the apex the basin is wide: measured, trial states at **35 to 90
    times** the yield limit still return, which is far beyond any load step a
    global solve would take.
    """

    #: unknown layout of the local problem: U^e, Gamma^e, chi, dgamma
    n_unknowns = 20

    def __init__(self, elastic: MicropolarHooke, cohesion: float,
                 friction: float = 0.0, dilation: float = 0.0,
                 hardening: float = 0.0, plastic_length: float | None = None):
        self.elastic = elastic
        phi, psi = np.radians(friction), np.radians(dilation)
        if psi > phi + 1e-12:
            raise ValueError(
                f"dilation {dilation} exceeds friction {friction}: the flow "
                f"rule would generate energy rather than dissipate it")
        self.friction, self.dilation = float(friction), float(dilation)
        self.alpha_dp = 6.0 * np.sin(phi) / (3.0 - np.sin(phi))
        self.beta_dp = 6.0 * np.sin(psi) / (3.0 - np.sin(psi))
        # cohesion may be per-particle: a shear-band study needs a weakened
        # row to seed localisation, and a uniform body localises wherever
        # round-off happens to break the symmetry, which is not comparable
        # between meshes
        self.k = 6.0 * np.asarray(cohesion, dtype=float) * np.cos(phi) \
            / (3.0 - np.sin(phi))
        #: scalar reference for tolerances and floors, which must not vary
        self._k_ref = float(np.max(self.k))
        self.H = float(hardening)
        self.l_p = float(plastic_length if plastic_length is not None
                         else elastic.length_bending)
        if self.l_p <= 0.0:
            raise ValueError(
                "plastic_length must be positive: it is what makes a couple "
                "stress commensurate with a stress inside the yield invariant, "
                "and the elastic bending length is zero for this material")
        self._D = elastic.D
        self._Dc = elastic.D_curvature

    def __repr__(self) -> str:
        return (f"MicropolarDruckerPrager(k={self._k_ref:.4g}, "
                f"phi={self.friction:g}, psi={self.dilation:g}, "
                f"H={self.H:g}, l_p={self.l_p:.4g})")

    # ------------------------------------------------------------------
    def _q_floor(self, q):
        """``q`` bounded away from zero *relative to the yield stress*.

        ``1/q^2`` appears throughout the Jacobians and a stress-free point has
        ``q = 0`` exactly. An absolute floor like 1e-300 makes ``1/q^2``
        overflow to ``inf``, and ``inf * 0`` is ``nan`` -- which then spreads
        from a point that was elastic and whose derivatives were never going to
        be used. A floor of ``1e-12 k`` is dimensionally sound and far below any
        state that can yield.
        """
        return np.maximum(q, 1e-12 * self._k_ref)

    def stresses(self, Q, F, Ue, Gam_e):
        """``(tau, nu)``: Kirchhoff stress and couple-stress.

        Note the asymmetry, which follows from the Coleman-Noll identification
        and is not a slip: ``tau`` is pushed forward with the *elastic*
        deformation gradient and ``nu`` with the **total** one.
        """
        B = np.einsum("abgp,ngp->nab", self._D, Ue - np.eye(3))
        S = np.einsum("abgp,ngp->nab", self._Dc, Gam_e)
        Fe = np.einsum("nia,nab->nib", Q, Ue)
        tau = np.einsum("nia,nab,njb->nij", Q, B, Fe)
        nu = np.einsum("nia,nab,njb->nij", Q, S, F)
        return tau, nu, B, S, Fe

    def invariants(self, tau, nu):
        """``(p, s, q)``. ``s`` is the *symmetrised* deviator -- the yield
        surface does not see the skew part of the stress, only the couple
        stress balances that."""
        p = np.einsum("nii->n", tau) / 3.0
        s = 0.5 * (tau + tau.transpose(0, 2, 1)) - p[:, None, None] * np.eye(3)
        q2 = 1.5 * (np.einsum("nij,nij->n", s, s)
                    + np.einsum("nij,nij->n", nu, nu) / self.l_p ** 2)
        return p, s, np.sqrt(np.maximum(q2, 0.0))

    def flow(self, s, nu, q):
        """``(dg/dtau, dg/dnu)``. The first is symmetric by construction."""
        qs = self._q_floor(q)[:, None, None]
        n_tau = 1.5 * s / qs + (self.beta_dp / 3.0) * np.eye(3)
        n_nu = 1.5 * nu / (self.l_p ** 2 * qs)
        return n_tau, n_nu

    def yield_function(self, p, q, chi, k=None):
        return q + self.alpha_dp * p - ((self.k if k is None else k)
                                        + self.H * chi)

    # ------------------------------------------------------------------
    def residuals(self, Q, F, Fe_tr, Gam_tr, chi_old, Ue, Gam_e, chi, dg,
                  k=None):
        """``(r_U, r_Gamma, r_chi, f)`` plus the working quantities the
        Jacobian needs, so the two never disagree about the state."""
        tau, nu, B, S, Fe = self.stresses(Q, F, Ue, Gam_e)
        p, s, q = self.invariants(tau, nu)
        n_tau, n_nu = self.flow(s, nu, q)

        expX, dexp = dexp_symmetric(-dg[:, None, None] * n_tau)
        r_U = Ue - np.einsum("nja,njk,nki->nai", Q, expX, Fe_tr)
        r_G = (Gam_e + dg[:, None, None]
               * np.einsum("nir,nij,nje->nre", Q, n_nu, F) - Gam_tr)
        r_chi = chi - chi_old - dg
        f = self.yield_function(p, q, chi, k)
        work = dict(tau=tau, nu=nu, B=B, S=S, Fe=Fe, p=p, s=s, q=q,
                    n_tau=n_tau, n_nu=n_nu, expX=expX, dexp=dexp)
        return r_U, r_G, r_chi, f, work

    def jacobian(self, Q, F, Fe_tr, Ue, Gam_e, dg, work):
        """``dr/dX``, ``(n, 20, 20)``, with ``X = [U^e, Gamma^e, chi, dgamma]``.

        Derived rather than transcribed. Two entries of the reference's
        Appendix B do not survive a finite-difference check as printed, and
        both are recorded in the tests: its Eq. B.14 is missing the ``1/3`` that
        ``dp/dU^e`` carries, and Eq. 148 of the main text has a wrong Levi-Civita
        index. Everything here is checked against central differences on
        :meth:`residuals`, which is the only thing that settles such questions.
        """
        n = Ue.shape[0]
        eye = np.eye(3)
        D, Dc, l2 = self._D, self._Dc, self.l_p ** 2
        B, Fe, s, nu, q = work["B"], work["Fe"], work["s"], work["nu"], work["q"]
        n_tau, n_nu, dexp = work["n_tau"], work["n_nu"], work["dexp"]
        qs = self._q_floor(q)

        # dp/dU^e. The 1/3 is real: p = tr(tau)/3.
        P1 = (2.0 * B + np.einsum("aatp->tp", D)[None]) / 3.0

        # ds/dU^e, symmetrised in its first pair
        t1 = np.einsum("nka,nlb,abtp->nkltp", Q, Fe, D)
        t2 = np.einsum("nka,nap,nlt->nkltp", Q, B, Q)
        A1 = 0.5 * (t1 + t1.transpose(0, 2, 1, 3, 4)
                    + t2 + t2.transpose(0, 2, 1, 3, 4))
        A1 = A1 - np.einsum("kl,ntp->nkltp", eye, P1)

        A2 = 1.5 * np.einsum("nkl,nkltp->ntp", s, A1) / qs[:, None, None]
        NG = np.einsum("nia,abtp,njb->nijtp", Q, Dc, F)
        A3 = (1.5 / l2) * np.einsum("nij,nijtp->ntp", nu, NG) / qs[:, None, None]

        inv_q = (1.0 / qs)[:, None, None, None, None]
        inv_q2 = (1.0 / qs ** 2)[:, None, None, None, None]
        dNt_dU = 1.5 * A1 * inv_q - 1.5 * np.einsum(
            "nkl,ntp->nkltp", s, A2) * inv_q2
        dNt_dG = -1.5 * np.einsum("nkl,ntp->nkltp", s, A3) * inv_q2
        dNn_dU = -(1.5 / l2) * np.einsum("nij,ntp->nijtp", nu, A2) * inv_q2
        dNn_dG = ((1.5 / l2) * NG * inv_q
                  - (1.5 / l2) * np.einsum("nij,ntp->nijtp", nu, A3) * inv_q2)

        # W absorbs the exponential's derivative and both push-forwards
        W = np.einsum("nja,njkmp,nki->naimp", Q, dexp, Fe_tr)
        g = dg[:, None, None, None, None]

        C = np.zeros((n, 20, 20))
        ident = np.einsum("at,ip->aitp", eye, eye).reshape(9, 9)
        C[:, 0:9, 0:9] = ident + (g * np.einsum(
            "naimp,nmptq->naitq", W, dNt_dU)).reshape(n, 9, 9)
        C[:, 0:9, 9:18] = (g * np.einsum(
            "naimp,nmptq->naitq", W, dNt_dG)).reshape(n, 9, 9)
        C[:, 0:9, 19] = np.einsum("naimp,nmp->nai", W, n_tau).reshape(n, 9)

        C[:, 9:18, 0:9] = (g * np.einsum(
            "nir,nijtq,nje->nretq", Q, dNn_dU, F)).reshape(n, 9, 9)
        C[:, 9:18, 9:18] = ident + (g * np.einsum(
            "nir,nijtq,nje->nretq", Q, dNn_dG, F)).reshape(n, 9, 9)
        C[:, 9:18, 19] = np.einsum("nir,nij,nje->nre", Q, n_nu, F).reshape(n, 9)

        C[:, 18, 18] = 1.0
        C[:, 18, 19] = -1.0

        C[:, 19, 0:9] = (A2 + self.alpha_dp * P1).reshape(n, 9)
        C[:, 19, 9:18] = A3.reshape(n, 9)
        C[:, 19, 18] = -self.H
        return C

    # ------------------------------------------------------------------
    def return_map(self, Q, F, Fe_tr, Gam_tr, chi_old,
                   tol: float = 1.0e-9, max_iter: int = 50) -> dict:
        """Elastic predictor, plastic corrector (reference Algorithm 1).

        Returns a dict with ``Ue``, ``Gam_e``, ``chi``, ``dgamma``, ``tau``,
        ``nu``, the boolean ``plastic`` mask and the iteration count. Points
        below yield are returned untouched, which is the whole point of the
        predictor: no work and no round-off for the elastic majority.

        The convergence test is the reference's, and it is deliberately in two
        parts: ``||r||`` and ``f`` differ by many orders of magnitude -- one is
        a strain and the other a stress -- so a single combined tolerance is
        dominated by whichever carries the larger units and silently stops
        testing the other.
        """
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        Ue = np.einsum("nja,njk->nak", Q, Fe_tr)      # trial: Q^T (F^e)^tr
        Gam_e = np.array(Gam_tr, dtype=float, copy=True)
        chi = np.array(chi_old, dtype=float, copy=True)
        dg = np.zeros(n)

        k_all = np.broadcast_to(np.asarray(self.k, dtype=float), (n,))
        r_U, r_G, r_chi, f, work = self.residuals(
            Q, F, Fe_tr, Gam_tr, chi_old, Ue, Gam_e, chi, dg, k_all)
        plastic = f > k_all * tol
        iterations = 0
        if not plastic.any():
            return dict(Ue=Ue, Gam_e=Gam_e, chi=chi, dgamma=dg,
                        tau=work["tau"], nu=work["nu"], plastic=plastic,
                        iterations=0, converged=True)

        idx = np.flatnonzero(plastic)
        Qa, Fa, Fea, Ga, ca = (Q[idx], F[idx], Fe_tr[idx], Gam_tr[idx],
                               chi_old[idx])
        Ua, Gea, cha, dga = Ue[idx], Gam_e[idx], chi[idx], dg[idx]
        ka = k_all[idx]
        converged = False

        # The three residual blocks have different units -- a stretch, a
        # curvature and a stress -- so a merit function built on their raw sum
        # is dominated by whichever carries the largest scale, and the line
        # search below would then be steered by one block alone.
        g_scale = np.maximum(np.abs(Ga).max(axis=(1, 2)), 1.0)

        def merit(rU, rG, rc, fv):
            return (np.einsum("nij,nij->n", rU, rU)
                    + np.einsum("nij,nij->n", rG, rG) / g_scale ** 2
                    + rc ** 2 + (fv / ka) ** 2)

        for iterations in range(1, max_iter + 1):
            r_U, r_G, r_chi, f, work = self.residuals(
                Qa, Fa, Fea, Ga, ca, Ua, Gea, cha, dga, ka)
            r = np.concatenate([r_U.reshape(-1, 9), r_G.reshape(-1, 9),
                                r_chi[:, None], f[:, None]], axis=1)
            p_bar = np.sqrt(np.einsum("ni,ni->n", r[:, :19], r[:, :19]))
            if (p_bar < tol).all() and (np.abs(f) < ka * tol).all():
                converged = True
                break

            C = self.jacobian(Qa, Fa, Fea, Ua, Gea, dga, work)
            dX = np.linalg.solve(C, -r)

            # Damped step. The undamped iteration limit-cycles rather than
            # diverging when the trial state is far outside the surface --
            # measured, it survives ~90x the yield limit undamped and oscillates
            # above that. Halving is enough; a full Wolfe search is not worth
            # its evaluations for a 20-unknown local problem.
            m0 = merit(r_U, r_G, r_chi, f)
            step = np.ones(len(idx))
            for _ in range(6):
                a = step[:, None]
                trial = self.residuals(
                    Qa, Fa, Fea, Ga, ca,
                    Ua + (a * dX[:, 0:9]).reshape(-1, 3, 3),
                    Gea + (a * dX[:, 9:18]).reshape(-1, 3, 3),
                    cha + step * dX[:, 18], dga + step * dX[:, 19], ka)
                bad = merit(*trial[:4]) > m0
                if not bad.any():
                    break
                step = np.where(bad, 0.5 * step, step)

            a = step[:, None]
            Ua = Ua + (a * dX[:, 0:9]).reshape(-1, 3, 3)
            Gea = Gea + (a * dX[:, 9:18]).reshape(-1, 3, 3)
            cha = cha + step * dX[:, 18]
            dga = dga + step * dX[:, 19]

        if not converged:
            stuck = np.flatnonzero((p_bar >= tol)
                                   | (np.abs(f) >= ka * tol))
            hint = ""
            if self.alpha_dp > 0.0:
                apex = self._k_ref / self.alpha_dp
                _, _, _, _, w0 = self.residuals(
                    Qa[stuck], Fa[stuck], Fea[stuck], Ga[stuck], ca[stuck],
                    np.einsum("nja,njk->nak", Qa[stuck], Fea[stuck]),
                    Ga[stuck], ca[stuck], np.zeros(len(stuck)), ka[stuck])
                beyond = int((w0["p"] > apex).sum())
                if beyond:
                    hint = (f"; {beyond} of them have a trial pressure beyond "
                            f"the cone apex at p = k/alpha_DP = {apex:.3e}, "
                            f"where the smooth Drucker-Prager surface has no "
                            f"admissible return at all. That is the geometry, "
                            f"not the solver -- cut the load step, or use a "
                            f"formulation with an apex return")
            raise RuntimeError(
                f"the micropolar return map did not converge in {max_iter} "
                f"iterations on {len(stuck)} of {len(idx)} points: "
                f"max ||r|| = {p_bar.max():.3e}, max |f| = "
                f"{np.abs(f).max():.3e} against k = {self._k_ref:.3e}{hint}")

        Ue[idx], Gam_e[idx], chi[idx], dg[idx] = Ua, Gea, cha, dga
        tau, nu, *_ = self.stresses(Q, F, Ue, Gam_e)
        return dict(Ue=Ue, Gam_e=Gam_e, chi=chi, dgamma=dg, tau=tau, nu=nu,
                    plastic=plastic, iterations=iterations, converged=True)

    # ------------------------------------------------------------------
    # the algorithmically consistent linearisation
    # ------------------------------------------------------------------
    #: layout of the constitutive model's *inputs*, for the auxiliary Jacobian
    input_slices = dict(F=slice(0, 9), Q=slice(9, 18), Fe_tr=slice(18, 27),
                        Gam_tr=slice(27, 36), chi_old=slice(36, 37))
    n_inputs = 37

    def auxiliary_jacobian(self, Q, F, Fe_tr, Ue, Gam_e, dg, work):
        """``T = dr/dI`` for ``I = {F, Q, (F^e)^tr, (Gamma^e)^tr, chi_old}``.

        ``(n, 20, 37)``. Combined with :meth:`jacobian` this gives the
        algorithmic Jacobian ``P = -C^-1 T`` -- the derivative of the return
        map's *outputs* with respect to its *inputs*, which is what makes the
        global tangent consistent through a plastic step.

        Two structural facts that cut most of the work, and are worth stating
        because they are not obvious from the residual forms:

        * **``tau`` does not depend on ``F``.** The Kirchhoff stress is pushed
          forward with the *elastic* deformation gradient, and ``F^e = Q U^e``
          is built from an unknown, not an input. So ``F`` reaches ``r_U`` only
          through ``q``, and only because the yield invariant mixes in ``nu``.
        * **``(F^e)^tr`` and ``(Gamma^e)^tr`` each appear in exactly one
          residual**, and only linearly.

        Derived rather than transcribed, then checked against central
        differences. Most of it reproduces the reference's Appendix C exactly --
        ``A4`` (Eq. C.3), ``A5`` (Eq. C.4), Eqs. C.1 and C.5 all fall out of
        this derivation unchanged, as do the corresponding lines of the
        published implementation.

        **One term does not.** The ``1/q`` piece of ``dr_Gamma/dF`` carries
        ``(S F^T F)``, because ``Q^T nu F = S F^T F`` -- not ``S``. The
        reference's Eq. C.6 and its ``constitutive/DP.m`` line 209 both use
        plain ``S``; its *own* line 211, for the ``Q``-block of the same
        quantity, uses ``(S*F')*F``, which is right. Measured against central
        differences: the derived form 5.4e-9, the printed form **1.1e-1**.
        Being a tangent error it costs the *global* Newton rate rather than the
        answer, which is how an implementation carries it and still looks fine.
        """
        n = Ue.shape[0]
        eye = np.eye(3)
        l2 = self.l_p ** 2
        tau, nu, S = work["tau"], work["nu"], work["S"]
        s, q, n_nu = work["s"], work["q"], work["n_nu"]
        expX, dexp = work["expX"], work["dexp"]
        qs = self._q_floor(q)

        QtT = np.einsum("nia,nij->naj", Q, tau)          # Q^T tau
        TQ = np.einsum("nij,njb->nib", tau, Q)           # tau Q
        Qtn = np.einsum("nia,nij->naj", Q, nu)           # Q^T nu
        QS = np.einsum("nia,nab->nib", Q, S)             # Q S

        # ds/dQ  (reference Eq. C.3)
        A4 = 0.5 * (np.einsum("km,ntl->nkltm", eye, QtT).transpose(0,1,2,4,3)
                    + np.einsum("lm,nkt->nkltm", eye, TQ).transpose(0,1,2,4,3)
                    + np.einsum("lm,ntk->nkltm", eye, QtT).transpose(0,1,2,4,3)
                    + np.einsum("km,nlt->nkltm", eye, TQ).transpose(0,1,2,4,3))
        trace_dQ = QtT.transpose(0, 2, 1) + TQ            # (Q^T tau)_tm + (tau Q)_mt
        A4 = A4 - (1.0 / 3.0) * np.einsum("kl,nmt->nklmt", eye, trace_dQ)
        dp_dQ = trace_dQ / 3.0

        # dq/dF and dq/dQ  (reference Eq. C.4)
        dq_dF = (1.5 / l2) * np.einsum("nim,nit->nmt", nu, QS) / qs[:, None, None]
        A5 = 1.5 * (np.einsum("nkl,nklmt->nmt", s, A4)
                    + np.einsum("nmj,ntj->nmt", nu, Qtn) / l2) / qs[:, None, None]

        iq = (1.0 / qs)[:, None, None, None, None]
        iq2 = (1.0 / qs ** 2)[:, None, None, None, None]
        dNt_dQ = 1.5 * A4 * iq - 1.5 * np.einsum("nkl,nmt->nklmt", s, A5) * iq2
        dNt_dF = -1.5 * np.einsum("nkl,nmt->nklmt", s, dq_dF) * iq2
        dNn_dQ = ((1.5 / l2) * np.einsum("im,ntj->nijmt", eye, Qtn) * iq
                  - (1.5 / l2) * np.einsum("nij,nmt->nijmt", nu, A5) * iq2)
        dNn_dF = ((1.5 / l2) * np.einsum("jm,nit->nijmt", eye, QS) * iq
                  - (1.5 / l2) * np.einsum("nij,nmt->nijmt", nu, dq_dF) * iq2)

        W = np.einsum("nja,njkmp,nki->naimp", Q, dexp, Fe_tr)
        g = dg[:, None, None, None, None]

        T = np.zeros((n, 20, self.n_inputs))
        sl = self.input_slices

        # --- r_U -------------------------------------------------------
        T[:, 0:9, sl["F"]] = (g * np.einsum(
            "naikl,nklmt->naimt", W, dNt_dF)).reshape(n, 9, 9)
        T[:, 0:9, sl["Q"]] = (
            g * np.einsum("naikl,nklmt->naimt", W, dNt_dQ)
            - np.einsum("at,nmi->naimt", eye,
                        np.einsum("nmk,nki->nmi", expX, Fe_tr))[:, :, :, :, :]
        ).reshape(n, 9, 9)
        T[:, 0:9, sl["Fe_tr"]] = -np.einsum(
            "nja,njk,ip->naikp", Q, expX, eye).reshape(n, 9, 9)

        # --- r_Gamma ---------------------------------------------------
        T[:, 9:18, sl["F"]] = (
            g * (np.einsum("nir,nijmt,nje->nremt", Q, dNn_dF, F)
                 + np.einsum("nir,nim,et->nremt", Q, n_nu, eye))
        ).reshape(n, 9, 9)
        T[:, 9:18, sl["Q"]] = (
            g * (np.einsum("nir,nijmt,nje->nremt", Q, dNn_dQ, F)
                 + np.einsum("rt,nmj,nje->nremt", eye, n_nu, F))
        ).reshape(n, 9, 9)
        T[:, 9:18, sl["Gam_tr"]] = -np.einsum(
            "ra,eb->reab", eye, eye).reshape(9, 9)[None]

        # --- r_chi and f -----------------------------------------------
        T[:, 18, sl["chi_old"]] = -1.0
        T[:, 19, sl["F"]] = dq_dF.reshape(n, 9)
        T[:, 19, sl["Q"]] = (A5 + self.alpha_dp * dp_dQ).reshape(n, 9)
        return T

    def algorithmic_jacobian(self, Q, F, Fe_tr, Ue, Gam_e, dg, work):
        """``P = -C^-1 T``, ``(n, 20, 37)``.

        Rows are ``{U^e, Gamma^e, chi, dgamma}`` and columns the inputs. Valid
        only **at convergence** of the local problem, where ``r`` is not merely
        small but stationary -- that is what lets ``dr = 0`` be differentiated
        to give ``C P = -T`` in the first place.
        """
        C = self.jacobian(Q, F, Fe_tr, Ue, Gam_e, dg, work)
        T = self.auxiliary_jacobian(Q, F, Fe_tr, Ue, Gam_e, dg, work)
        return -np.linalg.solve(C, T)
