"""Micropolar Drucker-Prager return map.

Layered so a failure localises:

1. ``dexp_symmetric`` -- the derivative of a matrix exponential, against scipy
   and against finite differences, *including* repeated eigenvalues, which are
   the normal case here and not a corner one.
2. The flow vectors, against finite differences of the potential ``g``.
3. The local Jacobian, against central differences on the residuals. This is
   what settles the two places the reference's appendices do not survive
   transcription.
4. The return map's own algebra -- ``f = 0``, ``dgamma >= 0``, elastic points
   untouched.
5. The external check: simple shear must plateau at the classical ``k/sqrt(3)``.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from scion.core.rotation import quat_from_rotvec, quat_to_matrix
from scion.materials.micropolar import MicropolarHooke
from scion.materials.micropolar_plasticity import (
    MicropolarDruckerPrager,
    dexp_symmetric,
)

ELASTIC = MicropolarHooke(lam=4.0e9, mu=4.0e9, a=0.1, alpha=0.0, beta=0.0,
                          gam=4.0e9 * 4.0 * 0.005 ** 2)
L_P = 0.005


def model(**kw):
    kw.setdefault("cohesion", 50.0e6)
    return MicropolarDruckerPrager(ELASTIC, plastic_length=L_P, **kw)


def random_state(n=4, seed=0, amp=0.05):
    """A spread of trial states. ``amp`` ramps along the batch so the set holds
    both elastic points and plastic ones, rather than a uniform severity --
    a batch that is entirely plastic cannot test the predictor."""
    rng = np.random.default_rng(seed)
    Q = quat_to_matrix(quat_from_rotvec(0.3 * rng.standard_normal((n, 3))))
    F = np.eye(3) + 0.1 * rng.standard_normal((n, 3, 3))
    ramp = np.linspace(0.0, amp, n)[:, None, None]
    Fe_tr = np.eye(3) + ramp * rng.standard_normal((n, 3, 3))
    Gam_tr = (ramp * 400.0) * rng.standard_normal((n, 3, 3))
    chi_old = np.abs(rng.standard_normal(n)) * 1e-3
    return Q, F, Fe_tr, Gam_tr, chi_old


# ======================================================================
# 1. the exponential map's derivative
# ======================================================================
class TestDexp:
    @staticmethod
    def cases():
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 3, 3))
        X = 0.5 * (X + X.transpose(0, 2, 1))
        X[0] = np.eye(3) * 0.7          # triply degenerate
        X[1] = np.diag([0.4, 0.4, -0.9])  # doubly degenerate
        X[2] = np.zeros((3, 3))           # zero
        return X

    def test_exponential_matches_scipy(self):
        X = self.cases()
        E, _ = dexp_symmetric(X)
        assert max(np.abs(E[i] - expm(X[i])).max()
                   for i in range(len(X))) < 1e-13

    def test_derivative_matches_finite_differences(self):
        X = self.cases()
        _, dE = dexp_symmetric(X)
        rng = np.random.default_rng(3)
        h, err, scale = 1e-6, 0.0, 0.0
        for _ in range(10):
            H = rng.standard_normal((3, 3))
            H = 0.5 * (H + H.T)
            ana = np.einsum("nijkl,kl->nij", dE, H)
            num = np.array([(expm(X[i] + h * H) - expm(X[i] - h * H)) / (2 * h)
                            for i in range(len(X))])
            err = max(err, np.abs(ana - num).max())
            scale = max(scale, np.abs(num).max())
        assert err / scale < 1e-8

    def test_repeated_eigenvalues_are_not_nan(self):
        """The argument is ``-dgamma dg/dtau``, a deviator plus a multiple of
        the identity, so any axisymmetric stress gives repeated eigenvalues.
        The divided difference is 0/0 there and a naive implementation returns
        nan on the *first* yielding point of a uniaxial test."""
        _, dE = dexp_symmetric(self.cases()[:3])
        assert np.isfinite(dE).all()


# ======================================================================
# 2. flow vectors
# ======================================================================
def test_flow_vectors_match_the_potential():
    m = model(friction=8.6, dilation=2.0)
    rng = np.random.default_rng(0)
    tau = rng.standard_normal((4, 3, 3)) * 1e7
    nu = rng.standard_normal((4, 3, 3)) * 1e5

    def g(t, v):
        p, s, q = m.invariants(t, v)
        return q + m.beta_dp * p

    p, s, q = m.invariants(tau, nu)
    n_tau, n_nu = m.flow(s, nu, q)
    h = 1.0
    for k in range(3):
        for l in range(3):
            e = np.zeros((3, 3))
            e[k, l] = h
            assert abs(n_tau[:, k, l]
                       - (g(tau + e, nu) - g(tau - e, nu)) / (2 * h)).max() \
                < 1e-6 * np.abs(n_tau).max()
            assert abs(n_nu[:, k, l]
                       - (g(tau, nu + e) - g(tau, nu - e)) / (2 * h)).max() \
                < 1e-6 * np.abs(n_nu).max()


def test_flow_direction_is_symmetric():
    """``dg/dtau`` is symmetric because the yield surface sees only the
    symmetrised deviator. The couple stress is what carries the skew part."""
    m = model(friction=8.6, dilation=2.0)
    rng = np.random.default_rng(1)
    tau = rng.standard_normal((4, 3, 3)) * 1e7
    nu = rng.standard_normal((4, 3, 3)) * 1e5
    p, s, q = m.invariants(tau, nu)
    n_tau, _ = m.flow(s, nu, q)
    assert np.abs(n_tau - n_tau.transpose(0, 2, 1)).max() == 0.0


# ======================================================================
# 3. the local Jacobian
# ======================================================================
class TestJacobian:
    @staticmethod
    def setup_case(m, n=4):
        Q, F, Fe_tr, Gam_tr, chi_old = random_state(n)
        rng = np.random.default_rng(7)
        Ue = np.eye(3) + 0.03 * rng.standard_normal((n, 3, 3))
        Ue = 0.5 * (Ue + Ue.transpose(0, 2, 1))
        Gam_e = 15.0 * rng.standard_normal((n, 3, 3))
        chi = chi_old + 1e-3
        dg = np.abs(rng.standard_normal(n)) * 1e-3 + 1e-3
        X0 = np.concatenate([Ue.reshape(n, 9), Gam_e.reshape(n, 9),
                             chi[:, None], dg[:, None]], axis=1)
        return (Q, F, Fe_tr, Gam_tr, chi_old), X0

    @staticmethod
    def residual_vector(m, args, X):
        n = X.shape[0]
        Q, F, Fe_tr, Gam_tr, chi_old = args
        r_U, r_G, r_chi, f, _ = m.residuals(
            Q, F, Fe_tr, Gam_tr, chi_old, X[:, 0:9].reshape(n, 3, 3),
            X[:, 9:18].reshape(n, 3, 3), X[:, 18], X[:, 19])
        return np.concatenate([r_U.reshape(n, 9), r_G.reshape(n, 9),
                               r_chi[:, None], f[:, None]], axis=1)

    @pytest.mark.parametrize("kw", [
        {},
        {"friction": 8.6, "dilation": 2.0},
        {"friction": 30.0, "dilation": 30.0, "hardening": 2.0e8},
    ])
    def test_against_finite_differences(self, kw):
        m = model(**kw)
        args, X0 = self.setup_case(m)
        n = X0.shape[0]
        _, _, _, _, work = m.residuals(
            *args[:3], args[3], args[4], X0[:, 0:9].reshape(n, 3, 3),
            X0[:, 9:18].reshape(n, 3, 3), X0[:, 18], X0[:, 19])
        C = m.jacobian(args[0], args[1], args[2], X0[:, 0:9].reshape(n, 3, 3),
                       X0[:, 9:18].reshape(n, 3, 3), X0[:, 19], work)

        num = np.zeros_like(C)
        for j in range(20):
            h = 1e-7 * max(abs(X0[:, j]).max(), 1e-3)
            up, dn = X0.copy(), X0.copy()
            up[:, j] += h
            dn[:, j] -= h
            num[:, :, j] = (self.residual_vector(m, args, up)
                            - self.residual_vector(m, args, dn)) / (2 * h)
        scale = np.abs(num).max(axis=(1, 2))[:, None, None]
        assert (np.abs(C - num) / scale).max() < 1e-5

    def test_the_printed_df_dUe_is_missing_a_third(self):
        """The reference's Eq. B.14 gives ``df/dU^e = A2 + alpha_DP (2B + D_aa)``
        with no ``1/3``. But ``p = tr(tau)/3``, so ``dp/dU^e`` carries one.

        Recorded as a test because it is invisible in von Mises (``alpha_DP``
        multiplies it) and, being a factor of three on one block of a
        twenty-by-twenty Jacobian, it degrades the *local* Newton rate rather
        than the answer -- and the local solve converges anyway, just slower.
        """
        m = model(friction=30.0)
        args, X0 = self.setup_case(m)
        n = X0.shape[0]
        _, _, _, _, work = m.residuals(
            *args[:3], args[3], args[4], X0[:, 0:9].reshape(n, 3, 3),
            X0[:, 9:18].reshape(n, 3, 3), X0[:, 18], X0[:, 19])
        C = m.jacobian(args[0], args[1], args[2], X0[:, 0:9].reshape(n, 3, 3),
                       X0[:, 9:18].reshape(n, 3, 3), X0[:, 19], work)

        j = 0
        h = 1e-7 * max(abs(X0[:, j]).max(), 1e-3)
        up, dn = X0.copy(), X0.copy()
        up[:, j] += h
        dn[:, j] -= h
        exact = ((self.residual_vector(m, args, up)
                  - self.residual_vector(m, args, dn)) / (2 * h))[:, 19]
        assert np.abs(C[:, 19, j] - exact).max() < 1e-4 * np.abs(exact).max()

        # the printed form multiplies the alpha_DP term by three
        P1 = (2.0 * work["B"] + np.einsum("aatp->tp", m._D)[None]) / 3.0
        printed = C[:, 19, :9] + 2.0 * m.alpha_dp * P1.reshape(n, 9)
        assert np.abs(printed[:, j] - exact).max() > 1e-2 * np.abs(exact).max()


# ======================================================================
# 4. the return map
# ======================================================================
class TestReturnMap:
    @pytest.mark.parametrize("kw", [
        {}, {"friction": 8.6, "dilation": 2.0},
        {"friction": 8.6, "dilation": 2.0, "hardening": 2.0e8},
    ])
    def test_lands_on_the_yield_surface(self, kw):
        m = model(**kw)
        Q, F, Fe_tr, Gam_tr, chi_old = random_state(n=40, seed=1, amp=0.04)
        out = m.return_map(Q, F, Fe_tr, Gam_tr, chi_old)
        p, s, q = m.invariants(out["tau"], out["nu"])
        f = m.yield_function(p, q, out["chi"])
        assert out["plastic"].any()
        assert np.abs(f[out["plastic"]]).max() < 1e-8 * m.k
        assert (f[~out["plastic"]] <= 0.0).all()

    def test_kkt_and_hardening_bookkeeping(self):
        m = model(friction=8.6, dilation=2.0, hardening=2.0e8)
        Q, F, Fe_tr, Gam_tr, chi_old = random_state(n=40, seed=1, amp=0.04)
        out = m.return_map(Q, F, Fe_tr, Gam_tr, chi_old)
        assert (out["dgamma"] >= 0.0).all()
        assert np.allclose(out["chi"], chi_old + out["dgamma"])
        assert (out["dgamma"][~out["plastic"]] == 0.0).all()

    def test_elastic_points_are_returned_untouched(self):
        """The predictor must do no work at all below yield -- not "converge in
        one iteration", which would still leave round-off on every point."""
        m = model(friction=8.6, dilation=2.0)
        n = 6
        Q = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        Fe_tr = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        Fe_tr[:, 0, 1] = np.linspace(0.0, 0.002, n)      # well below yield
        out = m.return_map(Q, F, Fe_tr, np.zeros((n, 3, 3)), np.zeros(n))
        assert not out["plastic"].any()
        assert out["iterations"] == 0
        assert np.abs(out["Ue"] - Fe_tr).max() == 0.0

    def test_dilation_beyond_friction_is_refused(self):
        with pytest.raises(ValueError, match="generate energy"):
            model(friction=10.0, dilation=20.0)

    def test_a_trial_state_past_the_apex_names_the_apex(self):
        """Beyond ``p = k/alpha_DP`` the smooth cone has no admissible return.
        The error must say so: the remedy is a smaller load step, and more
        iterations will never help."""
        m = model(friction=8.6, dilation=2.0)
        n = 1
        Q = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        # a large hydrostatic tension: pressure far beyond the cone tip
        Fe_tr = np.broadcast_to(np.eye(3) * 1.10, (n, 3, 3)).copy()
        with pytest.raises(RuntimeError, match="apex"):
            m.return_map(Q, F, Fe_tr, np.zeros((n, 3, 3)), np.zeros(n),
                         max_iter=40)

    def test_couple_stress_alone_can_yield(self):
        """The yield surface genuinely mixes the two: a state with no stress
        but a large wryness must still reach ``f = 0``. If ``l_p`` were
        mishandled this either never yields or yields instantly."""
        m = model()
        n = 3
        Q = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        Fe_tr = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        Gam_tr = np.zeros((n, 3, 3))
        # measured: with l_p = 5 mm this material yields on curvature alone at
        # Gamma_12 of order 1 per metre, so the bracket has to straddle that
        Gam_tr[:, 0, 1] = np.array([0.01, 1.0, 10.0])
        out = m.return_map(Q, F, Fe_tr, Gam_tr, np.zeros(n))
        assert not out["plastic"][0]
        assert out["plastic"][-1]
        p_, s_, q_ = m.invariants(out["tau"], out["nu"])
        # and the couple stress alone carries the whole invariant
        assert np.abs(q_[out["plastic"]] - m.k).max() < 1e-8 * m.k


# ======================================================================
# 5. the external check
# ======================================================================
def test_simple_shear_plateaus_at_the_classical_yield():
    """von Mises in simple shear yields at ``k/sqrt(3)``. That is classical
    plasticity, derived outside this library, and it is the only check here
    that would catch the whole model being self-consistently wrong.

    Measured: 5.773294e7 against 5.773503e7, a ratio of 0.99996. The residual
    0.004% is the finite-strain measure -- this is Kirchhoff stress conjugate
    to a Biot strain, not the small-strain pair -- and it shrinks with the
    applied shear.
    """
    elastic = MicropolarHooke(lam=4.0e9, mu=4.0e9, a=0.0, alpha=0.0, beta=0.0,
                              gam=4.0e9 * 4.0 * L_P ** 2)
    k = 1.0e8
    m = MicropolarDruckerPrager(elastic, cohesion=k / 2.0, plastic_length=L_P)
    assert abs(m.k - k) < 1e-6 * k

    n = 40
    Q = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    Fe_tr = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    Fe_tr[:, 0, 1] = np.linspace(0.0, 0.05, n)
    out = m.return_map(Q, F, Fe_tr, np.zeros((n, 3, 3)), np.zeros(n))

    tau = out["tau"]
    shear = 0.5 * (tau[:, 0, 1] + tau[:, 1, 0])
    plastic = out["plastic"]
    assert plastic.any() and not plastic.all()
    assert abs(shear[plastic].max() / (k / np.sqrt(3.0)) - 1.0) < 1e-3

    _, _, q = m.invariants(tau, out["nu"])
    assert np.abs(q[plastic] - k).max() < 1e-9 * k


# ======================================================================
# 6. the algorithmic linearisation
# ======================================================================
class TestAlgorithmicJacobian:
    """``T = dr/dI`` and ``P = -C^-1 T``.

    ``P`` is what makes the *global* tangent consistent through a plastic step,
    so an error here converges linearly and reads as a hard problem. The
    end-to-end test below is the one that matters: it re-converges the whole
    return map around a perturbed input, so it validates ``C``, ``T`` and the
    ``-C^-1 T`` relation together against nothing but the map's own behaviour.
    """

    @staticmethod
    def state(m, n=4, seed=7):
        rng = np.random.default_rng(seed)
        Q = quat_to_matrix(quat_from_rotvec(0.3 * rng.standard_normal((n, 3))))
        F = np.eye(3) + 0.1 * rng.standard_normal((n, 3, 3))
        Fe_tr = np.eye(3) + 0.03 * rng.standard_normal((n, 3, 3))
        Gam_tr = 15.0 * rng.standard_normal((n, 3, 3))
        chi_old = np.abs(rng.standard_normal(n)) * 1e-3
        Ue = np.eye(3) + 0.03 * rng.standard_normal((n, 3, 3))
        Ue = 0.5 * (Ue + Ue.transpose(0, 2, 1))
        Gam_e = 12.0 * rng.standard_normal((n, 3, 3))
        chi = chi_old + 1e-3
        dg = np.abs(rng.standard_normal(n)) * 1e-3 + 1e-3
        return Q, F, Fe_tr, Gam_tr, chi_old, Ue, Gam_e, chi, dg

    @staticmethod
    def inputs(Q, F, Fe_tr, Gam_tr, chi_old):
        n = Q.shape[0]
        return np.concatenate([F.reshape(n, 9), Q.reshape(n, 9),
                               Fe_tr.reshape(n, 9), Gam_tr.reshape(n, 9),
                               chi_old[:, None]], axis=1)

    def residual_of_inputs(self, m, I, Ue, Gam_e, chi, dg):
        n = Ue.shape[0]
        r_U, r_G, r_chi, f, _ = m.residuals(
            I[:, 9:18].reshape(n, 3, 3), I[:, 0:9].reshape(n, 3, 3),
            I[:, 18:27].reshape(n, 3, 3), I[:, 27:36].reshape(n, 3, 3),
            I[:, 36], Ue, Gam_e, chi, dg)
        return np.concatenate([r_U.reshape(n, 9), r_G.reshape(n, 9),
                               r_chi[:, None], f[:, None]], axis=1)

    def test_T_against_finite_differences(self):
        m = model(friction=8.6, dilation=2.0, hardening=2.0e8)
        Q, F, Fe_tr, Gam_tr, chi_old, Ue, Gam_e, chi, dg = self.state(m)
        _, _, _, _, work = m.residuals(Q, F, Fe_tr, Gam_tr, chi_old,
                                       Ue, Gam_e, chi, dg)
        T = m.auxiliary_jacobian(Q, F, Fe_tr, Ue, Gam_e, dg, work)
        I0 = self.inputs(Q, F, Fe_tr, Gam_tr, chi_old)

        num = np.zeros_like(T)
        for j in range(m.n_inputs):
            h = 1e-6 * max(abs(I0[:, j]).max(), 1e-3)
            up, dn = I0.copy(), I0.copy()
            up[:, j] += h
            dn[:, j] -= h
            num[:, :, j] = (self.residual_of_inputs(m, up, Ue, Gam_e, chi, dg)
                            - self.residual_of_inputs(m, dn, Ue, Gam_e, chi,
                                                      dg)) / (2 * h)
        scale = np.abs(num).max(axis=(1, 2))[:, None, None]
        assert (np.abs(T - num) / scale).max() < 1e-6

    def test_the_wryness_F_block_needs_S_F_T_F_not_S(self):
        """``dr_Gamma/dF``'s ``1/q`` term carries ``(S F^T F)``, not ``S``.

        Both the reference's Eq. C.6 and its published implementation
        (``constitutive/DP.m`` line 209) use plain ``S`` here -- and its *own*
        neighbouring block for the same quantity, line 211, uses
        ``T22a = (S*F')*F``, which is the correct form. So the two disagree
        internally, and the paper agrees with the wrong one.

        Measured: the derived form matches central differences at **5.4e-9**,
        the printed form at **1.1e-1**. Being a tangent error it costs the
        *global* Newton rate rather than the answer, which is why the code can
        carry it and still produce plausible results.
        """
        m = model(friction=8.6, dilation=2.0, hardening=2.0e8)
        Q, F, Fe_tr, Gam_tr, chi_old, Ue, Gam_e, chi, dg = self.state(m)
        n = Q.shape[0]
        _, _, _, _, work = m.residuals(Q, F, Fe_tr, Gam_tr, chi_old,
                                       Ue, Gam_e, chi, dg)
        T = m.auxiliary_jacobian(Q, F, Fe_tr, Ue, Gam_e, dg, work)
        I0 = self.inputs(Q, F, Fe_tr, Gam_tr, chi_old)

        exact = np.zeros((n, 9, 9))
        for j in range(9):
            h = 1e-5 * max(abs(I0[:, j]).max(), 1e-3)
            up, dn = I0.copy(), I0.copy()
            up[:, j] += h
            dn[:, j] -= h
            exact[:, :, j] = (
                self.residual_of_inputs(m, up, Ue, Gam_e, chi, dg)[:, 9:18]
                - self.residual_of_inputs(m, dn, Ue, Gam_e, chi, dg)[:, 9:18]
            ) / (2 * h)

        mine = T[:, 9:18, 0:9]
        assert np.abs(mine - exact).max() < 1e-6 * np.abs(exact).max()

        # and the printed form is wrong by ~10%, so this test has real power
        S, nu, q = work["S"], work["nu"], work["q"]
        l2 = m.l_p ** 2
        QS = np.einsum("nia,nab->nib", Q, S)
        dq_dF = (1.5 / l2) * np.einsum("nim,nit->nmt", nu, QS) / q[:, None, None]
        SFtF = np.einsum("nrb,njb,nje->nre", S, F, F)
        delta = (1.5 * dg[:, None, None, None, None]
                 / (l2 * q[:, None, None, None, None] ** 2)
                 * (np.einsum("nre,nmt->nremt", SFtF, dq_dF)
                    - np.einsum("nre,nmt->nremt", S, dq_dF)))
        printed = mine + delta.reshape(n, 9, 9)
        assert np.abs(printed - exact).max() > 1e-2 * np.abs(exact).max()

    @pytest.mark.parametrize("kw", [
        {}, {"friction": 8.6, "dilation": 2.0},
        {"friction": 8.6, "dilation": 2.0, "hardening": 2.0e8},
    ])
    def test_P_differentiates_the_converged_return_map(self, kw):
        """The end-to-end check: perturb an input, re-converge the *whole*
        return map, and compare the change in its outputs against ``P``. This
        validates ``C``, ``T`` and ``P = -C^-1 T`` together."""
        m = model(**kw)
        rng = np.random.default_rng(11)
        n = 3
        Q = quat_to_matrix(quat_from_rotvec(0.2 * rng.standard_normal((n, 3))))
        F = np.eye(3) + 0.05 * rng.standard_normal((n, 3, 3))
        Fe_tr = np.eye(3) + 0.02 * rng.standard_normal((n, 3, 3))
        Gam_tr = 8.0 * rng.standard_normal((n, 3, 3))
        chi_old = np.zeros(n)

        def outputs(F_, Q_, Fe_, G_, c_):
            o = m.return_map(Q_, F_, Fe_, G_, c_, tol=1e-13, max_iter=80)
            return np.concatenate([o["Ue"].reshape(n, 9),
                                   o["Gam_e"].reshape(n, 9),
                                   o["chi"][:, None], o["dgamma"][:, None]],
                                  axis=1), o

        X0, o = outputs(F, Q, Fe_tr, Gam_tr, chi_old)
        assert o["plastic"].all()
        _, _, _, _, work = m.residuals(Q, F, Fe_tr, Gam_tr, chi_old, o["Ue"],
                                       o["Gam_e"], o["chi"], o["dgamma"])
        P = m.algorithmic_jacobian(Q, F, Fe_tr, o["Ue"], o["Gam_e"],
                                   o["dgamma"], work)
        I0 = self.inputs(Q, F, Fe_tr, Gam_tr, chi_old)

        num = np.zeros_like(P)
        for j in range(m.n_inputs):
            h = 1e-6 * max(abs(I0[:, j]).max(), 1e-3)
            for sign, store in ((+1, "up"), (-1, "dn")):
                I = I0.copy()
                I[:, j] += sign * h
                vals = outputs(I[:, 0:9].reshape(n, 3, 3),
                               I[:, 9:18].reshape(n, 3, 3),
                               I[:, 18:27].reshape(n, 3, 3),
                               I[:, 27:36].reshape(n, 3, 3), I[:, 36])[0]
                num[:, :, j] += sign * vals / (2 * h)
        scale = np.abs(num).max(axis=(1, 2))[:, None, None]
        assert (np.abs(P - num) / scale).max() < 1e-5


def test_the_local_jacobian_is_singular_at_an_unstressed_non_dilatant_point():
    """``C`` has no inverse when ``psi = 0`` *and* ``H = 0`` *and* ``q = 0``.

    With ``beta_DP = 0`` the flow direction ``1.5 s/q + beta_DP I/3`` vanishes
    entirely at zero stress, so ``dr_U/dgamma``, ``dr_Gamma/dgamma``,
    ``df/dchi`` and ``df/dgamma`` are *all* zero at once and the ``chi`` and
    ``dgamma`` columns become linearly dependent.

    Every one of those three conditions is ordinary: perfect plasticity is the
    standard shear-band setting, zero dilatancy is one of the three cases in a
    Roscoe-angle sweep, and an unstressed point is how every analysis starts.
    The consequence is that ``P = -C^-1 T`` must be solved **only where the
    point is plastic** -- an unstressed point is elastic and needs the elastic
    sensitivities anyway, so excluding it is both correct and what keeps the
    solve non-singular.
    """
    elastic = MicropolarHooke.from_technical(
        E=2.5 * 8.0e6, nu=0.25, N=0.3015, l_t=0.004, l_b=0.002, psi=1.5)
    eye = np.broadcast_to(np.eye(3), (1, 3, 3)).copy()
    zero = np.zeros((1, 3, 3))

    def condition(dilation, hardening, Ue, Gam):
        m = MicropolarDruckerPrager(elastic, cohesion=100.0e3, friction=30.0,
                                    dilation=dilation, hardening=hardening,
                                    plastic_length=0.002)
        _, _, _, _, work = m.residuals(eye, eye, Ue, Gam, np.zeros(1), Ue, Gam,
                                       np.zeros(1), np.zeros(1))
        C = m.jacobian(eye, eye, Ue, Ue, Gam, np.zeros(1), work)
        return np.linalg.cond(C)[0]

    # all three conditions together: singular
    assert not np.isfinite(condition(0.0, 0.0, eye, zero))
    # relax any one of them and it is merely ill-conditioned
    assert np.isfinite(condition(10.0, 0.0, eye, zero))
    assert np.isfinite(condition(0.0, 1.0e5, eye, zero))
    assert np.isfinite(condition(0.0, 0.0, eye * 1.001, zero + 1.0e-3))
