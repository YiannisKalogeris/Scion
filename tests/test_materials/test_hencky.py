"""
Tests for the principal-stretch material family.

Each model is checked against something computed *independently* of it: the
published closed form for Fixed Corotated, a finite-differenced strain energy
for Hencky, and the analytic yield surface for the two return maps. Comparing a
model against a rearrangement of itself proves only that the algebra was copied
consistently.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core import small_linalg as sla
from scion.materials.hencky import (
    DruckerPragerSand,
    FixedCorotated,
    HenckyElastic,
    PrincipalPlasticity,
    PrincipalPlasticState,
    VonMisesHencky,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20250819)


def random_F(rng, n=32, scale=0.2):
    return sla.identity(n, 3) + scale * rng.standard_normal((n, 3, 3))


ELASTIC = [
    pytest.param(HenckyElastic(2.0, 3.0), id="hencky"),
    pytest.param(FixedCorotated(2.0, 3.0), id="fixed-corotated"),
]


# ---------------------------------------------------------------------------
# signed_svd
# ---------------------------------------------------------------------------

class TestSignedSVD:
    def test_reconstructs_the_input(self, rng):
        F = random_F(rng, 200, 0.3)
        U, s, Vt = sla.signed_svd(F)
        assert np.allclose(np.einsum("pij,pj,pjk->pik", U, s, Vt), F, atol=1e-13)

    def test_both_factors_are_rotations(self, rng):
        F = random_F(rng, 200, 0.3)
        U, _, Vt = sla.signed_svd(F)
        assert np.allclose(np.linalg.det(U), 1.0)
        assert np.allclose(np.linalg.det(Vt), 1.0)

    def test_inverted_input_keeps_rotations_and_signs_a_stretch(self):
        """A reflection in the factors would rotate the stress the wrong way."""
        F = np.diag([-1.0, 1.0, 1.0])[None]
        U, s, Vt = sla.signed_svd(F)

        assert np.isclose(np.linalg.det(U)[0], 1.0)
        assert np.isclose(np.linalg.det(Vt)[0], 1.0)
        assert (s < 0).sum() == 1                      # the inversion shows up here
        assert np.allclose(np.einsum("pij,pj,pjk->pik", U, s, Vt), F)

    def test_raw_svd_would_have_given_a_reflection(self):
        """Guards the reason the wrapper exists."""
        F = np.diag([-1.0, 1.0, 1.0])[None]
        U, _, Vt = np.linalg.svd(F)
        assert np.linalg.det(U)[0] * np.linalg.det(Vt)[0] < 0

    def test_identity_is_a_fixed_point(self):
        U, s, Vt = sla.signed_svd(sla.identity(4, 3))
        assert np.allclose(s, 1.0)


# ---------------------------------------------------------------------------
# Elastic laws
# ---------------------------------------------------------------------------

class TestElasticLaws:
    @pytest.mark.parametrize("model", ELASTIC)
    def test_stress_free_at_identity(self, model):
        assert np.allclose(model.kirchhoff(sla.identity(4, 3)), 0.0, atol=1e-12)

    @pytest.mark.parametrize("model", ELASTIC)
    def test_frame_indifference(self, model, rng):
        """tau(QF) = Q tau(F) Q^T for any rotation Q."""
        th = 0.7
        c, s = np.cos(th), np.sin(th)
        Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        F = random_F(rng)

        rotated = model.kirchhoff(np.einsum("ij,pjk->pik", Q, F))
        conjugated = np.einsum("ij,pjk,lk->pil", Q, model.kirchhoff(F), Q)
        assert np.allclose(rotated, conjugated, atol=1e-12)

    @pytest.mark.parametrize("model", ELASTIC)
    def test_kirchhoff_is_symmetric(self, model, rng):
        tau = model.kirchhoff(random_F(rng))
        assert np.allclose(tau, sla.transpose(tau), atol=1e-12)

    @pytest.mark.parametrize("model", ELASTIC)
    def test_tau_matches_the_energy_derivative(self, model, rng):
        """tau = (dW/dF) F^T, finite-differenced."""
        F0 = random_F(rng, 1, 0.15)
        h = 1e-6
        dWdF = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Fp, Fm = F0.copy(), F0.copy()
                Fp[0, i, j] += h
                Fm[0, i, j] -= h
                dWdF[i, j] = (model.energy(Fp)[0] - model.energy(Fm)[0]) / (2 * h)

        expected = dWdF @ F0[0].T
        assert np.allclose(model.kirchhoff(F0)[0], expected, atol=1e-6)

    def test_fixed_corotated_matches_its_published_form(self, rng):
        """P = 2 mu (F - R) + lambda J (J-1) F^-T, with R from polar."""
        m = FixedCorotated(mu=3.0, lam=5.0)
        F = random_F(rng, 40)
        P, _ = m.pk1(F)

        U, _, Vt = sla.signed_svd(F)
        R = np.einsum("pij,pjk->pik", U, Vt)
        J = sla.det(F)
        expected = (2.0 * m.mu * (F - R)
                    + (m.lam * J * (J - 1.0))[:, None, None]
                    * sla.inv_transpose(F, J))
        assert np.allclose(P, expected, rtol=1e-11, atol=1e-11)

    def test_hencky_principal_stress_is_linear_in_log_strain(self):
        """The property the closed-form return maps depend on."""
        m = HenckyElastic(mu=2.0, lam=3.0)
        s1 = np.array([[1.1, 0.9, 1.05]])
        t1 = m.principal_kirchhoff(s1)
        t2 = m.principal_kirchhoff(s1 ** 2)      # doubles the log strain
        assert np.allclose(t2, 2.0 * t1, rtol=1e-12)

    def test_hencky_stiffens_without_bound_under_compression(self):
        """Where Green-strain StVK loses stability instead."""
        m = HenckyElastic(mu=1.0, lam=1.0)
        press = [abs(m.principal_kirchhoff(np.full((1, 3), s))[0, 0])
                 for s in (0.5, 0.2, 0.05)]
        assert press[0] < press[1] < press[2]

    def test_fixed_corotated_survives_inversion(self):
        """Neo-Hookean's log J would be nan here."""
        F = np.diag([-0.5, 1.0, 1.0])[None]
        tau = FixedCorotated(1.0, 1.0).kirchhoff(F)
        assert np.all(np.isfinite(tau))

    @pytest.mark.parametrize("cls", [HenckyElastic, FixedCorotated])
    def test_from_E_nu(self, cls):
        m = cls.from_E_nu(1000.0, 0.25)
        assert np.isclose(m.mu, 400.0)
        assert np.isclose(m.lam, 400.0)

    @pytest.mark.parametrize("model", ELASTIC)
    def test_wave_speed_factor(self, model):
        assert np.isclose(model.wave_speed_factor,
                          np.sqrt(model.lam + 2.0 * model.mu))


# ---------------------------------------------------------------------------
# Drucker-Prager sand
# ---------------------------------------------------------------------------

class TestDruckerPragerSand:
    def test_alpha_reproduces_the_reference_constant(self):
        """CK-MPM hard-codes phi = 30 deg as 0.816496580927726*2*0.5/(3-0.5)."""
        expected = 0.816496580927726 * 2.0 * 0.5 / (3.0 - 0.5)
        assert np.isclose(DruckerPragerSand(30.0).alpha, expected, rtol=1e-14)

    def test_alpha_grows_with_friction_angle(self):
        a = [DruckerPragerSand(p).alpha for p in (10.0, 30.0, 45.0)]
        assert a[0] < a[1] < a[2]

    def test_confined_shear_below_yield_is_elastic(self):
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        eps = np.array([[-0.2 + 0.01, -0.2, -0.2 - 0.01]])
        out, log_jp = dp.project(eps.copy(), np.zeros(1))
        assert np.allclose(out, eps)
        assert np.allclose(log_jp, 0.0)

    def test_yielded_state_lands_exactly_on_the_cone(self):
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        k = (3.0 * dp.lam + 2.0 * dp.mu) / (2.0 * dp.mu)

        for c, d in [(0.05, 0.2), (0.02, 0.3), (0.10, 0.6)]:
            eps = np.array([[-c + d, -c, -c - d]])
            out, _ = dp.project(eps.copy(), np.zeros(1))
            trace = out.sum()
            dev = out - trace / 3.0
            residual = np.linalg.norm(dev) + k * trace * dp.alpha
            assert abs(residual) < 1e-14, (c, d, residual)

    def test_the_return_is_purely_deviatoric(self):
        """Pressure is unchanged; only the shear is shed."""
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        eps = np.array([[-0.05 + 0.2, -0.05, -0.05 - 0.2]])
        out, _ = dp.project(eps.copy(), np.zeros(1))
        assert np.isclose(out.sum(), eps.sum(), atol=1e-15)

    def test_more_confinement_carries_more_shear(self):
        """The defining frictional behaviour."""
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        shear = 0.2

        loose = np.array([[-0.05 + shear, -0.05, -0.05 - shear]])
        tight = np.array([[-0.20 + shear, -0.20, -0.20 - shear]])
        out_loose, _ = dp.project(loose.copy(), np.zeros(1))
        out_tight, _ = dp.project(tight.copy(), np.zeros(1))

        assert not np.allclose(out_loose, loose)   # yields
        assert np.allclose(out_tight, tight)       # still elastic

    def test_expansion_projects_to_the_tip_and_banks_the_volume(self):
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0, beta=1.0)
        eps = np.array([[0.1, 0.1, 0.1]])
        out, log_jp = dp.project(eps.copy(), np.zeros(1))

        assert np.allclose(out, 0.0)               # zero strain -> zero stress
        assert np.isclose(log_jp[0], 0.3)          # the whole expansion is banked

    def test_beta_controls_how_much_volume_is_remembered(self):
        eps = np.array([[0.1, 0.1, 0.1]])
        for beta in (0.0, 0.5, 1.0):
            dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0, beta=beta)
            _, log_jp = dp.project(eps.copy(), np.zeros(1))
            assert np.isclose(log_jp[0], beta * 0.3)

    def test_banked_volume_delays_recompression(self):
        """A dilated pile must close its voids before it carries load again."""
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        eps = np.array([[-0.05, -0.05, -0.05]])    # mildly compressive

        fresh, _ = dp.project(eps.copy(), np.zeros(1))
        dilated, _ = dp.project(eps.copy(), np.array([0.5]))

        assert np.allclose(fresh, eps)             # elastic, carries load
        assert np.allclose(dilated, 0.0)           # still open, carries nothing

    def test_cohesion_shifts_the_tip(self):
        dp = DruckerPragerSand(30.0, cohesion=0.05, mu=1.0, lam=1.0)
        out, _ = dp.project(np.array([[0.2, 0.2, 0.2]]), np.zeros(1))
        assert np.allclose(out, 0.05)

    def test_batched_states_take_independent_branches(self):
        dp = DruckerPragerSand(30.0, mu=1.0, lam=1.0)
        eps = np.array([
            [0.10, 0.10, 0.10],                    # expansion -> tip
            [-0.2 + 0.01, -0.2, -0.2 - 0.01],      # elastic
            [-0.05 + 0.2, -0.05, -0.05 - 0.2],     # yields
        ])
        out, log_jp = dp.project(eps.copy(), np.zeros(3))

        assert np.allclose(out[0], 0.0)
        assert np.allclose(out[1], eps[1])
        assert not np.allclose(out[2], eps[2])
        assert np.isclose(out[2].sum(), eps[2].sum())


# ---------------------------------------------------------------------------
# von Mises in Hencky strain
# ---------------------------------------------------------------------------

class TestVonMisesHencky:
    def test_below_yield_is_elastic(self):
        vm = VonMisesHencky(yield_stress=5.0, mu=100.0)
        eps = np.array([[0.001, -0.0005, -0.0005]])
        out, _ = vm.project(eps.copy(), np.zeros(1))
        assert np.allclose(out, eps)

    def test_deviatoric_stress_is_capped_at_the_yield_value(self):
        mu, tau_y = 100.0, 5.0
        vm = VonMisesHencky(tau_y, mu=mu)
        for mag in (0.05, 0.2, 1.0):
            eps = np.array([[mag, -0.5 * mag, -0.5 * mag]])
            out, _ = vm.project(eps.copy(), np.zeros(1))
            norm = 2.0 * mu * np.linalg.norm(out - out.mean())
            assert np.isclose(norm, tau_y, rtol=1e-12)

    def test_yielding_does_not_touch_the_volumetric_part(self):
        vm = VonMisesHencky(5.0, mu=100.0)
        eps = np.array([[0.3, -0.1, -0.1]])
        out, _ = vm.project(eps.copy(), np.zeros(1))
        assert np.isclose(out.sum(), eps.sum(), atol=1e-15)

    def test_return_direction_is_radial(self):
        """The deviator keeps its direction and only loses magnitude."""
        vm = VonMisesHencky(5.0, mu=100.0)
        eps = np.array([[0.3, -0.05, -0.25]])
        out, _ = vm.project(eps.copy(), np.zeros(1))

        d_in = eps - eps.mean()
        d_out = out - out.mean()
        cos = (d_in @ d_out.T)[0, 0] / (np.linalg.norm(d_in) * np.linalg.norm(d_out))
        assert np.isclose(cos, 1.0, atol=1e-12)

    def test_from_von_mises_applies_the_sqrt_two_thirds(self):
        """A handbook sigma_y used raw would overpredict strength by 22%."""
        vm = VonMisesHencky.from_von_mises(250e6)
        assert np.isclose(vm.yield_stress, np.sqrt(2.0 / 3.0) * 250e6)

    def test_log_jp_is_passed_through_untouched(self):
        vm = VonMisesHencky(5.0, mu=100.0)
        banked = np.array([0.3])
        _, out = vm.project(np.array([[0.3, -0.1, -0.2]]), banked)
        assert np.allclose(out, banked)


# ---------------------------------------------------------------------------
# The stateful model
# ---------------------------------------------------------------------------

class TestPrincipalPlasticity:
    def test_state_starts_undeformed(self):
        st = PrincipalPlasticState.identity(5)
        assert np.allclose(sla.det(st.F_e), 1.0)
        assert np.allclose(st.log_jp, 0.0)
        assert st.n == 5

    def test_elastic_constants_are_pushed_into_the_return_map(self):
        """Otherwise the yield surface is sized against defaults of 1.0."""
        elastic = HenckyElastic(mu=7.0, lam=11.0)
        rm = DruckerPragerSand(30.0)
        PrincipalPlasticity(elastic, rm)
        assert rm.mu == 7.0 and rm.lam == 11.0

    def test_no_deformation_gives_no_stress(self):
        m = PrincipalPlasticity(HenckyElastic(1e5, 1e5), DruckerPragerSand(30.0))
        sigma, st = m.update(None, np.zeros((3, 3, 3)), 0.01,
                             PrincipalPlasticState.identity(3))
        assert np.allclose(sigma, 0.0, atol=1e-9)
        assert np.allclose(sla.det(st.F_e), 1.0)

    def test_the_input_state_is_never_mutated(self):
        m = PrincipalPlasticity(HenckyElastic(1e5, 1e5), DruckerPragerSand(30.0))
        st = PrincipalPlasticState.identity(2)
        before = st.F_e.copy()
        L = np.tile(np.eye(3) * -0.3, (2, 1, 1))
        m.update(None, L, 0.05, st)
        assert np.array_equal(st.F_e, before)

    def test_sand_carries_no_tension(self):
        m = PrincipalPlasticity(HenckyElastic(1e5, 1e5), DruckerPragerSand(30.0))
        L = np.tile(np.eye(3) * 0.5, (1, 1, 1))
        sigma, st = m.update(None, L, 0.1, PrincipalPlasticState.identity(1))

        assert np.allclose(sigma, 0.0, atol=1e-9)
        assert st.log_jp[0] > 0.0                  # the volume is banked

    def test_sand_resists_compression(self):
        m = PrincipalPlasticity(HenckyElastic(1e5, 1e5), DruckerPragerSand(30.0))
        L = np.tile(-np.eye(3) * 0.5, (1, 1, 1))
        sigma, _ = m.update(None, L, 0.1, PrincipalPlasticState.identity(1))
        assert np.trace(sigma[0]) / 3.0 < -1.0     # compressive mean stress

    def test_stress_is_cauchy_not_kirchhoff(self):
        """sigma = tau / J; under compression J < 1 so they differ measurably."""
        elastic = HenckyElastic(1e3, 1e3)
        m = PrincipalPlasticity(elastic, VonMisesHencky(1e9))   # never yields
        L = np.tile(-np.eye(3) * 0.5, (1, 1, 1))
        sigma, st = m.update(None, L, 0.1, PrincipalPlasticState.identity(1))

        J = sla.det(st.F_e)
        tau = elastic.kirchhoff(st.F_e)
        assert np.allclose(sigma, tau / J[:, None, None], rtol=1e-11)
        assert J[0] < 0.95                          # genuinely different

    def test_a_state_that_never_yields_matches_pure_elasticity(self):
        elastic = FixedCorotated(1e3, 1e3)
        m = PrincipalPlasticity(elastic, VonMisesHencky(1e12))
        L = np.zeros((1, 3, 3))
        L[0, 0, 0] = -0.2

        _, st = m.update(None, L, 0.1, PrincipalPlasticState.identity(1))
        expected = np.eye(3) + 0.1 * L[0]
        assert np.allclose(st.F_e[0], expected, atol=1e-12)

    def test_name_reports_both_halves(self):
        m = PrincipalPlasticity(HenckyElastic(1.0, 1.0), DruckerPragerSand(30.0))
        assert m.name == "hencky+drucker-prager-sand"

    def test_state_repr_is_informative(self):
        st = PrincipalPlasticState.identity(4)
        assert "4 points" in repr(st)
