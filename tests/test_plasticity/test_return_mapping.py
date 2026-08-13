"""
Verification for the return map.

Three kinds of check. The stress update is measured against the closed-form
uniaxial response, where every quantity has an analytic value. The consistent
tangent is verified with the same finite-difference tool used on the structural
tangent -- a material-point residual is a residual. And the singular cases are
pinned: the cone apex, and the combination for which no solution exists.

python -m pytest tests/test_plasticity/test_return_mapping.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.plasticity import (
    Associative,
    DruckerPrager,
    LinearIsotropic,
    LinearKinematic,
    NoKinematic,
    NonAssociative,
    PerfectPlasticity,
    Swift,
    Voce,
    VonMises,
)
from simulacrum.materials.linear_elastic import isotropic_stiffness_mandel
from simulacrum.materials.notation import from_mandel, to_mandel
from simulacrum.materials.plasticity.return_mapping import (
    PlasticState,
    return_map,
)
from simulacrum.solvers.tangent_check import check_tangent

E, NU, Y, H = 210e3, 0.3, 250.0, 2000.0
C_EL = isotropic_stiffness_mandel(E, NU)
BULK = E / (3.0 * (1.0 - 2.0 * NU))


def _tensor(**kw):
    e = np.zeros((1, 3, 3))
    for k, v in kw.items():
        i, j = int(k[1]), int(k[2])
        e[0, i, j] = e[0, j, i] = v
    return to_mandel(e)


def _sigma(result):
    return from_mandel(result.sigma)[0]


def _uniaxial_pull(iso, n_steps=200, de=2e-5, crit=None, **kw):
    """Pull in uniaxial tension, solving for the lateral strain that keeps the
    transverse stresses zero. Returns (eps_11, sigma_11, alpha) per step."""
    crit = crit or VonMises()
    st = PlasticState.zeros(1)
    e11 = el = 0.0
    out = []
    for _ in range(n_steps):
        e11 += de
        for _ in range(40):
            eps = _tensor(e00=e11, e11=el, e22=el)
            r = return_map(eps, st, crit, iso, C_EL, **kw)
            s = _sigma(r)
            if abs(s[1, 1]) < 1e-9:
                break
            h = 1e-9
            s2 = _sigma(return_map(_tensor(e00=e11, e11=el + h, e22=el + h),
                                   st, crit, iso, C_EL, **kw))
            el -= s[1, 1] / ((s2[1, 1] - s[1, 1]) / h)
        out.append((e11, s[0, 0], r.state.alpha[0]))
        st = r.state
    return np.array(out)


# ---------------------------------------------------------------------------
# Elastic response
# ---------------------------------------------------------------------------

def test_below_yield_the_response_is_exactly_elastic():
    st = PlasticState.zeros(1)
    eps = _tensor(e00=1e-4, e01=5e-5)
    r = return_map(eps, st, VonMises(), LinearIsotropic(Y, H), C_EL)

    assert not r.plastic[0]
    assert np.abs(r.sigma - eps @ C_EL.T).max() < 1e-9
    assert np.abs(r.C_alg[0] - C_EL).max() < 1e-9
    assert r.state.alpha[0] == 0.0
    assert np.abs(r.state.eps_p).max() == 0.0


def test_the_committed_state_is_not_modified():
    """A discarded Newton iterate must not advance the plastic strain."""
    st = PlasticState.zeros(1)
    before = (st.eps_p.copy(), st.alpha.copy(), st.beta.copy())
    return_map(_tensor(e00=0.01), st, VonMises(), LinearIsotropic(Y, H), C_EL)
    assert np.array_equal(st.eps_p, before[0])
    assert np.array_equal(st.alpha, before[1])
    assert np.array_equal(st.beta, before[2])


# ---------------------------------------------------------------------------
# Uniaxial response against closed form
# ---------------------------------------------------------------------------

def test_uniaxial_yields_at_the_yield_stress():
    h = _uniaxial_pull(LinearIsotropic(Y, H))
    i = int(np.argmax(h[:, 1] > Y + 1e-6))
    assert h[i, 0] == pytest.approx(Y / E, rel=0.02)


def test_uniaxial_hardening_follows_the_prescribed_curve():
    """sigma = Y + H eps_p exactly, which also confirms the 2/3 convention in
    alpha: the equivalent plastic strain equals the axial plastic strain."""
    h = _uniaxial_pull(LinearIsotropic(Y, H))
    e11, s11, alpha = h[-1]
    eps_p = e11 - s11 / E
    assert s11 == pytest.approx(Y + H * eps_p, rel=1e-8)
    assert alpha == pytest.approx(eps_p, rel=1e-8)


def test_uniaxial_elastoplastic_slope_is_EH_over_E_plus_H():
    h = _uniaxial_pull(LinearIsotropic(Y, H))
    slope = (h[-1, 1] - h[-20, 1]) / (h[-1, 0] - h[-20, 0])
    assert slope == pytest.approx(E * H / (E + H), rel=1e-4)


def test_perfect_plasticity_caps_the_stress_at_the_yield_stress():
    h = _uniaxial_pull(PerfectPlasticity(Y))
    assert h[-1, 1] == pytest.approx(Y, rel=1e-8)
    assert h[-1, 0] > 3 * Y / E                # well past yield


def test_plastic_flow_is_volume_preserving_for_von_mises():
    st = PlasticState.zeros(1)
    r = return_map(_tensor(e00=0.01, e11=-0.003, e22=-0.003),
                   st, VonMises(), LinearIsotropic(Y, H), C_EL)
    assert r.plastic[0]
    trace = from_mandel(r.state.eps_p)[0].trace()
    assert abs(trace) < 1e-12


@pytest.mark.parametrize("iso", [Voce(Y, 400.0, 20.0),
                                 Swift.from_yield(Y, 600.0, 0.25)],
                         ids=["voce", "swift"])
def test_nonlinear_hardening_reaches_its_own_curve(iso):
    """The converged stress must lie on kappa(alpha), whatever the law."""
    h = _uniaxial_pull(iso, n_steps=150)
    e11, s11, alpha = h[-1]
    assert s11 == pytest.approx(float(iso.kappa(np.array([alpha]))[0]), rel=1e-6)


def test_kinematic_hardening_produces_a_bauschinger_effect():
    """Load forward into plasticity, reverse, and check that yielding on
    reversal happens well before the virgin yield stress in compression.

    With perfect isotropic behaviour and a linear back stress the numbers are
    exact: forward flow stops at ``Y + q(beta)`` and reverse flow begins at
    ``-Y + q(beta)``, so the elastic span stays ``2Y`` while its centre moves.
    A purely isotropic model would instead widen the span and yield *later*.
    """
    crit, iso = VonMises(), PerfectPlasticity(Y)
    kin = LinearKinematic(20000.0)
    st = PlasticState.zeros(1)
    el = 0.0

    def pull(e11, el, st):
        """One step at fixed axial strain, solving for the lateral strain that
        keeps the transverse stress zero."""
        for _ in range(30):
            r = return_map(_tensor(e00=e11, e11=el, e22=el), st, crit, iso,
                           C_EL, kinematic=kin)
            s = _sigma(r)
            if abs(s[1, 1]) < 1e-9:
                break
            h = 1e-9
            s2 = _sigma(return_map(_tensor(e00=e11, e11=el + h, e22=el + h),
                                   st, crit, iso, C_EL, kinematic=kin))
            el -= s[1, 1] / ((s2[1, 1] - s[1, 1]) / h)
        return r, s, el

    e11 = 0.0
    for _ in range(100):
        e11 += 4e-5
        r, s, el = pull(e11, el, st)
        st = r.state
    forward = s[0, 0]

    back = float(np.sqrt(1.5 * np.sum(from_mandel(st.beta)[0] ** 2)))
    assert forward == pytest.approx(Y + back, rel=1e-6)
    assert back > 0.0

    reverse = None
    for k in range(1, 600):
        r, s, el = pull(e11 - k * 4e-5, el, st)
        if r.plastic[0]:
            reverse = s[0, 0]
            break

    # the reverse yield point is detected on a discrete loading step, so it can
    # overshoot by up to one increment (E * 4e-5 ~ 8 MPa); 2 MPa is comfortably
    # inside that and still far tighter than the effect being measured (~51)
    assert reverse is not None
    assert reverse == pytest.approx(-Y + back, abs=2.0)
    assert reverse > -Y                    # yields early: the Bauschinger effect
    assert forward - reverse == pytest.approx(2 * Y, abs=2.0)


# ---------------------------------------------------------------------------
# Consistent tangent
# ---------------------------------------------------------------------------

def _tangent_case(crit, iso, flow=None, kin=None, state=None):
    st = state or PlasticState.zeros(1)
    eps0 = _tensor(e00=0.004, e11=-0.0012, e22=-0.0012, e01=0.001)[0]

    def residual(e):
        return return_map(e[None, :], st, crit, iso, C_EL, flow, kin).sigma[0]

    def tangent(e):
        return return_map(e[None, :], st, crit, iso, C_EL, flow, kin).C_alg[0]

    return check_tangent(residual, tangent, eps0, rtol=2e-5)


@pytest.mark.parametrize("crit,iso,flow,kin,label", [
    (VonMises(), LinearIsotropic(Y, H), None, None, "j2-linear"),
    (VonMises(), PerfectPlasticity(Y), None, None, "j2-perfect"),
    (VonMises(), Voce(Y, 400.0, 20.0), None, None, "j2-voce"),
    (VonMises(), Swift.from_yield(Y, 600.0, 0.25), None, None, "j2-swift"),
    (VonMises(), LinearIsotropic(Y, H), None, LinearKinematic(1500.0),
     "j2-combined"),
    (DruckerPrager.from_mohr_coulomb(30.0), LinearIsotropic(50.0, 500.0),
     None, None, "dp-associative"),
    (DruckerPrager.from_mohr_coulomb(30.0), LinearIsotropic(50.0, 500.0),
     NonAssociative.drucker_prager(10.0), None, "dp-non-associative"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_consistent_tangent_matches_finite_differences(crit, iso, flow, kin,
                                                       label):
    """The tangent is exact for the algorithm as implemented, not merely for
    the continuum rate equations. Getting this wrong costs quadratic Newton
    convergence and nothing else, which is why it needs an explicit test."""
    report = _tangent_case(crit, iso, flow, kin)
    assert report.passed, f"{label}: {report}"


def test_tangent_is_symmetric_for_associative_and_not_otherwise():
    """The consequence the assembler has to act on: a non-associative model
    stored in a symmetric format loses half its tangent.

    The cohesion is chosen so the state returns to the smooth part of the cone.
    At the apex the tangent is a rank-one volumetric operator and therefore
    symmetric whatever the flow rule, so an apex state would test nothing.
    """
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    iso = LinearIsotropic(500.0, 500.0)
    assert _tangent_case(crit, iso, Associative()).symmetry < 1e-10
    assert _tangent_case(crit, iso,
                         NonAssociative.drucker_prager(10.0)).symmetry > 0.1


def test_elastic_points_return_the_elastic_tangent():
    st = PlasticState.zeros(3)
    eps = np.zeros((3, 6))
    eps[0, 0] = 1e-5                          # elastic
    eps[1, 0] = 0.01                          # plastic
    eps[2, 0] = 2e-5                          # elastic
    r = return_map(eps, st, VonMises(), LinearIsotropic(Y, H), C_EL)
    assert list(r.plastic) == [False, True, False]
    assert np.abs(r.C_alg[0] - C_EL).max() < 1e-9
    assert np.abs(r.C_alg[2] - C_EL).max() < 1e-9
    assert np.abs(r.C_alg[1] - C_EL).max() > 1.0


# ---------------------------------------------------------------------------
# The cone apex
# ---------------------------------------------------------------------------

def test_tensile_states_return_to_the_cone_apex():
    """A hydrostatic tension well beyond the apex must land exactly on the tip,
    with zero deviator, however far beyond it starts."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    c = 50.0
    p_apex = float(crit.apex_pressure(c))

    for strain in (0.002, 0.004, 0.008):
        eps = _tensor(e00=strain, e11=strain, e22=strain)
        r = return_map(eps, PlasticState.zeros(1), crit,
                       PerfectPlasticity(c), C_EL)
        s = _sigma(r)
        assert r.apex[0]
        assert np.trace(s) / 3.0 == pytest.approx(p_apex, rel=1e-8)
        dev = s - np.trace(s) / 3.0 * np.eye(3)
        assert np.abs(dev).max() < 1e-9


@pytest.mark.parametrize("iso,expected_rank", [
    (PerfectPlasticity(50.0), 0),
    (LinearIsotropic(50.0, 500.0), 1),
], ids=["perfect", "hardening"])
def test_apex_tangent_is_volumetric_and_loses_all_deviatoric_stiffness(
        iso, expected_rank):
    """At the tip the deviatoric response vanishes entirely. With hardening the
    volumetric response survives as a rank-one operator; with perfect
    plasticity the stress is pinned at the apex whatever the strain, so the
    tangent is exactly zero and the element has no stiffness at all -- which is
    worth knowing before a global solve meets a singular matrix."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    eps = _tensor(e00=0.004, e11=0.004, e22=0.004)
    r = return_map(eps, PlasticState.zeros(1), crit, iso, C_EL)
    assert r.apex[0]

    C = r.C_alg[0]
    assert np.abs(C[3:, :]).max() < 1e-9          # no shear response
    tol = 1e-9 * max(np.abs(C).max(), 1.0)
    assert np.linalg.matrix_rank(C, tol=tol) == expected_rank


def test_zero_dilatancy_beyond_the_apex_is_reported_as_inadmissible():
    """Deviatoric flow cannot change the pressure, so a trial state beyond the
    apex has no admissible return at all. This is a property of the model, not
    a solver failure, and the message says so."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    eps = _tensor(e00=0.004, e11=0.004, e22=0.004)
    with pytest.raises(RuntimeError, match="dilatancy"):
        return_map(eps, PlasticState.zeros(1), crit, PerfectPlasticity(50.0),
                   C_EL, flow=NonAssociative.volume_preserving())


# ---------------------------------------------------------------------------
# Batching and unsupported combinations
# ---------------------------------------------------------------------------

def test_batched_points_are_independent():
    """Each quadrature point must be integrated on its own state."""
    st = PlasticState.zeros(4)
    eps = np.zeros((4, 6))
    eps[:, 0] = [1e-5, 0.005, 0.01, 2e-5]
    r = return_map(eps, st, VonMises(), LinearIsotropic(Y, H), C_EL)

    for k in range(4):
        single = return_map(eps[k:k + 1], PlasticState.zeros(1), VonMises(),
                            LinearIsotropic(Y, H), C_EL)
        assert np.abs(r.sigma[k] - single.sigma[0]).max() < 1e-10
        assert np.abs(r.C_alg[k] - single.C_alg[0]).max() < 1e-6


def test_nonlinear_kinematic_hardening_is_refused_clearly():
    """Armstrong-Frederick cannot be eliminated into the relative-stress form,
    so it is rejected rather than silently approximated."""
    from simulacrum.materials.plasticity import ArmstrongFrederick
    with pytest.raises(NotImplementedError, match="nonlinear"):
        return_map(_tensor(e00=0.01), PlasticState.zeros(1), VonMises(),
                   LinearIsotropic(Y, H), C_EL,
                   kinematic=ArmstrongFrederick(20000.0, 100.0))
