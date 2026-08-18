"""
Verification for the hardening laws.

Isotropic laws are checked against their own derivative and against the
properties that decide whether a return map can use them; kinematic laws are
checked against the Bauschinger effect and their saturation limits.

python -m pytest tests/test_plasticity/test_hardening.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.plasticity.hardening import (
    ArmstrongFrederick,
    LinearIsotropic,
    LinearKinematic,
    Ludwik,
    NoKinematic,
    PerfectPlasticity,
    Swift,
    Voce,
    get_isotropic,
    get_kinematic,
)

ISO = [
    PerfectPlasticity(250.0),
    LinearIsotropic(250.0, 2000.0),
    Voce(250.0, 400.0, 20.0),
    Swift.from_yield(250.0, 600.0, 0.25),
    Ludwik(250.0, 600.0, 0.25),
]


def _uniaxial_normal():
    """von Mises normal for uniaxial tension: diag(1, -1/2, -1/2)."""
    n = np.zeros((1, 3, 3))
    n[0, 0, 0], n[0, 1, 1], n[0, 2, 2] = 1.0, -0.5, -0.5
    return n


def _q(t):
    """von Mises equivalent of a deviatoric tensor."""
    return np.sqrt(1.5 * np.einsum("nij,nij->n", t, t))


# ---------------------------------------------------------------------------
# Isotropic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", ISO, ids=[l.name for l in ISO])
def test_modulus_is_the_derivative_of_kappa(law):
    a = np.linspace(1e-4, 0.5, 200)      # skip alpha=0: Ludwik is floored there
    h = 1e-7
    fd = (law.kappa(a + h) - law.kappa(a - h)) / (2 * h)
    H = law.dkappa_dalpha(a)
    assert np.abs(fd - H).max() < 1e-6 * max(np.abs(H).max(), 1.0)


@pytest.mark.parametrize("law", ISO, ids=[l.name for l in ISO])
def test_kappa_is_non_decreasing_for_hardening_laws(law):
    a = np.linspace(0.0, 0.5, 200)
    assert np.all(np.diff(law.kappa(a)) >= -1e-12)


@pytest.mark.parametrize("law", ISO, ids=[l.name for l in ISO])
def test_initial_yield_is_recovered_at_zero_plastic_strain(law):
    assert law.kappa(np.array([0.0]))[0] == pytest.approx(250.0, rel=1e-6)


def test_perfect_plasticity_never_hardens():
    law = PerfectPlasticity(250.0)
    a = np.linspace(0.0, 10.0, 50)
    assert np.abs(law.kappa(a) - 250.0).max() < 1e-12
    assert np.abs(law.dkappa_dalpha(a)).max() == 0.0
    assert law.linear


def test_linear_isotropic_matches_its_closed_form():
    law = LinearIsotropic(250.0, 2000.0)
    a = np.linspace(0.0, 0.5, 20)
    assert np.abs(law.kappa(a) - (250.0 + 2000.0 * a)).max() < 1e-10
    assert law.linear


def test_voce_saturates_and_the_modulus_decays_to_zero():
    """The material becomes asymptotically perfectly plastic, so a return map
    must not assume H is bounded away from zero."""
    law = Voce(250.0, 400.0, 20.0)
    assert law.kappa(np.array([1e3]))[0] == pytest.approx(400.0, rel=1e-9)
    assert law.dkappa_dalpha(np.array([1e3]))[0] == pytest.approx(0.0, abs=1e-9)
    info = law.check()
    assert info["saturates"] and not info["softening"]


def test_swift_has_a_finite_initial_modulus():
    """This is the whole reason Swift exists rather than Ludwik."""
    law = Swift.from_yield(250.0, 600.0, 0.25)
    H0 = law.dkappa_dalpha(np.array([0.0]))[0]
    assert np.isfinite(H0) and H0 > 0
    assert law.kappa(np.array([0.0]))[0] == pytest.approx(250.0)


def test_ludwik_modulus_is_singular_at_the_origin_and_floored():
    """H = K n alpha^(n-1) diverges for n < 1, at exactly the state every point
    is in at first yield. The floor is a numerical device; if it shows up in
    results, Swift is the right model."""
    law = Ludwik(250.0, 600.0, 0.25, alpha_floor=1e-8)
    H0 = law.dkappa_dalpha(np.array([0.0]))[0]
    assert np.isfinite(H0)
    assert H0 == pytest.approx(600.0 * 0.25 * 1e-8 ** (-0.75))
    # away from the origin the floor has no effect
    a = np.array([1e-3])
    assert law.dkappa_dalpha(a)[0] == pytest.approx(600.0 * 0.25 * a[0] ** -0.75)


def test_softening_is_allowed_but_reported():
    """Negative H is physical and destroys ellipticity; the model does not
    refuse it, but check() flags it so a caller can warn."""
    law = LinearIsotropic(250.0, -500.0)
    assert law.check()["softening"]
    assert law.dkappa_dalpha(np.array([0.1]))[0] < 0


def test_only_perfect_and_linear_claim_closed_form_solvability():
    assert PerfectPlasticity(1.0).linear and LinearIsotropic(1.0, 1.0).linear
    assert not Voce(1.0, 2.0, 1.0).linear
    assert not Swift(1.0, 1e-3, 0.3).linear


# ---------------------------------------------------------------------------
# Kinematic
# ---------------------------------------------------------------------------

def test_no_kinematic_leaves_the_surface_centred():
    k = NoKinematic()
    beta = np.zeros((4, 3, 3))
    assert np.abs(k.update(beta, _uniaxial_normal().repeat(4, 0),
                           np.full(4, 0.01))).max() == 0.0


def test_linear_kinematic_grows_the_equivalent_back_stress_as_C_alpha():
    """The 2/3 convention: q(beta) = C alpha, while beta_11 = (2/3) C alpha.
    Calibrating against the component instead is a factor-1.5 error."""
    C, n, dalpha = 2000.0, _uniaxial_normal(), 0.001
    law = LinearKinematic(C)
    beta = np.zeros((1, 3, 3))
    for _ in range(100):
        beta = law.update(beta, n, np.array([dalpha]))
    alpha = 100 * dalpha
    assert _q(beta)[0] == pytest.approx(C * alpha, rel=1e-10)
    assert beta[0, 0, 0] == pytest.approx((2.0 / 3.0) * C * alpha, rel=1e-10)


def test_kinematic_back_stress_stays_deviatoric():
    """A deviatoric flow direction must not generate a pressure in beta."""
    law = LinearKinematic(2000.0)
    beta = law.update(np.zeros((1, 3, 3)), _uniaxial_normal(), np.array([0.05]))
    assert abs(np.trace(beta[0])) < 1e-12


def test_armstrong_frederick_saturates():
    law = ArmstrongFrederick(C=20000.0, gamma=100.0)
    n = _uniaxial_normal()
    beta = np.zeros((1, 3, 3))
    for _ in range(2000):
        beta = law.update(beta, n, np.array([0.001]))
    assert _q(beta)[0] == pytest.approx(law.saturation(), rel=1e-6)
    assert law.saturation() == pytest.approx(20000.0 / 100.0)


def test_armstrong_frederick_implicit_update_cannot_overshoot():
    """An explicit update overshoots the saturation limit once gamma*dalpha
    approaches one; the implicit form is bounded for any step."""
    law = ArmstrongFrederick(C=20000.0, gamma=100.0)
    n = _uniaxial_normal()
    for dalpha in (1e-3, 1.0, 1e3, 1e6):
        beta = law.update(np.zeros((1, 3, 3)), n, np.array([dalpha]))
        assert _q(beta)[0] <= law.saturation() * (1 + 1e-9)


@pytest.mark.parametrize("law", [LinearKinematic(2000.0),
                                 ArmstrongFrederick(20000.0, 100.0)],
                         ids=["linear", "armstrong-frederick"])
def test_dbeta_ddalpha_is_the_derivative_of_the_update(law):
    """Needed by the consistent tangent; an error here shows up only as slow
    Newton convergence."""
    n = _uniaxial_normal()
    beta = 30.0 * n.copy()
    d = np.array([0.002])
    h = 1e-8
    fd = (law.update(beta, n, d + h) - law.update(beta, n, d - h)) / (2 * h)
    assert np.abs(fd - law.dbeta_ddalpha(beta, n, d)).max() < 1e-5


def test_bauschinger_effect():
    """After forward yielding, reverse yield occurs below the initial yield
    stress by twice the back stress. A purely isotropic model predicts the
    opposite, so this is the property that requires a kinematic component."""
    C, y0 = 2000.0, 250.0
    law = LinearKinematic(C)
    n = _uniaxial_normal()
    alpha = 0.05
    beta = law.update(np.zeros((1, 3, 3)), n, np.array([alpha]))

    forward = y0 + _q(beta)[0]          # stress at the end of forward loading
    reverse = -y0 + _q(beta)[0]         # first yield on reversal
    assert abs(forward - reverse) == pytest.approx(2 * y0)
    assert abs(reverse) < y0            # yields earlier than a virgin material


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def test_registries_resolve_and_reject():
    assert isinstance(get_isotropic("linear", kappa_0=1.0, H=1.0),
                      LinearIsotropic)
    assert isinstance(get_kinematic("prager", C=1.0), LinearKinematic)
    assert isinstance(get_kinematic("af", C=1.0, gamma=1.0), ArmstrongFrederick)
    with pytest.raises(NotImplementedError):
        # Zerilli-Armstrong: a real rate- and temperature-dependent law that
        # is genuinely not implemented. Johnson-Cook used to be the example
        # here and now resolves, which is what adding it was for.
        get_isotropic("zerilli-armstrong")
    with pytest.raises(NotImplementedError):
        get_kinematic("chaboche")


def test_swift_rejects_a_zero_offset():
    with pytest.raises(ValueError, match="alpha_0"):
        Swift(K=600.0, alpha_0=0.0, n=0.25)
