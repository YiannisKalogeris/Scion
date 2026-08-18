"""
Verification for the cyclic curve and the notch rules.

Every analytic derivative here gets a finite-difference check, and every notch
rule gets checked against the conservation statement it is derived from --
``sigma eps`` for Neuber, strain energy density for Glinka -- rather than
against a reference value, so a change to the solver cannot quietly move the
answer.

python -m pytest tests/test_fatigue/test_cyclic.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.fatigue.curves import StrainLife
from scion.fatigue.cyclic import (
    RambergOsgood,
    glinka,
    glinka_range,
    neuber,
    neuber_range,
)

E = 210.0e3
RO = RambergOsgood(E=E, K=1200.0, n=0.15)
STRESS = np.array([50.0, 200.0, 400.0, 700.0, -400.0])


# ---------------------------------------------------------------------------
# Ramberg-Osgood
# ---------------------------------------------------------------------------

def test_the_derivative_is_the_derivative_of_the_strain():
    h = 1e-4
    fd = (RO.strain(STRESS + h) - RO.strain(STRESS - h)) / (2.0 * h)
    assert np.allclose(fd, RO.dstrain_dsigma(STRESS), rtol=1e-6)


def test_the_energy_density_is_the_integral_under_the_curve():
    """``dW/dsigma = sigma deps/dsigma``. Checked on the magnitude, since ``W``
    is an area and even in the stress while the strain is odd."""
    h = 1e-4
    s = np.abs(STRESS)
    fd = (RO.energy_density(s + h) - RO.energy_density(s - h)) / (2.0 * h)
    assert np.allclose(fd, s * RO.dstrain_dsigma(s), rtol=1e-6)


def test_the_curve_inverts():
    assert np.allclose(RO.stress(RO.strain(STRESS)), STRESS, rtol=1e-9,
                       atol=1e-9)


def test_the_curve_is_odd():
    assert np.allclose(RO.strain(-STRESS), -RO.strain(STRESS))


def test_the_elastic_line_bounds_the_stress_from_above():
    """The property the inversion bracket relies on: the plastic term is
    non-negative, so ``sigma <= E eps`` always."""
    e = RO.strain(STRESS)
    assert np.all(np.abs(RO.stress(e)) <= np.abs(E * e) + 1e-9)


def test_massing_doubles_the_curve():
    """``d_eps(d_sigma) == 2 eps(d_sigma/2)``, which is Massing's hypothesis and
    the only reason the range-form notch rules need no separate derivation."""
    d = np.array([100.0, 400.0, 900.0])
    assert np.allclose(RO.hysteresis_strain_range(d), 2.0 * RO.strain(0.5 * d))


def test_an_exponent_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="n must"):
        RambergOsgood(E=E, K=1200.0, n=0.0)
    with pytest.raises(ValueError, match="n must"):
        RambergOsgood(E=E, K=1200.0, n=1.5)


# ---------------------------------------------------------------------------
# Compatibility with the strain-life constants
# ---------------------------------------------------------------------------

def test_the_compatibility_relations_reproduce_the_plastic_strain():
    """``n' = b/c`` and ``K' = sigma_f'/eps_f'^n'`` are derived by requiring the
    cyclic curve and the strain-life curve to describe the same loops. So the
    plastic strain amplitude at a given life must come out the same read off
    either one -- which is the statement, not the derivation, and therefore
    worth asserting.
    """
    sl = StrainLife(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60)
    ro = RambergOsgood.from_strain_life(sl)

    assert ro.n == pytest.approx(sl.b / sl.c)
    assert ro.K == pytest.approx(sl.sigma_f / sl.eps_f ** ro.n)

    n = np.array([1.0e2, 1.0e4, 1.0e6])
    sigma_a = sl.stress_amplitude(n)
    from_cyclic = ro.strain(sigma_a) - sigma_a / E      # plastic part
    assert np.allclose(from_cyclic, sl.plastic_amplitude(n), rtol=1e-9)


# ---------------------------------------------------------------------------
# Notch rules
# ---------------------------------------------------------------------------

ELASTIC = np.array([300.0, 600.0, 1200.0, -600.0])


def test_neuber_conserves_the_stress_strain_product():
    s, e = neuber(ELASTIC, RO)
    assert np.allclose(s * e, ELASTIC ** 2 / E, rtol=1e-9)


def test_glinka_conserves_the_strain_energy_density():
    s, _ = glinka(ELASTIC, RO)
    assert np.allclose(RO.energy_density(s), ELASTIC ** 2 / (2.0 * E),
                       rtol=1e-9)


def test_both_rules_return_a_point_on_the_cyclic_curve():
    """The corrected pair has to lie on the material's curve; conserving the
    right quantity while landing off the curve would be meaningless."""
    for rule in (neuber, glinka):
        s, e = rule(ELASTIC, RO)
        assert np.allclose(RO.strain(s), e, rtol=1e-9)


def test_plasticity_relaxes_the_stress_and_raises_the_strain():
    s, e = neuber(ELASTIC, RO)
    assert np.all(np.abs(s) <= np.abs(ELASTIC) + 1e-9)
    assert np.all(np.abs(e) >= np.abs(ELASTIC) / E - 1e-12)


def test_neuber_predicts_more_local_strain_than_glinka():
    """Which is why Neuber is the conservative choice and the pair brackets the
    measurement."""
    _, en = neuber(ELASTIC, RO)
    _, eg = glinka(ELASTIC, RO)
    assert np.all(np.abs(en) >= np.abs(eg))
    assert np.abs(en[2]) > 1.05 * np.abs(eg[2])       # and not trivially so


def test_both_rules_are_the_identity_while_the_response_is_elastic():
    """Far below yield the plastic term is negligible and the correction must
    do nothing measurable, or every high-cycle result would be perturbed."""
    small = np.array([20.0])
    for rule in (neuber, glinka):
        s, e = rule(small, RO)
        assert s[0] == pytest.approx(small[0], rel=1e-4)
        assert e[0] == pytest.approx(small[0] / E, rel=1e-4)


def test_the_sign_of_the_elastic_stress_is_carried_through():
    for rule in (neuber, glinka):
        s, e = rule(np.array([-600.0]), RO)
        assert s[0] < 0.0 and e[0] < 0.0


def test_the_range_form_is_exactly_twice_the_amplitude_form():
    """Under Massing, doubling both axes multiplies both sides of each
    conservation statement by four, so the doubled solution is the solution.
    That identity is why there is one solver here and not two."""
    for rule, rule_range in ((neuber, neuber_range), (glinka, glinka_range)):
        s, e = rule(ELASTIC, RO)
        ds, de = rule_range(2.0 * ELASTIC, RO)
        assert np.allclose(ds, 2.0 * s, rtol=1e-9)
        assert np.allclose(de, 2.0 * e, rtol=1e-9)


def test_a_stress_concentration_factor_multiplies_the_elastic_input():
    nominal = np.array([200.0])
    a, _ = neuber(nominal, RO, kt=3.0)
    b, _ = neuber(3.0 * nominal, RO)
    assert a[0] == pytest.approx(b[0], rel=1e-12)
