"""
Verification for the life curves and the mean-stress corrections.

The external benchmark is EN 1993-1-9, whose detail-category geometry is fixed
by the standard and reproduced here exactly. The rest are inverse round trips --
the analogue, for a curve, of the finite-difference check a derivative gets:
``cycles`` and ``amplitude`` are derived separately, and a sign or a factor of
two in either shows up as a failure to compose to the identity.

python -m pytest tests/test_fatigue/test_curves.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.fatigue.curves import (
    Basquin,
    CoffinManson,
    StrainLife,
    TwoSlopeSN,
    get_curve,
)
from scion.fatigue.mean_stress import (
    Gerber,
    Goodman,
    ModifiedMorrow,
    Morrow,
    MorrowStress,
    NoCorrection,
    NoStrainCorrection,
    SmithWatsonTopper,
    Soderberg,
    SWT,
    Walker,
    get_mean_stress,
    get_strain_mean_stress,
)

E, SIGMA_U, SIGMA_Y = 210.0e3, 600.0, 420.0
LIVES = np.array([1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])

SL = dict(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60)


# ---------------------------------------------------------------------------
# External benchmark: EN 1993-1-9
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", [36.0, 71.0, 90.0, 160.0])
def test_eurocode3_detail_category_geometry(category):
    """The standard fixes three points exactly, and they are definitions rather
    than fits: the category is the stress range at 2x10^6 with m = 3, the
    constant-amplitude limit sits at 5x10^6, and the cut-off at 10^8 with m = 5
    between them.
    """
    c = TwoSlopeSN.eurocode3(category)

    # the category itself: a *range*, so halved into an amplitude
    assert c.cycles(np.array([0.5 * category])) == pytest.approx(2.0e6)

    # the constant-amplitude fatigue limit
    d_d = (2.0 / 5.0) ** (1.0 / 3.0) * category
    assert 2.0 * c.a_k == pytest.approx(d_d)
    assert c.cycles(np.array([c.a_k])) == pytest.approx(5.0e6)

    # the cut-off, five decades on at the shallower slope
    d_l = (5.0 / 100.0) ** (1.0 / 5.0) * d_d
    assert 2.0 * c.cutoff_amplitude == pytest.approx(d_l)
    assert np.isinf(c.cycles(np.array([0.999 * c.cutoff_amplitude])))


def test_eurocode3_slopes_are_three_then_five():
    """Read off the curve rather than off the constructor arguments."""
    c = TwoSlopeSN.eurocode3(90.0)
    for n1, n2, m in ((1.0e5, 1.0e6, 3.0), (1.0e7, 5.0e7, 5.0)):
        a1, a2 = c.amplitude(np.array([n1, n2]))
        got = np.log(n2 / n1) / np.log(a1 / a2)
        assert got == pytest.approx(m, rel=1e-9)


def test_the_detail_category_is_a_range_and_the_halving_is_visible():
    """A factor of two here is the single most likely error in the module, so
    it is asserted rather than assumed: the amplitude at 2x10^6 is half the
    category, not the category."""
    c = TwoSlopeSN.eurocode3(160.0)
    assert c.amplitude(np.array([2.0e6]))[0] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

def _curves():
    return [
        Basquin(sigma_f=1200.0, b=-0.095),
        Basquin.from_slope(m=3.0, amplitude_ref=100.0, cycles_ref=2.0e6),
        Basquin.estimate_steel(SIGMA_U),
        TwoSlopeSN.eurocode3(90.0),
        TwoSlopeSN.haibach(80.0, 5.0e6, m=3.0),
        CoffinManson(eps_f=0.45, c=-0.60),
        StrainLife(**SL),
        StrainLife(**SL, mean_stress="morrow"),
        StrainLife(**SL, mean_stress="modified-morrow"),
    ]


@pytest.mark.parametrize("curve", _curves(), ids=lambda c: repr(c)[:28])
def test_cycles_inverts_amplitude(curve):
    """``cycles(amplitude(N)) == N``. Catches a sign on an exponent, a missing
    reversal factor, and a wrong branch in a piecewise curve, none of which
    produce an implausible number on their own."""
    n = LIVES[LIVES < 5.0e6]                  # inside every curve's live range
    got = curve.cycles(curve.amplitude(n))
    assert np.allclose(got, n, rtol=1e-6)


@pytest.mark.parametrize("curve", _curves(), ids=lambda c: repr(c)[:28])
def test_life_decreases_with_amplitude(curve):
    a = curve.amplitude(LIVES)
    n = curve.cycles(a)
    assert np.all(np.diff(n[np.isfinite(n)]) > 0.0)     # a is decreasing
    assert np.all(np.diff(a) < 0.0)


def test_basquin_matches_its_closed_form():
    c = Basquin(sigma_f=1200.0, b=-0.095)
    n = np.array([1.0e4])
    assert c.amplitude(n)[0] == pytest.approx(1200.0 * (2.0e4) ** -0.095)
    assert c.cycles(c.amplitude(n))[0] == pytest.approx(1.0e4)


def test_basquin_from_points_passes_through_both():
    c = Basquin.from_points((450.0, 1.0e3), (200.0, 1.0e6))
    assert c.cycles(np.array([450.0]))[0] == pytest.approx(1.0e3, rel=1e-9)
    assert c.cycles(np.array([200.0]))[0] == pytest.approx(1.0e6, rel=1e-9)


def test_the_code_slope_and_basquins_exponent_are_reciprocal():
    """``b = -1/m``. The reversal factor cancels in a ratio of lives, so no
    factor of two belongs in this relation -- which is exactly why it is easy
    to insert one."""
    c = Basquin.from_slope(m=3.0, amplitude_ref=100.0, cycles_ref=2.0e6)
    assert c.slope == pytest.approx(3.0)
    assert c.b == pytest.approx(-1.0 / 3.0)
    a1, a2 = c.amplitude(np.array([1.0e5, 1.0e8]))
    assert (a1 / a2) ** 3 == pytest.approx(1000.0, rel=1e-9)


def test_an_endurance_limit_gives_infinite_life_below_it_only():
    c = Basquin(sigma_f=1200.0, b=-0.095, endurance_limit=100.0)
    assert np.isinf(c.cycles(np.array([99.9]))[0])
    assert np.isfinite(c.cycles(np.array([100.1]))[0])


def test_a_positive_exponent_is_refused():
    with pytest.raises(ValueError, match="negative"):
        Basquin(sigma_f=1000.0, b=0.1)
    with pytest.raises(ValueError, match="negative"):
        CoffinManson(eps_f=0.5, c=0.5)


def test_haibach_uses_the_shallower_slope_beyond_the_knee():
    c = TwoSlopeSN.haibach(80.0, 5.0e6, m=3.0)
    assert c.m2 == pytest.approx(5.0)
    assert c.cutoff_amplitude is None
    assert np.isfinite(c.cycles(np.array([1.0]))[0])   # never runs out


def test_a_steeper_second_slope_is_refused():
    """It would make small cycles more damaging than large ones."""
    with pytest.raises(ValueError, match="shallower"):
        TwoSlopeSN(80.0, 5.0e6, m1=5.0, m2=3.0)


# ---------------------------------------------------------------------------
# Strain-life
# ---------------------------------------------------------------------------

def test_the_transition_life_is_where_the_two_terms_cross():
    c = StrainLife(**SL)
    nt = c.transition_life()
    assert c.elastic_amplitude(nt) == pytest.approx(c.plastic_amplitude(nt),
                                                    rel=1e-9)
    assert c.amplitude(nt) == pytest.approx(2.0 * c.elastic_amplitude(nt),
                                            rel=1e-9)


def test_the_two_terms_dominate_on_opposite_sides_of_the_transition():
    c = StrainLife(**SL)
    nt = c.transition_life()
    assert c.plastic_amplitude(0.01 * nt) > 10.0 * c.elastic_amplitude(0.01 * nt)
    assert c.elastic_amplitude(100.0 * nt) > 10.0 * c.plastic_amplitude(100.0 * nt)


def test_the_elastic_term_is_exactly_basquin():
    """Not approximately: the same two constants. A stress-life and a
    strain-life assessment of the same high-cycle history must therefore agree,
    and any discrepancy is in the counting rather than in the material data."""
    c = StrainLife(**SL)
    b = c.basquin()
    n = np.array([1.0e5, 1.0e7])
    assert np.allclose(b.cycles(c.stress_amplitude(n)), n, rtol=1e-9)
    assert np.allclose(c.stress_amplitude(n), E * c.elastic_amplitude(n))


def test_strain_life_agrees_with_basquin_where_the_plastic_term_is_negligible():
    """The two families converge at long life by construction, so this checks
    the root solve rather than the physics."""
    c = StrainLife(**SL)
    n = np.array([1.0e8])
    eps_a = c.amplitude(n)                      # total, mostly elastic
    from_strain = c.cycles(eps_a)
    from_stress = c.basquin().cycles(E * eps_a)
    assert from_strain[0] == pytest.approx(n[0], rel=1e-6)
    assert from_stress[0] == pytest.approx(n[0], rel=0.15)


def test_c_must_be_steeper_than_b():
    with pytest.raises(ValueError, match="steeper"):
        StrainLife(E=E, sigma_f=1000.0, b=-0.6, eps_f=0.45, c=-0.095)


def test_the_uniform_material_law_is_self_consistent():
    """The constants are Baeumel and Seeger's; only their arithmetic is checked
    here, since the estimate itself is not something the library can verify."""
    c = StrainLife.uniform_material_law(sigma_u=SIGMA_U, E=E)
    assert c.sigma_f == pytest.approx(1.5 * SIGMA_U)
    assert c.b == pytest.approx(-0.087)
    assert c.eps_f == pytest.approx(0.59)      # sigma_u/E < 0.003, so psi = 1
    assert c.c == pytest.approx(-0.58)
    assert np.allclose(c.cycles(c.amplitude(LIVES)), LIVES, rtol=1e-6)


def test_runout_is_reported_rather_than_extrapolated():
    """The equation has no endurance limit, so a life beyond max_reversals is
    arithmetic and not prediction; it comes back as inf."""
    c = StrainLife(**SL, max_reversals=1.0e10)
    assert np.isinf(c.cycles(np.array([1.0e-9]))[0])


def test_an_amplitude_beyond_the_curve_fails_immediately():
    c = StrainLife(**SL)
    assert c.cycles(np.array([10.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# Strain-life mean stress
# ---------------------------------------------------------------------------

def test_swt_reproduces_the_fully_reversed_life_exactly():
    """A strong check, because it is an algebraic identity rather than a
    tolerance: multiplying the strain-life equation by the stress-life one gives
    the SWT curve, so a fully reversed cycle placed on one lands on the other.
    """
    plain = StrainLife(**SL)
    swt = StrainLife(**SL, mean_stress="swt")
    n = np.array([1.0e3, 1.0e5, 1.0e7])
    eps_a = plain.amplitude(n)
    s_max = plain.stress_amplitude(n)             # fully reversed: max = amp
    assert np.allclose(swt.cycles(eps_a, max_stress=s_max), n, rtol=1e-6)


def test_morrow_reduces_to_the_plain_curve_at_zero_mean():
    plain = StrainLife(**SL)
    for name in ("morrow", "modified-morrow"):
        c = StrainLife(**SL, mean_stress=name)
        eps_a = plain.amplitude(LIVES)
        assert np.allclose(c.cycles(eps_a, mean_stress=0.0), LIVES, rtol=1e-6)


@pytest.mark.parametrize("name", ["morrow", "modified-morrow", "swt"])
def test_a_tensile_mean_shortens_life_and_a_compressive_one_extends_it(name):
    c = StrainLife(**SL, mean_stress=name)
    eps_a = np.full(3, 0.004)
    means = np.array([-200.0, 0.0, 200.0])
    kw = {"max_stress": eps_a * E * 0.5 + means} if name == "swt" else {}
    n = c.cycles(eps_a, mean_stress=means, **kw)
    assert n[0] > n[1] > n[2]


def test_morrow_refuses_a_mean_stress_above_the_fatigue_strength():
    """The elastic term goes negative there and the curve stops being monotone,
    so the equation has no meaningful root. Reported as zero life, not as
    whatever the solver last held."""
    c = StrainLife(**SL, mean_stress="morrow")
    assert c.cycles(np.array([0.004]), mean_stress=SL["sigma_f"] + 1.0)[0] == 0.0


def test_swt_needs_the_maximum_stress_and_says_so():
    c = StrainLife(**SL, mean_stress="swt")
    with pytest.raises(ValueError, match="max_stress"):
        c.cycles(np.array([0.004]), mean_stress=100.0)


def test_swt_reports_a_wholly_compressive_loop_as_harmless():
    """A real prediction, and a known over-severity of the parameter, not a
    numerical guard."""
    c = StrainLife(**SL, mean_stress="swt")
    assert np.isinf(c.cycles(np.array([0.004]), max_stress=-50.0)[0])


def test_only_swt_reads_the_maximum_stress():
    assert SmithWatsonTopper().needs_max_stress
    assert not Morrow().needs_max_stress
    assert not NoStrainCorrection().needs_stress
    assert Morrow().needs_stress


# ---------------------------------------------------------------------------
# Stress-life mean stress
# ---------------------------------------------------------------------------

def test_every_correction_is_the_identity_at_zero_mean():
    for c in (NoCorrection(), Goodman(SIGMA_U), Soderberg(SIGMA_Y),
              Gerber(SIGMA_U), MorrowStress(1000.0), Walker(0.6), SWT()):
        got = c.equivalent_amplitude(np.array([150.0]), 0.0)
        assert got[0] == pytest.approx(150.0, rel=1e-12), c.name


def test_the_conservatism_ordering_of_the_linear_family():
    """Soderberg (yield) is more severe than Goodman (ultimate), which is more
    severe than Gerber (parabola through the same endpoints)."""
    a, m = np.array([150.0]), np.array([200.0])
    sod = Soderberg(SIGMA_Y).equivalent_amplitude(a, m)[0]
    good = Goodman(SIGMA_U).equivalent_amplitude(a, m)[0]
    ger = Gerber(SIGMA_U).equivalent_amplitude(a, m)[0]
    assert sod > good > ger > a[0]


def test_a_mean_stress_at_the_static_strength_gives_zero_life():
    """Not a negative amplitude and not a sign flip: the part failed on the
    first cycle and the fatigue question is the wrong one."""
    c = Goodman(SIGMA_U)
    assert np.isinf(c.equivalent_amplitude(np.array([10.0]), SIGMA_U)[0])
    assert np.isinf(c.equivalent_amplitude(np.array([10.0]), 2.0 * SIGMA_U)[0])
    curve = Basquin(sigma_f=1200.0, b=-0.095, mean_stress=c)
    assert curve.cycles(np.array([10.0]), mean_stress=SIGMA_U)[0] == 0.0


def test_the_linear_family_takes_no_credit_for_a_compressive_mean_by_default():
    """Goodman extrapolated to large negative means predicts unbounded benefit
    and Gerber predicts damage, both without evidence. The default truncates."""
    a, m = np.array([150.0]), np.array([-300.0])
    assert Goodman(SIGMA_U).equivalent_amplitude(a, m)[0] == pytest.approx(150.0)
    assert Gerber(SIGMA_U).equivalent_amplitude(a, m)[0] == pytest.approx(150.0)

    lenient = Goodman(SIGMA_U, compressive="extrapolate")
    assert lenient.equivalent_amplitude(a, m)[0] < 150.0
    harmful = Gerber(SIGMA_U, compressive="extrapolate")
    assert harmful.equivalent_amplitude(a, m)[0] > 150.0     # the defect


def test_walker_and_swt_do_take_that_credit():
    """Their power-law form is fitted across R ratios including negative ones,
    so the compressive branch is measured rather than extrapolated."""
    a, m = np.array([150.0]), np.array([-100.0])
    assert SWT().equivalent_amplitude(a, m)[0] < 150.0
    assert Walker(0.6).equivalent_amplitude(a, m)[0] < 150.0


def test_swt_is_walker_at_gamma_one_half():
    a, m = np.array([100.0, 150.0]), np.array([50.0, -20.0])
    assert np.allclose(SWT().equivalent_amplitude(a, m),
                       Walker(0.5).equivalent_amplitude(a, m))
    assert np.allclose(SWT().equivalent_amplitude(a, m),
                       np.sqrt((m + a) * a))


def test_walker_at_gamma_one_is_no_correction_at_all():
    a, m = np.array([100.0]), np.array([200.0])
    assert Walker(1.0).equivalent_amplitude(a, m)[0] == pytest.approx(100.0)


def test_a_loop_that_never_goes_into_tension_is_harmless_under_walker():
    a, m = np.array([50.0]), np.array([-200.0])
    assert SWT().equivalent_amplitude(a, m)[0] == 0.0


def test_an_unknown_compressive_mode_is_refused():
    with pytest.raises(ValueError, match="compressive"):
        Goodman(SIGMA_U, compressive="maybe")


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def test_registries_resolve_and_reject():
    assert isinstance(get_curve("basquin", sigma_f=1000.0, b=-0.1), Basquin)
    assert isinstance(get_curve("s-n", sigma_f=1000.0, b=-0.1), Basquin)
    assert isinstance(get_mean_stress("goodman", sigma_u=600.0), Goodman)
    assert isinstance(get_strain_mean_stress("swt"), SmithWatsonTopper)
    assert isinstance(get_strain_mean_stress("modified_morrow"), ModifiedMorrow)

    with pytest.raises(NotImplementedError):
        get_curve("weibull")
    with pytest.raises(NotImplementedError):
        get_mean_stress("findley")


def test_the_two_mean_stress_families_are_not_interchangeable():
    """Applying a Goodman correction to a strain-life curve is a category
    error, and the message says which registry to use instead."""
    with pytest.raises(NotImplementedError, match="get_mean_stress"):
        get_strain_mean_stress("goodman")


def test_a_curve_declares_which_channel_it_counts():
    assert Basquin(sigma_f=1.0, b=-0.1).variable == "stress"
    assert StrainLife(**SL).variable == "strain"
    assert CoffinManson(0.5, -0.6).variable == "plastic-strain"
