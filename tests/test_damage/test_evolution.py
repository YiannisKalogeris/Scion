"""
Verification for the damage evolution laws.

Derivatives against finite differences, and every closed-form quantity the laws
advertise -- fracture energy, residual stress, peak stress -- against a
numerical integration or scan of the same law. Those closed forms are what
calibration is built on, so an error in one of them is an error in every
result that used it.

python -m pytest tests/test_damage/test_evolution.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
# not np.trapz / np.trapezoid: NumPy renamed one to the other in 2.0 and
# removed the old name, so either spelling pins the library to one major
# version. SciPy is already a hard dependency and carries both.
from scipy.integrate import trapezoid

from scion.materials.damage.evolution import (
    ExponentialSoftening,
    LinearSoftening,
    MazarsLaw,
    get_damage_law,
)

E = 30.0e3
K0 = 1.0e-4

LAWS = [
    LinearSoftening(K0, 1.0e-2),
    ExponentialSoftening(K0, alpha=1.0, beta=500.0),
    ExponentialSoftening(K0, alpha=0.8, beta=500.0),
    MazarsLaw(K0, A=1.0, B=2.0e4),
    MazarsLaw(K0, A=0.9, B=2.0e4),
]


def _above(law, n=400):
    """A kappa range strictly inside the softening branch."""
    return np.linspace(1.001 * law.kappa_0, law._sample_max(), n)


# ---------------------------------------------------------------------------
# Derivatives and shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_the_slope_is_the_derivative_of_the_damage(law):
    k = _above(law)
    h = 1.0e-11
    fd = (law.D(k + h) - law.D(k - h)) / (2.0 * h)
    analytic = law.dD_dkappa(k)
    assert np.abs(fd - analytic).max() < 1e-5 * np.abs(analytic).max()


@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_damage_is_zero_below_the_threshold_and_monotone_above(law):
    k = np.linspace(0.0, law._sample_max(), 500)
    D = law.D(k)
    assert np.all(D[k <= law.kappa_0] == 0.0)
    assert np.all(np.diff(D) >= -1e-14)
    assert D.min() >= 0.0 and D.max() <= 1.0


@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_the_slope_vanishes_outside_the_active_range(law):
    """Below the threshold and once D has saturated. A nonzero slope in either
    place would put a spurious rank-one term into the consistent tangent at
    points that are not damaging."""
    assert law.dD_dkappa(np.array([0.5 * law.kappa_0]))[0] == 0.0
    assert law.dD_dkappa(np.array([1.0e6]))[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_the_reported_fracture_energy_is_the_area_under_the_curve(law):
    """Calibration rests entirely on this number.

    Integrated over the law's own ``_sample_max``, which is where the decaying
    part is spent -- a fixed range would either miss the corner of the linear
    law or waste points on the tail of the exponential ones.

    A law with a residual stress has no finite area, and the meaningful check
    is then the *rate* at which the area grows: doubling the range must add the
    residual stress times the extra range and nothing else, because the
    decaying part is already gone.
    """
    end = law._sample_max()
    k1 = np.linspace(0.0, end, 400_001)
    area_1 = trapezoid(law.stress(k1, E), k1)

    if np.isfinite(law.fracture_energy(E)):
        assert area_1 == pytest.approx(law.fracture_energy(E), rel=2e-4)
        return

    k2 = np.linspace(0.0, 2.0 * end, 800_001)
    area_2 = trapezoid(law.stress(k2, E), k2)
    assert area_2 - area_1 == pytest.approx(law.residual_stress(E) * end,
                                            rel=1e-3)


@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_the_reported_residual_stress_is_the_limit_of_the_curve(law):
    far = np.array([law.kappa_0 + 500.0
                    / getattr(law, "beta", getattr(law, "B", 1.0))])
    assert law.stress(far, E)[0] == pytest.approx(law.residual_stress(E),
                                                  abs=1e-9 * E * law.kappa_0)


def test_a_residual_stress_means_an_infinite_fracture_energy():
    """Reported as inf rather than as a large finite number, because the
    material genuinely never releases -- and a large finite number would be
    used for calibration without anyone noticing."""
    partial = ExponentialSoftening(K0, alpha=0.8, beta=500.0)
    assert np.isinf(partial.fracture_energy(E))
    assert partial.residual_stress(E) == pytest.approx(0.2 * E * K0)

    complete = ExponentialSoftening(K0, alpha=1.0, beta=500.0)
    assert np.isfinite(complete.fracture_energy(E))
    assert complete.residual_stress(E) == 0.0


# ---------------------------------------------------------------------------
# Linear softening: the law with a closed form for everything
# ---------------------------------------------------------------------------

def test_linear_softening_makes_the_stress_linear_not_the_damage():
    """The reason the law is written the way it is. The stress runs straight
    from E kappa_0 to exactly zero, which is what gives the 1-D bar a closed
    form to verify against."""
    law = LinearSoftening(K0, 1.0e-2)
    k = np.linspace(K0, 1.0e-2, 200)
    s = law.stress(k, E)
    assert s[0] == pytest.approx(E * K0)
    assert s[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.diff(s, 2), 0.0, atol=1e-9)      # straight line


def test_linear_softening_reaches_full_damage_at_a_finite_strain():
    law = LinearSoftening(K0, 1.0e-2)
    assert law.D(np.array([1.0e-2]))[0] == pytest.approx(1.0)
    assert law.D(np.array([2.0e-2]))[0] == 1.0             # clipped, not >1


def test_the_fracture_energy_calibration_round_trips():
    law = LinearSoftening.from_fracture_energy(K0, E, g_f=0.15)
    assert law.fracture_energy(E) == pytest.approx(0.15)
    assert law.kappa_u == pytest.approx(2.0 * 0.15 / (E * K0))


def test_an_ultimate_strain_below_the_threshold_is_refused():
    with pytest.raises(ValueError, match="kappa_u"):
        LinearSoftening(1.0e-3, 1.0e-4)


# ---------------------------------------------------------------------------
# The Mazars hardening trap
# ---------------------------------------------------------------------------

def test_mazars_keeps_hardening_when_b_kappa_0_is_below_one():
    """The strength is then not E kappa_0 and a calibration built on that
    assumption is wrong by the overshoot -- a factor of 7.7 at these
    parameters, not a rounding difference."""
    law = MazarsLaw(kappa_0=1.0e-4, A=1.0, B=500.0)
    assert law.B * law.kappa_0 < 1.0
    assert not law.softens_immediately()

    k = np.linspace(law.kappa_0, law.kappa_0 + 12.0 / law.B, 200_001)
    sampled = law.stress(k, E).max()
    assert law.peak_stress(E) == pytest.approx(sampled, rel=1e-5)
    assert law.peak_stress(E) / (E * law.kappa_0) == pytest.approx(7.7, rel=0.02)


def test_mazars_softens_at_once_when_b_kappa_0_exceeds_one():
    law = MazarsLaw(kappa_0=1.0e-4, A=1.0, B=2.0e4)
    assert law.softens_immediately()
    assert law.peak_stress(E) == pytest.approx(E * law.kappa_0)


@pytest.mark.parametrize("law", LAWS, ids=lambda l: repr(l)[:34])
def test_the_peak_stress_is_the_largest_stress_the_law_reaches(law):
    k = np.linspace(0.0, law._sample_max(), 200_001)
    assert law.stress(k, E).max() <= law.peak_stress(E) * (1.0 + 1e-6)


# ---------------------------------------------------------------------------

def test_check_reports_the_properties_that_decide_usability():
    info = LinearSoftening(K0, 1.0e-2).check(E)
    assert info["monotone"] and info["softens_immediately"]
    assert info["D_max"] == pytest.approx(1.0)
    assert info["residual_stress"] == 0.0
    assert info["fracture_energy"] == pytest.approx(0.5 * E * K0 * 1.0e-2)


def test_the_registry_resolves_and_rejects():
    assert isinstance(get_damage_law("peerlings", kappa_0=K0),
                      ExponentialSoftening)
    assert isinstance(get_damage_law("linear", kappa_0=K0, kappa_u=1e-2),
                      LinearSoftening)
    with pytest.raises(NotImplementedError):
        get_damage_law("lemaitre", kappa_0=K0)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError, match="kappa_0"):
        LinearSoftening(0.0, 1e-2)
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSoftening(K0, alpha=1.5)
    with pytest.raises(ValueError, match="beta"):
        ExponentialSoftening(K0, beta=-1.0)
    with pytest.raises(ValueError, match="A must"):
        MazarsLaw(K0, A=0.0)
