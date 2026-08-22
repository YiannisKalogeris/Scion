"""Calibration of the biaxial band-angle estimator against known answers.

This exists because two earlier estimators on this problem were *wrong in a way
that read as a physical result*: the recorded "band angle 66 degrees, no trend
with psi" was the instrument, not the model. An estimator that reduces a band
to a centroid cannot work when the band is 24 mm wide in a 25 mm quadrant, and
nothing about its output says so. The only defence is a synthetic band whose
angle is known, in the same geometry.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.mpm.micropolar_biaxial import (HEIGHT, WIDTH,
                                             band_orientation)

SUB = 1.25e-3                      # the h = 2.5 mm run, 2 particles per cell
WIDE = 0.024 / np.sqrt(12.0)       # sigma of a band 24 mm across, as measured


def lattice():
    axes = [np.arange(SUB / 2, e, SUB) for e in (WIDTH, HEIGHT, 2 * SUB)]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")],
                    axis=1)


def straight_band(X, angle_deg, sigma=WIDE):
    t = np.radians(angle_deg)
    normal = np.array([-np.sin(t), np.cos(t)])
    return np.exp(-(X[:, :2] @ normal) ** 2 / (2 * sigma ** 2))


@pytest.mark.parametrize("true", [45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
def test_recovers_a_known_angle(true):
    """Within one degree over the whole range the sweep has to discriminate.

    Roscoe and Coulomb differ by 10 degrees at psi = 10, so an estimator with a
    several-degree error settles nothing.
    """
    X = lattice()
    angle, _ = band_orientation(X, straight_band(X, true))
    assert_allclose(angle, true, atol=1.0)


def test_is_independent_of_the_threshold():
    """The property that distinguishes selecting on ``|grad chi|`` from
    selecting on ``chi``. A chi-threshold picks the band *core*, a plateau where
    the gradient vanishes, so it selects the cells with no orientation
    information -- measured 14 to 31 degrees of swing on the real field."""
    X = lattice()
    chi = straight_band(X, 55.0)
    got = [band_orientation(X, chi, frac=f)[0]
           for f in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert np.nanmax(got) - np.nanmin(got) < 0.5


def test_a_band_filling_the_domain_is_still_measured():
    """The failure that killed both predecessors. At 24 mm across a 25 mm
    quadrant a centroid-based fit reads ~70 degrees whatever the truth."""
    X = lattice()
    angle, _ = band_orientation(X, straight_band(X, 50.0, sigma=WIDE))
    assert_allclose(angle, 50.0, atol=1.0)


def test_the_stretch_changes_the_frame_not_just_the_number():
    """A band direction is a line element and pushes forward with ``F``. At the
    measured 20%-strain stretches this is a 14-degree difference, so reporting
    the wrong frame loses the answer on its own."""
    X = lattice()
    chi = straight_band(X, 59.2)
    ref, _ = band_orientation(X, chi)
    dfm, _ = band_orientation(X, chi, stretch=(1.2921, 0.7807))
    assert_allclose(ref, 59.2, atol=1.0)
    assert_allclose(dfm, 45.3, atol=1.5)
    assert ref - dfm > 10.0


def test_a_uniform_field_has_no_angle():
    """No band means no orientation, and the estimator must say so rather than
    invent one -- the covariance estimator's original sin was returning the
    *specimen's* aspect ratio for a uniformly yielding sample."""
    X = lattice()
    angle, _ = band_orientation(X, np.ones(len(X)))
    assert not np.isfinite(angle)


def test_the_push_forward_must_carry_the_shear_terms():
    """A diagonal stretch is not the deformation gradient.

    Fitting each coordinate separately -- ``polyfit(X[:,0], x[:,0])`` and the
    same for y -- gives only the diagonal and silently drops the shear. On the
    real biaxial field at 20% strain the band-averaged ``F`` is
    ``[1.352 -0.217; 0.195 0.769]`` and the omission moves the angle by 11
    degrees, which is wider than the Roscoe-to-Coulomb gap the whole sweep
    exists to resolve. Asserted against that measured tensor.
    """
    X = lattice()
    chi = straight_band(X, 59.2)
    F = np.array([[1.352, -0.217], [0.195, 0.769]])
    full, _ = band_orientation(X, chi, stretch=F)
    diag, _ = band_orientation(X, chi, stretch=(F[0, 0], F[1, 1]))
    # the synthetic lattice reads 58.9 rather than exactly 59.2, and the
    # diagonal push amplifies that offset, so these are the values this
    # construction gives -- the size of the gap is the content, not the digits
    assert_allclose(full, 55.9, atol=1.5)
    assert_allclose(diag, 43.3, atol=1.5)
    assert full - diag > 8.0
