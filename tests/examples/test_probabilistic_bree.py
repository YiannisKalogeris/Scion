"""
The ray construction of the probabilistic Bree example.

Everything else in that example is covered elsewhere -- the closed-form
boundaries in ``tests/test_shakedown/test_bree.py`` and the Heaviside
quadrature in ``tests/test_uq/test_dpim_h.py``. What is only here is the
mapping from material properties to a point of the Bree plane, and that is
exactly where the surface-versus-range convention for ``sigma_T`` enters. A
draft of the accompanying manuscript had it inconsistent, so it gets a test.

python -m pytest tests/examples/test_probabilistic_bree.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.shakedown.probabilistic_bree import (
    ALPHA_MEAN,
    E_MEAN,
    NU,
    RAYS,
    YIELD_MEAN,
    distribution,
    ray_coordinates,
    ray_means,
    thermal_stress,
)
from scion.shakedown import transition_loads


def test_each_ray_lands_where_it_was_asked_to_at_mean_properties():
    """``ray_means`` and ``ray_coordinates`` are inverse, so the ray drawn on
    the deterministic diagram is the ray the random variables are built
    around."""
    for name, (x_hat, y_hat) in RAYS.items():
        delta_T, sigma_p = ray_means(x_hat, y_hat)
        theta = np.array([[YIELD_MEAN, E_MEAN, ALPHA_MEAN, delta_T, sigma_p]])
        got_x, got_y = ray_coordinates(theta)

        assert got_x.item() == pytest.approx(x_hat, rel=1e-12), name
        assert got_y.item() == pytest.approx(y_hat, rel=1e-12), name


def test_the_two_rays_exercise_the_regimes_they_are_chosen_for():
    """Ray A crosses ``y = 2`` into alternating plasticity and ratchets later;
    ray B leaves through ``y = 4(1 - x)`` and ratchets at once. If both took
    the same branch the example would measure one case twice."""
    a = transition_loads(*RAYS["A"])
    b = transition_loads(*RAYS["B"])

    assert not a.alternating_is_empty
    assert float(a.shakedown) < float(a.ratchet)
    assert b.alternating_is_empty
    assert float(b.shakedown) == pytest.approx(float(b.ratchet))


def test_the_thermal_stress_is_the_surface_value():
    """``sigma_T = E alpha DT / [2(1 - nu)]``, so a realization at the
    shakedown boundary ``y = 2`` has a *surface* thermal stress of twice the
    yield stress and a through-thickness variation of four times it. The other
    reading moves every Bree boundary by a factor of two."""
    sigma_t = thermal_stress(E_MEAN, ALPHA_MEAN, 100.0)

    assert sigma_t == pytest.approx(E_MEAN * ALPHA_MEAN * 100.0
                                    / (2.0 * (1.0 - NU)))
    # the value that puts a mean-property realization exactly at y = 2
    delta_T, _ = ray_means(0.0, 2.0)
    assert thermal_stress(E_MEAN, ALPHA_MEAN, delta_T) \
        == pytest.approx(2.0 * YIELD_MEAN)


def test_the_distribution_has_the_tabulated_moments():
    """Lognormals built from moments, not from the underlying normal's
    parameters -- the trap ``LogNormal.from_moments`` exists for."""
    dist = distribution(*ray_means(*RAYS["A"]))
    rng = np.random.default_rng(0)
    sample = dist.sample(200_000, rng=rng)

    assert dist.dim == 5
    for i, (mean, cov) in enumerate([(YIELD_MEAN, 0.08), (E_MEAN, 0.05),
                                     (ALPHA_MEAN, 0.05)]):
        assert sample[:, i].mean() == pytest.approx(mean, rel=0.01)
        assert sample[:, i].std() / mean == pytest.approx(cov, rel=0.05)


def test_no_realization_of_either_ray_has_a_negative_coordinate():
    """``DT`` and ``sigma_P`` are normal with a 10% coefficient of variation,
    so a negative draw is ten standard deviations out. It cannot be ruled out
    in principle, which is why the example counts and reports clipped samples
    rather than assuming; this pins that the count is zero at the sample sizes
    used."""
    for name, (x_hat, y_hat) in RAYS.items():
        dist = distribution(*ray_means(x_hat, y_hat))
        theta = dist.sample(200_000, rng=np.random.default_rng(1))
        x, y = ray_coordinates(theta)

        assert x.min() > 0.0, name
        assert y.min() > 0.0, name
