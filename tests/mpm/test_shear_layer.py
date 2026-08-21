"""Shear-band regularisation: does the band have a width of its own?

The claim the whole micropolar chain was built for. The *full* three-mesh
refinement study lives in ``examples/mpm/micropolar_shear_layer.py`` and takes
about 40 minutes; what is asserted here is the single discriminating comparison
at the coarsest mesh, which runs in about two, following the same split as the
curved-cantilever benchmark.

Measured by the example, 100 mm layer, cells of 10 / 5 / 2.5 mm:

    micropolar (l_b = 20 mm)   band 70 -> 65 -> 65 mm    spread 1.08x
                               in cells   7 -> 13 -> 26
                               reaction spread 0.17%,  chi_max spread 1.15x

    no internal length         band 20 ->  5 ->  5 mm    spread 4.00x
                               in cells   2 ->  1 ->  2
                               reaction spread 11.8%,   chi_max spread 3.57x

A width that holds in *millimetres* while the cell size halves is the material
setting the scale. A width that holds in *cells* is the mesh setting it.
"""

import numpy as np
import pytest

from examples.mpm.micropolar_shear_layer import (
    LENGTH_CLASSICAL,
    LENGTH_MICROPOLAR,
    band_profile,
    run,
)

COARSE = 10


@pytest.fixture(scope="module")
def solved():
    out = {}
    for name, l_b in (("micropolar", LENGTH_MICROPOLAR),
                      ("classical", LENGTH_CLASSICAL)):
        particles, stepper, X, h, curve, _ = run(COARSE, l_b)
        prof, width = band_profile(particles, X, h)
        out[name] = dict(width=width, h=h, prof=prof, curve=curve,
                         chi=particles.hardening.max(),
                         converged=all(r.converged for r in stepper.history),
                         steps=len(stepper.history))
    return out


def test_both_runs_complete(solved):
    for name, r in solved.items():
        assert r["converged"], name
        assert r["chi"] > 0.0, name           # something actually yielded


def test_the_internal_length_widens_the_band(solved):
    """With ``l_b`` the band spans several cells; without it, one or two.

    This is the whole point of the regularisation and it must be visible even
    at the coarsest mesh, where ``l_b`` is only two cells across.
    """
    assert solved["micropolar"]["width"] > 3.0 * solved["classical"]["width"]
    assert solved["micropolar"]["width"] / solved["micropolar"]["h"] >= 4.0
    assert solved["classical"]["width"] / solved["classical"]["h"] <= 3.0


def test_the_unregularised_band_concentrates_more_plastic_strain(solved):
    """Dissipation collapsing into a thinner zone means a larger strain inside
    it. Measured at the coarsest mesh alone the ratio is ~3x, and it *grows*
    with refinement (3.57x across the three meshes) -- which is the unphysical
    part: there is no limit to it."""
    assert solved["classical"]["chi"] > 2.0 * solved["micropolar"]["chi"]


def test_the_profile_is_smooth_with_a_length_and_abrupt_without(solved):
    """The shape, not just the width. With ``l_b`` the plastic multiplier decays
    smoothly to the layer's faces; without it, cells adjacent to the band carry
    essentially nothing."""
    for name, r in solved.items():
        prof = r["prof"] / r["prof"].max()
        edges = np.concatenate([prof[:2], prof[-2:]]).mean()
        if name == "micropolar":
            assert edges > 0.05, prof
        else:
            assert edges < 0.02, prof


def test_localisation_costs_the_unregularised_solve_more_steps(solved):
    """Not a correctness claim, a robustness one, and the reference makes it
    too: untreated localisation converges poorly. Measured 215 load steps
    against 120, both for the same requested increment -- the difference is
    step cut-back."""
    assert solved["classical"]["steps"] > solved["micropolar"]["steps"]
