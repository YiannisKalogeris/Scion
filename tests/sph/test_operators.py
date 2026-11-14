"""
The discrete operators.

Two properties carry the weight here and both are identities rather than
tolerances: the corrected gradient reproduces a linear field exactly, and the
diffusion operator conserves the species exactly. Everything else an SPH
operator can be checked for -- that it is smooth, that it looks like a
Laplacian, that refinement improves it -- is satisfied by formulations that
are wrong by a constant.
"""

import numpy as np
import pytest

from scion.discretization.sph import (WendlandC2, build_neighbors,
                                      kernel_pair_values, lattice,
                                      laplacian_rate,
                                      linear_gradient_correction,
                                      number_density, pair_conductance,
                                      sph_gradient)

DP = 0.02
H = 1.3 * DP
LOWER, UPPER = [0.0, 0.0], [2.0, 0.4]


# NOT named ``setup``: pytest still honours the nose-style module-level
# ``setup(module)`` hook and would call this with the module as its first
# argument, which fails inside the lattice with "module object is not
# iterable" and looks like a bug in the library
def build_case(periodic=(False, False), spacing=DP, jitter=0.0, seed=0):
    kernel = WendlandC2()
    h = 1.3 * spacing
    x, volumes = lattice(LOWER, UPPER, spacing, periodic=periodic)
    if jitter:
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-jitter, jitter, size=x.shape) * spacing
    pairs = build_neighbors(x, kernel.support(h), lower=LOWER, upper=UPPER,
                            periodic=periodic)
    w, dw = kernel_pair_values(pairs, kernel, h, 2)
    return kernel, h, x, volumes, pairs, w, dw


def test_correction_makes_the_linear_gradient_exact():
    """The identity the correction exists for, and the control beside it: a
    correction that did nothing would pass the first assertion alone if the
    tolerance were loose enough."""
    _, _, x, volumes, pairs, _, dw = build_case()
    B = linear_gradient_correction(pairs, volumes, dw, 2, epsilon=0.0)
    slope = np.array([0.3, -0.7])
    phi = x @ slope
    corrected = sph_gradient(pairs, phi, volumes, dw, 2, B)
    bare = sph_gradient(pairs, phi, volumes, dw, 2)
    assert np.abs(corrected - slope).max() < 1e-12
    assert np.abs(bare - slope).max() > 1e-2


def test_correction_is_exact_on_a_disordered_set_too():
    """Not a property of the lattice: the renormalisation tensor is built
    from the actual neighbourhood, so the identity survives jitter. If it
    held only on a lattice it would be measuring the lattice."""
    _, _, x, volumes, pairs, _, dw = build_case(jitter=0.3)
    B = linear_gradient_correction(pairs, volumes, dw, 2, epsilon=0.0)
    slope = np.array([-1.1, 0.4])
    corrected = sph_gradient(pairs, x @ slope, volumes, dw, 2, B)
    assert np.abs(corrected - slope).max() < 1e-10


def test_the_regularisation_costs_seven_digits_of_that_identity():
    """SPHinXsys regularises the inverse with ``sqrt(machine eps)``, and that
    is a real cost, measured rather than assumed: the linear-consistency
    identity degrades from round-off to about 1e-7. Worth knowing before
    reading a convergence study that bottoms out there."""
    _, _, x, volumes, pairs, _, dw = build_case()
    slope = np.array([0.3, -0.7])
    phi = x @ slope
    exact = linear_gradient_correction(pairs, volumes, dw, 2, epsilon=0.0)
    damped = linear_gradient_correction(pairs, volumes, dw, 2)  # sqrt(eps)
    err_exact = np.abs(sph_gradient(pairs, phi, volumes, dw, 2, exact) - slope).max()
    err_damped = np.abs(sph_gradient(pairs, phi, volumes, dw, 2, damped) - slope).max()
    assert err_exact < 1e-12
    assert 1e-9 < err_damped < 1e-5


def test_partition_of_unity_holds_inside_and_fails_at_a_free_surface():
    """The number density is near one away from a boundary and visibly short
    at one -- which is the deficit the correction matrix exists to undo, so a
    test that only looked at the interior would not see it."""
    kernel, h, x, volumes, pairs, w, _ = build_case()
    unity = number_density(pairs, volumes, w) + kernel.W(0.0, h, 2) * volumes
    interior = (x[:, 1] > 4 * DP) & (x[:, 1] < 0.4 - 4 * DP) & \
               (x[:, 0] > 4 * DP) & (x[:, 0] < 2.0 - 4 * DP)
    surface = x[:, 1] < DP          # the first particle row, at y = dp/2
    assert np.abs(unity[interior] - 1.0).max() < 0.02
    assert unity[surface].max() < 0.85


@pytest.mark.parametrize("corrected", [True, False])
def test_diffusion_conserves_the_species_exactly(corrected):
    """Structural, from storing each pair once and updating both ends with
    opposite signs -- so it holds with or without the correction, and a drift
    above round-off is a bookkeeping bug in the pair loop rather than
    inaccuracy."""
    _, _, x, volumes, pairs, _, dw = build_case(periodic=(True, False), jitter=0.2)
    B = linear_gradient_correction(pairs, volumes, dw, 2) if corrected else None
    phi = np.sin(3.0 * x[:, 0]) + x[:, 1]
    weights = pair_conductance(pairs, dw, np.full(len(pairs), 1e-4), B)
    rate = laplacian_rate(pairs, phi, volumes, weights)
    drift = float((rate * volumes).sum())
    scale = float(np.abs(rate * volumes).sum())
    assert abs(drift) < 1e-14 * scale


def test_a_uniform_field_has_no_rate():
    """``phi_i - phi_j`` is identically zero, so this is exact; it is the
    cheapest check that no stray self-term has crept into the sum."""
    _, _, x, volumes, pairs, _, dw = build_case(periodic=(True, False))
    B = linear_gradient_correction(pairs, volumes, dw, 2)
    weights = pair_conductance(pairs, dw, np.full(len(pairs), 1e-4), B)
    rate = laplacian_rate(pairs, np.full(len(x), 0.7), volumes, weights)
    assert np.abs(rate).max() == 0.0


def test_laplacian_recovers_a_known_second_derivative():
    """``phi = sin(pi x)`` on the periodic axis has ``lap phi = -pi^2 phi``.
    Checked on the mid-line, where the free surfaces in ``y`` are fifteen
    smoothing lengths away and do not contribute."""
    _, _, x, volumes, pairs, _, dw = build_case(periodic=(True, False), spacing=0.005)
    B = linear_gradient_correction(pairs, volumes, dw, 2)
    phi = np.sin(np.pi * x[:, 0])
    rate = laplacian_rate(pairs, phi, volumes,
                          pair_conductance(pairs, dw, np.ones(len(pairs)), B))
    mid = np.abs(x[:, 1] - 0.2) < 0.05
    expected = -np.pi**2 * phi[mid]
    assert np.abs(rate[mid] - expected).max() < 0.02 * np.abs(expected).max()
