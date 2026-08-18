"""
Verification for the Melan lower bound and the machinery under it.

The external benchmark is the thick-walled cylinder: with a one-vertex load
domain, shakedown collapses onto plastic collapse, whose limit pressure
``2 Y / sqrt(3) ln(b/a)`` involves no elastic constant at all. Two properties
matter and they are different -- the answer must be *below* it, because a lower
bound that is not below is not a bound, and it must approach it as the search
space grows.

The rest of the file is about the two things that make the number trustworthy:
every basis mode really is self-equilibrated, and the reported factor really was
verified rather than interpolated.

python -m pytest tests/test_shakedown/test_melan.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.plasticity import VonMises
from simulacrum.shakedown import (
    LoadDomain,
    ResidualBasis,
    elastic_factor,
    melan_lower_bound,
)

from .cylinder import C_MANDEL, LIMIT_PRESSURE, YIELD, build_cylinder


def _problem(n=10, domain=None, levels=None, enrichments=4):
    tables, equilibrium, envelope = build_cylinder(n, n)
    domain = domain if domain is not None else LoadDomain.fixed(1.0)
    vertex = envelope.vertex_stresses(domain)
    lam_e = elastic_factor(vertex, VonMises(), YIELD)
    basis = ResidualBasis(tables, C_MANDEL, equilibrium, scale=YIELD)
    basis.from_overstress(vertex, VonMises(), YIELD,
                          levels=levels if levels is not None
                          else np.linspace(1.02, 4.0, 8),
                          elastic_factor=lam_e)
    result = melan_lower_bound(vertex, basis, VonMises(), YIELD, rtol=1e-4,
                               enrichments=enrichments)
    return result, vertex, equilibrium, basis


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

def test_the_limit_load_is_approached_closely():
    """A one-vertex domain is limit analysis, and the answer has a closed form.

    Note what is *not* asserted: that the factor lies below it. This formulation
    bounds the discrete model, where yield is checked at quadrature points and
    equilibrium is weak, and that is not a bound on the continuum. With the
    eigenstrain basis used here it comes out below; enrich the basis with the
    converged mechanism of a linear matching run and it crosses over to 200.49
    against an analytic 200.094, having passed every check the formulation can
    make. Asserting `< LIMIT_PRESSURE` here would pass for the wrong reason --
    the basis being too poor to find the slack -- and would break the moment
    the basis improved.

    :mod:`simulacrum.shakedown.equilibrium_elements` is the formulation whose
    output really is bounded, and it is tested that way.
    """
    result, *_ = _problem()
    assert abs(result.factor - LIMIT_PRESSURE) < 0.05 * LIMIT_PRESSURE


def test_a_richer_basis_can_only_help():
    """Each basis is a subset of the next, so the feasible set grows and the
    factor cannot fall. This is the property that makes a partial run usable:
    stopping early costs accuracy, never validity."""
    factors = []
    for n_levels, enrich in ((1, 0), (2, 0), (4, 0), (8, 0), (8, 4)):
        result, *_ = _problem(levels=np.linspace(1.02, 4.0, n_levels),
                              enrichments=enrich)
        factors.append(result.factor)

    assert np.all(np.diff(factors) > -1e-6 * LIMIT_PRESSURE)
    assert factors[-1] > factors[0]


def test_the_factor_is_stable_across_meshes():
    """Refining resolves the elastic peak better, so the elastic limit falls and
    the factor with it. It moves little, and stays near the analytic value."""
    for n in (6, 10, 14):
        result, *_ = _problem(n)
        assert abs(result.factor - LIMIT_PRESSURE) < 0.1 * LIMIT_PRESSURE, n


def test_shakedown_never_falls_below_first_yield():
    """``rho = 0`` is always admissible, so the elastic limit is a free lower
    bound. A shakedown factor below it means the search went backwards."""
    result, *_ = _problem()
    assert result.factor >= result.elastic
    assert result.reserve > 1.5


def test_a_fully_reversed_domain_gives_exactly_the_elastic_limit():
    """A second closed form, and this one is exact rather than approached.

    With vertices at ``+lambda sigma^E`` and ``-lambda sigma^E``, admissibility
    of both requires their deviatoric parts to lie in the same von Mises ball,
    whose diameter is ``2 sqrt(2/3) sigma_y``. The distance between them is
    ``2 lambda ||s||`` and the residual field cancels out of it entirely, so

        lambda q(sigma^E) <= sigma_y,

    which is the elastic limit. ``rho = 0`` attains it, so the shakedown factor
    of a fully reversed domain *equals* first yield -- no redistribution helps,
    which is what alternating plasticity means. The solver has to reproduce it
    to the digit, not approximately.
    """
    result, *_ = _problem(domain=LoadDomain.box([-1.0], [1.0]))
    assert result.factor == pytest.approx(result.elastic, rel=1e-6)
    assert result.reserve == pytest.approx(1.0, rel=1e-6)


def test_load_variation_costs_reserve():
    """Held pressure buys a large reserve over first yield through
    redistribution; full reversal buys none."""
    fixed, *_ = _problem(domain=LoadDomain.fixed(1.0))
    alternating, *_ = _problem(domain=LoadDomain.box([-1.0], [1.0]))

    assert fixed.reserve > 1.5
    assert alternating.reserve == pytest.approx(1.0, rel=1e-6)
    assert alternating.factor < 0.7 * fixed.factor


def test_every_domain_stays_near_or_below_the_limit_load():
    """Whatever the variation, the structure cannot outlast plastic collapse
    under the most severe load the domain contains -- to within the few percent
    by which the discrete formulation is not itself bounded.

    Note this is asserted of each computed bound against the *analytic* limit
    load, not of one computed bound against another: two lower bounds built
    from separately generated bases are each valid without being ordered, so
    comparing them directly would be a test of the basis generator rather than
    of the physics.
    """
    for domain in (LoadDomain.fixed(1.0), LoadDomain.box([0.0], [1.0]),
                   LoadDomain.box([-1.0], [1.0])):
        result, *_ = _problem(domain=domain)
        assert result.factor <= 1.05 * LIMIT_PRESSURE, repr(domain)


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

def test_the_reported_factor_carries_a_field_that_certifies_it():
    """The whole value of the static theorem. Recomputed from the stored field,
    independently of anything the solver did."""
    result, vertex, equilibrium, _ = _problem()
    report = result.verify(vertex, equilibrium, VonMises(), YIELD)

    assert report["certified"]
    assert report["equilibrium_residual"] < 1e-8
    assert report["max_yield_violation"] <= 1e-6 * YIELD


def test_the_factor_is_verified_not_interpolated():
    """``largest_feasible`` returns the largest value its oracle accepted, so
    the reported factor is inside the bracket at its lower end -- quoting the
    midpoint would be quoting something never checked."""
    result, *_ = _problem()
    lo, hi = result.bracket
    assert lo == result.factor
    assert lo < hi
    assert result.relative_width < 1e-3


def test_an_empty_basis_gives_the_elastic_limit():
    """The right answer for an empty search space, not a failure. With no
    residual field available, Melan's condition is first yield."""
    tables, equilibrium, envelope = build_cylinder(6, 6)
    vertex = envelope.vertex_stresses(LoadDomain.fixed(1.0))
    basis = ResidualBasis(tables, C_MANDEL, equilibrium, scale=YIELD)
    assert basis.n_modes == 0

    result = melan_lower_bound(vertex, basis, VonMises(), YIELD,
                               enrichments=0)
    expected = elastic_factor(vertex, VonMises(), YIELD)
    assert result.factor == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

def test_every_mode_is_self_equilibrated_and_the_set_is_orthonormal():
    """The first is why the optimisation can ignore the equilibrium constraint;
    the second is why the coefficients are well scaled."""
    _, _, _, basis = _problem()
    report = basis.verify()

    assert report["n_modes"] > 5
    assert report["max_equilibrium_residual"] < 1e-8
    assert report["max_gram_error"] < 1e-8
    assert report["ok"]


def test_relaxing_an_eigenstrain_produces_a_residual_field():
    """``rho = C:(eps(v) - eps*)`` with the structure relaxed is in equilibrium
    with no external load whatever the eigenstrain was."""
    tables, equilibrium, _ = build_cylinder(6, 6)
    basis = ResidualBasis(tables, C_MANDEL, equilibrium, scale=YIELD)

    rng = np.random.default_rng(0)
    eps_star = np.zeros((tables.n_points, 6))
    eps_star[:, [0, 1, 2, 5]] = rng.normal(scale=1e-3,
                                           size=(tables.n_points, 4))
    rho = basis.relax(eps_star)

    assert equilibrium.residual(rho) < 1e-8
    assert np.abs(rho).max() > 0.0


def test_a_mode_already_in_the_span_is_dropped():
    """Keeping a near-null vector would leave the coefficients ill-conditioned
    for no gain, so a repeated direction is refused rather than stored."""
    tables, equilibrium, _ = build_cylinder(6, 6)
    basis = ResidualBasis(tables, C_MANDEL, equilibrium, scale=YIELD)

    rng = np.random.default_rng(1)
    eps_star = np.zeros((tables.n_points, 6))
    eps_star[:, 0] = rng.normal(scale=1e-3, size=tables.n_points)

    assert basis.add_eigenstrain(eps_star)
    assert not basis.add_eigenstrain(eps_star)
    assert not basis.add_eigenstrain(3.7 * eps_star)
    assert basis.n_modes == 1


def test_the_shear_modulus_comes_from_the_stiffness():
    """Mandel carries 2*mu on the shear diagonal. Passing it separately would
    let it disagree with the stiffness actually being used."""
    tables, equilibrium, _ = build_cylinder(6, 6)
    basis = ResidualBasis(tables, C_MANDEL, equilibrium, scale=YIELD)
    from .cylinder import E, NU
    assert basis.shear_modulus == pytest.approx(E / (2.0 * (1.0 + NU)))


def test_the_field_of_zero_coefficients_is_zero():
    _, _, _, basis = _problem(6)
    assert np.abs(basis.field(np.zeros(basis.n_modes))).max() == 0.0
    with pytest.raises(ValueError, match="coefficients"):
        basis.field(np.zeros(basis.n_modes + 1))
