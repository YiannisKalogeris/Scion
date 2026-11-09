"""
Verification for the Residual Stress Decomposition Method.

RSDM answers a different question from the other two methods here: not whether a
load domain is safe, but what asymptotic state a *given* cyclic history settles
into. So the checks are about the three states and the machinery that
distinguishes them, rather than about a bound.

The anchor is still the thick cylinder, where the two failure loads are known in
closed form: the structure collapses at ``p_lim = 2 Y ln(b/a) / sqrt(3)``, and
alternates plastically at ``2 p_e``, twice the elastic limit, because a fully
reversed elastic stress range of ``2 sigma_y`` is what the residual field cannot
help with.

python -m pytest tests/test_shakedown/test_rsdm.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.materials.plasticity import VonMises
from scion.shakedown import (
    CyclicProgram,
    ElasticEnvelope,
    LoadDomain,
    SelfEquilibrium,
    elastic_factor,
    rsdm,
)
from scion.materials.plasticity.hardening import PerfectPlasticity
from scion.materials.plasticity.return_mapping import PlasticState, return_map
from scion.shakedown.rsdm import RSDMResult, _deviator, _effective

from .cylinder import C_MANDEL, LIMIT_PRESSURE, YIELD, build_cylinder


def _setup(n=5, n_time=8):
    tables, equilibrium, envelope = build_cylinder(n, n)
    program = CyclicProgram.proportional(1, n_time)
    history = program.elastic_history(envelope)
    elastic = elastic_factor(envelope.vertex_stresses(LoadDomain.fixed(1.0)),
                             VonMises(), YIELD)
    return tables, equilibrium, history, elastic


def _run(gamma_ratio, max_iter=3000, n=5, n_time=8):
    tables, equilibrium, history, elastic = _setup(n, n_time)
    result = rsdm(gamma_ratio * elastic * history, tables, equilibrium.free,
                  C_MANDEL, YIELD, max_iter=max_iter, tol=1e-11)
    return result, tables, equilibrium, history, elastic


# ---------------------------------------------------------------------------
# The three asymptotic states
# ---------------------------------------------------------------------------

def test_a_wholly_elastic_history_shakes_down_immediately():
    """Below first yield there is no plasticity to redistribute, so the fixed
    point is reached at once and the residual stress is identically zero."""
    result, *_ = _run(0.5, max_iter=50)

    assert result.converged
    assert result.state == "shakedown"
    assert result.phi == 0.0
    assert np.abs(result.residual).max() == 0.0


def test_a_history_above_first_yield_still_shakes_down():
    """The interesting case: plasticity occurs, a residual field builds up, and
    the steady state is elastic. Convergence is linear and wants hundreds of
    iterations -- each cheap, since the stiffness is factorised once."""
    result, _, _, _, elastic = _run(1.2)

    assert result.converged
    assert result.state == "shakedown"
    assert result.phi < 1e-6
    assert np.abs(result.plastic_stress).max() < 1e-6 * YIELD
    assert np.abs(result.constant_residual).max() > 0.0


def test_above_the_collapse_load_it_reports_ratchetting():
    """And it is right to: no admissible state exists there, which the yield
    check confirms independently of anything the iteration decided."""
    tables, equilibrium, history, elastic = _setup()
    gamma = 1.08 * LIMIT_PRESSURE
    result = rsdm(gamma * history, tables, equilibrium.free, C_MANDEL, YIELD,
                  max_iter=1500, tol=1e-11)

    assert result.state == "ratchetting"
    assert not result.converged

    worst = max(float((_effective(gamma * history[t] + result.constant_residual)
                       - YIELD).max()) for t in range(history.shape[0]))
    assert worst > 1e-3 * YIELD


def test_the_state_is_judged_relative_to_the_yield_stress():
    """A converged shakedown state leaves a plastic stress at round-off, around
    1e-9 where the yield stress is a few hundred. An absolute threshold would
    call that ratchetting and misclassify every converged run."""
    result, *_ = _run(1.2)
    assert 0.0 < np.abs(result.plastic_stress).max() < 1e-6 * YIELD
    assert result.state == "shakedown"

    result.scale = 1e-12                       # pretend the yield stress is tiny
    assert result.state != "shakedown"


# ---------------------------------------------------------------------------
# Above the shakedown limit there is no fixed point at all
# ---------------------------------------------------------------------------

def _constant_at(gamma, n_iter, n=5, n_time=8):
    """``|a_0/2|_inf`` after exactly ``n_iter`` iterations from cold.

    ``tol=0`` disables the convergence break, so the run is the first
    ``n_iter`` terms of the sequence and two budgets are directly comparable.
    """
    tables, equilibrium, history, _ = _setup(n, n_time)
    result = rsdm(gamma * history, tables, equilibrium.free, C_MANDEL, YIELD,
                  max_iter=n_iter, tol=0.0)
    return float(np.abs(result.constant_residual).max())


def test_below_the_limit_the_constant_term_settles():
    """The fixed point exists there, so doubling the budget changes nothing.

    400 and not 200: the sequence is still arriving at 200, where doubling
    moves it 1.1e-6. From 400 the relative move is 1.5e-12, which is the
    settled regime this test is about.
    """
    short = _constant_at(0.70 * LIMIT_PRESSURE, 400)
    long = _constant_at(0.70 * LIMIT_PRESSURE, 800)

    assert abs(long - short) < 1e-9 * short


def test_above_the_limit_the_constant_term_DRIFTS_without_bound():
    """Which is why RSDM-S is absent, and why the fix is not acceleration.

    The Fourier series presupposes a periodic asymptotic state and ratchetting
    is not periodic, so there is nothing for the iteration to converge to. The
    sharp form of the statement: the constant term passes the yield stress,
    which no admissible residual field can do, and keeps going.
    """
    short = _constant_at(1.20 * LIMIT_PRESSURE, 400)
    long = _constant_at(1.20 * LIMIT_PRESSURE, 800)

    assert long > 1.2 * short                 # ~1.33x per doubling, measured
    assert long > YIELD                       # 2.03 Y at 800, 3.69 Y at 3200


def test_the_growth_ratio_separates_the_two_where_state_cannot():
    """``state`` needs ~1600 iterations to stop reporting failure on a
    structure that shakes down; the growth ratio needs none, because it asks
    whether the sequence is going anywhere rather than where it got to."""
    def growth(ratio, n):
        return (_constant_at(ratio * LIMIT_PRESSURE, 2 * n)
                / _constant_at(ratio * LIMIT_PRESSURE, n))

    assert growth(0.70, 400) == pytest.approx(1.0, abs=1e-9)
    assert growth(1.20, 200) > 1.05


# ---------------------------------------------------------------------------
# Plastic strain accumulation and the asymptotic-cycle indicators
# ---------------------------------------------------------------------------

def _reversed_program(n_time=8):
    """A fully reversed history, ``alpha(tau + 1/2) = -alpha(tau)``.

    The proportional hump never changes sign, so it cannot produce alternating
    plasticity. This one can, and its antisymmetry is what makes the net
    plastic strain vanish exactly rather than approximately.
    """
    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    return CyclicProgram(np.sin(2.0 * np.pi * tau)[:, None], tau)


def _run_reversed(multiple, n=5, n_time=8, max_iter=3000):
    tables, equilibrium, envelope = build_cylinder(n, n)
    p_e = elastic_factor(envelope.vertex_stresses(LoadDomain.fixed(1.0)),
                         VonMises(), YIELD)
    history = _reversed_program(n_time).elastic_history(envelope)
    return rsdm(multiple * p_e * history, tables, equilibrium.free, C_MANDEL,
                YIELD, max_iter=max_iter, tol=1e-11)


def test_the_plastic_strain_rate_is_the_closest_point_projection():
    """The identity the whole accumulation rests on, and the reason no return
    map had to be added.

    The formulation writes the plastic strain rate from the closest-point
    projection of the stress onto the admissible set. For von Mises that
    projection keeps the hydrostatic part and scales the deviator by
    ``sigma_Y/sigma_eq``, so the difference is exactly the radial excess RSDM
    already computes. Checked against the library's own return map, which is
    independently validated against the thick cylinder.
    """
    rng = np.random.default_rng(0)
    sigma = rng.normal(0.0, 400.0, (400, 6))          # Mandel, mostly yielding
    effective = _effective(sigma)
    excess = effective > YIELD
    assert excess.sum() > 300                         # the test must bite

    scale = np.where(excess, 1.0 - YIELD / np.maximum(effective, 1e-300), 0.0)
    radial = scale[:, None] * _deviator(sigma)

    mapped = return_map(np.linalg.solve(C_MANDEL, sigma.T).T,
                        PlasticState.zeros(len(sigma)), VonMises(),
                        PerfectPlasticity(YIELD), C_MANDEL)

    assert np.array_equal(mapped.plastic, excess)
    reference = mapped.state.eps_p
    assert np.abs(np.linalg.solve(C_MANDEL, radial.T).T - reference).max() \
        < 1e-14 * np.abs(reference).max()


def test_the_plastic_strain_is_exactly_incompressible():
    """``tr(eps^p) = 0`` to round-off, not to a tolerance: the radial excess is
    a deviator and the compliance of an isotropic material maps deviators to
    deviators. A nonzero trace is a bug, which is what makes this sharp."""
    result = _run_reversed(1.6)
    trace = np.abs(result.plastic_strain[..., :3].sum(axis=-1)).max()

    assert trace < 1e-14 * np.abs(result.plastic_strain).max()


def test_shakedown_carries_no_plastic_strain_in_the_cycle():
    """All three indicators vanish together, which is the definition."""
    result, *_ = _run(1.2)
    assert result.state == "shakedown"

    assert result.cyclic_amplitude < 1e-8
    assert result.ratchet_rate < 1e-8 * result.plastic_strain_scale
    assert result.plastic_strain_range < 1e-8 * result.plastic_strain_scale


def test_a_fully_reversed_history_has_NO_net_plastic_strain():
    """Alternating plasticity, and the distinction from ratchetting is exact
    rather than thresholded.

    With ``sigma^E(tau + 1/2) = -sigma^E(tau)`` the mean residual stress
    vanishes by symmetry, so the total stress is antisymmetric too, so the
    radial excess is -- and it sums to zero over the cycle. Measured 3.0e-13
    of the shear yield strain against a plastic strain range of 0.088, i.e.
    eleven orders of magnitude of separation, where `state` has to guess with
    a `1e-3 * peak` threshold.

    Also pins the two indicators against each other: ``r`` must be the norm
    of the last entry of the accumulated history, or the ``dt`` in one of them
    does not match the other.
    """
    result = _run_reversed(1.6)

    assert result.cyclic_amplitude > 1.0                    # genuinely cycling
    assert result.plastic_strain_range > 0.05 * result.plastic_strain_scale
    assert result.ratchet_rate < 1e-9 * result.plastic_strain_scale
    assert result.regime() == "alternating"

    net = np.linalg.norm(result.plastic_strain[-1], axis=-1).max()
    assert result.ratchet_rate == pytest.approx(net, rel=1e-12)


def test_the_plastic_strain_range_grows_with_the_load():
    """And is zero below first yield. Monotone, because a larger reversed
    amplitude puts more of the cycle outside the yield surface."""
    ranges = [_run_reversed(m).plastic_strain_range / YIELD
              for m in (0.8, 1.2, 1.6)]

    assert ranges[0] == 0.0
    assert ranges[1] < ranges[2]


def test_the_amplitude_is_volume_weighted_where_phi_is_not():
    """``A`` integrates over the body; ``phi`` counts quadrature points. On a
    graded mesh they are different numbers, and only the first is a property
    of the structure rather than of the discretisation."""
    a = np.array([[[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]]])
    b = np.zeros_like(a)
    weights = np.array([1.0, 3.0])                  # one point four times wider
    result = RSDMResult(a0=np.zeros((2, 6)), a=a, b=b,
                        residual=np.zeros((1, 2, 6)),
                        plastic_stress=np.zeros((1, 2, 6)),
                        compliance=np.linalg.inv(C_MANDEL),
                        weights=weights, sigma_0=2.0)

    assert result.cyclic_amplitude == pytest.approx(np.sqrt(4.0) / 2.0)


def test_the_regime_separates_elastic_from_shakedown():
    """Both carry zero plastic strain in the asymptotic cycle, so the
    indicators alone cannot tell them apart; the residual stress field can,
    because an elastic history never builds one."""
    elastic = _run_reversed(0.8)
    assert elastic.regime() == "elastic"
    assert np.abs(elastic.constant_residual).max() == 0.0

    redistributed, *_ = _run(1.2)
    assert redistributed.regime() == "shakedown"
    assert np.abs(redistributed.constant_residual).max() > 0.0


# ---------------------------------------------------------------------------
# The machinery
# ---------------------------------------------------------------------------

def test_the_residual_field_is_self_equilibrated_at_every_cycle_point():
    """By construction: the rate solves ``K r_dot = INT B^T sigma_p dV`` and is
    then ``D B r_dot - sigma_p``, whose internal force cancels exactly. It is
    the property that makes the field a legitimate residual stress rather than
    just a correction."""
    result, _, equilibrium, _, _ = _run(1.2, max_iter=400)

    for t in range(result.residual.shape[0]):
        assert equilibrium.residual(result.residual[t]) < 1e-8
    assert equilibrium.residual(result.constant_residual) < 1e-8


def test_the_fourier_series_reproduces_the_stored_residual():
    """Eq. (10) evaluated from the coefficients has to match the field the
    iteration carried, or the two halves of the fixed point are out of step."""
    result, *_ = _run(1.2, max_iter=300)

    n_time = result.residual.shape[0]
    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    k = np.arange(1, result.a.shape[0] + 1)
    rebuilt = (0.5 * result.a0[None]
               + np.einsum("kt,kpi->tpi", np.cos(2 * np.pi * np.outer(k, tau)),
                           result.a)
               + np.einsum("kt,kpi->tpi", np.sin(2 * np.pi * np.outer(k, tau)),
                           result.b))
    assert np.abs(rebuilt - result.residual).max() < 1e-8 * max(
        float(np.abs(result.residual).max()), 1.0)


def test_at_shakedown_only_the_constant_term_survives():
    """The definition of shaking down: the residual stress stops varying in
    time. Every trigonometric coefficient goes to zero and ``a_0/2`` is the
    whole field."""
    result, *_ = _run(1.2)

    assert np.abs(result.a).max() < 1e-6 * np.abs(result.a0).max()
    assert np.abs(result.b).max() < 1e-6 * np.abs(result.a0).max()
    spread = np.abs(result.residual - result.constant_residual).max()
    assert spread < 1e-6 * np.abs(result.constant_residual).max()


def test_warm_starting_reaches_the_same_fixed_point():
    """The outer loops that use RSDM warm-start it, so the answer must not
    depend on where it started -- only how quickly it gets there."""
    cold, tables, equilibrium, history, elastic = _run(1.2)
    warm = rsdm(1.2 * elastic * history, tables, equilibrium.free, C_MANDEL,
                YIELD, max_iter=3000, tol=1e-11,
                initial=(cold.a0, cold.a, cold.b))

    assert warm.converged
    assert np.abs(warm.constant_residual - cold.constant_residual).max() \
        < 1e-6 * np.abs(cold.constant_residual).max()


# ---------------------------------------------------------------------------
# The load program
# ---------------------------------------------------------------------------

def test_the_proportional_program_is_a_smooth_hump():
    """Smooth rather than triangular: a corner would need many harmonics to
    represent and RSDM keeps only one or two."""
    program = CyclicProgram.proportional(1, n_time=64)
    alpha = program.multipliers[:, 0]

    assert alpha.min() == pytest.approx(0.0, abs=1e-12)
    assert alpha.max() == pytest.approx(1.0, rel=1e-3)
    assert program.peak_index() == 32
    assert np.abs(np.diff(alpha, 2)).max() < 0.02


def test_a_vertex_program_visits_every_vertex_of_the_domain():
    """Convexity then extends the result to the whole domain, which is what
    lets a history-based method answer a domain-based question."""
    domain = LoadDomain.box([0.0, 0.0], [1.0, 2.0])
    program = CyclicProgram.through_vertices(domain, per_leg=5)

    assert program.n_cases == 2
    for vertex in domain.vertices:
        assert np.abs(program.multipliers - vertex).sum(axis=1).min() < 1e-12


def test_the_elastic_history_is_the_multiplier_combination():
    tables, equilibrium, envelope = build_cylinder(4, 4)
    program = CyclicProgram.proportional(1, 6)
    history = program.elastic_history(envelope)

    assert history.shape == (6, tables.n_points, 6)
    peak = history[program.peak_index()]
    assert np.allclose(peak, program.multipliers[program.peak_index(), 0]
                       * envelope.stresses[0])


def test_a_mismatched_program_is_refused():
    with pytest.raises(ValueError, match="tau and multipliers"):
        CyclicProgram(np.zeros((5, 1)), tau=np.zeros(4))
