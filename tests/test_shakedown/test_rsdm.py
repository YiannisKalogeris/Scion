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

from simulacrum.materials.plasticity import VonMises
from simulacrum.shakedown import (
    CyclicProgram,
    ElasticEnvelope,
    LoadDomain,
    SelfEquilibrium,
    elastic_factor,
    rsdm,
)
from simulacrum.shakedown.rsdm import _effective

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
