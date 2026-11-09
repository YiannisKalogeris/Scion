"""
Verification for the asymptotic cycle with yield enforced exactly.

This is the module that gives a plastic strain *magnitude*, so the checks are
about magnitudes, against closed forms, and about the two properties the
penalty form of RSDM cannot have: yield satisfied at round-off, and an answer
independent of the cycle sampling.

The Bree strip is the benchmark rather than the thick cylinder, and
deliberately: it pins a whole two-parameter map in closed form, including a
plastic strain range and -- derived in ``bree.py`` -- a ratchet rate.

python -m pytest tests/test_shakedown/test_cyclic.py -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scion.discretization.fem.tables import QuadratureTables
from scion.materials.plasticity import VonMises
from scion.materials.plasticity.hardening import get_isotropic
from scion.shakedown import (
    CyclicProgram,
    ElasticEnvelope,
    LoadDomain,
    SelfEquilibrium,
    cyclic_steady_state,
    cyclic_strip,
    elastic_factor,
    linear_matching,
    plastic_strain_range_exact,
    ratchet_rate_exact,
    rsdm,
    rsdm_strip,
    solve_elastic,
    transition_loads,
)
from scion.shakedown.cyclic import CyclicConvergenceError, _internal_force

from .cylinder import (
    C_MANDEL,
    LIMIT_PRESSURE,
    YIELD,
    bore_pressure_forces,
    build_cylinder,
    quarter_annulus,
)

#: alternating-plasticity rays, where ``plastic_strain_range_exact`` applies
AP_RAYS = [(0.0, 3.0, 1.0), (0.0, 2.5, 1.0), (0.0, 4.0, 1.0), (0.1, 0.8, 2.8),
           (0.1, 0.8, 3.4), (0.05, 1.0, 3.0), (0.15, 2.0, 1.2),
           (0.15, 2.0, 1.8), (0.2, 3.0, 1.0), (0.3, 2.0, 1.2),
           (0.02, 5.0, 1.0)]


# ---------------------------------------------------------------------------
# The magnitude, which is the whole point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, lam", AP_RAYS)
def test_the_plastic_strain_range_is_the_closed_form_EXACTLY(x, y, lam):
    """Not "to engineering accuracy": to round-off.

    ``Delta eps^p = (lambda sigma_T - 2 sigma_Y)/E'`` at the face, derived in
    ``plastic_strain_range_exact``. Every ray here agrees to below 1e-14
    relative, worst measured 9.3e-16, which is what distinguishes a constraint
    from a penalty -- the penalty form returns 0.084261 for an exact 1.0 on the
    first ray and is checked doing so below.
    """
    result = cyclic_strip(x, y, lam)
    exact = float(plastic_strain_range_exact(y, lam))
    assert result.plastic_strain_range == pytest.approx(exact, rel=1e-14)


def test_the_penalty_form_does_NOT_get_the_magnitude():
    """The comparison that justifies this module existing at all.

    Both solve the same problem and agree on the regime; only one gets the
    size. If this ever starts passing at the exact value, the 2012 iteration
    has been reformulated and the whole of ``cyclic.py`` can be reconsidered.
    """
    exact = float(plastic_strain_range_exact(3.0, 1.0))
    assert exact == pytest.approx(1.0)
    assert cyclic_strip(0.0, 3.0, 1.0).plastic_strain_range == \
        pytest.approx(exact, rel=1e-14)
    penalty = rsdm_strip(0.0, 3.0, 1.0)
    assert penalty.plastic_strain_range == pytest.approx(0.0843, abs=1e-3)
    # but the two agree on the regime, which is the scale-free part
    assert penalty.regime() == cyclic_strip(0.0, 3.0, 1.0).regime()


def test_yield_is_satisfied_at_round_off_and_not_merely_approximately():
    """The structural difference from the penalty form.

    There the residue outside the surface *is* the plastic strain rate, so it
    is finite by construction. Here it is a projection tolerance, and for a
    scalar stress the projection is a clip, so the violation is an exact zero.
    """
    for x, y, lam in AP_RAYS:
        assert cyclic_strip(x, y, lam).max_yield_violation == 0.0


# ---------------------------------------------------------------------------
# Independence of the discretisation, which the penalty form does not have
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_time", [8, 12, 16, 24, 32, 64])
def test_the_range_does_not_depend_on_the_cycle_sampling(n_time):
    """``rsdm_strip``'s own docstring carries an ``n_t`` table because its
    answer moves; this one does not move at all.

    It must not: only the extremes of the cycle enter, which is the same
    statement that makes Melan's condition waveform-blind.
    """
    result = cyclic_strip(0.1, 0.8, 3.0, n_time=n_time)
    assert result.plastic_strain_range == pytest.approx(0.4, rel=1e-12)


@pytest.mark.parametrize("n_thickness", [11, 21, 51, 101, 401])
def test_the_range_does_not_depend_on_the_thickness_quadrature(n_thickness):
    """The range is attained at the face, which trapezoid puts a node on."""
    result = cyclic_strip(0.1, 0.8, 3.0, n_thickness=n_thickness)
    assert result.plastic_strain_range == pytest.approx(0.4, rel=1e-12)


def test_the_two_waveforms_give_the_same_answer():
    hump = cyclic_strip(0.1, 0.8, 3.0, waveform="hump")
    tri = cyclic_strip(0.1, 0.8, 3.0, waveform="triangle")
    assert hump.plastic_strain_range == pytest.approx(
        tri.plastic_strain_range, rel=1e-12)


@pytest.mark.parametrize("n_time", [9, 11, 25])
def test_an_odd_cycle_sampling_is_REFUSED_rather_than_rounded(n_time):
    """Both waveforms peak at ``tau = 1/2``, which an odd count never samples.

    The strip is then under-loaded and the answer is low -- 0.9264 / 0.9506 /
    0.9904 for an exact 1.0 at 9 / 11 / 25 -- and it *converges*, so it reads
    as a discretisation rather than an error. That is why it raises: the even
    counts are already exact at 8, so there is nothing to trade away.
    """
    with pytest.raises(ValueError, match="even"):
        cyclic_strip(0.0, 3.0, 1.0, n_time=n_time)


# ---------------------------------------------------------------------------
# The ratchet rate, against the closed form derived in bree.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, lam", [
    (0.3, 1.5, 1.6), (0.15, 2.0, 2.0), (0.02, 5.0, 3.5), (0.1, 0.8, 4.0),
    (0.05, 2.0, 4.0), (0.05, 1.0, 5.36656), (0.125, 2.0, 3.0),
])
def test_the_ratchet_rate_is_the_derived_closed_form(x, y, lam):
    """``r = 2(lambda^2 x y - 1) sigma_Y / E'`` on the R2 branch.

    Derived in ``bree.ratchet_rate_exact``, and it vanishes identically on
    ``x y lambda^2 = 1`` -- which is the one boundary of the Bree diagram the
    Melan argument cannot reach, so this validates that curve from the plastic
    strain rather than by bisecting an indicator.
    """
    result = cyclic_strip(x, y, lam, n_thickness=2001, max_cycles=400,
                          tol=1e-11)
    exact = float(ratchet_rate_exact(x, y, lam))
    # 3.5e-6 is the worst of the seven at this thickness quadrature, and it
    # falls to 1e-8 by n_thickness = 25601: the residual is the elastic-plastic
    # front's position, not the formula.
    assert result.ratchet_rate == pytest.approx(exact, rel=5e-6)


def test_that_closed_form_is_refused_ON_the_branch_boundary():
    """``y = 4x`` exactly is excluded by measurement, not by caution.

    That ray passes through the corner where the alternating band closes, so
    both mechanisms are marginal, and it converges to 1.00011965 times the
    formula over ``n_thickness`` from 101 to 25601 -- a converged
    disagreement. One step inside, at ``y = 4.4x``, the ratio is 1.0000000.
    """
    with pytest.raises(NotImplementedError, match="strict"):
        ratchet_rate_exact(0.25, 1.0, 2.2)
    inside = cyclic_strip(0.22, 1.0, 2.2, n_thickness=2001, max_cycles=400)
    assert inside.ratchet_rate == pytest.approx(
        float(ratchet_rate_exact(0.22, 1.0, 2.2)), rel=2e-6)


@pytest.mark.parametrize("x, y", [(0.15, 2.0), (0.1, 0.8), (0.05, 1.0)])
def test_the_ratchet_rate_is_EXACTLY_zero_up_to_lambda_rr(x, y):
    """Not thresholded. Below and at ``lambda_rr`` the cycle closes, so the
    net plastic strain is zero by cancellation rather than by being small."""
    lam_rr = float(transition_loads(x, y).ratchet)
    for factor in (0.9, 0.99, 0.999, 1.0):
        result = cyclic_strip(x, y, lam_rr * factor, n_thickness=401,
                              max_cycles=400)
        assert result.ratchet_rate < 1e-12
    just_above = cyclic_strip(x, y, lam_rr * 1.001, n_thickness=401,
                              max_cycles=400)
    assert just_above.ratchet_rate == pytest.approx(4.0e-3, rel=0.05)


@pytest.mark.parametrize("x, y", [(0.4, 0.5), (0.6, 0.3), (0.5, 1.0)])
def test_on_the_R1_branch_the_range_EQUALS_the_net(x, y):
    """Which is the definition of one-way ratchetting.

    The ``y < 4x`` rays leave shakedown across ``y = 4(1-x)`` with no
    alternating band, so there is no reversed plasticity and every increment
    adds to the same total. Asserted as an exact equality because it is one.
    """
    lam_sd = float(transition_loads(x, y).shakedown)
    result = cyclic_strip(x, y, lam_sd * 1.01, n_thickness=401,
                          max_cycles=400)
    assert result.ratchet_rate > 1e-3
    assert result.ratchet_rate == pytest.approx(
        result.plastic_strain_range, rel=1e-12)


def test_the_closed_form_range_FAILS_once_the_ray_ratchets():
    """Its own docstring predicted this before it was measured.

    ``plastic_strain_range_exact`` is the thermal part alone, derived on the
    assumption that the cycle closes. Past ``lambda_rr`` it does not, the net
    drift enters the range, and the formula reads 2.25 times low. Recorded as
    a test so the benchmark cannot quietly be applied outside its regime.
    """
    result = cyclic_strip(0.25, 1.0, 2.5, max_cycles=400)
    exact_ap = float(plastic_strain_range_exact(1.0, 2.5))
    assert result.regime() == "ratchetting"
    assert result.plastic_strain_range / exact_ap == pytest.approx(2.25, rel=0.02)


# ---------------------------------------------------------------------------
# Collapse, which is an answer and not a failure
# ---------------------------------------------------------------------------

def test_the_strip_refuses_past_its_limit_load():
    """``lambda x_hat = 1``: the membrane stress alone reaches yield, the whole
    thickness is plastic and the section cannot carry the force.

    Bracketed rather than asserted at a point, because the statement is that
    the refusal lands *on* the limit load.
    """
    cyclic_strip(0.6, 0.3, 1.0 / 0.6 - 1e-5, max_cycles=400)
    with pytest.raises(RuntimeError, match="collapse"):
        cyclic_strip(0.6, 0.3, 1.0 / 0.6 + 1e-5, max_cycles=400)


# ---------------------------------------------------------------------------
# The regime classification
# ---------------------------------------------------------------------------

def test_the_regime_agrees_with_the_closed_form_over_a_sweep():
    """Twenty rays and load factors against ``transition_loads``.

    The elastic/shakedown split is excluded: both carry no plastic strain, so
    the indicators cannot see it and ``regime`` reads it off the residual
    field instead, which is tested separately.
    """
    checked = 0
    for x_hat, y_hat in [(0.1, 0.8), (0.05, 1.0), (0.15, 2.0), (0.02, 5.0),
                         (0.4, 0.5)]:
        loads = transition_loads(x_hat, y_hat)
        for lam in (0.5 * float(loads.shakedown),
                    0.95 * float(loads.shakedown),
                    1.2 * float(loads.shakedown),
                    1.1 * float(loads.ratchet)):
            if lam * x_hat >= 1.0:
                continue
            got = cyclic_strip(x_hat, y_hat, lam, n_thickness=401,
                               max_cycles=400).regime()
            want = str(loads.regime_at(lam))
            if want in ("elastic", "shakedown"):
                assert got in ("elastic", "shakedown")
            else:
                assert got == want, (x_hat, y_hat, lam, got, want)
            checked += 1
    assert checked >= 18


def test_the_range_is_a_better_alternating_indicator_than_the_amplitude():
    """Which is why ``CyclicState.regime`` differs from ``RSDMResult.regime``.

    Below the boundary the range is *identically* zero where the amplitude is
    round-off, and above it the range responds ten times more strongly. Both
    halves are asserted, since only the pair is the argument.
    """
    lam_sd = float(transition_loads(0.1, 2.0).shakedown)
    below = cyclic_strip(0.1, 2.0, lam_sd * 0.9999)
    above = cyclic_strip(0.1, 2.0, lam_sd * 1.001)
    assert below.plastic_strain_range == 0.0
    assert below.cyclic_amplitude > 0.0            # round-off, not zero
    assert (above.plastic_strain_range / above.plastic_strain_scale
            > 9.0 * above.cyclic_amplitude)


# ---------------------------------------------------------------------------
# The finite element path
# ---------------------------------------------------------------------------

def _cylinder(n=5, bbar=False, n_time=8):
    """The thick cylinder, with the integration rule as a parameter.

    ``build_cylinder`` fixes full integration, which is right for Melan and
    RSDM and wrong here -- so this rebuilds it rather than importing it.
    """
    domain = quarter_annulus(n, n)
    tables = QuadratureTables(domain, thickness=1.0, bbar=bbar)
    dm = tables.vector_dofs
    nodes = domain.coordinates
    on_x = np.where(nodes[:, 1] < 1e-7)[0]
    on_y = np.where(nodes[:, 0] < 1e-7)[0]
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x)[1::2],
                                      dm.element_dofs(on_y)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)
    u, = solve_elastic(tables, C_MANDEL, free, [bore_pressure_forces(domain, dm)])
    envelope = ElasticEnvelope(tables, C_MANDEL).add_case(u)
    history = CyclicProgram.proportional(1, n_time).elastic_history(envelope)
    factor = elastic_factor(envelope.vertex_stresses(LoadDomain.fixed(1.0)),
                            VonMises(), YIELD)
    return tables, SelfEquilibrium(tables, free), history, factor


def test_the_residual_field_is_self_equilibrated_at_every_cycle_point():
    """``INT B^T rho dV = 0`` is the whole content of the Newton solve, so this
    checks that what the Newton converged on is what was intended."""
    tables, eqm, history, factor = _cylinder()
    scale = 0.9 * LIMIT_PRESSURE / factor
    result = cyclic_steady_state(scale * factor * history, tables, eqm.free,
                                 C_MANDEL, YIELD)
    n_dof = tables.vector_dofs.n_total
    reference = max(np.abs(_internal_force(
        tables, history[history.shape[0] // 2] * scale * factor,
        n_dof, tables.n_voigt)[eqm.free]).max(), 1e-300)
    for t in range(result.stress.shape[0]):
        force = _internal_force(tables, result.residual[t], n_dof,
                                tables.n_voigt)
        assert np.abs(force[eqm.free]).max() < 1e-8 * reference


def test_below_shakedown_the_cycle_carries_no_plastic_strain():
    tables, eqm, history, factor = _cylinder()
    result = cyclic_steady_state(0.9 * LIMIT_PRESSURE * history, tables,
                                 eqm.free, C_MANDEL, YIELD)
    assert result.plastic_strain_range == 0.0
    assert result.ratchet_rate == 0.0
    assert result.regime() in ("elastic", "shakedown")
    assert result.max_yield_violation < 1e-12


def test_an_elastic_history_develops_no_residual_field_at_all():
    """Which is how ``elastic`` is separated from ``shakedown``: the
    indicators are zero in both, so only the residual field distinguishes
    them."""
    tables, eqm, history, factor = _cylinder()
    result = cyclic_steady_state(0.5 * factor * history, tables, eqm.free,
                                 C_MANDEL, YIELD)
    assert result.regime() == "elastic"
    assert np.abs(result.constant_residual).max() < 1e-10 * YIELD


def test_b_bar_lands_on_the_Koiter_bound_and_full_integration_LOCKS():
    """The validation of the finite element path, and its sharpest trap.

    Past the limit load of the discrete model no admissible stress field
    equilibrates the peak of the cycle, so the refusal *is* the answer. The
    Koiter bound is computed here rather than quoted, because this is the one
    genuinely independent check the finite element path has and a figure in a
    docstring is not a check. Fully integrated the same mesh carries 1.179,
    locked 17% high, with nothing in that run looking wrong -- and note LMM
    cannot be run there at all for comparison, since it requires b-bar, which
    is why the locking has to be read against the b-bar row.

    Bracketed coarsely to keep the test cheap; the fine bisection (1.00747) is
    in the module docstring.
    """
    tables, eqm, history, factor = _cylinder(bbar=True)
    scale = LIMIT_PRESSURE / factor
    ok = cyclic_steady_state(1.005 * scale * factor * history, tables,
                             eqm.free, C_MANDEL, YIELD, max_cycles=20)
    assert ok.regime() in ("elastic", "shakedown")
    with pytest.raises(CyclicConvergenceError, match="collapse|equilibrate"):
        cyclic_steady_state(1.02 * scale * factor * history, tables, eqm.free,
                            C_MANDEL, YIELD, max_cycles=20)

    koiter = linear_matching(
        ElasticEnvelope(tables, C_MANDEL).add_case(
            solve_elastic(tables, C_MANDEL, eqm.free,
                          [bore_pressure_forces(
                              quarter_annulus(5, 5),
                              tables.vector_dofs)])[0]
        ).vertex_stresses(LoadDomain.pulsating(1)),
        tables, eqm.free, YIELD)
    assert koiter.converged
    bound = koiter.factor / LIMIT_PRESSURE
    assert bound == pytest.approx(1.00730, abs=1e-4)
    # the refusal is bracketed in (1.005, 1.02) and the bound is inside it
    assert 1.005 < bound < 1.02

    full, eqf, hist_f, factor_f = _cylinder(bbar=False)
    # LMM has no answer on fully integrated tables, so the locked row has
    # nothing to be compared against. Asserted explicitly rather than with a
    # bare `pytest.raises`, which would also pass if LMM rejected the tables
    # for some unrelated reason: what is claimed is that it RUNS and fails to
    # produce a number.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = linear_matching(
                ElasticEnvelope(full, C_MANDEL).add_case(
                    solve_elastic(full, C_MANDEL, eqf.free,
                                  [bore_pressure_forces(quarter_annulus(5, 5),
                                                        full.vector_dofs)])[0]
                ).vertex_stresses(LoadDomain.pulsating(1)),
                full, eqf.free, YIELD, allow_locking=True, max_iter=20)
            assert not res.converged or not np.isfinite(res.factor)
        except (RuntimeError, ValueError):
            pass            # refusing outright is equally acceptable

    locked = cyclic_steady_state(1.10 * (LIMIT_PRESSURE / factor_f) * factor_f
                                 * hist_f, full, eqf.free, C_MANDEL, YIELD,
                                 max_cycles=20)
    assert locked.regime() in ("elastic", "shakedown")
    assert locked.max_yield_violation < 1e-10


def test_the_shakedown_residual_field_is_NOT_unique():
    """RSDM and this method reach different members of the admissible set.

    Expecting them to agree was a mistake: any self-equilibrated field keeping
    every cycle point admissible is a shakedown state, and RSDM's iteration
    leaves ``a_0`` wherever its own path got to once ``sigma_p`` reaches zero.
    What is checkable is that *both* are admissible and that the difference is
    itself self-equilibrated -- two valid answers rather than one answer and
    one error.
    """
    tables, eqm, _, _ = _cylinder()
    cyl_tables, cyl_eqm, cyl_env = build_cylinder(5, 5)
    history = CyclicProgram.proportional(1, 8).elastic_history(cyl_env)
    factor = elastic_factor(
        cyl_env.vertex_stresses(LoadDomain.fixed(1.0)), VonMises(), YIELD)
    loaded = 0.9 * (LIMIT_PRESSURE / factor) * factor * history

    exact = cyclic_steady_state(loaded, cyl_tables, cyl_eqm.free, C_MANDEL,
                                YIELD)
    penalty = rsdm(loaded, cyl_tables, cyl_eqm.free, C_MANDEL, YIELD,
                   max_iter=4000, tol=1e-12)

    a, b = exact.constant_residual, penalty.constant_residual
    difference = np.abs(a - b).max() / np.abs(b).max()
    assert 1e-3 < difference < 1e-1        # measured 2.73e-2: real, not noise

    n_dof = cyl_tables.vector_dofs.n_total
    nv = cyl_tables.n_voigt
    for field in (a, b, a - b):
        force = _internal_force(cyl_tables, field, n_dof, nv)
        assert np.abs(force[cyl_eqm.free]).max() < 1e-6 * YIELD

    assert exact.max_yield_violation < 1e-12


def test_a_history_that_is_not_six_components_is_refused():
    """The in-plane reduction would drop the out-of-plane constraint, and the
    formulation needs ``sigma^E`` whole: ``D^-1 sigma^E`` is what carries the
    thermal eigenstrain and makes ``eps_33 = 0`` come out rather than be
    imposed."""
    tables, eqm, _, _ = _cylinder()
    with pytest.raises(ValueError, match="Mandel"):
        cyclic_steady_state(np.zeros((4, tables.n_points, 4)), tables,
                            eqm.free, C_MANDEL, YIELD)


def _reversed_history(history):
    """``sin(2 pi tau)`` on the same elastic field as ``_cylinder``.

    The pulsating program has an *empty* alternating band -- shakedown gives
    way directly to collapse -- so it offers no load level carrying a nonzero
    plastic strain range with a periodic state. A fully reversed history does.
    The hump peaks at ``tau = 1/2``, so the single load case is recovered from
    the middle sample and re-multiplied.
    """
    n_time = history.shape[0]
    case = history[n_time // 2]
    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    return np.sin(2.0 * np.pi * tau)[:, None, None] * case


def test_a_uniform_yield_field_reproduces_the_scalar_path_bit_for_bit():
    """A point-varying yield stress enters as a pointwise rescaling of the
    strain, so the scalar path is the special case rather than a separate
    branch. Asserted as an exact equality because that is what it is: the
    scaling is by one at every point, and any difference would mean the two
    code paths are not the same calculation.

    The load is above the shakedown limit on purpose. Below it the plastic
    strain is identically zero and the comparison would pass for a solver
    that had stopped computing."""
    tables, eqm, history, factor = _cylinder(bbar=True)
    load = 1.005 * LIMIT_PRESSURE * history
    scalar = cyclic_steady_state(load, tables, eqm.free, C_MANDEL, YIELD,
                                 max_cycles=20)
    field = cyclic_steady_state(load, tables, eqm.free, C_MANDEL,
                                np.full(tables.n_points, YIELD),
                                max_cycles=20)
    assert np.abs(scalar.stress - field.stress).max() == 0.0
    assert scalar.plastic_strain_range == field.plastic_strain_range
    assert scalar.ratchet_rate == field.ratchet_rate
    assert scalar.cycles == field.cycles
    assert scalar.regime() == field.regime()


def test_yield_holds_against_the_local_value_of_a_varying_field():
    """The point of the rescaling: admissibility is pointwise, so a smooth
    field must leave the violation at round-off measured against each point's
    own yield stress rather than against any average of them."""
    tables, eqm, history, factor = _cylinder(bbar=True)
    xq = tables.interpolate(tables.domain.coordinates[:, 0])
    sigma_y = YIELD * (1.0 + 0.3 * np.sin(5.0 * xq / np.abs(xq).max()))
    # 0.75 rather than 0.95: weakening the wall by 30% somewhere lowers the
    # collapse load of the structure with it, so the reference for the load
    # level is the weakened body and not the uniform one
    result = cyclic_steady_state(0.75 * LIMIT_PRESSURE * history, tables,
                                 eqm.free, C_MANDEL, sigma_y, max_cycles=40)
    assert result.max_yield_violation < 1.0e-12
    assert (result.effective <= sigma_y[None, :] * (1.0 + 1.0e-9)).all()


def test_the_transition_loads_are_degree_one_in_the_yield_field():
    """Scaling the yield field and the load together by the same factor is
    the same problem, so every reported strain scales exactly. This is the
    check that the rescaling is exact rather than merely plausible -- it would
    fail for any formulation that mixed the field into a single scalar."""
    tables, eqm, history, factor = _cylinder(bbar=True)
    xq = tables.interpolate(tables.domain.coordinates[:, 0])
    sigma_y = YIELD * (1.0 + 0.2 * np.cos(4.0 * xq / np.abs(xq).max()))
    load = 0.6 * LIMIT_PRESSURE * _reversed_history(history)
    base = cyclic_steady_state(load, tables, eqm.free, C_MANDEL, sigma_y,
                               max_cycles=60)
    assert base.plastic_strain_range > 0.0
    for k in (0.5, 1.7):
        scaled = cyclic_steady_state(k * load, tables, eqm.free, C_MANDEL,
                                     k * sigma_y, max_cycles=60)
        assert scaled.plastic_strain_range == pytest.approx(
            k * base.plastic_strain_range, rel=1.0e-12)


def test_a_varying_yield_field_with_hardening_is_refused():
    """The rescaling is exact only because perfect plasticity has no scale in
    ``kappa``. With ``kappa(alpha)`` the scaling and the hardening do not
    commute, and silently applying it would be wrong in the one case a reader
    would most want to trust."""
    tables, eqm, history, _ = _cylinder(bbar=True)
    sigma_y = np.full(tables.n_points, YIELD)
    sigma_y[0] *= 0.5
    with pytest.raises(NotImplementedError, match="hardening"):
        cyclic_steady_state(history, tables, eqm.free, C_MANDEL, sigma_y,
                            isotropic=get_isotropic("linear", kappa_0=YIELD,
                                                    H=10.0))


def test_plastic_incompressibility_is_exact_in_the_finite_element_path():
    """A deviatoric flow rule maps a deviator to a deviator, so a nonzero
    trace is a bug and not a tolerance."""
    tables, eqm, history, factor = _cylinder(bbar=True)
    result = cyclic_steady_state(1.005 * LIMIT_PRESSURE * history, tables,
                                 eqm.free, C_MANDEL, YIELD, max_cycles=20)
    trace = result.plastic_strain[..., :3].sum(axis=-1)
    assert np.abs(trace).max() < 1e-14


def test_the_amplitude_matches_RSDM_s_definition_by_Parseval():
    """``A`` is defined on the Fourier coefficients there and computed from the
    history here, so the two are the same quantity only if Parseval is applied
    correctly -- which is worth checking on a field whose harmonic content is
    known: a shakedown state has none, so ``A`` must be at round-off."""
    tables, eqm, history, factor = _cylinder()
    result = cyclic_steady_state(0.9 * LIMIT_PRESSURE * history, tables,
                                 eqm.free, C_MANDEL, YIELD)
    assert result.cyclic_amplitude < 1e-12
    # and on the strip, where the alternating state has real content
    strip = cyclic_strip(0.0, 3.0, 1.0)
    rho = strip.residual
    centred = rho - rho.mean(axis=0, keepdims=True)
    by_hand = np.sqrt(2.0 * float((strip.weights[None, :, None]
                                   * centred ** 2).sum()) / rho.shape[0])
    assert strip.cyclic_amplitude == pytest.approx(by_hand / strip.sigma_0)


@pytest.mark.parametrize("x, y, lam", [
    (0.05, 2.0, 4.0), (0.15, 2.0, 2.2), (0.1, 0.8, 4.0), (0.02, 5.0, 3.5),
    (0.3, 1.5, 1.6), (0.05, 1.0, 5.0), (0.125, 2.0, 3.0),
])
def test_the_R2_mechanism_has_the_structure_the_derivation_assumes(x, y, lam):
    """The two relations behind ``r = 2(lambda^2 x y - 1)``, checked separately.

    At the hot extreme the converged cycle is tension-plastic above ``u``,
    compression-plastic below ``v``, elastic between. The derivation in
    ``ratchet_rate_exact`` needs exactly two facts about those fronts, and they
    are pinned by *different* arguments, so they get different tolerances:

    ``u + v = -2a`` is force equilibrium -- the elastic core is linear and
    antisymmetric about its own midpoint, so it contributes nothing to the
    thickness-mean stress -- and holds to machine precision whatever the
    quadrature. ``u - v = 2/b`` is the statement that the core spans
    ``2 sigma_Y`` at gradient ``b``, which is a front *position* and so is
    resolved only to the thickness spacing.

    Checking the mechanism rather than only the rate is what makes this a test
    of the derivation instead of a second copy of the measurement.
    """
    n = 4001
    a, b = lam * x, lam * y
    result = cyclic_strip(x, y, lam, n_thickness=n, max_cycles=400)
    zeta = np.linspace(-1.0, 1.0, n)
    hot = int(np.argmax(np.abs(result.elastic[:, -1, 0] - lam * x)))
    stress = result.stress[hot, :, 0]
    u = zeta[stress > 1.0 - 1e-9].min()
    v = zeta[stress < -1.0 + 1e-9].max()

    spacing = 2.0 / (n - 1)
    assert u + v == pytest.approx(-2.0 * a, abs=1e-12)
    # each front is located to within one interval, so their DIFFERENCE has a
    # two-interval budget -- the one-interval tolerance written first here
    # failed three of these seven rays by 7.1e-4 against 5.0e-4, which was the
    # error budget being wrong rather than the relation
    assert u - v == pytest.approx(2.0 / b, abs=2.0 * spacing + 1e-12)
    assert result.ratchet_rate == pytest.approx(2.0 * (a * b - 1.0), rel=3e-6)
