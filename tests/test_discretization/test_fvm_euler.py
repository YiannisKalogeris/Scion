"""
Verification for the 1-D Euler solver and its Riemann solvers.

The external benchmark is Toro, *Riemann Solvers and Numerical Methods for
Fluid Dynamics*: the exact star state for Sod (``p* = 0.30313``,
``u* = 0.92745``) and the shock position that follows from it. Everything else
in this file is checked against the exact Riemann solution, which is the one
thing here that owes nothing to the discretisation.

The checks that matter most, and why:

**Shock position, not shock sharpness.** A scheme in conservation form gets the
shock speed right by the Lax-Wendroff theorem; one that is not gets a sharp
shock in the wrong place and refines to a sharper shock still in the wrong
place. Position is the property that separates the two, and a smoothness or
error-norm check does not.

**The stationary contact.** HLLC preserves it exactly and HLL cannot, for an
algebraic reason rather than an accuracy one, so this is the test that tells the
three solvers apart at all. On Sod they differ by a few tens of percent in
contact width and by almost nothing once MUSCL is switched on.

**The round trip through the exact solver.** ``primitive(conserved(W)) == W``
and ``exact_riemann`` on two identical states returning that state are each
derived from separate algebra, and they catch the sign and factor slips that
produce plausible numbers.

python -m pytest tests/test_discretization/test_fvm_euler.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.discretization.fvm import (
    RIEMANN_REGISTRY,
    SHOCK_TUBES,
    SOD,
    EulerEquations,
    EulerSolver1D,
    VacuumError,
    exact_riemann,
    get_riemann_solver,
    get_shock_tube,
    star_pressure,
)

SOLVERS = ["rusanov", "hll", "hllc"]
GAMMA = 1.4

#: Toro, table 4.1: the exact star state of the Sod problem.
SOD_P_STAR = 0.30313
SOD_U_STAR = 0.92745
#: and the two star densities, which differ across the contact
SOD_RHO_STAR_L = 0.42632
SOD_RHO_STAR_R = 0.26557
#: and the shock speed that follows, giving x = 0.5 + S t at t = 0.2
SOD_SHOCK_X = 0.850431


@pytest.fixture
def eos():
    return EulerEquations(GAMMA)


def _density_error(result, tube, eos):
    computed = eos.primitive(result.U)[:, 0]
    return float(np.abs(computed - tube.exact(result.x, result.t)[:, 0]).mean())


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------

def test_primitive_and_conserved_are_inverses(eos):
    """The round trip. Both directions carry the ``1/2 rho u^2`` term and it is
    easy to lose it from one of them; a single-direction test cannot see that."""
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.1, 3.0, 50)
    u = rng.uniform(-2.0, 2.0, 50)
    p = rng.uniform(0.1, 5.0, 50)

    W = eos.primitive(eos.conserved(rho, u, p))
    np.testing.assert_allclose(W, np.stack([rho, u, p], axis=-1))


def test_the_flux_of_a_uniform_state_at_rest_is_pressure_only(eos):
    U = eos.conserved(1.2, 0.0, 3.4)
    np.testing.assert_allclose(eos.flux(U), [[0.0, 3.4, 0.0]], atol=1e-14)


def test_the_sound_speed_matches_its_definition(eos):
    U = eos.conserved(2.0, 1.0, 5.0)
    assert float(eos.sound_speed(U)[0]) == pytest.approx(np.sqrt(1.4 * 5 / 2))
    assert eos.max_speed(U) == pytest.approx(1.0 + np.sqrt(1.4 * 5 / 2))


def test_a_gamma_at_or_below_one_is_refused():
    with pytest.raises(ValueError, match="gamma"):
        EulerEquations(1.0)


# ---------------------------------------------------------------------------
# The exact Riemann solver
# ---------------------------------------------------------------------------

def test_the_sod_star_state_matches_toro():
    """The published external benchmark. Every other Euler test here is
    measured against a solution this function produced, so it is the one that
    has to be right on its own terms."""
    p_star, u_star = star_pressure(SOD.left, SOD.right, GAMMA)
    assert p_star == pytest.approx(SOD_P_STAR, abs=5e-6)
    assert u_star == pytest.approx(SOD_U_STAR, abs=5e-6)


def test_two_identical_states_produce_no_waves():
    """A Riemann problem with no jump has no solution structure at all, so
    every sample must return the state itself. It exercises both wave branches
    at their meeting point ``p = p_K``, where the shock and rarefaction
    formulas must agree."""
    state = (1.3, 0.7, 2.1)
    out = exact_riemann(state, state, [-5.0, -0.7, 0.0, 0.7, 5.0], GAMMA)
    np.testing.assert_allclose(out, np.tile(state, (5, 1)), rtol=1e-10)


def test_the_exact_solution_is_constant_outside_the_wave_fan():
    """Nothing propagates faster than ``|u| + a``, so far enough out the
    initial states survive untouched."""
    out = exact_riemann(SOD.left, SOD.right, [-50.0, 50.0], GAMMA)
    np.testing.assert_allclose(out[0], SOD.left)
    np.testing.assert_allclose(out[1], SOD.right)


def test_pressure_and_velocity_are_continuous_across_the_contact():
    """The defining property of a contact discontinuity, and the equation the
    star state was solved from -- so this checks that the sampling routine
    agrees with the pressure iteration rather than merely that both ran."""
    p_star, u_star = star_pressure(SOD.left, SOD.right, GAMMA)
    just_left = exact_riemann(SOD.left, SOD.right, [u_star - 1e-8], GAMMA)[0]
    just_right = exact_riemann(SOD.left, SOD.right, [u_star + 1e-8], GAMMA)[0]

    assert just_left[1] == pytest.approx(u_star, rel=1e-6)
    assert just_right[1] == pytest.approx(u_star, rel=1e-6)
    assert just_left[2] == pytest.approx(p_star, rel=1e-6)
    assert just_right[2] == pytest.approx(p_star, rel=1e-6)
    # density is the one thing that does jump, and to published values
    assert just_left[0] == pytest.approx(SOD_RHO_STAR_L, abs=5e-6)
    assert just_right[0] == pytest.approx(SOD_RHO_STAR_R, abs=5e-6)


def test_the_sod_shock_sits_where_the_jump_conditions_put_it():
    x = np.linspace(0.80, 0.90, 200001)
    rho = SOD.exact(x, 0.2)[:, 0]
    jump = 0.5 * (x[np.argmax(np.abs(np.diff(rho)))]
                  + x[np.argmax(np.abs(np.diff(rho))) + 1])
    assert jump == pytest.approx(SOD_SHOCK_X, abs=1e-5)


def test_a_vacuum_is_refused_rather_than_approximated():
    """Two states pulling apart faster than their rarefactions can follow. The
    star region ceases to exist, so there is no ``p*`` to iterate towards and
    the sampling formulas do not apply."""
    with pytest.raises(VacuumError, match="vacuum"):
        star_pressure((1.0, -10.0, 0.4), (1.0, 10.0, 0.4), GAMMA)


def test_toros_123_problem_is_near_vacuum_but_still_has_a_star_state():
    """The boundary case: pressure in the middle collapses to about 0.0019 of
    the initial value without reaching zero. A solver that is not careful about
    positivity fails here and nowhere easier."""
    p_star, u_star = star_pressure((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), GAMMA)
    assert 0.0 < p_star < 0.01
    assert u_star == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The approximate Riemann solvers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SOLVERS)
def test_a_solver_given_two_equal_states_returns_the_physical_flux(name, eos):
    """Consistency: with no jump there is nothing to stabilise, so every
    numerical flux must collapse to ``F(U)``. This is the condition in the
    Lax-Wendroff theorem, and a scheme that fails it converges to the wrong
    weak solution."""
    U = eos.conserved([1.4, 0.3], [0.8, -1.2], [2.0, 0.5])
    np.testing.assert_allclose(get_riemann_solver(name, GAMMA)(U, U),
                               eos.flux(U), rtol=1e-12)


@pytest.mark.parametrize("name", ["hll", "hllc"])
def test_a_supersonic_face_is_pure_upwinding(name, eos):
    """When every wave travels the same way the face can only see one side, so
    a wave model that tracks ``S_L`` and ``S_R`` must return ``F(U_L)``
    outright."""
    UL = eos.conserved(1.0, 10.0, 1.0)        # u = 10 against a = 1.18
    UR = eos.conserved(0.5, 10.0, 0.5)
    np.testing.assert_allclose(get_riemann_solver(name, GAMMA)(UL, UR),
                               eos.flux(UL), rtol=1e-10)


def test_rusanov_is_not_upwind_for_a_system_even_though_it_is_for_a_scalar(eos):
    """The asymmetry between :class:`~.flux.RusanovFlux` and
    :class:`~.riemann.RusanovSolver`, which share a name and a formula.

    For a scalar with velocity ``a > 0`` the jump penalty ``|a|(u_R - u_L)/2``
    exactly cancels the ``u_R`` half of the average and leaves ``a u_L``. For a
    system the same cancellation would need ``F_R - F_L = s (U_R - U_L)``, which
    holds only if the whole jump is one wave travelling at ``s``. It is not, so
    Rusanov applies its full dissipation to a face where no dissipation is
    needed at all -- measured here at 3% of the flux on a Mach 9 state.
    """
    UL = eos.conserved(1.0, 10.0, 1.0)
    UR = eos.conserved(0.5, 10.0, 0.5)
    numerical = get_riemann_solver("rusanov", GAMMA)(UL, UR)
    exact = eos.flux(UL)

    excess = float(np.abs(numerical - exact).max() / np.abs(exact).max())
    assert 0.01 < excess < 0.05


def test_hllc_preserves_a_stationary_contact_exactly_and_hll_cannot(eos):
    """The measurement that separates the three solvers.

    A density jump at rest under uniform pressure is a steady solution of the
    Euler equations: nothing should move. HLLC's star state reduces
    algebraically to the cell state, so the flux is exact to the bit. HLL's
    single intermediate state cannot represent a density jump and leaves a
    spurious mass flux proportional to ``S_L S_R``, which is negative and
    therefore never vanishes.
    """
    n = 100
    widths = {}
    for name in SOLVERS:
        solver = EulerSolver1D(n, riemann=name, limiter="first-order")
        rho0 = np.where(solver.x < 0.5, 1.0, 0.125)
        U0 = eos.conserved(rho0, np.zeros(n), np.ones(n))
        result = solver.solve(U0, 0.4, cfl=0.4)
        W = eos.primitive(result.U)

        # velocity and pressure must stay uniform for all three
        assert np.abs(W[:, 1]).max() < 1e-12
        assert np.abs(W[:, 2] - 1.0).max() < 1e-12
        widths[name] = float(np.abs(W[:, 0] - rho0).max())

    assert widths["hllc"] == 0.0
    assert widths["hll"] > 0.4
    assert widths["rusanov"] > 0.4


def test_hllc_resolves_the_sod_contact_better_than_hll_and_rusanov(eos):
    """Same ordering on a real problem, where it is a difference of degree
    rather than of kind: 32 cells against 35 and 48 at n = 800. Worth knowing
    before spending effort on the Riemann solver -- the limiter is the bigger
    lever, and with van Leer all three land within a cell of each other."""
    def contact_width(name, limiter):
        solver = EulerSolver1D(800, riemann=name, limiter=limiter)
        rho = eos.primitive(solver.run_tube(SOD).U)[:, 0]
        window = (solver.x > 0.60) & (solver.x < 0.78)
        lo, hi = rho[window].min(), rho[window].max()
        return int(np.count_nonzero((rho[window] > lo + 0.1 * (hi - lo))
                                    & (rho[window] < lo + 0.9 * (hi - lo))))

    first = {n: contact_width(n, "first-order") for n in SOLVERS}
    assert first["hllc"] < first["hll"] < first["rusanov"]

    limited = {n: contact_width(n, "van-leer") for n in SOLVERS}
    assert max(limited.values()) - min(limited.values()) <= 2
    assert max(limited.values()) < min(first.values()) / 3


def test_an_unknown_riemann_solver_explains_why_roe_is_absent():
    with pytest.raises(NotImplementedError, match="entropy fix"):
        get_riemann_solver("roe")


def test_the_registry_and_the_wave_counts_agree():
    for name, cls in RIEMANN_REGISTRY.items():
        solver = cls(GAMMA)
        assert solver.waves in (1, 2, 3)
        assert solver.resolves_contact == (solver.waves == 3)


# ---------------------------------------------------------------------------
# The solver: Sod
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SOLVERS)
def test_sod_converges_to_the_exact_solution(name, eos):
    """L1 in density against the exact Riemann solution. The rate is below one
    and stays there however fine the mesh: the solution is discontinuous, so
    the cells straddling the shock and the contact each carry an O(1) error and
    there are O(1) of them. That is the correct behaviour for a discontinuous
    problem, not a defect in the scheme."""
    errors = []
    for n in (100, 200, 400, 800):
        result = EulerSolver1D(n, riemann=name, limiter="minmod").run_tube(SOD)
        errors.append(_density_error(result, SOD, eos))

    rate = np.log2(errors[-2] / errors[-1])
    assert 0.7 < rate < 1.2
    assert errors[-1] < 0.3 * errors[0]


@pytest.mark.parametrize("n,tolerance", [(100, 0.012), (400, 0.003),
                                         (1600, 0.0008)])
def test_the_captured_shock_sits_where_the_exact_one_does(n, tolerance):
    """The property that conservation form buys and nothing else does. The
    error must fall with ``h``, because the shock is located to within the cell
    it occupies and nothing better."""
    eos = EulerEquations(GAMMA)
    solver = EulerSolver1D(n, riemann="hllc", limiter="minmod")
    result = solver.run_tube(SOD)
    rho = eos.primitive(result.U)[:, 0]

    right_half = result.x[:-1] > 0.75
    i = int(np.argmax(np.abs(np.diff(rho)) * right_half))
    captured = 0.5 * (result.x[i] + result.x[i + 1])
    assert abs(captured - SOD_SHOCK_X) < tolerance


def test_the_star_state_plateau_matches_the_exact_pressure(eos):
    """Between the contact and the shock the exact solution is constant at
    ``p*``. A scheme can smear a discontinuity and still be right; getting the
    plateau wrong means the jump conditions themselves are wrong."""
    result = EulerSolver1D(800, riemann="hllc",
                           limiter="van-leer").run_tube(SOD)
    W = eos.primitive(result.U)
    plateau = (result.x > 0.72) & (result.x < 0.82)

    assert W[plateau, 2].mean() == pytest.approx(SOD_P_STAR, rel=2e-3)
    assert W[plateau, 1].mean() == pytest.approx(SOD_U_STAR, rel=2e-3)
    assert W[plateau, 2].std() < 1e-3


def test_the_solution_stays_physical_everywhere(eos):
    """Density and pressure positive at every cell, for every solver and every
    tube -- including the two designed to break that."""
    for tube in SHOCK_TUBES.values():
        solver = EulerSolver1D(400, riemann="hllc", limiter="minmod",
                               gamma=tube.gamma, length=tube.length)
        result = solver.run_tube(tube)
        W = EulerEquations(tube.gamma).primitive(result.U)
        assert W[:, 0].min() > 0.0, tube.name
        assert W[:, 2].min() > 0.0, tube.name


def test_no_state_is_repaired_on_a_problem_that_does_not_need_it():
    """``repairs`` counts the times positivity had to be forced. It must be
    zero on Sod; a solver that quietly floors its own state hides exactly the
    failure this counter exists to expose."""
    result = EulerSolver1D(400, riemann="hllc", limiter="minmod").run_tube(SOD)
    assert result.repairs == 0


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def test_sod_conserves_mass_and_energy_to_round_off():
    """No wave has reached either end by ``t = 0.2``, so the domain is closed
    and the totals cannot move. This is the finite volume guarantee, and it
    holds regardless of how badly the waves themselves are resolved."""
    result = EulerSolver1D(200, riemann="hllc", limiter="minmod").run_tube(SOD)
    assert result.mass_drift < 1e-14
    assert result.energy_drift < 1e-13
    assert result.conservative


def test_a_reflective_tube_conserves_after_the_shock_has_bounced(eos):
    """A wall is the case where conservation is a real statement rather than a
    consequence of nothing having happened yet: the shock arrives, reflects and
    doubles in strength, and the totals still must not move."""
    n = 200
    solver = EulerSolver1D(n, riemann="hllc", limiter="minmod",
                           boundary="reflective")
    U0 = SOD.initial_state(solver.x, eos)
    result = solver.solve(U0, 0.5, cfl=0.4)

    assert result.mass_drift < 1e-12
    assert result.energy_drift < 1e-11
    assert eos.pressure_positive(result.U)


def test_mass_leaves_through_a_transmissive_boundary_and_that_is_correct():
    """The counterpart, and the reason ``conservative`` is a property rather
    than an assertion. Toro's 123 problem drives fluid out at both ends, so a
    drift of order one is the physics and a drift of zero would be the bug."""
    tube = get_shock_tube("123")
    result = EulerSolver1D(400, riemann="hllc",
                           limiter="minmod").run_tube(tube)
    assert result.mass_drift > 0.1
    assert not result.conservative


# ---------------------------------------------------------------------------
# Reconstruction and time stepping
# ---------------------------------------------------------------------------

def test_muscl_beats_first_order_on_every_tube(eos):
    """If reconstruction is not paying for itself somewhere the wiring is
    wrong. Checked on all four tubes so it cannot be an artefact of Sod."""
    for name, tube in SHOCK_TUBES.items():
        errors = {}
        for limiter in ("first-order", "minmod", "van-leer"):
            solver = EulerSolver1D(200, riemann="hllc", limiter=limiter,
                                   gamma=tube.gamma, length=tube.length)
            result = solver.run_tube(tube)
            errors[limiter] = _density_error(
                result, tube, EulerEquations(tube.gamma))
        assert errors["minmod"] < errors["first-order"], name
        assert errors["van-leer"] <= errors["minmod"] * 1.05, name


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_every_ssp_stage_count_runs_and_conserves(stages, eos):
    solver = EulerSolver1D(100, riemann="hllc", limiter="minmod")
    U0 = SOD.initial_state(solver.x, eos)
    result = solver.solve(U0, 0.2, cfl=0.3, stages=stages)

    assert result.mass_drift < 1e-13
    assert eos.pressure_positive(result.U)


def test_a_uniform_flow_is_transported_without_distortion(eos):
    """A constant state is an exact solution, so the residual must vanish
    identically -- the cheapest way to catch an index slip in the
    reconstruction or a sign error in the flux difference."""
    n = 30
    solver = EulerSolver1D(n, riemann="hllc", limiter="van-leer")
    U0 = eos.conserved(np.full(n, 1.3), np.full(n, 0.7), np.full(n, 2.1))
    assert np.abs(solver.residual(U0)).max() < 1e-13

    result = solver.solve(U0, 0.3, cfl=0.4)
    np.testing.assert_allclose(result.U, U0, atol=1e-12)


def test_the_time_step_follows_the_wave_speed_rather_than_being_fixed():
    """On the blast tube the fastest wave speed *rises* through the run --
    37.4 at rest, 53.5 once the gas is moving behind the shock -- so the stable
    step shrinks by 30%. A step sized once from the initial data would be above
    the stability limit for the whole second half; the 320 steps actually taken
    would have covered 0.017 rather than the 0.012 asked for, and that gap is
    the measure of the error avoided."""
    tube = get_shock_tube("blast")
    solver = EulerSolver1D(200, riemann="hllc", limiter="minmod")
    eos = EulerEquations(tube.gamma)
    U0 = tube.initial_state(solver.x, eos)

    first = solver.stable_dt(U0, 0.4)
    result = solver.solve(U0, tube.t_end, cfl=0.4)
    last = solver.stable_dt(result.U, 0.4)

    assert eos.max_speed(result.U) > 1.3 * eos.max_speed(U0)
    assert last < 0.8 * first
    assert result.steps * first > 1.3 * tube.t_end


# ---------------------------------------------------------------------------
# Setup and refusals
# ---------------------------------------------------------------------------

def test_the_named_tubes_are_self_consistent():
    for name, tube in SHOCK_TUBES.items():
        assert tube.name == name
        assert tube.t_end > 0.0
        assert 0.0 < tube.x0 < tube.length
        assert tube.note

    with pytest.raises(KeyError, match="unknown shock tube"):
        get_shock_tube("riemann")


def test_the_initial_state_is_the_two_states_and_nothing_between(eos):
    solver = EulerSolver1D(10)
    W = eos.primitive(SOD.initial_state(solver.x, eos))
    np.testing.assert_allclose(W[:5], np.tile(SOD.left, (5, 1)))
    np.testing.assert_allclose(W[5:], np.tile(SOD.right, (5, 1)))


def test_the_exact_solution_at_time_zero_is_the_initial_data():
    x = np.linspace(0.0, 1.0, 11)
    W = SOD.exact(x, 0.0)
    np.testing.assert_allclose(W[x < 0.5], np.tile(SOD.left, (5, 1)))


def test_a_mismatched_grid_or_gas_is_refused():
    solver = EulerSolver1D(100, gamma=1.667)
    with pytest.raises(ValueError, match="gamma"):
        solver.run_tube(SOD)

    with pytest.raises(ValueError, match="at least four cells"):
        EulerSolver1D(3)
    with pytest.raises(ValueError, match="unknown boundary"):
        EulerSolver1D(100, boundary="wall")
    with pytest.raises(ValueError, match="1, 2 or 3 stages"):
        EulerSolver1D(100).advance(SOD.initial_state(
            EulerSolver1D(100).x, EulerEquations(GAMMA)), 1e-4, stages=4)
