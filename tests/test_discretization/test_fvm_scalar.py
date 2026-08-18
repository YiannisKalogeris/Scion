"""
Verification for the scalar finite volume assembler and its limiters.

Four things are checked, and they are chosen because each fails differently:

**Conservation** to round-off. This is the property the method exists for and
the one no other discretisation here has. It holds structurally -- every
interior face enters two cells with opposite signs -- so a drift above 1e-14 is
not inaccuracy, it is a bookkeeping bug in the flux loop.

**Observed order.** One for upwind, two for central. A scheme with a mistyped
coefficient still converges; the slope is what catches it.

**The Peclet limit.** Central differencing oscillates above ``Pe = 2`` and
upwind never does, and both are consequences of the matrix rather than of the
mesh being coarse. The oscillation must *not* go away under refinement at fixed
``Pe``.

**Sweby's TVD region and the limiters.** Asserted on the functions directly and
then demonstrated on a transported square wave, where the unlimited scheme
overshoots by a third of the jump and every limiter overshoots by nothing.

python -m pytest tests/test_discretization/test_fvm_scalar.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from simulacrum.discretization.fvm import (
    FVMAssembler,
    LIMITER_REGISTRY,
    NoLimiter,
    get_flux,
    get_limiter,
    peclet_limit,
    slope_ratio,
)

LIMITERS = sorted(set(LIMITER_REGISTRY))
SECOND_ORDER = ["minmod", "van-leer", "van-albada", "mc", "superbee"]

PERIODIC = {"left": "periodic", "right": "periodic"}


def _order(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])]


def _layer(n, flux, a=1.0, D=0.01, L=1.0):
    """Steady boundary layer ``a u' = D u''``, u(0)=0, u(L)=1."""
    fv = FVMAssembler(None, (n,), spacing=L / n, velocity=a, diffusivity=D,
                      flux=flux)
    system = fv.assemble(conditions={"left": ("dirichlet", 0.0),
                                     "right": ("dirichlet", 1.0)})
    return fv, spsolve(system.matrix.tocsc(), system.rhs)


def _layer_exact_averages(fv, a=1.0, D=0.01, L=1.0):
    """Exact *cell averages*, not point values.

    Comparing a finite volume answer to a point sample of the exact solution
    confuses an O(h^2) quadrature difference with a discretisation error, which
    is enough to spoil a second-order measurement.
    """
    k = a * L / D
    xf = fv.faces(0)
    antiderivative = np.exp(k * xf) / k - xf
    return (np.diff(antiderivative) / np.diff(xf)) / (np.exp(k) - 1.0)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("limiter", LIMITERS)
def test_periodic_advection_conserves_the_total_to_round_off(limiter):
    """No flux crosses a periodic boundary, so the total cannot move at all --
    not to truncation error, to round-off, for every scheme and every limiter
    including the ones that oscillate."""
    n = 100
    fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=1.0,
                      flux="rusanov", limiter=limiter)
    x = fv.centres(0)
    u0 = np.exp(-100.0 * (x - 0.3) ** 2) + (np.abs(x - 0.7) < 0.1)

    u, _ = fv.solve_unsteady(u0, 1.0, PERIODIC, cfl=0.4, stages=3)
    assert abs(fv.total(u) - fv.total(u0)) < 1e-13


def test_conservation_holds_in_two_dimensions_too():
    n = 24
    fv = FVMAssembler(None, (n, n), spacing=1.0 / n, velocity=(0.7, -0.4),
                      flux="rusanov", limiter="van-leer")
    rng = np.random.default_rng(0)
    u0 = rng.random((n, n))
    conditions = {"left": "periodic", "right": "periodic",
                  "bottom": "periodic", "top": "periodic"}

    u, _ = fv.solve_unsteady(u0, 0.5, conditions, cfl=0.3)
    assert abs(fv.total(u) - fv.total(u0)) < 1e-12 * abs(fv.total(u0))


def test_the_total_changes_only_through_the_two_boundary_faces():
    """Every interior face cancels between its two cells, so the whole sum
    collapses to the two ends -- and those are cheap enough to write out by
    hand, which makes this an independent check of the telescoping rather than
    a restatement of it.

    ``first-order`` disables the reconstruction so the boundary face states are
    the ghost values themselves: at the inflow the reflected ghost
    ``2 u_b - u_0``, at the zero-gradient outflow the last cell.
    """
    n = 20
    a, u_b = 1.3, 0.4
    fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=a, flux="upwind",
                      limiter="first-order")
    u = np.random.default_rng(1).random(n)
    conditions = {"left": ("dirichlet", u_b), "right": "neumann"}

    expected = a * u[-1] - a * (2.0 * u_b - u[0])
    assert fv.boundary_flux(u, conditions) == pytest.approx(expected)

    # and a forward Euler step moves the total by exactly that, times dt
    after = u + 1.0 * fv.divergence(u, conditions)
    assert fv.total(after) - fv.total(u) == pytest.approx(-expected)


# ---------------------------------------------------------------------------
# Observed order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flux,expected", [("upwind", 1.0), ("central", 2.0)])
def test_the_steady_scheme_converges_at_its_advertised_order(flux, expected):
    """Manufactured solution ``u = sin(pi x)`` for ``a u' - D u'' = s``, at a
    Peclet number low enough that neither scheme is in trouble."""
    a, D = 1.0, 0.5
    exact = lambda x: np.sin(np.pi * x)
    source = lambda x: (a * np.pi * np.cos(np.pi * x)
                        + D * np.pi ** 2 * np.sin(np.pi * x))
    conditions = {"left": ("dirichlet", 0.0), "right": ("dirichlet", 0.0)}

    errors = []
    for n in (20, 40, 80, 160):
        fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=a,
                          diffusivity=D, flux=flux)
        system = fv.assemble(source=source, conditions=conditions)
        u = spsolve(system.matrix.tocsc(), system.rhs)
        errors.append(float(np.abs(u - exact(fv.centres(0))).max()))

    assert _order(errors)[-1] == pytest.approx(expected, abs=0.15)


def test_two_dimensional_diffusion_is_second_order():
    exact = lambda x, y: np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)
    source = lambda x, y: 5.0 * np.pi ** 2 * exact(x, y)
    conditions = {s: ("dirichlet", 0.0)
                  for s in ("left", "right", "bottom", "top")}

    errors = []
    for n in (8, 16, 32, 64):
        fv = FVMAssembler(None, (n, n), spacing=1.0 / n, velocity=0.0,
                          diffusivity=1.0, flux="central")
        system = fv.assemble(source=source, conditions=conditions)
        u = spsolve(system.matrix.tocsc(), system.rhs)
        errors.append(float(np.abs(u - exact(*fv.coordinates().T)).max()))

    assert _order(errors)[-1] == pytest.approx(2.0, abs=0.1)


def test_central_reaches_second_order_on_the_exponential_boundary_layer():
    """The classical closed-form advection-diffusion solution,
    ``(exp(Pe x/L) - 1)/(exp(Pe) - 1)``, compared against exact cell averages."""
    errors = []
    for n in (20, 40, 80, 160):
        fv, u = _layer(n, "central", D=0.1)
        errors.append(float(np.abs(u - _layer_exact_averages(fv, D=0.1)).max()))

    assert _order(errors)[-1] == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# The Peclet limit
# ---------------------------------------------------------------------------

def test_the_cell_peclet_number_is_reported_and_is_infinite_without_diffusion():
    fv = FVMAssembler(None, (10,), spacing=0.1, velocity=2.0, diffusivity=0.05)
    assert fv.peclet() == pytest.approx(4.0)
    assert peclet_limit(1.0, 0.0, 0.1) == float("inf")
    assert FVMAssembler(None, (10,), spacing=0.1,
                        velocity=1.0).peclet() == float("inf")


@pytest.mark.parametrize("n,oscillates", [(10, True), (25, True),
                                          (100, False), (400, False)])
def test_central_oscillates_above_peclet_two_and_only_above_it(n, oscillates):
    """The oscillation is a property of the operator, not of resolution. It
    appears exactly where ``Pe`` crosses two and vanishes exactly when
    refinement pushes it back below."""
    fv, u = _layer(n, "central")
    assert (fv.peclet() > 2.0) is oscillates
    # the exact solution is monotone between 0 and 1; anything outside is
    # spurious. bool() is not decoration -- `np.False_ is False` is False.
    assert bool(u.min() < -1e-9 or u.max() > 1.0 + 1e-9) is oscillates


@pytest.mark.parametrize("n", [10, 25, 50, 100, 400])
def test_upwind_stays_monotone_at_every_peclet_number(n):
    """The reason to accept first-order accuracy. Upwinding makes the operator
    an M-matrix, and an M-matrix cannot produce a new extremum whatever the
    Peclet number."""
    fv, u = _layer(n, "upwind")
    assert u.min() > -1e-13
    assert u.max() < 1.0 + 1e-13
    assert np.all(np.diff(u) > -1e-13)


def test_upwind_converges_on_the_layer_even_though_it_is_only_first_order():
    errors = []
    for n in (100, 400, 1600):
        fv, u = _layer(n, "upwind")
        errors.append(float(np.abs(u - _layer_exact_averages(fv)).max()))
    assert errors[-1] < 0.3 * errors[0]


# ---------------------------------------------------------------------------
# Fluxes
# ---------------------------------------------------------------------------

def test_rusanov_reduces_to_upwind_for_a_scalar_with_constant_velocity():
    """The point of Rusanov is that it needs no sign test, so on the one case
    where the sign is obvious the two must agree exactly."""
    left = np.array([1.0, -2.0, 0.5])
    right = np.array([3.0, 0.0, -1.5])
    for a in (1.3, -1.3):
        speed = np.full(3, a)
        np.testing.assert_allclose(get_flux("rusanov")(left, right, speed),
                                   get_flux("upwind")(left, right, speed))


@pytest.mark.parametrize("name", ["central", "upwind", "rusanov"])
def test_every_flux_is_exact_on_a_uniform_state(name):
    """A constant field must produce a constant flux, hence zero divergence:
    the discrete analogue of ``div(a c) = 0``. It is the cheapest way to catch
    an index slip in the reconstruction."""
    n = 12
    fv = FVMAssembler(None, (n,), spacing=0.1, velocity=0.8, flux=name,
                      limiter="van-leer")
    u = np.full(n, 2.5)
    assert np.abs(fv.divergence(u, PERIODIC)).max() < 1e-14


def test_an_unknown_flux_points_at_the_riemann_solvers():
    with pytest.raises(NotImplementedError, match="riemann"):
        get_flux("hllc")


# ---------------------------------------------------------------------------
# Limiters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SECOND_ORDER)
def test_every_limiter_lies_in_the_sweby_tvd_region(name):
    assert get_limiter(name).is_tvd()


@pytest.mark.parametrize("name", SECOND_ORDER)
def test_every_limiter_passes_through_one(name):
    """``phi(1) = 1`` is the second-order condition. Without it a limiter clips
    smooth extrema and falls back towards first order where it did not need
    to."""
    assert get_limiter(name).is_second_order()


def test_the_unlimited_and_first_order_references_behave_as_labelled():
    """Both are deliberately outside the useful set: one is second order and
    not TVD, the other TVD and not second order. Keeping them is what lets the
    tests above mean something."""
    assert NoLimiter().is_second_order() and not NoLimiter().is_tvd()
    first = get_limiter("first-order")
    assert first.is_tvd() and not first.is_second_order()


def test_minmod_and_superbee_bound_the_region_from_below_and_above():
    r = np.linspace(0.0, 4.0, 400)
    minmod, superbee = get_limiter("minmod")(r), get_limiter("superbee")(r)
    for name in ("van-leer", "van-albada", "mc"):
        phi = get_limiter(name)(r)
        assert np.all(phi >= minmod - 1e-12)
        assert np.all(phi <= superbee + 1e-12)


def test_the_slope_ratio_is_one_where_the_solution_is_linear():
    """A limiter only leaves a linear profile alone because ``r`` is exactly
    one there; if the ratio is off, every limiter clips a straight line."""
    u = 3.0 * np.arange(10) - 1.0
    np.testing.assert_allclose(slope_ratio(u), np.ones(10))


def test_a_flat_region_does_not_produce_a_division_by_zero():
    u = np.full(8, 2.0)
    r = slope_ratio(u)
    assert np.all(np.isfinite(r))


def test_an_unknown_limiter_is_refused():
    with pytest.raises(NotImplementedError, match="Unknown limiter"):
        get_limiter("weno")


# ---------------------------------------------------------------------------
# TVD in practice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SECOND_ORDER + ["first-order"])
def test_a_transported_square_wave_gains_no_new_extremum(name):
    """The property the Sweby region buys, demonstrated rather than asserted.
    A square wave advected once around a periodic domain must stay inside its
    own bounds."""
    n = 100
    fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=1.0,
                      flux="rusanov", limiter=name)
    u0 = (np.abs(fv.centres(0) - 0.5) < 0.15).astype(float)

    u, _ = fv.solve_unsteady(u0, 1.0, PERIODIC, cfl=0.4, stages=3)
    assert u.max() <= u0.max() + 1e-12
    assert u.min() >= u0.min() - 1e-12


def test_the_unlimited_scheme_is_the_one_that_overshoots():
    """Without this the test above proves nothing -- a scheme that returns the
    initial data unchanged would pass it too."""
    n = 100
    fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=1.0,
                      flux="rusanov", limiter="none")
    u0 = (np.abs(fv.centres(0) - 0.5) < 0.15).astype(float)

    u, _ = fv.solve_unsteady(u0, 1.0, PERIODIC, cfl=0.4, stages=3)
    assert u.max() > u0.max() + 0.2


def test_the_limiters_are_ordered_by_how_much_they_diffuse():
    """Minmod hugs the lower edge of the Sweby region and superbee the upper,
    so on a profile with both a smooth bump and a jump they come out in a fixed
    order. This pins the *identity* of each limiter, which ``phi(1) = 1`` and
    the TVD bound between them do not."""
    n = 100
    x = np.linspace(0.005, 0.995, n)
    u0 = np.exp(-100.0 * (x - 0.3) ** 2) + (np.abs(x - 0.7) < 0.1)

    errors = {}
    for name in ("first-order", "minmod", "van-leer", "mc", "superbee"):
        fv = FVMAssembler(None, (n,), spacing=1.0 / n, velocity=1.0,
                          flux="rusanov", limiter=name)
        u, _ = fv.solve_unsteady(u0, 1.0, PERIODIC, cfl=0.4, stages=3)
        errors[name] = float(np.abs(u - u0).mean())

    ordered = ["first-order", "minmod", "van-leer", "mc", "superbee"]
    assert [errors[k] for k in ordered] == sorted(
        (errors[k] for k in ordered), reverse=True)


# ---------------------------------------------------------------------------
# Grid bookkeeping and refusals
# ---------------------------------------------------------------------------

def test_cells_tile_the_domain_and_centres_sit_at_half_steps():
    """The difference from the FDM assembler, which counts interior points and
    puts the boundary a full step beyond the first unknown."""
    fv = FVMAssembler(None, (4,), spacing=0.25, origin=1.0)
    np.testing.assert_allclose(fv.centres(0), [1.125, 1.375, 1.625, 1.875])
    np.testing.assert_allclose(fv.faces(0), [1.0, 1.25, 1.5, 1.75, 2.0])
    assert fv.n_cells == 4
    assert fv.length(0) == pytest.approx(1.0)


def test_coordinates_are_in_solve_order():
    fv = FVMAssembler(None, (2, 3), spacing=(1.0, 10.0))
    points = fv.coordinates()
    assert points.shape == (6, 2)
    np.testing.assert_allclose(points[:3, 1], [5.0, 15.0, 25.0])
    assert fv.recover_fields(np.arange(6))["primary"].shape == (2, 3)


def test_the_matrix_matches_the_operator_it_was_probed_from():
    """``assemble`` builds the matrix by probing the same routine the explicit
    stepper calls, so the two can never disagree -- which is the point of doing
    it that way instead of writing the stencil out twice."""
    n = 9
    fv = FVMAssembler(None, (n,), spacing=0.3, velocity=1.4, diffusivity=0.2,
                      flux="upwind")
    conditions = {"left": ("dirichlet", 2.0), "right": "neumann"}
    system = fv.assemble(source=None, conditions=conditions)

    rng = np.random.default_rng(3)
    u = rng.normal(size=n)
    np.testing.assert_allclose(system.matrix @ u - system.rhs,
                               -fv._divergence(u, conditions, width=1),
                               atol=1e-12)


def test_the_steady_matrix_is_the_negative_laplacian_for_pure_diffusion():
    h = 0.2
    fv = FVMAssembler(None, (4,), spacing=h, velocity=0.0, diffusivity=1.0,
                      flux="central")
    A = fv.assemble(conditions=fv.default_conditions("dirichlet")
                    ).matrix.toarray()
    # interior row is the familiar (-1, 2, -1)/h^2; the Dirichlet row is
    # (3, -1)/h^2, because the reflected ghost puts the wall half a cell away
    # and doubles that face's gradient
    np.testing.assert_allclose(A[1], np.array([-1.0, 2.0, -1.0, 0.0]) / h ** 2)
    np.testing.assert_allclose(A[0], np.array([3.0, -1.0, 0.0, 0.0]) / h ** 2)


def test_a_half_periodic_direction_is_refused():
    fv = FVMAssembler(None, (6,), spacing=0.1)
    with pytest.raises(ValueError, match="periodic at both ends"):
        fv.divergence(np.zeros(6), {"left": "periodic", "right": "neumann"})


def test_a_missing_or_unknown_boundary_condition_is_refused():
    fv = FVMAssembler(None, (6,), spacing=0.1)
    with pytest.raises(KeyError, match="left"):
        fv.divergence(np.zeros(6), {"right": "neumann"})
    with pytest.raises(ValueError, match="unknown boundary"):
        fv.divergence(np.zeros(6), {"left": "robin", "right": "neumann"})


def test_unsupported_dimensions_and_degenerate_grids_are_refused():
    with pytest.raises(NotImplementedError, match="3-D"):
        FVMAssembler(None, (4, 4, 4))
    with pytest.raises(ValueError, match="at least two cells"):
        FVMAssembler(None, (1,))
    with pytest.raises(ValueError, match="1, 2 or 3 stages"):
        FVMAssembler(None, (6,), spacing=0.1).advance(
            np.zeros(6), 0.01, PERIODIC, stages=4)


def test_the_stable_step_accounts_for_diffusion_as_well_as_advection():
    """The diffusive limit scales as ``h^2``, so on a fine mesh it sets the
    step however small the diffusivity is. Sizing from the advective limit
    alone is stable on a coarse grid and blows up on a fine one."""
    coarse = FVMAssembler(None, (10,), spacing=0.1, velocity=1.0,
                          diffusivity=1e-3)
    fine = FVMAssembler(None, (10000,), spacing=1e-4, velocity=1.0,
                        diffusivity=1e-3)
    assert coarse.stable_dt() == pytest.approx(0.4 / (10.0 + 0.2))
    # advection alone would allow 4e-5 here; diffusion cuts it to 1.9e-6
    advective_only = 0.4 * 1e-4 / 1.0
    assert fine.stable_dt() == pytest.approx(0.4 / (1e4 + 2e5))
    assert fine.stable_dt() < 0.05 * advective_only
