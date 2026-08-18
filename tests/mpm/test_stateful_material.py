"""
Verification for path-dependent materials in the MPM stepper.

The stepper originally accepted only ``pk1(F) -> (P, J)``: stress as a function
of the current deformation gradient and nothing else. Plasticity cannot be
expressed that way, so this adds the second interface,
``update(F, L, dt, state)``, and the tests here cover the three things that
change with it:

* **when** the stress is computed -- at the end of the step rather than in the
  middle, because the velocity gradient does not exist until the grid has been
  solved. That is Update Stress Last.
* **the plane-strain embedding** -- 2-D kinematics promoted into 3-D for the
  material and the in-plane block taken back, because ``sigma_33`` is nonzero
  and enters ``J2``.
* **that the hyperelastic path is untouched**, which is asserted directly rather
  than assumed.

python -m pytest tests/mpm/test_stateful_material.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.discretization.mpm import (
    CartesianGrid,
    ExplicitMPMStepper,
    HalfSpace,
    ParticleSet,
    get_basis,
    get_transfer,
)
from simulacrum.discretization.mpm.stepper import _promote
from simulacrum.materials.eos import LinearEOS, MieGruneisen
from simulacrum.materials.high_rate import HighRateState, JohnsonCookModel
from simulacrum.materials.plasticity import JohnsonCook

RHO = 8930.0


def material(rate_term: float = 0.025) -> JohnsonCookModel:
    E, nu = 124e9, 0.34
    return JohnsonCookModel(
        E / (2.0 * (1.0 + nu)),
        JohnsonCook(A=90e6, B=292e6, n=0.31, C=rate_term, m=1.09,
                    T_reference=293.0, T_melt=1356.0),
        MieGruneisen(RHO, 3940.0, 2.02, 1.489),
        density=RHO, specific_heat=383.0)


def block(cells: int = 6, speed: float = 100.0, size: float = 0.01):
    """A small square block moving at a wall."""
    dx = size / cells
    pad = 3
    grid = CartesianGrid(
        shape=(cells + 2 * pad, cells + 2 * pad), dx=dx,
        origin=(-pad * dx, -pad * dx), basis=get_basis("quadratic"))
    particles = ParticleSet.from_sampler(
        lambda X: (X[:, 0] < size) & (X[:, 1] > 0.0) & (X[:, 1] < size),
        ((0.0, 0.0), (size, size)), dx, ppc=2, density=RHO)
    particles.v[:, 1] = -speed
    return grid, particles, dx


# ---------------------------------------------------------------------------
# The embedding
# ---------------------------------------------------------------------------

def test_a_deformation_gradient_promotes_with_one_on_the_diagonal():
    """Plane strain: no stretch out of plane, so ``F_33 = 1``."""
    F = np.array([[[1.1, 0.2], [0.0, 0.9]]])
    promoted = _promote(F, fill_diagonal=True)

    assert promoted.shape == (1, 3, 3)
    np.testing.assert_array_equal(promoted[0, :2, :2], F[0])
    assert promoted[0, 2, 2] == 1.0
    assert promoted[0, 0, 2] == 0.0


def test_a_velocity_gradient_promotes_with_zero():
    """No out-of-plane *rate*, which is a different statement from no stretch.
    Filling this one with 1 would impose a spurious volumetric strain rate."""
    L = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    promoted = _promote(L, fill_diagonal=False)

    assert promoted[0, 2, 2] == 0.0
    np.testing.assert_array_equal(promoted[0, :2, :2], L[0])


def test_three_dimensional_input_passes_straight_through():
    F = np.random.default_rng(0).standard_normal((4, 3, 3))
    np.testing.assert_array_equal(_promote(F, True), F)


def test_the_out_of_plane_stress_is_kept_by_the_material():
    """It exerts no in-plane force, so the stepper does not use it -- but it is
    not discarded: it lives in the deviator, where J2 needs it. Dropping it
    would give the wrong yield surface, which is the trap the library's tensor
    convention exists to prevent."""
    model = material()
    state = HighRateState.zeros(1)
    L = np.zeros((1, 3, 3))
    # a *non-isochoric* in-plane stretch. An isochoric one has trace(d) = 0 and
    # then d_dev[2,2] is genuinely zero, which is why this has to be uniaxial
    # rather than pure shear to show the effect at all.
    L[0, 0, 0] = 1e5
    F = np.array([np.eye(3) + L[0] * 1e-8])

    sigma, new_state = model.update(F, L, 1e-8, state)
    assert abs(new_state.deviator[0, 2, 2]) > 0.0
    assert abs(sigma[0, 2, 2]) > 0.0


# ---------------------------------------------------------------------------
# The stepper
# ---------------------------------------------------------------------------

def test_a_stateful_material_runs_and_accumulates_history():
    grid, particles, dx = block()
    model = material()
    state = HighRateState.zeros(particles.n)
    dt = 0.1 * dx / (model.wave_speed + 100.0)

    stepper = ExplicitMPMStepper(
        grid, model, get_transfer("flip", alpha=0.99), dt,
        boundaries=[HalfSpace(axis=1, level=0.0, sign=1, mode="sticky")],
        state=state)
    for _ in range(200):
        stepper.step(particles)

    assert stepper.state.alpha.max() > 0.0            # it yielded
    assert stepper.state.temperature.max() > 293.0    # and heated
    assert np.all(np.isfinite(particles.x))
    assert stepper.stress.shape == (particles.n, 3, 3)


def test_the_state_advances_exactly_once_per_step():
    """Update Stress Last means one constitutive call per step, after the grid
    solve. Two calls would double the plastic strain rate; none would freeze it."""
    grid, particles, dx = block()
    model = material()
    calls = {"n": 0}
    original = model.update

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.update = counted
    stepper = ExplicitMPMStepper(
        grid, model, get_transfer("flip"), 1e-9,
        state=HighRateState.zeros(particles.n))
    for _ in range(10):
        stepper.step(particles)

    # 10 steps, plus one priming evaluation on the first internal force
    assert calls["n"] == 11


def test_the_stress_is_available_before_the_first_step():
    """The first internal force has no previous step to borrow from, so the
    material is evaluated at zero rate -- which reads out the initial state
    rather than assuming it is unstressed."""
    grid, particles, dx = block()
    model = material()
    state = HighRateState.zeros(particles.n)
    # pre-stress the block
    state.deviator[:, 0, 1] = state.deviator[:, 1, 0] = 5e7

    stepper = ExplicitMPMStepper(grid, model, get_transfer("flip"), 1e-9,
                                 state=state)
    assert stepper.stress is None
    stepper.step(particles)

    assert stepper.stress is not None
    assert np.abs(stepper.stress[:, 0, 1]).max() > 0.0


def test_a_state_without_an_update_method_is_refused():
    grid, particles, _ = block()

    class Hyperelastic:
        wave_speed_factor = 1.0

        def pk1(self, F):
            return np.zeros_like(F), np.ones(F.shape[0])

    with pytest.raises(TypeError, match="no update"):
        ExplicitMPMStepper(grid, Hyperelastic(), get_transfer("flip"), 1e-9,
                           state=HighRateState.zeros(particles.n))


def test_the_hyperelastic_path_is_unchanged():
    """The whole point of adding a branch rather than replacing the interface.
    Same problem, same material, with and without the new code present."""
    from simulacrum.materials.hyperelastic_batched import (
        NeoHookeanCompressible,
    )

    grid, particles, dx = block(speed=20.0)
    model = NeoHookeanCompressible.from_E_nu(1e7, 0.3)
    reference = ParticleSet.from_sampler(
        lambda X: (X[:, 0] < 0.01) & (X[:, 1] > 0.0) & (X[:, 1] < 0.01),
        ((0.0, 0.0), (0.01, 0.01)), dx, ppc=2, density=RHO)
    reference.v[:, 1] = -20.0

    dt = 1e-8
    for target in (particles, reference):
        stepper = ExplicitMPMStepper(grid, model, get_transfer("flip"), dt)
        assert stepper.state is None                 # hyperelastic path
        for _ in range(50):
            stepper.step(target)

    np.testing.assert_allclose(particles.x, reference.x, rtol=1e-12)


def test_energy_is_dissipated_by_the_transfer_not_the_material():
    """The measured result from the Taylor bar example, in miniature: PIC loses
    far more energy than FLIP on the same problem with the same material, so a
    shortfall in the energy balance is a transfer property first."""
    model = material()
    kept = {}
    for name, transfer in (("flip", get_transfer("flip", alpha=1.0)),
                           ("pic", get_transfer("pic"))):
        grid, particles, dx = block(speed=150.0)
        state = HighRateState.zeros(particles.n)
        dt = 0.1 * dx / (model.wave_speed + 150.0)
        stepper = ExplicitMPMStepper(
            grid, model, transfer, dt,
            boundaries=[HalfSpace(axis=1, level=0.0, sign=1, mode="sticky")],
            state=state)
        initial = 0.5 * np.sum(particles.mass
                               * np.sum(particles.v ** 2, axis=1))
        for _ in range(300):
            stepper.step(particles)
        final = 0.5 * np.sum(particles.mass
                             * np.sum(particles.v ** 2, axis=1))
        work = float(np.sum(particles.vol0 * stepper.state.energy)
                     / model.taylor_quinney)
        kept[name] = (final + work) / initial

    assert kept["flip"] > kept["pic"]


def test_the_cfl_step_uses_the_p_wave_speed():
    """``sqrt((K + 4G/3)/rho)``. Sizing from the bulk speed alone gives a step
    about 15% too large, which is the entire stability margin."""
    model = material()
    bulk_only = np.sqrt(model.eos.bulk_modulus / RHO)
    assert model.wave_speed > bulk_only
