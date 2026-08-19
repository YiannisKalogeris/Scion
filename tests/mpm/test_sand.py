"""
Drucker-Prager sand driven through the MPM stepper.

The unit-level checks on the return map live in
``tests/test_materials/test_hencky.py``; these are the integration ones -- that
the model survives the stateful stepper path, that a pile actually arrests, and
that the arrest angle responds to the friction angle.

Kept small and coarse deliberately: the physics validation is the example
(``examples/mpm/sand_column_collapse.py``), and these only have to catch a
regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC,
    Box,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
)
from scion.discretization.mpm.output import ParticleOutput
from scion.materials.hencky import (
    DruckerPragerSand,
    HenckyElastic,
    PrincipalPlasticity,
    PrincipalPlasticState,
)

NY = 32
DX = 1.0 / NY
RHO0 = 1500.0
GRAVITY = np.array([0.0, -9.81])


def column(friction_angle=30.0, aspect=2.0, r0=4 * DX):
    grid = CartesianGrid(shape=(3 * NY, NY), dx=DX)
    material = PrincipalPlasticity(
        HenckyElastic.from_E_nu(1.0e6, 0.3),
        DruckerPragerSand(friction_angle=friction_angle),
    )
    lo = 2 * DX

    def region(p):
        return ((p[:, 0] > lo) & (p[:, 0] < lo + r0)
                & (p[:, 1] > lo) & (p[:, 1] < lo + aspect * r0))

    particles = ParticleSet.from_sampler(
        region, ((0.0, 0.0), (3.0, 1.0)), DX, ppc=2,
        velocity=(0.0, 0.0), density=RHO0,
    )
    walls = Box(lo=(lo, lo), hi=(3.0 - 2 * DX, 1.0 - 2 * DX), mode="sticky")
    c = material.wave_speed_factor / np.sqrt(RHO0)
    stepper = ExplicitMPMStepper(
        grid, material, APIC(), dt=0.3 * DX / c,
        body_force=GRAVITY, boundaries=[walls],
        state=PrincipalPlasticState.identity(particles.n),
    )
    return grid, material, particles, stepper, lo


def surface_angle(particles, lo):
    x = particles.x[:, 0] - lo
    y = particles.x[:, 1] - lo
    edges = np.arange(0.0, x.max() + DX, DX)
    which = np.digitize(x, edges)
    top = np.array([[edges[i - 1] + 0.5 * DX, y[which == i].max()]
                    for i in np.unique(which) if (which == i).sum() >= 2])
    h = top[:, 1]
    band = (h > 0.15 * h.max()) & (h < 0.85 * h.max())
    if band.sum() < 3:
        return float("nan")
    return float(np.degrees(np.arctan(abs(
        np.polyfit(top[band, 0], top[band, 1], 1)[0]))))


class TestSandInTheStepper:
    def test_the_column_runs_and_stays_finite(self):
        _, _, particles, stepper, _ = column()
        with quiet():
            stepper.run(particles, n_steps=400)
        assert np.all(np.isfinite(particles.x))
        assert np.all(np.isfinite(particles.v))
        assert np.all(np.isfinite(stepper.state.F_e))

    def test_the_pile_arrests(self):
        """The point of a frictional material: it stops."""
        _, _, particles, stepper, _ = column()
        with quiet():
            stepper.run(particles, n_steps=600)
        peak = np.abs(particles.v).max()
        with quiet():
            stepper.run(particles, n_steps=2400)
        assert np.abs(particles.v).max() < 0.1 * peak

    def test_it_spreads_sideways_and_settles_lower(self):
        _, _, particles, stepper, lo = column()
        w0 = particles.x[:, 0].max() - particles.x[:, 0].min()
        h0 = particles.x[:, 1].max()
        with quiet():
            stepper.run(particles, n_steps=3000)
        assert particles.x[:, 0].max() - particles.x[:, 0].min() > 1.5 * w0
        assert particles.x[:, 1].max() < h0

    def test_a_steeper_friction_angle_holds_a_steeper_pile(self):
        angles = []
        for phi in (20.0, 40.0):
            _, _, particles, stepper, lo = column(friction_angle=phi)
            with quiet():
                stepper.run(particles, n_steps=3000)
            angles.append(surface_angle(particles, lo))
        assert angles[0] < angles[1], angles

    def test_no_particle_escapes_the_domain(self):
        _, _, particles, stepper, lo = column()
        with quiet():
            stepper.run(particles, n_steps=3000)
        assert particles.x[:, 0].min() >= lo - DX
        assert particles.x[:, 1].min() >= lo - DX

    def test_mass_is_conserved(self):
        _, _, particles, stepper, _ = column()
        m0 = particles.mass.sum()
        with quiet():
            stepper.run(particles, n_steps=1000)
        assert np.isclose(particles.mass.sum(), m0)

    def test_the_state_grows_a_plastic_history(self):
        """If nothing ever yielded, the model would be doing no work."""
        _, _, particles, stepper, _ = column()
        with quiet():
            stepper.run(particles, n_steps=1500)
        J_e = np.linalg.det(stepper.state.F_e)
        assert not np.allclose(J_e, 1.0)


class TestParticleOutputDimensionality:
    def test_two_dimensional_output_without_a_body_force(self, tmp_path):
        """``body_force`` defaulted to ``zeros(3)`` and broke every 2-D run
        that did not pass one, inside the verbose momentum diagnostic."""
        _, _, particles, stepper, _ = column()
        out = ParticleOutput(tmp_path, verbose=True)
        out(0, 0.0, particles)          # raised ValueError on the broadcast
        assert (tmp_path / "particles_000000.vtu").exists()

    def test_body_force_is_still_honoured_when_given(self, tmp_path):
        _, _, particles, stepper, _ = column()
        out = ParticleOutput(tmp_path, body_force=GRAVITY, verbose=True)
        out(0, 0.0, particles)
        assert np.allclose(out.body_force, GRAVITY)
