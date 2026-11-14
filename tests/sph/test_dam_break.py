"""
SPHinXsys example 9: the dam break under static confinement.

The external reference is their own recorded total mechanical energy, which
starts at exactly 1.0 for these parameters -- mass ``rho0 LL LH = 2`` with
its centroid at ``LH/2`` -- and is sampled every 200 advection steps. Both
implementations use the same advection time-step rule, so the comparison is
made at matched *iteration counts* rather than at matched times, which is
what their file records.
"""

import numpy as np
import pytest

from examples.sph.dam_break_confinement import (build, escaped,
                                                mechanical_energy)

# SPHinXsys master, tests/2d_examples/test_2d_static_confinement/
#   regression_test_tool/WaterBody_TotalMechanicalEnergy_Run_0_result.xml.
# Snapshot k is written after 200 k advection steps.
SPHINXSYS_ENERGY = [
    1.0000000000, 1.0032425273, 1.0036760970, 0.9868928907, 0.9684260893,
    0.9550402415, 0.9439867673, 0.8943590902, 0.7947102583, 0.7260802942,
    0.6865256173, 0.5828486932, 0.5560872702, 0.5226176797, 0.4991483766,
    0.4898095786, 0.4782088748, 0.4687178064, 0.4619079916, 0.4544349968,
    0.4503254308, 0.4455502929]


def test_initial_energy_matches_the_reference_exactly():
    """No dynamics in this one: it pins the geometry, the lattice and the
    particle mass against their recorded value."""
    solver = build()
    assert mechanical_energy(solver) == pytest.approx(SPHINXSYS_ENERGY[0], rel=1e-12)
    assert len(solver.positions) == 3200


def test_the_column_starts_inside_the_tank_and_outside_the_obstacle():
    solver = build()
    phi = solver.confinement[0].signed_distance(solver.positions)
    assert (phi < 0.0).all()


@pytest.fixture(scope="module")
def early_dam_break():
    """200 advection steps, which is their first recorded snapshot."""
    solver = build()
    counter = {"steps": 0}
    history = []

    def record(sol):
        counter["steps"] += 1
        if counter["steps"] % 200 == 0:
            history.append((counter["steps"], sol.time, mechanical_energy(sol),
                            escaped(sol)))

    # advance in slices so the callback sees every advection step
    while counter["steps"] < 200:
        solver.advance(solver.time + 0.1, reference_speed=2.0, callback=record)
    return solver, history


def test_matches_the_reference_energy_at_the_first_snapshot(early_dam_break):
    _, history = early_dam_break
    steps, _, energy, _ = history[0]
    assert steps == 200
    assert energy == pytest.approx(SPHINXSYS_ENERGY[1], rel=0.02)


def test_no_particle_leaves_the_tank(early_dam_break):
    """The property the whole level-set boundary exists to deliver. A wall
    that is slightly wrong shows up here long before it shows up in the
    energy."""
    solver, history = early_dam_break
    assert all(out == 0 for _, _, _, out in history)
    assert escaped(solver) == 0


def test_energy_does_not_grow(early_dam_break):
    """An inviscid dam break can only lose energy. The surface bounding is
    the part that could inject it -- it moves particles against the pressure
    -- so this is the check on that, not on the fluid."""
    _, history = early_dam_break
    assert max(e for _, _, e, _ in history) < 1.01 * SPHINXSYS_ENERGY[0]
