"""
The diffusion solver, against two things outside this library.

1. **SPHinXsys's own published observations at ``t = 0``.** Those eleven
   numbers depend on the kernel, its normalisation, the lattice, the particle
   volumes and the Shepard interpolation, and on nothing else -- no time
   integration, no operator. Reproducing them pins the entire spatial
   discretisation against an independent implementation.
2. **The closed-form solution of the continuum problem.** The initial
   condition depends on ``x`` alone and the only boundaries cutting the field
   are periodic, so the exact solution is a difference of error functions
   plus a spreading Gaussian, and the mid-line is fifteen smoothing lengths
   from either free surface. The *rate* at which the error falls is the
   property that cannot be reached by tuning anything.
"""

import numpy as np
import pytest

from examples.sph.heat_transfer_2d import (build, convergence_time_step,
                                           exact_profile, mid_line, observers)
from scion.discretization.sph import (DiffusionSolver, DirectionalDiffusivity,
                                      IsotropicDiffusivity, lattice)

# tests/2d_examples/test_2d_diffusion/regression_test_tool/
#   TemperatureObserver_Phi_ensemble_averaged_mean_variance.xml, Snapshot_0,
# from SPHinXsys master (Apache-2.0). Transcribed here rather than read from
# the source tree, which is a staging directory and not a dependency.
SPHINXSYS_INITIAL_OBSERVATIONS = np.array([
    0.0, 0.0, 0.94302593126241574, 0.0, 0.0, 4.9871608334479541e-252,
    1.573557537176933e-101, 6.2734694041436527e-19, 0.035497009580211382,
    1.5828162115099197e-47, 9.6766950053827561e-160])


def test_matches_sphinxsys_initial_observations():
    """The external benchmark. Phi_2 agrees to all seventeen published
    digits; the rest are sums of exponentials spanning 250 decades and agree
    to 2.4e-14 relative, which is the printed precision of the reference."""
    solver, phi, _ = build()
    got = solver.observe(observers(), phi)
    reference = SPHINXSYS_INITIAL_OBSERVATIONS
    nonzero = reference > 0.0
    assert (got[~nonzero] == 0.0).all()
    relative = np.abs(got[nonzero] - reference[nonzero]) / reference[nonzero]
    assert relative.max() < 1e-13
    assert got[2] == reference[2]          # bit for bit


def test_second_order_against_the_closed_form():
    """At a *fixed* time step, so this is the spatial order alone. Letting
    ``dt`` shrink with ``h`` would report the smaller of the two orders and
    hide which one it was."""
    end_time, levels, errors = 1.0, (1, 2), []
    dt = convergence_time_step(levels, end_time)
    n_steps = int(np.ceil(end_time / dt))
    for level in levels:
        solver, phi, _ = build(level)
        for _ in range(n_steps):
            phi = solver.step(phi, dt)
        row = mid_line(solver)
        exact = exact_profile(solver.positions[row, 0], n_steps * dt)
        errors.append(np.abs(phi[row] - exact).max())
    rate = np.log2(errors[0] / errors[1])
    assert rate > 1.8, f"observed order {rate:.2f} from errors {errors}"


def test_species_is_conserved_over_a_whole_run():
    """Not just per step: the integral is a property of the pair loop, and
    the two-stage scheme cannot break it because each stage conserves."""
    solver, phi, _ = build()
    before = float((phi * solver.volumes).sum())
    dt = solver.time_step()
    for _ in range(3):
        phi = solver.step(phi, dt)
    after = float((phi * solver.volumes).sum())
    assert abs(after - before) < 1e-14 * before


def test_the_reference_time_step_is_inside_the_measured_stability_limit():
    """SPHinXsys's ``dt = 0.5 h^2/(D dim)`` is a smoothing-length heuristic,
    not a bound on this operator. Measured here against the spectral radius
    of the assembled Laplacian, it sits 12% inside -- real margin, but not
    much, and a kernel or ``h/dp`` change could cross it without warning."""
    solver, _, _ = build()
    rule = solver.time_step()
    limit = solver.stable_time_step()
    assert rule < limit
    assert rule / limit > 0.8


def test_a_uniform_field_is_stationary():
    solver, _, _ = build()
    phi = np.full(len(solver.positions), 0.7)
    for _ in range(5):
        phi = solver.step(phi, solver.time_step())
    assert np.abs(phi - 0.7).max() < 1e-15


def test_isotropic_projection_is_the_coefficient():
    """``1/(e . D^-1 e)`` must collapse to ``d`` in every direction, which is
    what catches an un-inverted or transposed tensor -- both of which give
    the right answer along the axes and the wrong one in between."""
    diffusivity = IsotropicDiffusivity(1e-4)
    angles = np.linspace(0.0, 2 * np.pi, 37)
    e = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert np.abs(diffusivity.inter_particle(e) - 1e-4).max() < 1e-19


def test_directional_projection_reads_the_principal_values():
    alpha = np.pi / 6.0
    along = np.array([[np.cos(alpha), np.sin(alpha)]])
    across = np.array([[-np.sin(alpha), np.cos(alpha)]])
    diffusivity = DirectionalDiffusivity(1e-4, 3e-4, (np.cos(alpha), np.sin(alpha)))
    assert diffusivity.inter_particle(along)[0] == pytest.approx(4e-4, rel=1e-12)
    assert diffusivity.inter_particle(across)[0] == pytest.approx(1e-4, rel=1e-12)
    assert diffusivity.reference() == pytest.approx(4e-4)


def test_a_bias_along_the_gradient_speeds_the_decay_and_across_it_does_not():
    """The anisotropic path on the real problem: with the field varying in
    ``x`` only, a bias along ``x`` must diffuse it faster than the isotropic
    case and a bias along ``y`` must leave it alone. A sign slip inside the
    projection swaps these two and nothing else notices."""
    end_time = 0.5
    peaks = {}
    for name, bias_direction in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
        x, volumes = lattice([0.0, 0.0], [2.0, 0.4], 0.02, periodic=(True, False))
        diffusivity = DirectionalDiffusivity(1e-4, 3e-4, bias_direction)
        solver = DiffusionSolver(x, volumes, diffusivity, h=1.3 * 0.02,
                                 lower=[0.0, 0.0], upper=[2.0, 0.4],
                                 periodic=(True, False))
        phi = np.sin(np.pi * x[:, 0])
        dt = 0.25 * solver.stable_time_step()
        for _ in range(int(np.ceil(end_time / dt))):
            phi = solver.step(phi, dt)
        peaks[name] = np.abs(phi).max()
    # exp(-D k^2 t) with k = pi: 4e-4 along x against 1e-4 across it
    assert peaks["x"] < peaks["y"]
    assert peaks["y"] == pytest.approx(np.exp(-1e-4 * np.pi**2 * end_time), rel=0.02)
    assert peaks["x"] == pytest.approx(np.exp(-4e-4 * np.pi**2 * end_time), rel=0.02)


def test_refuses_a_lattice_that_does_not_close_on_a_periodic_axis():
    """A box side that is not a whole number of spacings leaves a short seam
    at the wrap, which shows up as a stationary defect in the solution rather
    than as an error."""
    with pytest.raises(ValueError, match="seam"):
        lattice([0.0, 0.0], [2.005, 0.4], 0.01, periodic=(True, False))


def test_run_is_the_step_loop():
    """``run`` exists so a caller does not write the loop, and it does not
    clip the last step to land on ``end_time`` -- the reference's clock
    overshoots and the comparison has to run the same one."""
    solver, phi, _ = build()
    dt = solver.time_step()
    manual = phi.copy()
    for _ in range(3):
        manual = solver.step(manual, dt)
    seen = []
    automatic = solver.run(phi, dt, 1.0, callback=lambda t, f: seen.append(t))
    assert len(seen) == 3
    assert seen[-1] == pytest.approx(3 * dt)
    assert np.array_equal(automatic, manual)
