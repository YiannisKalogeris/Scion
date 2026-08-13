"""
Elastic block dropped onto a floor.

Exercises the parts the two-disk benchmark does not: body force and grid
boundary conditions.

Run:  python -m examples.mpm.elastic_block_drop
      python -m examples.mpm.elastic_block_drop --interactive
      python -m examples.mpm.elastic_block_drop --no-vtk

Outputs
-------
results/block_drop/particles.pvd        ParaView / OVITO time series
results/block_drop/particles_grid.pvd   background-grid mass (debug)
results/block_drop/drop.png             PyVista contact sheet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simulacrum.discretization.mpm import (
    APIC,
    Box,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
)
from simulacrum.discretization.mpm.output import ParticleOutput, particle_fields
from simulacrum.materials.hyperelastic_batched import NeoHookeanCompressible

N_GRID = 48
DX = 1.0 / N_GRID
E, NU, RHO0 = 5.0e4, 0.3, 1.0e3
GRAVITY = np.array([0.0, -9.81])
N_STEPS = 4000

FIELD = "pressure"           # any key produced by particle_fields
RESULTS = Path(__file__).resolve().parent / "results" / "block_drop"


def build():
    grid = CartesianGrid(shape=(N_GRID, N_GRID), dx=DX)
    material = NeoHookeanCompressible.from_E_nu(E, NU)

    def block(p):
        return ((p[:, 0] > 0.35) & (p[:, 0] < 0.65)
                & (p[:, 1] > 0.55) & (p[:, 1] < 0.85))

    particles = ParticleSet.from_sampler(
        block, ((0.0, 0.0), (1.0, 1.0)), DX, ppc=2,
        velocity=(0.0, 0.0), density=RHO0,
    )

    # floor and side walls; "separating" lets the block rebound but not stick
    walls = Box(lo=(2 * DX, 2 * DX), hi=(1.0 - 2 * DX, 1.0 - 2 * DX),
                mode="separating")

    stepper = ExplicitMPMStepper(
        grid, material, APIC(), dt=2.0e-4,
        body_force=GRAVITY, boundaries=[walls],
    )
    return grid, material, particles, stepper


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(snapshots, field_name=FIELD, path=None, interactive=False,
           point_size=9.0, domain=(0.0, 1.0)):
    """Contact sheet of the collected frames.

    Unlike the two-disk figure this pins the camera to the full domain, so the
    block visibly falls instead of every panel auto-zooming onto it.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed; skipping the figure "
              "(the .pvd series is still written)")
        return None

    times = sorted(snapshots)
    n = len(times)
    lo = min(float(np.min(snapshots[t][1])) for t in times)
    hi = max(float(np.max(snapshots[t][1])) for t in times)
    a, b = domain

    plotter = pv.Plotter(
        off_screen=not interactive,
        shape=(1, n),
        window_size=(300 * n, 340),
        border=False,
    )
    for k, t in enumerate(times):
        pos, scal = snapshots[t]
        cloud = pv.PolyData(np.column_stack([pos, np.zeros(len(pos))]))
        cloud.point_data[field_name] = scal

        plotter.subplot(0, k)
        # outline of the container, so the fall is readable
        plotter.add_mesh(
            pv.Rectangle([[a, a, 0.0], [b, a, 0.0], [b, b, 0.0]]),
            style="wireframe", color="grey", line_width=1,
        )
        plotter.add_mesh(
            cloud, scalars=field_name, cmap="coolwarm", clim=(lo, hi),
            render_points_as_spheres=True, point_size=point_size,
            show_scalar_bar=(k == n - 1),
            scalar_bar_args=dict(title=field_name, vertical=True,
                                 position_x=0.80, width=0.06),
        )
        plotter.add_text(f"t = {t:.2f} s", font_size=9)
        plotter.view_xy()
        plotter.camera.parallel_projection = True
        plotter.camera.focal_point = ((a + b) / 2, (a + b) / 2, 0.0)
        plotter.camera.parallel_scale = 0.55 * (b - a)

    if interactive:
        plotter.show()
        return None

    path = Path(path or RESULTS / "drop.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path))
    plotter.close()
    print(f"figure -> {path}")
    return path


# ---------------------------------------------------------------------------

def main(write_vtk: bool = True, make_figure: bool = True,
         interactive: bool = False):
    grid, material, particles, stepper = build()
    weight = particles.mass.sum() * abs(GRAVITY[1])
    print(f"particles: {particles.n}   dt: {stepper.dt}   "
          f"CFL: {stepper.cfl_dt(particles, RHO0):.2e}")
    print(f"total weight: {weight:.4f} N")

    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    # grid=grid also dumps the nodal mass as .vti; body_force lets the momentum
    # diagnostic subtract the gravity impulse instead of reporting nonsense
    vtk_out = ParticleOutput(RESULTS, material=material, grid=grid,
                             body_force=GRAVITY) if write_vtk else None

    def log(step, t, pts):
        fields = particle_fields(pts, material)
        snapshots[round(t, 3)] = (pts.x.copy(), fields[FIELD].copy())
        if vtk_out is not None:
            vtk_out(step, t, pts)
        else:
            y = pts.x[:, 1]
            print(f"  t={t:5.3f}  y in [{y.min():.4f}, {y.max():.4f}]   "
                  f"KE = {pts.kinetic_energy():.4e}   "
                  f"minJ = {pts.jacobian().min():.4f}")

    stepper.run(particles, n_steps=N_STEPS, callback=log, callback_every=500)

    if make_figure:
        render(snapshots, interactive=interactive)
    return snapshots


if __name__ == "__main__":
    main(
        write_vtk="--no-vtk" not in sys.argv,
        make_figure="--no-figure" not in sys.argv,
        interactive="--interactive" in sys.argv,
    )
