"""
Sulsky two-disk collision -- the standard MPM verification benchmark.

Two elastic disks are launched at each other, collide, and separate. No gravity
and no boundary conditions, so total linear momentum is conserved exactly. That
single number is the cheapest end-to-end check of the transfer operators.

Reference: Sulsky, Zhou & Schreyer (1995), Comput. Phys. Commun. 87, 236-252.

Run:  python -m examples.mpm.two_disks
      python -m examples.mpm.two_disks --interactive
      python -m examples.mpm.two_disks --no-vtk

Outputs
-------
results/two_disks/particles.pvd   ParaView / OVITO time series
results/two_disks/collision.png   PyVista contact sheet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scion.discretization.mpm import (
    APIC,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
    QuadraticBSpline,
)
from scion.discretization.mpm.output import ParticleOutput, particle_fields
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

# --- problem data ----------------------------------------------------------
N_GRID = 64
DX = 1.0 / N_GRID
E, NU, RHO0 = 1.0e3, 0.3, 1.0e3
RADIUS = 0.2
C1, C2 = np.array([0.3, 0.3]), np.array([0.7, 0.7])
V1, V2 = np.array([0.1, 0.1]), np.array([-0.1, -0.1])
T_END = 3.0

FIELD = "von_mises"          # any key produced by particle_fields
RESULTS = Path(__file__).resolve().parent / "results" / "two_disks"


def build():
    basis = QuadraticBSpline()
    grid = CartesianGrid(shape=(N_GRID, N_GRID), dx=DX, basis=basis)
    material = NeoHookeanCompressible.from_E_nu(E, NU)

    bounds = ((0.0, 0.0), (1.0, 1.0))
    disk1 = ParticleSet.from_sampler(
        lambda p: np.linalg.norm(p - C1, axis=1) < RADIUS,
        bounds, DX, ppc=4, velocity=V1, density=RHO0,
    )
    disk2 = ParticleSet.from_sampler(
        lambda p: np.linalg.norm(p - C2, axis=1) < RADIUS,
        bounds, DX, ppc=4, velocity=V2, density=RHO0,
    )
    particles = disk1.concatenate(disk2)

    stepper = ExplicitMPMStepper(grid, material, APIC(), dt=1.0e-3)
    return grid, material, particles, stepper


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(snapshots, field_name=FIELD, path=None, interactive=False,
           point_size=6.0):
    """Contact sheet of the collected frames, one subplot per snapshot.

    One shared colour range across all panels is essential: PyVista rescales
    per subplot otherwise, which makes a steady field look like it is pulsing.
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

    plotter = pv.Plotter(
        off_screen=not interactive,
        shape=(1, n),
        window_size=(340 * n, 380),
        border=False,
    )
    for k, t in enumerate(times):
        pos, scal = snapshots[t]
        # PyVista wants 3-D points; MPM here is 2-D, so pad z with zeros
        cloud = pv.PolyData(np.column_stack([pos, np.zeros(len(pos))]))
        cloud.point_data[field_name] = scal

        plotter.subplot(0, k)
        plotter.add_mesh(
            cloud, scalars=field_name, cmap="inferno", clim=(lo, hi),
            render_points_as_spheres=True, point_size=point_size,
            show_scalar_bar=(k == n - 1),
            scalar_bar_args=dict(title=field_name, vertical=True,
                                 position_x=0.82, width=0.06),
        )
        plotter.add_text(f"t = {t:.1f} s", font_size=9)
        plotter.view_xy()

    if interactive:
        plotter.show()
        return None

    path = Path(path or RESULTS / "collision.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path))
    plotter.close()
    print(f"figure -> {path}")
    return path


# ---------------------------------------------------------------------------

def main(write_vtk: bool = True, make_figure: bool = True,
         interactive: bool = False):
    grid, material, particles, stepper = build()
    print(f"particles: {particles.n}   grid: {N_GRID}x{N_GRID}   dt: {stepper.dt}")
    print(f"CFL suggestion: {stepper.cfl_dt(particles, RHO0):.2e}")

    p0 = particles.linear_momentum()
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    vtk_out = ParticleOutput(RESULTS, material=material, name="particles",
                             verbose=False) if write_vtk else None

    def log(step, t, pts):
        fields = particle_fields(pts, material)
        snapshots[round(t, 3)] = (pts.x.copy(), fields[FIELD].copy())
        if vtk_out is not None:
            vtk_out(step, t, pts)
        dp = np.linalg.norm(pts.linear_momentum() - p0)
        print(f"  t={t:5.2f}  |p - p0| = {dp:.3e}   "
              f"KE = {pts.kinetic_energy():.6e}   minJ = {pts.jacobian().min():.4f}")

    stepper.run(particles, n_steps=int(T_END / stepper.dt),
                callback=log, callback_every=int(0.5 / stepper.dt))

    if make_figure:
        render(snapshots, interactive=interactive)
    return snapshots


if __name__ == "__main__":
    main(
        write_vtk="--no-vtk" not in sys.argv,
        make_figure="--no-figure" not in sys.argv,
        interactive="--interactive" in sys.argv,
    )
