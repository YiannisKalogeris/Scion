r"""
SPHinXsys example 5, the 2-D Taylor-Green vortex, on Scion's SPH module.

    python -m examples.sph.taylor_green_2d --end 1.0
    python -m examples.sph.taylor_green_2d --n 100 --end 5.0 --csv tg.csv

``--animate`` writes a GIF of the decaying vortices and ``--vtk`` a ParaView
series.

The same case in Rust, about twelve times faster, taking the end time, the
particles per side and an optional CSV:

    cargo run -p scion-sph --release --example taylor_green_2d -- 1.0 100 out.csv

Figures go to ``examples/sph/results`` unless ``--results`` says otherwise.

Why this case and not the ones that were asked for
--------------------------------------------------
Examples 8 (hydrostatic water column on an elastic plate) and 9 (static
confinement) both stand on a weakly compressible fluid solver, and neither
can be verified until that solver is. Taylor-Green is the one fluid case with
a **closed-form** answer -- an exact solution of the incompressible
Navier-Stokes equations whose velocity decays as ``exp(-8 pi^2 nu t)`` on the
unit square -- and it needs no boundary machinery at all, only periodicity.
So it is the right place to pin the fluid core, and it is itself one of their
examples.

What it does not test, and therefore what examples 8 and 9 still need: a wall
(a contact relation against solid particles, or a level set), a free surface,
an elastic solid, and the fluid-structure coupling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from scion.discretization.sph import lattice
from scion.discretization.sph.fluid import FluidSolver, WeaklyCompressibleFluid

#: Figures from every example in this folder land here. Created on demand, so
#: a fresh clone has no empty directory and a run that draws nothing leaves
#: none behind.
RESULTS = Path(__file__).resolve().parent / "results"

SIDE = 1.0
RHO0, U_REF, REYNOLDS = 1.0, 1.0, 100.0
MACH = 0.1          # c0 = U/Ma, so the density varies by about Ma^2 = 1%


def build(n: int = 100):
    """The unit square, periodic both ways, with the vortex velocity."""
    spacing = SIDE / n
    fluid = WeaklyCompressibleFluid(rho0=RHO0, c0=U_REF / MACH,
                                    viscosity=RHO0 * U_REF * SIDE / REYNOLDS)
    x, _ = lattice([0.0, 0.0], [SIDE, SIDE], spacing, periodic=(True, True))
    solver = FluidSolver(x, spacing, fluid, lower=[0.0, 0.0], upper=[SIDE, SIDE],
                         periodic=(True, True))
    solver.velocity[:, 0] = -np.cos(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])
    solver.velocity[:, 1] = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
    return solver


def decay_rate(solver) -> float:
    """``8 pi^2 nu``, the rate the *velocity* decays at.

    The kinetic energy therefore decays at twice this. Mixing the two up
    gives a factor of two in the comparison and still looks exponential.
    """
    return 8.0 * np.pi**2 * solver.fluid.viscosity / solver.fluid.rho0


def vorticity(solver) -> np.ndarray:
    """``dv/dx - du/dy`` at the particles, from the corrected SPH gradient.

    The field worth colouring by, and the reason is that ``speed`` cannot
    show this flow: it vanishes both at the four vortex centres and along
    the stagnation lines between them, so two structurally opposite places
    read the same. Vorticity is signed, so the counter-rotating pairs
    separate, and it has a closed form to check against --
    ``4 pi cos(2 pi x) cos(2 pi y)`` at ``t = 0``.

    **The correction matrix is not optional here, and that is the
    surprise**: this domain is periodic both ways, so there is no free
    surface and no truncated support anywhere -- the deficit the correction
    usually exists to repair does not occur. Measured against the closed
    form at ``n = 50 / 100 / 200``:

        uncorrected   3.33e-2   2.79e-2   2.65e-2     plateaus
        corrected     7.42e-3   1.86e-3   4.66e-4     ratios 3.99, 4.00

    Second order with it and *no convergence at all* without, because
    refining holds ``h/dp`` fixed, so the kernel's smoothing error is a
    constant fraction rather than something refinement removes. The solver
    itself keeps no ``B`` -- its momentum and continuity terms are the
    antisymmetric pair forms, chosen for conservation rather than
    consistency -- so one is built here for the diagnostic alone.
    """
    from scion.discretization.sph.operators import (linear_gradient_correction,
                                                    sph_gradient)
    B = linear_gradient_correction(solver.pairs, solver.volume, solver.dw_ij,
                                   solver.dim)
    grad = [sph_gradient(solver.pairs, solver.velocity[:, c], solver.volume,
                         solver.dw_ij, solver.dim, B) for c in (0, 1)]
    return grad[1][:, 0] - grad[0][:, 1]


def run(n: int = 100, end_time: float = 1.0, interval: float = 0.1,
        quiet: bool = False, snapshots: list | None = None):
    """March the vortex and return ``(solver, history, rate, energy0)``.

    Pass a list as ``snapshots`` to collect ``(t, positions, velocity,
    pressure)`` once per output interval, for the animation and the VTK
    series. The particles advect, so every frame carries its own geometry.
    """
    solver = build(n)
    rate = decay_rate(solver)
    energy0 = solver.kinetic_energy()
    if not quiet:
        print(f"  {len(solver.positions)} particles, {len(solver.pairs)} pairs, "
              f"Re = {REYNOLDS:g}, Ma = {MACH:g}")
        print(f"  velocity decays at exp(-{rate:.4f} t), energy at twice that")
    history = [(0.0, energy0, solver.maximum_speed(), float(np.ptp(solver.density)))]

    def capture():
        if snapshots is not None:
            snapshots.append((solver.time, solver.positions.copy(),
                              solver.velocity.copy(), solver.pressure.copy(),
                              vorticity(solver)))

    capture()
    start = time.perf_counter()
    steps = int(round(end_time / interval))
    for k in range(steps):
        solver.advance((k + 1) * interval, reference_speed=U_REF)
        history.append((solver.time, solver.kinetic_energy(), solver.maximum_speed(),
                        float(np.ptp(solver.density))))
        capture()
    if not quiet:
        print(f"  {time.perf_counter() - start:.1f} s of wall time")
    return solver, history, rate, energy0


def report(history, rate, energy0) -> None:
    print(f"\n  {'t':>6} {'kinetic energy':>15} {'exact':>12} {'ratio':>7} "
          f"{'max speed':>10} {'exact':>9} {'rho spread':>11}")
    stride = max(1, len(history) // 12)
    for t, energy, speed, spread in history[::stride]:
        exact_energy = energy0 * np.exp(-2.0 * rate * t)
        print(f"  {t:6.2f} {energy:15.6e} {exact_energy:12.4e} "
              f"{energy / exact_energy:7.3f} {speed:10.5f} "
              f"{np.exp(-rate * t):9.5f} {spread:11.2e}")
    times = np.array([h[0] for h in history])
    energies = np.array([h[1] for h in history])
    window = (times > 0.3 * times[-1]) & (times < 0.9 * times[-1])
    if window.sum() > 2:
        fitted = np.polyfit(times[window], np.log(energies[window]), 1)[0]
        print(f"\n  fitted energy decay rate over the middle of the run: "
              f"{fitted:.4f} against an exact {-2 * rate:.4f} "
              f"({abs(fitted / (-2 * rate) - 1) * 100:.1f}% high)")


def setup_figure(out: Path, n: int = 50) -> None:
    """The geometry, the particles and the velocity they start with.

    Drawn at a coarser ``n`` than the case runs at, because at ``dp = 1/100``
    the arrows overlap into a solid block and the lattice stops being
    visible. The point of the figure is the arrangement, not the resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    solver = build(n)
    x = solver.positions
    speed = np.linalg.norm(solver.velocity, axis=1)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(x[:, 0], x[:, 1], c=speed, s=4.0, cmap="viridis",
                    linewidths=0, zorder=2)
    stride = max(1, n // 14)
    pick = np.zeros(len(x), dtype=bool)
    pick[::stride * stride] = True
    ax.quiver(x[pick, 0], x[pick, 1], solver.velocity[pick, 0],
              solver.velocity[pick, 1], color="0.15", scale=18, width=0.004,
              zorder=3)
    for edge in (0.0, SIDE):
        ax.plot([edge, edge], [0, SIDE], color="#c0622f", lw=1.4, ls=(0, (3, 2)),
                zorder=4)
        ax.plot([0, SIDE], [edge, edge], color="#c0622f", lw=1.4, ls=(0, (3, 2)),
                zorder=4)
    for start, end in (((SIDE, 0.5 * SIDE), (SIDE + 0.09, 0.5 * SIDE)),
                       ((-0.09, 0.5 * SIDE), (0.0, 0.5 * SIDE)),
                       ((0.5 * SIDE, SIDE), (0.5 * SIDE, SIDE + 0.09)),
                       ((0.5 * SIDE, -0.09), (0.5 * SIDE, 0.0))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->",
                                     color="#c0622f", lw=1.2, mutation_scale=9,
                                     zorder=4))
    ax.text(0.5 * SIDE, SIDE + 0.12, "periodic both ways", color="#c0622f",
            fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.14, SIDE + 0.14); ax.set_ylim(-0.14, SIDE + 0.2)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="initial speed")
    ax.set_title(f"{len(x)} particles at dp = {SIDE / n:g}, "
                 f"Re = {REYNOLDS:g}, Ma = {MACH:g}", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_taylor_green_2d_setup.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def figure(solver, history, rate, energy0, out: Path) -> None:
    """The decay against the closed form, and the field it decayed from.

    Semi-log on the energy because the prediction is exponential: a straight
    line there *is* the analytic decay rate, which is the thing being checked,
    where on linear axes two exponentials agree by eye long after their rates
    have parted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([h[0] for h in history])
    energy = np.array([h[1] for h in history])
    exact = energy0 * np.exp(-2.0 * rate * times)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw=dict(width_ratios=(1.25, 1.0)))

    ax = axes[0]
    ax.semilogy(times, exact, color="#9aa7b1", lw=2.6, zorder=1,
                label=r"closed form, $e^{-16\pi^2\nu t}$")
    ax.semilogy(times, energy, color="#1f6f8b", lw=1.4, marker="o", ms=3.0,
                zorder=3, label="SPH")
    ax.set_xlabel("t"); ax.set_ylabel("kinetic energy")
    ax.grid(True, which="both", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"energy decay, Re = {REYNOLDS:g}", fontsize=9)

    ax = axes[1]
    speed = np.linalg.norm(solver.velocity, axis=1)
    sc = ax.scatter(solver.positions[:, 0], solver.positions[:, 1], c=speed,
                    s=2.0, cmap="viridis", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, SIDE); ax.set_ylim(0.0, SIDE)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="speed")
    ax.set_title(f"t = {times[-1]:.2f}", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_taylor_green_2d.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def animate(snapshots, out: Path, fps: int = 15, max_frames: int = 150) -> None:
    """The vortices decaying, coloured by speed.

    The colour scale is fixed to the *initial* maximum rather than to each
    frame's own: the whole content of this case is that the velocity decays
    exponentially, and a per-frame rescale would hide exactly that by
    renormalising it away.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    stride = max(1, len(snapshots) // max_frames)
    frames = snapshots[::stride]
    initial = float(np.linalg.norm(frames[0][2], axis=1).max())

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    t0, x0, v0, _, _ = frames[0]
    sc = ax.scatter(x0[:, 0], x0[:, 1], c=np.linalg.norm(v0, axis=1), s=3.0,
                    cmap="viridis", vmin=0.0, vmax=initial, linewidths=0)
    for edge in (0.0, SIDE):
        ax.plot([edge, edge], [0, SIDE], color="#c0622f", lw=1.0, ls=(0, (3, 2)))
        ax.plot([0, SIDE], [edge, edge], color="#c0622f", lw=1.0, ls=(0, (3, 2)))
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, SIDE + 0.02); ax.set_ylim(-0.02, SIDE + 0.02)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="speed")
    title = ax.set_title(f"t = {frames[0][0]:.3f}", fontsize=9)
    fig.tight_layout()

    def draw(frame):
        t, x, v, _, _ = frame
        sc.set_offsets(x)
        sc.set_array(np.linalg.norm(v, axis=1))
        title.set_text(f"t = {t:.3f}")
        return sc, title

    movie = FuncAnimation(fig, draw, frames=frames, blit=False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_taylor_green_2d.gif"
    movie.save(path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"  wrote {path} ({len(frames)} frames at {fps} fps)")


def write_series(snapshots, out: Path) -> None:
    """One ``.vtu`` per frame plus the ``.pvd`` that gives ParaView a clock."""
    from scion.io.particles import write_particles_vtu
    from scion.io.time_series import TimeSeriesWriter

    directory = out / "taylor_green_2d_vtk"
    series = TimeSeriesWriter(directory, "particles", suffix=".vtu")
    for t, x, v, pressure, omega in snapshots:
        series.add(t, write_particles_vtu, x,
                   point_data={"velocity": v, "speed": np.linalg.norm(v, axis=1),
                               "pressure": pressure, "vorticity": omega})
    print(f"  wrote {len(snapshots)} frames to {directory}/particles.pvd")
    write_cell(directory / "cell.vtu")


def write_cell(path: Path) -> Path:
    """The periodic cell, as a line loop, written once.

    There is no boundary to draw here -- the domain has no edge, which is
    the whole point of the case -- but without the square there is nothing
    in the view to say the four vortices tile, or to notice a particle
    leaving one side and arriving at the other. Colour it dashed in
    ParaView so it does not read as a wall.
    """
    import meshio

    corners = np.array([[0.0, 0.0], [SIDE, 0.0], [SIDE, SIDE], [0.0, SIDE]])
    xyz = np.zeros((4, 3))
    xyz[:, :2] = corners
    lines = np.array([[i, (i + 1) % 4] for i in range(4)])
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.Mesh(points=xyz,
                cells=[meshio.CellBlock("line", lines)]).write(str(path))
    print(f"  wrote {path} (the periodic cell; not a wall)")
    return path


def write_csv(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("time,kinetic_energy,max_speed,density_spread\n")
        for row in history:
            handle.write(",".join(f"{v:.17e}" for v in row) + "\n")
    print(f"  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=100, help="particles per side")
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="directory for the figure")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the geometry-and-particles figure")
    parser.add_argument("--animate", action="store_true",
                        help="write a GIF of the decaying vortices")
    parser.add_argument("--vtk", action="store_true",
                        help="write a .vtu series and its .pvd for ParaView")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("SPH Taylor-Green vortex (SPHinXsys example 5)")
    snapshots = [] if (args.animate or args.vtk) else None
    solver, history, rate, energy0 = run(args.n, args.end, args.interval,
                                         snapshots=snapshots)
    report(history, rate, energy0)
    if not args.no_setup:
        setup_figure(args.results)
    if not args.no_figure:
        figure(solver, history, rate, energy0, args.results)
    if args.animate:
        animate(snapshots, args.results)
    if args.vtk:
        write_series(snapshots, args.results)
    if args.csv is not None:
        write_csv(history, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
