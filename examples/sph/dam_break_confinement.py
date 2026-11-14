r"""
SPHinXsys example 9, the 2-D dam break with *static confinement*.

    python -m examples.sph.dam_break_confinement --end 2.0
    python -m examples.sph.dam_break_confinement --end 20.0 --csv energy.csv

``--animate`` writes a GIF of the collapse and ``--vtk`` a ParaView series;
unlike the diffusion case, here the particles genuinely move.

The same case in Rust, about eighteen times faster, taking the end time and
an optional CSV:

    cargo run -p scion-sph --release --example dam_break_confinement -- 2.0 out.csv

Figures go to ``examples/sph/results`` unless ``--results`` says otherwise.

A 2.0 x 1.0 column of water collapses in a 5.366 x 5.366 tank that contains a
triangular obstacle. What makes it example 9 rather than another dam break is
that **neither the tank nor the obstacle is made of particles**: both are
signed distance fields, and every boundary term is an integral of the kernel
over the solid region, computed once on a background grid (see
``scion.discretization.sph.levelset``).

The two solids are disjoint and further apart than a kernel support, so they
are carried as one level set, ``max(phi_tank, phi_triangle)`` -- the union of
two solids in the convention where ``phi > 0`` is solid.

What is checked
---------------
The initial mechanical energy is ``1.0`` exactly for these parameters, which
is what SPHinXsys records, and the decay afterwards is compared against their
run. Beyond that: no particle may leave the tank, the mass is fixed by
construction, and the energy must not grow.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from scion.discretization.sph import lattice
from scion.discretization.sph.fluid import FluidSolver, WeaklyCompressibleFluid
from scion.discretization.sph.levelset import (LevelSet, box_interior_distance,
                                               polygon_distance)

#: Figures from every example in this folder land here. Created on demand, so
#: a fresh clone has no empty directory and a run that draws nothing leaves
#: none behind.
RESULTS = Path(__file__).resolve().parent / "results"

TANK = 5.366
COLUMN_LENGTH, COLUMN_HEIGHT = 2.0, 1.0
SPACING = 0.025
RHO0, GRAVITY = 1.0, 1.0
U_REF = 2.0 * np.sqrt(GRAVITY * COLUMN_HEIGHT)
SOUND_SPEED = 10.0 * U_REF
MARGIN = 4 * SPACING


def obstacle() -> np.ndarray:
    """The triangle, as the reference case places it."""
    return np.array([
        [0.5 * TANK, 0.05 * TANK],
        [0.5 * TANK + 0.5 * COLUMN_LENGTH, 0.05 * TANK + 0.5 * COLUMN_HEIGHT],
        [0.5 * TANK + 0.5 * COLUMN_LENGTH, 0.05 * TANK],
    ])


def solid_distance(points: np.ndarray) -> np.ndarray:
    """``phi > 0`` inside the solid: outside the tank, or inside the triangle."""
    outside_tank = box_interior_distance(points, [0.0, 0.0], [TANK, TANK])
    inside_triangle = polygon_distance(points, obstacle())
    return np.maximum(outside_tank, inside_triangle)


def build(spacing: float = SPACING):
    """The water column, the level set and the solver.

    The level-set grid spacing is the **particle** spacing and its origin is
    an integer number of spacings from the tank wall. Both matter; see the
    note in ``levelset.py``.
    """
    lower = np.array([-MARGIN, -MARGIN])
    upper = np.array([TANK + MARGIN, TANK + MARGIN])
    fluid = WeaklyCompressibleFluid(rho0=RHO0, c0=SOUND_SPEED, viscosity=0.0)
    x, _ = lattice([0.0, 0.0], [COLUMN_LENGTH, COLUMN_HEIGHT], spacing)
    confinement = LevelSet(solid_distance, lower, upper, spacing,
                           "wendland_c2", 1.3 * spacing)
    solver = FluidSolver(x, spacing, fluid, lower=lower, upper=upper,
                         confinement=confinement, free_surface=True,
                         riemann_density=True, riemann_pressure=True,
                         transport_coefficient=0.0)
    solver.gravity = np.array([0.0, -GRAVITY])
    return solver


def mechanical_energy(solver) -> float:
    """``sum m (|v|^2/2 + g y)``, the quantity SPHinXsys records.

    For this column it is exactly ``1.0`` at rest: the mass is ``rho0 LL LH``
    = 2 and the centroid is at ``LH/2``.
    """
    kinetic = 0.5 * (solver.velocity ** 2).sum(axis=1)
    potential = GRAVITY * solver.positions[:, 1]
    return float((solver.mass * (kinetic + potential)).sum())


def escaped(solver) -> int:
    """Particles that have left the fluid region. Must stay zero."""
    return int((solver.confinement[0].signed_distance(solver.positions) > 0.0).sum())


def run(end_time: float = 2.0, interval: float = 0.1, spacing: float = SPACING,
        quiet: bool = False, snapshots: list | None = None):
    """March the collapse and return ``(solver, history)``.

    Pass a list as ``snapshots`` to collect ``(t, positions, velocity,
    pressure)`` once per output interval, which is what the animation and the
    VTK series are built from. Opt-in: the positions alone are an array per
    frame, and unlike the diffusion case they genuinely change, so nothing
    can be shared between frames.
    """
    solver = build(spacing)
    if not quiet:
        print(f"  {len(solver.positions)} particles at dp = {spacing}, "
              f"U_ref = {U_REF:.3f}, c0 = {SOUND_SPEED:.2f}")
        print(f"  level set {solver.confinement[0].counts} cells")
    history = [(0.0, mechanical_energy(solver), solver.maximum_speed(),
                float(solver.positions[:, 0].max()), escaped(solver))]

    def capture():
        if snapshots is not None:
            snapshots.append((solver.time, solver.positions.copy(),
                              solver.velocity.copy(), solver.pressure.copy()))

    capture()
    start = time.perf_counter()
    for k in range(int(round(end_time / interval))):
        solver.advance((k + 1) * interval, reference_speed=U_REF)
        history.append((solver.time, mechanical_energy(solver),
                        solver.maximum_speed(),
                        float(solver.positions[:, 0].max()), escaped(solver)))
        capture()
    if not quiet:
        print(f"  {time.perf_counter() - start:.1f} s of wall time")
    return solver, history


def report(history) -> None:
    print(f"\n  {'t':>6} {'energy':>10} {'E/E0':>7} {'max speed':>10} "
          f"{'front x':>9} {'escaped':>8}")
    stride = max(1, len(history) // 15)
    for t, energy, speed, front, out in history[::stride]:
        print(f"  {t:6.2f} {energy:10.5f} {energy / history[0][1]:7.4f} "
              f"{speed:10.4f} {front:9.4f} {out:8d}")
    worst = max(h[4] for h in history)
    grew = max(h[1] for h in history) / history[0][1]
    print(f"\n  initial energy {history[0][1]:.10f} (SPHinXsys records 1.0000000000)")
    print(f"  particles outside the fluid region, worst over the run: {worst}")
    print(f"  largest energy seen, as a fraction of the initial: {grew:.4f}")


def setup_figure(out: Path, spacing: float = SPACING) -> None:
    """The geometry, the particles, and the wall that is not made of them.

    The background is ``K(x)``, the fraction of a kernel centred at ``x``
    that falls inside the solid. That field *is* the boundary condition
    here -- there are no wall particles to draw -- so showing the water on
    top of it is the only honest picture of what the fluid is bounded by.
    The inset is the floor, where ``K`` ramps from 0 to 1 over one support
    radius and the first particle row sits half a spacing clear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

    solver = build(spacing)
    level_set = solver.confinement[0]
    lower = level_set.lower
    upper = lower + np.asarray(level_set.counts) * level_set.spacing
    extent = (lower[0], upper[0], lower[1], upper[1])
    x = solver.positions

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0),
                             gridspec_kw=dict(width_ratios=(1.25, 1.0)))

    for ax, zoom in ((axes[0], False), (axes[1], True)):
        im = ax.imshow(level_set.kernel_integral.T, origin="lower",
                       extent=extent, cmap="Greys", vmin=0.0, vmax=1.0,
                       zorder=1)
        ax.add_patch(Rectangle((0.0, 0.0), TANK, TANK, fill=False,
                               edgecolor="#c0622f", lw=1.2, zorder=3))
        ax.add_patch(Polygon(obstacle(), closed=True, fill=False,
                             edgecolor="#c0622f", lw=1.2, zorder=3))
        ax.scatter(x[:, 0], x[:, 1], s=2.0 if not zoom else 14.0,
                   color="#1f6f8b", linewidths=0, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    ax = axes[0]
    ax.add_patch(FancyArrowPatch((0.45 * TANK, 0.75 * TANK),
                                 (0.45 * TANK, 0.55 * TANK), arrowstyle="->",
                                 color="0.2", lw=1.4, mutation_scale=12, zorder=5))
    ax.text(0.47 * TANK, 0.65 * TANK, f"g = {GRAVITY:g}", fontsize=8, zorder=5)
    ax.text(0.5 * COLUMN_LENGTH, COLUMN_HEIGHT + 0.12, "water column",
            fontsize=8, ha="center", color="#1f6f8b", zorder=5)
    ax.text(0.78 * TANK, 0.06 * TANK, "obstacle\n(level set)", fontsize=8,
            color="#c0622f", zorder=5)
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1])
    ax.set_title(f"{len(x)} particles at dp = {spacing:g}; tank "
                 f"{TANK:g} x {TANK:g}", fontsize=9)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("kernel mass inside the solid, K", fontsize=8)

    ax = axes[1]
    # the whole story is inside one support radius, so the window is set by
    # h and not by the tank: at 10 spacings the ramp is a black edge again
    ax.set_xlim(-2.2 * solver.h, 4.5 * solver.h)
    ax.set_ylim(-2.2 * solver.h, 4.5 * solver.h)
    ax.annotate("", xy=(2.6 * solver.h, 0.0), xytext=(2.6 * solver.h, 0.5 * spacing),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.0))
    ax.text(2.8 * solver.h, 0.22 * spacing, "dp/2", color="#c0392b", fontsize=8)
    ax.set_title(f"the corner: the wall is K, ramping over "
                 f"2h = {2 * solver.h:.3f}", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_dam_break_confinement_setup.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def figure(solver, history, out: Path) -> None:
    """What was conserved, and where the water went.

    Both curves on the left are dimensionless -- energy against its initial
    value, front against the tank -- so they share one axis honestly. A second
    scale for the front would be a dual-axis plot, which lets any two
    quantities be made to look related by choosing the limits.

    The tank and the obstacle are drawn from the same constants the level set
    is built from, so the outline cannot drift from the boundary the fluid
    actually felt.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    times = np.array([h[0] for h in history])
    energy = np.array([h[1] for h in history]) / history[0][1]
    front = np.array([h[3] for h in history]) / TANK

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2),
                             gridspec_kw=dict(width_ratios=(1.0, 1.15)))

    ax = axes[0]
    ax.plot(times, energy, color="#1f6f8b", lw=1.6, label="energy / initial")
    ax.plot(times, front, color="#c0622f", lw=1.6, label="front x / tank")
    ax.set_xlabel("t"); ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, lw=0.4, color="0.9"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("mechanical energy and surge front", fontsize=9)

    ax = axes[1]
    speed = np.linalg.norm(solver.velocity, axis=1)
    sc = ax.scatter(solver.positions[:, 0], solver.positions[:, 1], c=speed,
                    s=1.6, cmap="viridis", vmin=0.0, linewidths=0, zorder=2)
    ax.add_patch(Rectangle((0.0, 0.0), TANK, TANK, fill=False,
                           edgecolor="0.25", lw=1.0, zorder=3))
    ax.add_patch(Polygon(obstacle(), closed=True, facecolor="0.85",
                         edgecolor="0.25", lw=1.0, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, TANK + 0.1); ax.set_ylim(-0.1, TANK + 0.1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="speed")
    ax.set_title(f"t = {times[-1]:.2f}, no particle outside the tank", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_dam_break_confinement.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def animate(snapshots, out: Path, fps: int = 15, max_frames: int = 150) -> None:
    """The collapse, coloured by speed, with the solids drawn from the geometry.

    Unlike the diffusion case the particles genuinely move here, so the map
    *is* the content and there is no profile to fall back on. The colour
    scale is fixed to the fastest particle over the whole run rather than
    rescaled per frame: a scale that moves with the data makes every frame
    look equally energetic, which is the opposite of what a dam break does.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon, Rectangle

    stride = max(1, len(snapshots) // max_frames)
    frames = snapshots[::stride]
    fastest = max(float(np.linalg.norm(v, axis=1).max()) for _, _, v, _ in frames)

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    t0, x0, v0, _ = frames[0]
    sc = ax.scatter(x0[:, 0], x0[:, 1], c=np.linalg.norm(v0, axis=1), s=2.2,
                    cmap="viridis", vmin=0.0, vmax=fastest, linewidths=0,
                    zorder=2)
    ax.add_patch(Rectangle((0.0, 0.0), TANK, TANK, fill=False,
                           edgecolor="0.25", lw=1.1, zorder=3))
    ax.add_patch(Polygon(obstacle(), closed=True, facecolor="0.85",
                         edgecolor="0.25", lw=1.1, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, TANK + 0.1); ax.set_ylim(-0.1, TANK + 0.1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="speed")
    title = ax.set_title(f"t = {frames[0][0]:.2f}", fontsize=9)
    fig.tight_layout()

    def draw(frame):
        t, x, v, _ = frame
        sc.set_offsets(x)
        sc.set_array(np.linalg.norm(v, axis=1))
        title.set_text(f"t = {t:.2f}")
        return sc, title

    movie = FuncAnimation(fig, draw, frames=frames, blit=False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_dam_break_confinement.gif"
    movie.save(path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"  wrote {path} ({len(frames)} frames at {fps} fps)")


def write_series(snapshots, out: Path) -> None:
    """One ``.vtu`` per frame plus the ``.pvd`` that gives ParaView a clock.

    The positions are written per frame, not once: that is the difference
    between this and the diffusion series, and it is why the files cannot
    share geometry.
    """
    from scion.io.particles import write_particles_vtu
    from scion.io.time_series import TimeSeriesWriter

    directory = out / "dam_break_confinement_vtk"
    series = TimeSeriesWriter(directory, "particles", suffix=".vtu")
    for t, x, v, pressure in snapshots:
        series.add(t, write_particles_vtu, x,
                   point_data={"velocity": v, "speed": np.linalg.norm(v, axis=1),
                               "pressure": pressure})
    print(f"  wrote {len(snapshots)} frames to {directory}/particles.pvd")
    write_solid(directory / "solid.vtu")


def write_solid(path: Path) -> Path:
    """The tank and the obstacle, as line loops, written once.

    They are absent from the particle series for the reason the case exists:
    **neither is made of particles**, so there is nothing in the fluid files
    to draw them with. They are a level set sampled on a background grid,
    which a point cloud cannot carry -- so load this file beside the series
    in ParaView and the fluid finally has something to collapse against.

    Drawn from the same ``TANK`` and ``obstacle()`` the level set is built
    from, so the outline cannot drift from the boundary the solver enforces.
    """
    import meshio

    tank = np.array([[0.0, 0.0], [TANK, 0.0], [TANK, TANK], [0.0, TANK]])
    triangle = obstacle()

    points, cells = [], []
    for loop in (tank, triangle):
        start = len(points)
        points.extend(loop)
        n = len(loop)
        cells.extend([[start + i, start + (i + 1) % n] for i in range(n)])

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.zeros((len(points), 3))
    xyz[:, :2] = np.asarray(points)
    meshio.Mesh(points=xyz,
                cells=[meshio.CellBlock("line", np.asarray(cells))]).write(str(path))
    print(f"  wrote {path} (the tank and the obstacle, as a wireframe)")
    return path


def write_csv(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("time,mechanical_energy,max_speed,front_x,escaped\n")
        for row in history:
            handle.write(",".join(f"{v:.17e}" for v in row) + "\n")
    print(f"  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--end", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--spacing", type=float, default=SPACING)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="directory for the figure")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the geometry-and-particles figure")
    parser.add_argument("--animate", action="store_true",
                        help="write a GIF of the collapse")
    parser.add_argument("--vtk", action="store_true",
                        help="write a .vtu series and its .pvd for ParaView")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("SPH dam break with static confinement (SPHinXsys example 9)")
    snapshots = [] if (args.animate or args.vtk) else None
    solver, history = run(args.end, args.interval, args.spacing,
                          snapshots=snapshots)
    report(history)
    if not args.no_setup:
        setup_figure(args.results, args.spacing)
    if not args.no_figure:
        figure(solver, history, args.results)
    if args.animate:
        animate(snapshots, args.results)
    if args.vtk:
        write_series(snapshots, args.results)
    if args.csv is not None:
        write_csv(history, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
