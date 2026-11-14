r"""
SPHinXsys example 6, the 2-D heat-transfer case, on Scion's SPH module.

    python -m examples.sph.heat_transfer_2d
    python -m examples.sph.heat_transfer_2d --refine 2 --convergence

The same case in Rust, which writes the observer history and the final field
as CSV instead of drawing anything:

    cargo run -p scion-sph --release --example heat_transfer_2d -- out.csv

Figures go to ``examples/sph/results`` unless ``--results`` says otherwise.

Three kinds of output, for three different jobs
-----------------------------------------------
``--setup`` and the result figure are **matplotlib**, because a committed
figure has to regenerate itself from the code that produced the numbers; a
screenshot cannot. ``--animate`` is the same drawing once per step, written as
a GIF. ``--vtk`` writes a ``.vtu`` per frame and a ``.pvd`` collection, which
is what to open in ParaView when the question is "what is this field doing"
rather than "what goes in the paper" -- arbitrary colour maps, probes and
slices, and the same format SPHinXsys itself writes.

The case, from
https://www.sphinxsys.org/html/examples/example6_2D_heat_transfer.html and
its source at ``tests/2d_examples/test_2d_diffusion``: a 2.0 x 0.4 block of
particles on a lattice at ``dp = H/40``, periodic along ``x`` and with free
surfaces along ``y``; Wendland C2 at ``h = 1.3 dp``; a directional
diffusivity ``D = d I + d_bias (b (x) b)`` with ``d = 1e-4`` and
``d_bias = 0``, so it is isotropic in this instance and the anisotropic path
is exercised by ``--bias``; corrected kernel gradients; two-stage Runge-Kutta
in time at their ``dt = 0.5 h^2 / (D dim)``; eleven observers along the
mid-line.

What makes this a verification rather than a demonstration
----------------------------------------------------------
The initial condition depends on ``x`` alone and the only boundaries that cut
the field are periodic, so the continuum problem is one-dimensional and its
solution is closed form: a unit block diffuses into a difference of error
functions and a Gaussian stays a Gaussian with variance ``sigma0^2 + 2 D t``.
The mid-line sits fifteen smoothing lengths from either free surface, so the
comparison there is against that exact solution and not against another
code's output. ``--convergence`` measures the rate, which is the property
that cannot be reached by tuning a tolerance.

The reference implementation is used for the one thing it can settle beyond
doubt: its own published ``t = 0`` observations, which depend on the kernel,
the lattice, the particle volumes and the Shepard interpolation and on
nothing else. Those are reproduced to round-off; see
``tests/sph/test_diffusion.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scion.discretization.sph import (DiffusionSolver, DirectionalDiffusivity,
                                      lattice)

#: Figures from every example in this folder land here. Created on demand, so
#: a fresh clone has no empty directory and a run that draws nothing leaves
#: none behind.
RESULTS = Path(__file__).resolve().parent / "results"

LENGTH, HEIGHT = 2.0, 0.4
D_COEFF = 1.0e-4
BIAS_DIRECTION = (np.cos(np.pi / 6.0), np.sin(np.pi / 6.0))
END_TIME = 1.0
N_OBSERVERS = 11


def initial_condition(x: np.ndarray) -> np.ndarray:
    """A unit block on ``[0.45, 0.55]`` and a narrow Gaussian at ``x = 1.5``.

    Written in the same order as the reference case, where the second branch
    overwrites the first; they do not overlap, but the order is part of the
    specification and copying it costs nothing.
    """
    phi = np.where((x[:, 0] >= 0.45) & (x[:, 0] <= 0.55), 1.0, 0.0)
    return np.where(x[:, 0] >= 1.0, np.exp(-2500.0 * (x[:, 0] - 1.5) ** 2), phi)


def exact_profile(x: np.ndarray, t: float, d_coeff: float = D_COEFF) -> np.ndarray:
    """The closed-form solution of the 1-D problem at time ``t``.

    The periodic images are omitted: at ``t = 1`` the diffusion length is
    ``sqrt(2 D t) = 0.014`` against a period of 2, so the nearest image
    contributes ``exp(-(L/2 sigma)^2) ~ 1e-2000``. Stated rather than assumed
    because it is the one approximation in the reference solution.
    """
    from scipy.special import erf

    if t <= 0.0:
        out = np.where((x >= 0.45) & (x <= 0.55), 1.0, 0.0)
        return np.where(x >= 1.0, np.exp(-2500.0 * (x - 1.5) ** 2), out)
    spread = np.sqrt(4.0 * d_coeff * t)
    block = 0.5 * (erf((0.55 - x) / spread) - erf((0.45 - x) / spread))
    sigma0_sq = 1.0 / 5000.0
    sigma_sq = sigma0_sq + 2.0 * d_coeff * t
    gaussian = np.sqrt(sigma0_sq / sigma_sq) * np.exp(-(x - 1.5) ** 2 / (2.0 * sigma_sq))
    return block + gaussian


def observers(height: float = HEIGHT, length: float = LENGTH) -> np.ndarray:
    """The reference case's eleven probe points, at 5% to 95% of the length.

    An *observer* is not a particle. It carries no mass, no volume and no
    physics; it is a position at which the field is interpolated and
    recorded, SPHinXsys's ``ObserverBody``. Nothing in the solve knows these
    exist -- removing them changes no result.

    They matter because these eleven numbers **are** the published reference:
    SPHinXsys's regression file records ``Phi`` here and nowhere else, so
    reproducing them is what pins this implementation against theirs.

    Why ``y = H/2`` and not anywhere else. The initial condition depends on
    ``x`` alone and the only boundaries cutting the field are periodic, so
    the solution does not vary with ``y`` and one line of probes along ``x``
    captures all of it. But the free surfaces at ``y = 0`` and ``y = H``
    truncate the kernel, and the operator is degraded within a support radius
    of them -- the partition of unity reads 0.79 in the first particle row
    against 1.00 in the interior. The mid-line sits **15.4 smoothing lengths**
    from either surface, so the probes read the interior solution, which is
    also what makes the comparison against the 1-D closed form legitimate.
    """
    xs = 0.05 * length + 0.9 * length * np.arange(N_OBSERVERS) / (N_OBSERVERS - 1)
    return np.stack([xs, np.full(N_OBSERVERS, 0.5 * height)], axis=1)


def build(refine: int = 1, bias: float = 0.0, correction: bool = True):
    """The solver and the initial field at ``dp = H / (40 * refine)``."""
    dp = HEIGHT / (40 * refine)
    x, volumes = lattice([0.0, 0.0], [LENGTH, HEIGHT], dp, periodic=(True, False))
    diffusivity = DirectionalDiffusivity(d_coeff=D_COEFF, bias_coeff=bias,
                                         bias_direction=BIAS_DIRECTION)
    solver = DiffusionSolver(x, volumes, diffusivity, h=1.3 * dp,
                             lower=[0.0, 0.0], upper=[LENGTH, HEIGHT],
                             periodic=(True, False), correction=correction)
    return solver, initial_condition(x), dp


def mid_line(solver) -> np.ndarray:
    """The particle row nearest ``y = H/2``; on an even lattice there are two
    equidistant rows and this takes the lower one."""
    y = solver.positions[:, 1]
    offset = np.abs(y - 0.5 * HEIGHT)
    return np.flatnonzero(offset <= offset.min() + 1e-12 * HEIGHT)


def run(refine: int = 1, bias: float = 0.0, end_time: float = END_TIME,
        steps_per_rule: int = 1, correction: bool = True, quiet: bool = False,
        snapshots: list | None = None):
    """March the case and return ``(solver, phi, history)``.

    ``steps_per_rule`` subdivides the reference time step; at 1 the clock is
    SPHinXsys's own, which at this resolution is three steps to ``t = 1`` and
    is within 12% of the operator's measured stability limit.

    Pass a list as ``snapshots`` to collect ``(t, phi)`` at every step, which
    is what the animation and the VTK series are built from. It is opt-in
    because the field is one array per step and the caller may want three
    steps or three thousand.
    """
    solver, phi, dp = build(refine, bias, correction)
    dt = solver.time_step() / steps_per_rule
    n_steps = int(np.ceil(end_time / dt))
    points = observers()
    if not quiet:
        print(f"  {len(phi)} particles at dp = {dp:.5f}, {len(solver.pairs)} pairs, "
              f"{solver.pairs.counts().mean():.1f} neighbours each")
        print(f"  dt = {dt:.6f} (rule), stability limit {solver.stable_time_step():.6f}, "
              f"{n_steps} steps to t = {n_steps * dt:.4f}")
    history = [(0.0, solver.observe(points, phi))]
    total0 = float((phi * solver.volumes).sum())
    if snapshots is not None:
        snapshots.append((0.0, phi.copy()))
    for _ in range(n_steps):
        phi = solver.step(phi, dt)
        history.append((history[-1][0] + dt, solver.observe(points, phi)))
        if snapshots is not None:
            snapshots.append((history[-1][0], phi.copy()))
    if not quiet:
        total = float((phi * solver.volumes).sum())
        print(f"  species integral {total0:.12e} -> {total:.12e} "
              f"(drift {abs(total - total0) / total0:.2e})")
    return solver, phi, history


def report(solver, phi, history) -> None:
    print("\n  observers along y = H/2")
    header = "".join(f"{p[0]:>12.2f}" for p in observers())
    print(f"  {'t':>8}{header}")
    for t, row in history:
        print(f"  {t:8.4f}" + "".join(f"{v:12.5f}" for v in row))

    row = mid_line(solver)
    t_end = history[-1][0]
    exact = exact_profile(solver.positions[row, 0], t_end)
    got = phi[row]
    print(f"\n  against the closed form at t = {t_end:.4f}: "
          f"max |phi - phi_exact| = {np.abs(got - exact).max():.3e}, "
          f"L1 = {(np.abs(got - exact) * (LENGTH / row.size)).sum():.3e}")


def convergence_time_step(levels, end_time: float = END_TIME) -> float:
    """One time step for the whole study, set by the FINEST level.

    The operator's stability limit scales as ``h^2``, so a step taken from
    the coarse level is unstable on the fine one -- by a factor of sixteen
    over two refinements, which does not degrade the answer gently, it
    returns 1e9. Taking the finest level's rule and quartering it keeps every
    level stable and the time error common to all of them, which is what
    makes the remaining differences spatial.
    """
    finest = HEIGHT / (40 * max(levels))
    return DirectionalDiffusivity(d_coeff=D_COEFF).time_step(1.3 * finest, 2) / 4.0


def convergence(levels=(1, 2, 4), end_time: float = END_TIME) -> None:
    """Observed order of accuracy against the closed form.

    The rate is the property that cannot be reached by tuning a tolerance,
    which is why this exists beside the single-mesh error.
    """
    dt = convergence_time_step(levels, end_time)
    n_steps = int(np.ceil(end_time / dt))
    print(f"\n  spatial convergence at dt = {dt:.5f}, {n_steps} steps")
    print(f"  {'dp':>9} {'particles':>10} {'max err':>11} {'rate':>6} "
          f"{'L1 err':>11} {'rate':>6}")
    previous = None
    for level in levels:
        solver, phi, dp = build(level)
        for _ in range(n_steps):
            phi = solver.step(phi, dt)
        row = mid_line(solver)
        err = np.abs(phi[row] - exact_profile(solver.positions[row, 0], n_steps * dt))
        worst, l1 = err.max(), float((err * dp).sum())
        if previous is None:
            rates = ("", "")
        else:
            rates = (f"{np.log2(previous[0] / worst):6.2f}",
                     f"{np.log2(previous[1] / l1):6.2f}")
        print(f"  {dp:9.5f} {len(phi):10d} {worst:11.3e} {rates[0]:>6} "
              f"{l1:11.3e} {rates[1]:>6}")
        previous = (worst, l1)


def figure(solver, phi, history, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = mid_line(solver)
    x = solver.positions[row, 0]
    order = np.argsort(x)
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6),
                             gridspec_kw=dict(height_ratios=(2.0, 1.0)))

    ax = axes[0]
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(history)))
    for (t, _), colour in zip(history, shades):
        ax.plot(x[order], exact_profile(x[order], t), color=colour, lw=2.4,
                alpha=0.35, zorder=1,
                label="closed form" if t == 0.0 else None)
    # the SPH field is only computed at the stored times, so the last one is
    # the only state still in hand; the rest are the observers
    ax.plot(x[order], phi[row][order], color="0.1", lw=1.0, zorder=3,
            label=f"SPH, t = {history[-1][0]:.3f}")
    ax.scatter(observers()[:, 0], history[-1][1], s=18, color="#c0392b",
               zorder=4, label="observers")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\phi$")
    ax.set_xlim(0.0, LENGTH); ax.legend(frameon=False, fontsize=8)
    ax.set_title("mid-line profile, every stored time", fontsize=9)

    ax = axes[1]
    sc = ax.scatter(solver.positions[:, 0], solver.positions[:, 1], c=phi,
                    s=1.2, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlim(0.0, LENGTH); ax.set_ylim(0.0, HEIGHT)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label=r"$\phi$")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_heat_transfer_2d.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  wrote {path}")


def setup_figure(out: Path, refine: int = 1) -> None:
    """The geometry, the particles, and what the boundaries do.

    The inset is the part that is actually about SPH: one particle, the
    circle of radius ``2h`` it integrates over, and the links to every
    neighbour inside it. A domain sketch alone would describe this case
    equally well if it were finite elements -- what distinguishes a particle
    method is which particles can see each other, and at the free surface
    that circle is visibly half empty, which is the deficit the corrected
    gradient exists to undo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch

    solver, phi, dp = build(refine)
    x = solver.positions
    support = solver.kernel.support(solver.h)

    fig = plt.figure(figsize=(7.6, 5.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.35), hspace=0.58)

    # -- the domain -----------------------------------------------------
    ax = fig.add_subplot(grid[0])
    sc = ax.scatter(x[:, 0], x[:, 1], c=phi, s=1.2, cmap="viridis",
                    vmin=0.0, vmax=1.0, linewidths=0, zorder=2)
    ax.plot([0, LENGTH], [0, 0], color="0.3", lw=1.4, zorder=3)
    ax.plot([0, LENGTH], [HEIGHT, HEIGHT], color="0.3", lw=1.4, zorder=3)
    for edge in (0.0, LENGTH):
        ax.plot([edge, edge], [0, HEIGHT], color="#c0622f", lw=1.4,
                ls=(0, (3, 2)), zorder=3)
    ax.add_patch(FancyArrowPatch((LENGTH, 0.5 * HEIGHT), (LENGTH + 0.12, 0.5 * HEIGHT),
                                 arrowstyle="->", color="#c0622f", lw=1.2,
                                 mutation_scale=9, zorder=4))
    ax.add_patch(FancyArrowPatch((-0.12, 0.5 * HEIGHT), (0.0, 0.5 * HEIGHT),
                                 arrowstyle="->", color="#c0622f", lw=1.2,
                                 mutation_scale=9, zorder=4))
    probes = observers()
    # bounded by the block: axhline would run out through the periodic label
    ax.plot([0.0, LENGTH], [0.5 * HEIGHT] * 2, color="#c0392b", lw=0.7,
            ls=(0, (4, 3)), zorder=4)
    ax.scatter(probes[:, 0], probes[:, 1], s=22, facecolor="none",
               edgecolor="#c0392b", lw=1.0, zorder=5)
    # how far the probe line sits from the surface that degrades the operator
    clearance = 0.5 * HEIGHT / solver.h
    ax.annotate("", xy=(0.19, HEIGHT), xytext=(0.19, 0.5 * HEIGHT),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=0.9),
                zorder=5)
    ax.text(0.205, 0.74 * HEIGHT, f"{clearance:.1f} h to the free surface",
            color="#c0392b", fontsize=8, va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.text(LENGTH + 0.16, 0.5 * HEIGHT, "periodic", color="#c0622f",
            fontsize=8, va="center")
    ax.text(0.5 * LENGTH, HEIGHT + 0.035, "free surface", color="0.3",
            fontsize=8, ha="center")
    ax.text(0.5 * LENGTH, -0.085, "free surface", color="0.3",
            fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.42, LENGTH + 0.62); ax.set_ylim(-0.12, HEIGHT + 0.09)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{len(x)} particles at dp = {dp:g}, h = 1.3 dp, "
                 f"support 2h = {support:.3f}", fontsize=9)
    bar = fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01)
    bar.set_label(r"initial $\phi$", fontsize=8)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="none",
                              mec="#c0392b",
                              label="11 observers: probe points, not particles; "
                                    r"$\phi$ is recorded here")],
              frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, -0.30))

    # -- one particle and everything it can see -------------------------
    ax = fig.add_subplot(grid[1])
    # a particle on the free surface, where the support is half empty
    centre = np.array([0.5 * LENGTH, 0.5 * dp])
    chosen = int(np.argmin(np.linalg.norm(x - centre, axis=1)))
    offset = x - x[chosen]
    near = np.linalg.norm(offset, axis=1) < support
    window = 1.35 * support
    shown = np.all(np.abs(offset) < window, axis=1)

    for j in np.flatnonzero(near):
        if j == chosen:
            continue
        ax.plot([x[chosen, 0], x[j, 0]], [x[chosen, 1], x[j, 1]],
                color="#9ecae1", lw=0.6, zorder=1)
    ax.scatter(x[shown, 0], x[shown, 1], s=9, color="0.75", zorder=2)
    ax.scatter(x[near, 0], x[near, 1], s=14, color="#1f6f8b", zorder=3)
    ax.scatter(*x[chosen], s=34, color="#c0392b", zorder=4)
    ax.add_patch(Circle(x[chosen], support, fill=False, edgecolor="#c0392b",
                        lw=1.1, zorder=4))
    ax.add_patch(Circle(x[chosen], solver.h, fill=False, edgecolor="#c0392b",
                        lw=0.8, ls=(0, (2, 2)), zorder=4))
    ax.plot([0, LENGTH], [0, 0], color="0.3", lw=1.4, zorder=5)
    # below and left, where the solid side of the support has no particles
    ax.annotate("h", x[chosen] + np.array([-0.78 * solver.h, -0.55 * solver.h]),
                color="#c0392b", fontsize=8)
    ax.annotate("2h", x[chosen] + np.array([0.72 * support, 0.72 * support]),
                color="#c0392b", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(x[chosen, 0] - window, x[chosen, 0] + window)
    ax.set_ylim(x[chosen, 1] - window, x[chosen, 1] + window)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"a particle at the free surface and its {int(near.sum()) - 1} "
                 f"neighbours; the support is cut by the boundary", fontsize=9)

    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_heat_transfer_2d_setup.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def animate(solver, snapshots, out: Path, fps: int = 12,
            max_frames: int = 90) -> None:
    """The field over the particles and its profile, as a GIF.

    **The particles do not move, and the colour map alone looks frozen.**
    Both are properties of the case rather than of the drawing: this is
    diffusion on a fixed particle set, so the positions are constant to
    0.0e+00, and over the reference second the field spreads by
    ``sqrt(2 D t) = 0.014``, which is 0.7% of the domain and 1.6% of the
    colour range between neighbouring frames -- under what the eye resolves
    in a filled colour map.

    So the lower panel carries the actual content: the mid-line profile,
    against the closed form at the same instant. There the block's corners
    visibly round and the narrow Gaussian visibly flattens, because a line
    plot resolves a 2% change that a colour map cannot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    stride = max(1, len(snapshots) // max_frames)
    frames = snapshots[::stride]
    x = solver.positions
    row = mid_line(solver)
    order = row[np.argsort(x[row, 0])]
    line_x = x[order, 0]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2),
                             gridspec_kw=dict(height_ratios=(1.0, 1.25),
                                              hspace=0.45))

    ax = axes[0]
    sc = ax.scatter(x[:, 0], x[:, 1], c=frames[0][1], s=1.6, cmap="viridis",
                    vmin=0.0, vmax=1.0, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, LENGTH); ax.set_ylim(0.0, HEIGHT)
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01, label=r"$\phi$")
    title = ax.set_title("", fontsize=9)

    ax = axes[1]
    exact_line, = ax.plot(line_x, exact_profile(line_x, 0.0), color="#9aa7b1",
                          lw=2.6, zorder=1, label="closed form")
    sph_line, = ax.plot(line_x, frames[0][1][order], color="#1f6f8b", lw=1.3,
                        zorder=2, label="SPH, y = H/2")
    ax.set_xlim(0.0, LENGTH); ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\phi$")
    ax.grid(True, lw=0.4, color="0.93"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    def draw(frame):
        t, field = frame
        sc.set_array(field)
        sph_line.set_ydata(field[order])
        exact_line.set_ydata(exact_profile(line_x, t))
        title.set_text(f"t = {t:.4f}   (the particles never move; only "
                       r"$\phi$ does)")
        return sc, sph_line, exact_line, title

    movie = FuncAnimation(fig, draw, frames=frames, blit=False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_heat_transfer_2d.gif"
    movie.save(path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"  wrote {path} ({len(frames)} frames at {fps} fps)")


def write_series(solver, snapshots, out: Path) -> None:
    """One ``.vtu`` per frame plus the ``.pvd`` that gives ParaView a clock.

    ``.vtu`` with vertex cells rather than ``.vtp``: see the note in
    ``scion.io.particles`` -- meshio cannot read PolyData, so a ``.vtp``
    series would be unreadable by anything in this repository that produced
    it.
    """
    from scion.io.particles import write_particles_vtu
    from scion.io.time_series import TimeSeriesWriter

    directory = out / "heat_transfer_2d_vtk"
    series = TimeSeriesWriter(directory, "particles", suffix=".vtu")
    for t, field in snapshots:
        series.add(t, write_particles_vtu, solver.positions,
                   point_data={"phi": field, "volume": solver.volumes})
    print(f"  wrote {len(snapshots)} frames to {directory}/particles.pvd")


def write_csv(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("time," + ",".join(f"phi_{i}" for i in range(N_OBSERVERS)) + "\n")
        for t, row in history:
            handle.write(f"{t:.17e}," + ",".join(f"{v:.17e}" for v in row) + "\n")
    print(f"  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--refine", type=int, default=1,
                        help="particles per H/40 in each direction")
    parser.add_argument("--bias", type=float, default=0.0,
                        help="anisotropic part of the diffusivity (0 = the reference case)")
    parser.add_argument("--substeps", type=int, default=1,
                        help="subdivisions of the reference time step")
    parser.add_argument("--no-correction", action="store_true",
                        help="drop the kernel-gradient correction")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="directory for the figure")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the geometry-and-particles figure")
    parser.add_argument("--animate", action="store_true",
                        help="write a GIF of the field; implies finer substeps")
    parser.add_argument("--vtk", action="store_true",
                        help="write a .vtu series and its .pvd for ParaView")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("SPH 2-D heat transfer (SPHinXsys example 6)")
    # the reference clock is three steps to t = 1, which is a four-frame
    # animation; say so rather than silently producing one
    substeps = args.substeps
    if (args.animate or args.vtk) and substeps == 1:
        substeps = 24
        print(f"  --substeps raised to {substeps} so the frames are worth watching")
    snapshots = [] if (args.animate or args.vtk) else None
    solver, phi, history = run(refine=args.refine, bias=args.bias,
                               steps_per_rule=substeps,
                               correction=not args.no_correction,
                               snapshots=snapshots)
    report(solver, phi, history)
    if not args.no_setup:
        setup_figure(args.results, refine=args.refine)
    if args.animate:
        animate(solver, snapshots, args.results)
    if args.vtk:
        write_series(solver, snapshots, args.results)
    if args.convergence:
        convergence()
    if not args.no_figure:
        figure(solver, phi, history, args.results)
    if args.csv is not None:
        write_csv(history, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
