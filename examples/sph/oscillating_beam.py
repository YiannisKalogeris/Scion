r"""
SPHinXsys example 13, the 2-D oscillating beam.

    python -m examples.sph.oscillating_beam --end 1.0
    python -m examples.sph.oscillating_beam --end 1.0 --csv tip.csv

``--animate`` writes a GIF of the ringing beam and ``--vtk`` a ParaView
series.

The same case in Rust, about thirty times faster, taking the end time and an
optional CSV:

    cargo run -p scion-sph --release --example oscillating_beam -- 1.0 out.csv

Figures go to ``examples/sph/results`` unless ``--results`` says otherwise.

A cantilever clamped into a thicker base is released from the first
Euler-Bernoulli mode shape and left to ring. Total-Lagrangian SPH, so the
neighbour list and the correction matrix are built once in the reference
configuration and never touched.

Why this case is a verification and not a demonstration
-------------------------------------------------------
The initial velocity is not arbitrary: it is the exact first mode of a
cantilever, ``kl = 1.875``, normalised so the tip moves at ``vf c0``. A beam
released from a pure mode oscillates in that mode, so the **period** is a
closed-form prediction and the only thing the solver can get wrong. A
frequency is a rate, so it cannot be matched by tuning an amplitude.

Three readings of "the" analytical frequency differ by a few percent and the
report prints all three rather than choosing: Euler-Bernoulli with the plane
stress modulus ``E``, with the plane strain modulus ``E/(1-nu^2)``, and with
Timoshenko's shear correction, which matters here because the beam is thick
(``L/h = 10``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from scion.discretization.sph import lattice
from scion.discretization.sph.elastic import (ElasticSolver,
                                              SaintVenantKirchhoffSolid)

#: Figures from every example in this folder land here. Created on demand, so
#: a fresh clone has no empty directory and a run that draws nothing leaves
#: none behind.
RESULTS = Path(__file__).resolve().parent / "results"

LENGTH, THICKNESS, INSERT = 0.2, 0.02, 0.06
SPACING = THICKNESS / 10.0
BASE_WIDTH = 4 * SPACING
RHO0, YOUNGS, POISSON = 1.0e3, 2.0e6, 0.3975
VELOCITY_FRACTION = 0.05
KL = 1.875                     # first cantilever eigenvalue
END_TIME = 1.0
#: the observer body carries its own smoothing length in the reference case
#: (``defineAdaptationRatios(1.15, 2.0)``), and the tip reading depends on it
OBSERVER_H_RATIO = 1.15


def mode_shape(x_hat: np.ndarray) -> np.ndarray:
    """The first cantilever mode, normalised to one at the tip."""
    M = np.sin(KL) + np.sinh(KL)
    N = np.cos(KL) + np.cosh(KL)
    Q = 2.0 * (np.cos(KL) * np.sinh(KL) - np.sin(KL) * np.cosh(KL))
    return (M * (np.cos(KL * x_hat) - np.cosh(KL * x_hat))
            - N * (np.sin(KL * x_hat) - np.sinh(KL * x_hat))) / Q


def _in_box(points, lower, upper):
    return np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)),
                  axis=1)


def beam_region(points):
    """The beam proper: ``x`` from the insert to the tip."""
    return _in_box(points, [-INSERT, -0.5 * THICKNESS],
                   [LENGTH, 0.5 * THICKNESS])


def base_region(points):
    """The thicker block the beam is inserted into."""
    return _in_box(points, [-INSERT - BASE_WIDTH, -0.5 * THICKNESS - BASE_WIDTH],
                   [0.0, 0.5 * THICKNESS + BASE_WIDTH])


def build(spacing: float = SPACING):
    """The body, the solver and the mask of clamped particles.

    The clamp is the base *minus* the beam -- a collar around the insert, not
    the whole base. Clamping the whole base would hold the first centimetre
    of the beam itself rigid and shorten the cantilever.
    """
    solid = SaintVenantKirchhoffSolid(RHO0, YOUNGS, POISSON)
    lower = [-INSERT - BASE_WIDTH, -0.5 * LENGTH]
    upper = [LENGTH + 3 * BASE_WIDTH, 0.5 * LENGTH]
    body = lambda p: beam_region(p) | base_region(p)
    x, volumes = lattice(lower, upper, spacing, inside=body)
    solver = ElasticSolver(x, volumes, solid, spacing=spacing)
    fixed = base_region(x) & ~beam_region(x)

    inside_beam = x[:, 0] > 0.0
    solver.velocity[inside_beam, 1] = (VELOCITY_FRACTION * solid.c0
                                       * mode_shape(x[inside_beam, 0] / LENGTH))
    return solver, fixed


def analytic_periods(solid: SaintVenantKirchhoffSolid) -> dict:
    """Three readings of the first-mode period, in seconds.

    They differ by a few percent and the difference is physics, not
    uncertainty: which one a thick beam should match is the question the
    measurement answers.
    """
    second_moment = THICKNESS ** 3 / 12.0
    mass_per_length = solid.rho0 * THICKNESS
    out = {}
    for name, modulus in (("plane stress", YOUNGS),
                          ("plane strain", YOUNGS / (1.0 - POISSON ** 2))):
        omega = (KL / LENGTH) ** 2 * np.sqrt(modulus * second_moment
                                             / mass_per_length)
        out[name] = 2.0 * np.pi / omega
    # Timoshenko's first-order correction for shear and rotary inertia
    modulus = YOUNGS / (1.0 - POISSON ** 2)
    radius = np.sqrt(second_moment / THICKNESS)
    shear_coefficient = 5.0 / 6.0
    omega = (KL / LENGTH) ** 2 * np.sqrt(modulus * second_moment / mass_per_length)
    correction = 1.0 + (KL / LENGTH) ** 2 * radius ** 2 * (
        1.0 + modulus / (shear_coefficient * solid.shear_modulus))
    out["Timoshenko"] = 2.0 * np.pi / (omega / np.sqrt(correction))
    return out


def measure_period(times, displacement) -> float:
    """From the zero crossings of the tip displacement.

    Averaging over every crossing rather than timing one cycle, so the figure
    is not a single-sample measurement of a quantity that drifts.
    """
    t = np.asarray(times)
    y = np.asarray(displacement)
    sign_change = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))
    if sign_change.size < 3:
        return float("nan")
    crossings = t[sign_change] + (t[sign_change + 1] - t[sign_change]) * (
        -y[sign_change] / (y[sign_change + 1] - y[sign_change]))
    return float(2.0 * np.mean(np.diff(crossings)))


def run(end_time: float = END_TIME, interval: float = 0.01,
        spacing: float = SPACING, quiet: bool = False,
        snapshots: list | None = None):
    """Ring the beam and return ``(solver, history)``.

    Pass a list as ``snapshots`` to collect ``(t, positions, displacement,
    velocity)`` once per sampling interval, for the animation and the VTK
    series. Total Lagrangian, so the *reference* positions never change --
    but the deformed ones are what moves, and those are what is stored.
    """
    solver, fixed = build(spacing)
    # the writers need it: the clamp is the boundary condition, and nothing
    # in the particle positions or the displacement shows where it is
    solver.clamped = fixed
    tip = np.array([[LENGTH, 0.0]])
    if not quiet:
        print(f"  {len(solver.positions)} particles at dp = {spacing}, "
              f"{len(solver.pairs)} pairs, {int(fixed.sum())} clamped")
        print(f"  c0 = {solver.solid.c0:.4f}, tip velocity "
              f"{VELOCITY_FRACTION * solver.solid.c0:.4f}")
    history = [(0.0, 0.0, solver.kinetic_energy(), solver.strain_energy())]

    def capture():
        if snapshots is not None:
            snapshots.append((solver.time, solver.positions.copy(),
                              solver.displacement(), solver.velocity.copy()))

    capture()
    start = time.perf_counter()
    steps = 0
    next_sample = interval
    while solver.time < end_time:
        dt = solver.time_step()
        solver.step(dt, fixed=fixed)
        steps += 1
        if solver.time >= next_sample:
            displacement = solver.probe(tip, solver.displacement()[:, 1],
                                        h_ratio=OBSERVER_H_RATIO)[0]
            history.append((solver.time, displacement, solver.kinetic_energy(),
                            solver.strain_energy()))
            capture()
            next_sample += interval
    if not quiet:
        print(f"  {steps} steps, {time.perf_counter() - start:.1f} s of wall time")
    return solver, history


def report(solver, history) -> None:
    times = np.array([h[0] for h in history])
    tip = np.array([h[1] for h in history])
    energy = np.array([h[2] + h[3] for h in history])
    print(f"\n  {'t':>6} {'tip dy':>11} {'kinetic':>11} {'strain':>11} {'total':>11}")
    stride = max(1, len(history) // 12)
    for row in history[::stride]:
        print(f"  {row[0]:6.3f} {row[1]:11.3e} {row[2]:11.4e} {row[3]:11.4e} "
              f"{row[2] + row[3]:11.4e}")

    measured = measure_period(times, tip)
    print(f"\n  measured period {measured:.5f} s  "
          f"(frequency {1.0 / measured:.4f} Hz)")
    for name, period in analytic_periods(solver.solid).items():
        print(f"    against {name:13s} {period:.5f} s -> "
              f"{100 * (measured / period - 1):+6.2f}%")
    print(f"  peak tip displacement {np.abs(tip).max():.5f} m "
          f"= {100 * np.abs(tip).max() / LENGTH:.2f}% of the length")
    print(f"  total energy drift over the run: "
          f"{100 * (energy[1:].max() / energy[1:].min() - 1):.2f}%")


def setup_figure(out: Path, spacing: float = SPACING) -> None:
    """The geometry, the particles, the clamp and the initial velocity.

    Two things here are easy to get wrong and both are visible rather than
    described: the clamp is the *collar* around the insert and not the whole
    base -- holding the base would shorten the cantilever -- and the initial
    velocity is the first Euler-Bernoulli mode, normalised to ``vf c0`` at
    the tip, which is why the case has a closed-form period at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    solver, fixed = build(spacing)
    x = solver.reference

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4),
                             gridspec_kw=dict(height_ratios=(1.0, 0.85),
                                              hspace=0.55))

    ax = axes[0]
    ax.scatter(x[~fixed, 0], x[~fixed, 1], s=5.0, color="#1f6f8b",
               linewidths=0, zorder=2)
    ax.scatter(x[fixed, 0], x[fixed, 1], s=5.0, color="#c0392b",
               linewidths=0, zorder=3)
    stride = max(1, int(round(0.01 / spacing)))
    # the lattice has rows at +-dp/2, never at y = 0, so a half-spacing
    # window selects nothing and the quiver silently disappears
    arrow = (x[:, 0] > 0.0) & (np.abs(x[:, 1]) < 0.75 * spacing)
    pick = np.flatnonzero(arrow)[::stride]
    ax.quiver(x[pick, 0], x[pick, 1], np.zeros(pick.size),
              solver.velocity[pick, 1], color="0.15", scale=150, width=0.003,
              zorder=4)
    ax.scatter([LENGTH], [0.0], s=34, facecolor="none", edgecolor="#c0392b",
               lw=1.2, zorder=5)
    ax.text(LENGTH, -0.5 * THICKNESS - 0.012, "tip observer", fontsize=8,
            ha="center", color="#c0392b")
    ax.set_aspect("equal")
    ax.set_xlim(-INSERT - 5 * BASE_WIDTH, LENGTH + 4 * BASE_WIDTH)
    ax.set_ylim(-0.028, 0.040)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{len(x)} particles at dp = {spacing:g}, "
                 f"{int(fixed.sum())} clamped; L = {LENGTH:g}, "
                 f"h = {THICKNESS:g}, insert {INSERT:g}", fontsize=9)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#1f6f8b", label="free"),
        Line2D([], [], marker="o", ls="none", color="#c0392b",
               label="clamped collar")],
        frameon=False, fontsize=8, loc="upper left", ncol=2)

    ax = axes[1]
    span = np.linspace(0.0, 1.0, 200)
    ax.plot(span * LENGTH, mode_shape(span), color="#1f6f8b", lw=1.6)
    ax.scatter(x[pick, 0], solver.velocity[pick, 1]
               / (VELOCITY_FRACTION * solver.solid.c0),
               s=10, color="0.25", zorder=3, label="particles")
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$v_y / (v_f c_0)$")
    ax.set_xlim(0.0, LENGTH)
    ax.grid(True, lw=0.4, color="0.93"); ax.set_axisbelow(True)
    ax.set_title(f"first cantilever mode, kl = {KL}, tip speed "
                 f"{VELOCITY_FRACTION * solver.solid.c0:.3f} m/s", fontsize=9)

    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_oscillating_beam_setup.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def figure(solver, history, out: Path) -> None:
    """The ring, and the shape it rings in.

    The measured period is drawn against the plane-stress Euler-Bernoulli one
    rather than quoted beside it, because the eye catches a frequency error
    that accumulates over four cycles long before it catches a 1% difference
    in a printed number.

    The right panel is at whatever time the run ended, not at the peak, so a
    short ``--end`` shows a nearly straight beam. That is the honest thing to
    draw: nothing is stored from the peak.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([h[0] for h in history])
    tip = np.array([h[1] for h in history])
    measured = measure_period(times, tip)
    periods = analytic_periods(solver.solid)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0),
                             gridspec_kw=dict(height_ratios=(1.35, 1.0)))

    ax = axes[0]
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=1)
    for k in range(1, 1 + int(times[-1] / periods["plane stress"])):
        ax.axvline(k * periods["plane stress"], color="#c0622f", lw=1.0,
                   ls=(0, (4, 3)), zorder=1,
                   label="Euler-Bernoulli period" if k == 1 else None)
    ax.plot(times, tip, color="#1f6f8b", lw=1.4, zorder=3, label="SPH tip")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"tip $\Delta y$ [m]")
    ax.set_xlim(0.0, times[-1])
    ax.grid(True, lw=0.4, color="0.93"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    title = "tip displacement"
    if np.isfinite(measured):
        title += (f"; measured period {measured:.4f} s against "
                  f"{periods['plane stress']:.4f} s")
    ax.set_title(title, fontsize=9)

    ax = axes[1]
    magnitude = np.linalg.norm(solver.displacement(), axis=1)
    ax.scatter(solver.reference[:, 0], solver.reference[:, 1], s=1.0,
               color="0.84", linewidths=0, zorder=1, label="reference")
    sc = ax.scatter(solver.positions[:, 0], solver.positions[:, 1],
                    c=magnitude, s=2.4, cmap="viridis", linewidths=0, zorder=2,
                    label="deformed")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label=r"$|u|$ [m]")
    # upper left is the one corner the beam never reaches, at rest or at full
    # deflection; lower left is the clamp and the legend lands on it
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=3)
    ax.set_title(f"configuration at t = {times[-1]:.3f} s", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_oscillating_beam.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def animate(solver, snapshots, out: Path, fps: int = 15,
            max_frames: int = 150) -> None:
    """The beam ringing, coloured by displacement magnitude.

    The reference lattice stays drawn underneath, because without it a
    cantilever at full deflection and one at rest look equally plausible and
    the amplitude -- 57% of the beam's length -- does not read.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    stride = max(1, len(snapshots) // max_frames)
    frames = snapshots[::stride]
    largest = max(float(np.linalg.norm(u, axis=1).max()) for _, _, u, _ in frames)
    reach = max(float(np.abs(x[:, 1]).max()) for _, x, _, _ in frames)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(solver.reference[:, 0], solver.reference[:, 1], s=1.4,
               color="0.86", linewidths=0, zorder=1)
    t0, x0, u0, _ = frames[0]
    sc = ax.scatter(x0[:, 0], x0[:, 1], c=np.linalg.norm(u0, axis=1), s=3.0,
                    cmap="viridis", vmin=0.0, vmax=largest, linewidths=0,
                    zorder=2)
    ax.set_aspect("equal")
    ax.set_xlim(-INSERT - 5 * BASE_WIDTH, LENGTH + 6 * BASE_WIDTH)
    ax.set_ylim(-1.12 * reach, 1.12 * reach)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label=r"$|u|$ [m]")
    title = ax.set_title(f"t = {frames[0][0]:.3f} s", fontsize=9)
    fig.tight_layout()

    def draw(frame):
        t, x, u, _ = frame
        sc.set_offsets(x)
        sc.set_array(np.linalg.norm(u, axis=1))
        title.set_text(f"t = {t:.3f} s")
        return sc, title

    movie = FuncAnimation(fig, draw, frames=frames, blit=False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_oscillating_beam.gif"
    movie.save(path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"  wrote {path} ({len(frames)} frames at {fps} fps)")


def write_series(solver, snapshots, out: Path) -> None:
    """One ``.vtu`` per frame plus the ``.pvd`` that gives ParaView a clock."""
    from scion.io.particles import write_particles_vtu
    from scion.io.time_series import TimeSeriesWriter

    clamped = solver.clamped.astype(np.int8)
    directory = out / "oscillating_beam_vtk"
    series = TimeSeriesWriter(directory, "particles", suffix=".vtu")
    for t, x, u, v in snapshots:
        series.add(t, write_particles_vtu, x,
                   point_data={"displacement": u,
                               "magnitude": np.linalg.norm(u, axis=1),
                               "velocity": v, "clamped": clamped})
    print(f"  wrote {len(snapshots)} frames to {directory}/particles.pvd")
    write_reference(directory / "reference.vtu")


def write_reference(path: Path) -> Path:
    """The undeformed outline -- the base block and the beam -- written once.

    This is the beam's counterpart of the dam break's tank file, and it is
    the one piece a total-Lagrangian run cannot show on its own: the
    reference configuration never appears in the series, because every
    frame carries the *deformed* positions. Without it there is nothing in
    the view to measure the deflection against.

    Note the clamp is **not** this outline. The held particles are the
    collar of the base block minus the beam's own cross-section, so the
    beam flexes inside the insert -- which is why the measured period sits
    about a percent above Euler-Bernoulli. ``clamped`` in the point data is
    what shows that; threshold on it in ParaView.
    """
    import meshio

    base = np.array([[-INSERT - BASE_WIDTH, -0.5 * THICKNESS - BASE_WIDTH],
                     [0.0, -0.5 * THICKNESS - BASE_WIDTH],
                     [0.0, 0.5 * THICKNESS + BASE_WIDTH],
                     [-INSERT - BASE_WIDTH, 0.5 * THICKNESS + BASE_WIDTH]])
    beam = np.array([[-INSERT, -0.5 * THICKNESS], [LENGTH, -0.5 * THICKNESS],
                     [LENGTH, 0.5 * THICKNESS], [-INSERT, 0.5 * THICKNESS]])

    points, cells = [], []
    for loop in (base, beam):
        start = len(points)
        points.extend(loop)
        n = len(loop)
        cells.extend([[start + i, start + (i + 1) % n] for i in range(n)])

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.zeros((len(points), 3))
    xyz[:, :2] = np.asarray(points)
    meshio.Mesh(points=xyz,
                cells=[meshio.CellBlock("line", np.asarray(cells))]).write(str(path))
    print(f"  wrote {path} (the undeformed base and beam)")
    return path


def write_csv(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("time,tip_dy,kinetic_energy,strain_energy\n")
        for row in history:
            handle.write(",".join(f"{v:.17e}" for v in row) + "\n")
    print(f"  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--end", type=float, default=END_TIME)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--spacing", type=float, default=SPACING)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="directory for the figure")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the geometry-and-particles figure")
    parser.add_argument("--animate", action="store_true",
                        help="write a GIF of the ringing beam")
    parser.add_argument("--vtk", action="store_true",
                        help="write a .vtu series and its .pvd for ParaView")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("SPH oscillating beam (SPHinXsys example 13)")
    snapshots = [] if (args.animate or args.vtk) else None
    solver, history = run(args.end, args.interval, args.spacing,
                          snapshots=snapshots)
    report(solver, history)
    if not args.no_setup:
        setup_figure(args.results, args.spacing)
    if not args.no_figure:
        figure(solver, history, args.results)
    if args.animate:
        animate(solver, snapshots, args.results)
    if args.vtk:
        write_series(solver, snapshots, args.results)
    if args.csv is not None:
        write_csv(history, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
