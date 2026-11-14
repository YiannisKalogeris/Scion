r"""
Where the time goes in the SPH diffusion case, NumPy against Rust.

    python -m examples.sph.benchmark
    python -m examples.sph.benchmark --refine 1 2 4 --steps 50

This one calls Rust itself, so there is no separate command to reproduce the
table. To run only the native half, at a given refinement and step count:

    cargo run -p scion-sph --release --example bench -- 4 50

Figures go to ``examples/sph/results`` unless ``--results`` says otherwise.

Two halves are timed separately because they scale differently: building the
neighbour list, the kernel values and the correction matrices happens once,
and the stepping loop is what a longer run pays for. The reference case is
three steps, so it is almost entirely setup -- which is the first thing to
know before asking whether the native path is worth anything here.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from examples.sph.heat_transfer_2d import HEIGHT, LENGTH, D_COEFF, initial_condition
from scion.discretization.sph import DiffusionSolver, IsotropicDiffusivity, lattice

ROOT = Path(__file__).resolve().parents[2]

#: Figures from every example in this folder land here. Created on demand, so
#: a fresh clone has no empty directory and a run that draws nothing leaves
#: none behind.
RESULTS = Path(__file__).resolve().parent / "results"


def time_numpy(refine: int, steps: int) -> dict:
    dp = HEIGHT / (40 * refine)
    x, volumes = lattice([0.0, 0.0], [LENGTH, HEIGHT], dp, periodic=(True, False))
    phi = initial_condition(x)
    start = time.perf_counter()
    solver = DiffusionSolver(x, volumes, IsotropicDiffusivity(D_COEFF), h=1.3 * dp,
                             lower=[0.0, 0.0], upper=[LENGTH, HEIGHT],
                             periodic=(True, False))
    setup = time.perf_counter() - start
    dt = solver.time_step()
    start = time.perf_counter()
    for _ in range(steps):
        phi = solver.step(phi, dt)
    stepping = time.perf_counter() - start
    return dict(particles=len(x), pairs=len(solver.pairs), setup=setup,
                stepping=stepping, per_step=stepping / steps)


def time_rust(refine: int, steps: int) -> dict | None:
    if shutil.which("cargo") is None:
        return None
    result = subprocess.run(
        ["cargo", "run", "-p", "scion-sph", "--release", "--quiet",
         "--example", "bench", "--", str(refine), str(steps)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return None
    fields = dict(re.findall(r"(\w+)=([\w.+-]+)", result.stdout))
    return dict(particles=int(fields["particles"]), pairs=int(fields["pairs"]),
                setup=float(fields["setup"]), stepping=float(fields["stepping"]),
                per_step=float(fields["per_step"]))


def setup_figure(out: Path, refine=(1, 2, 4)) -> None:
    """The one geometry this script has: the same block at three resolutions.

    There is no new case here -- the benchmark times the diffusion example --
    so what the figure can usefully show is the thing being varied. The
    panels are a fixed physical window, so the particles get denser rather
    than the picture getting bigger, which is the point of a refinement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.sph.heat_transfer_2d import HEIGHT, LENGTH, build

    window = 0.08
    fig, axes = plt.subplots(1, len(refine), figsize=(2.5 * len(refine), 2.9),
                             sharex=True, sharey=True)
    for ax, level in zip(np.atleast_1d(axes), refine):
        solver, phi, dp = build(level)
        x = solver.positions
        near = (x[:, 0] < window) & (x[:, 1] < window)
        ax.scatter(x[near, 0], x[near, 1], s=7.0, color="#1f6f8b", linewidths=0)
        ax.plot([0, window], [0, 0], color="0.3", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlim(-0.004, window); ax.set_ylim(-0.004, window)
        ax.set_xlabel("x")
        ax.set_title(f"dp = {dp:g}\n{len(x)} particles, "
                     f"{len(solver.pairs)} pairs", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("y")
    fig.suptitle(f"the diffusion block at {len(refine)} resolutions, "
                 f"same {window:g} x {window:g} window", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_benchmark_setup.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def figure(rows, steps: int, out: Path) -> None:
    """Both halves against particle count, on log-log axes.

    Log-log because what is being read is a *slope*: a straight line says the
    cost is a power of the particle count, and the vertical gap between the
    two lines is the ratio at that size. On linear axes the small cases
    collapse onto the origin and the ratio has to be read off a legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    particles = np.array([row[0]["particles"] for row in rows], dtype=float)
    have_rust = all(row[1] is not None for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for ax, key, label in ((axes[0], "per_step", "per step [ms]"),
                           (axes[1], "setup", "setup [s]")):
        scale = 1e3 if key == "per_step" else 1.0
        ax.loglog(particles, [scale * row[0][key] for row in rows],
                  color="#1f6f8b", lw=1.6, marker="o", ms=4.5, label="NumPy")
        if have_rust:
            ax.loglog(particles, [scale * row[1][key] for row in rows],
                      color="#c0622f", lw=1.6, marker="s", ms=4.5, label="Rust")
        ax.set_xlabel("particles"); ax.set_ylabel(label)
        ax.grid(True, which="both", lw=0.4, color="0.92")
        ax.set_axisbelow(True)

    if have_rust:
        axes[0].legend(frameon=False, fontsize=8)
        # the gain is the vertical gap, so write it where the gap is
        for row in rows:
            gain = row[0]["per_step"] / row[1]["per_step"]
            axes[0].annotate(f"{gain:.1f}x",
                             (row[0]["particles"], 1e3 * row[1]["per_step"]),
                             textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=7, color="#c0622f")
    axes[0].set_title(f"stepping, {steps} steps", fontsize=9)
    axes[1].set_title("neighbour list, kernels and correction", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sph_benchmark.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="directory for the figure")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the geometry-and-particles figure")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print(f"  {args.steps} steps per run\n")
    print(f"  {'particles':>10} {'pairs':>9} | {'setup (s)':>20} | "
          f"{'per step (ms)':>22}")
    print(f"  {'':>10} {'':>9} | {'numpy':>9} {'rust':>9} | "
          f"{'numpy':>9} {'rust':>9} {'gain':>5}")
    rows = []
    for refine in args.refine:
        numpy_run = time_numpy(refine, args.steps)
        rust_run = time_rust(refine, args.steps)
        rows.append((numpy_run, rust_run))
        if rust_run is None:
            print(f"  {numpy_run['particles']:10d} {numpy_run['pairs']:9d} | "
                  f"{numpy_run['setup']:9.3f} {'--':>9} | "
                  f"{1e3 * numpy_run['per_step']:9.3f} {'--':>9} {'--':>5}")
            continue
        assert rust_run["pairs"] == numpy_run["pairs"], "the two found different pairs"
        gain = numpy_run["per_step"] / rust_run["per_step"]
        print(f"  {numpy_run['particles']:10d} {numpy_run['pairs']:9d} | "
              f"{numpy_run['setup']:9.3f} {rust_run['setup']:9.3f} | "
              f"{1e3 * numpy_run['per_step']:9.3f} {1e3 * rust_run['per_step']:9.3f} "
              f"{gain:5.2f}")
    if not args.no_setup:
        setup_figure(args.results, tuple(args.refine))
    if not args.no_figure:
        figure(rows, args.steps, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
