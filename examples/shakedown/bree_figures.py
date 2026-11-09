#!/usr/bin/env python3
r"""
Figures for the probabilistic Bree example, written straight into the paper.

    python -m examples.shakedown.bree_figures --out _incoming/figures
    python -m examples.shakedown.bree_figures --out _incoming/figures --quick

Generated, not drawn by hand, for the same reason the configurational-forces
tables are: a figure that is produced by the code cannot drift from it. Each
PDF carries a provenance line in its metadata saying which point count and
sample size produced it, and ``--quick`` shortens everything and says so in
the figure itself -- do not paste a ``--quick`` figure into the manuscript.

Colour
------
Four regimes, so a categorical palette in fixed order, validated for colour
vision deficiency rather than chosen by eye: worst adjacent pair separates by
dE 9.2 under deuteranopia and 9.6 under tritanopia, with a normal-vision floor
of 24.0. The aqua sits below 3:1 against white, so the stacked bands are
*directly labelled* as well as keyed -- identity is never carried by colour
alone. The regime map uses a single-hue sequential ramp, light to dark, since
there the quantity is a magnitude and not an identity.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scion.shakedown import rsdm_strip, transition_loads
from scion.uq.density import (band_probability, exceedance_probability,
                              gf_points, mixed_density,
                              reflected_density)

from examples.shakedown.probabilistic_bree import (
    RAYS,
    crude_points,
    distribution,
    monte_carlo,
    ray_coordinates,
    ray_means,
    regime_curves,
    scrambled_sobol,
)

# validated categorical palette, in regime order
REGIME_COLOUR = {"E": "#2a78d6", "S": "#1baf7a", "AP": "#eb6834",
                 "R": "#6250d6"}
REGIME_LABEL = {"E": "elastic", "S": "shakedown",
                "AP": "alternating plasticity", "R": "ratcheting"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": INK_2, "axes.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.labelcolor": INK, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

COL1, COL2 = 3.4, 7.0            # single- and double-column widths, inches


def _finish(ax, grid="y"):
    ax.grid(axis=grid, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path, name: str, provenance: str) -> Path:
    path = out / f"{name}.pdf"
    fig.savefig(path, metadata={"Subject": provenance})
    plt.close(fig)
    print(f"  {path}")
    return path


# ---------------------------------------------------------------------------
# Fig. 1 -- the deterministic diagram
# ---------------------------------------------------------------------------

def figure_bree_diagram(out: Path, quick: bool) -> None:
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.82))
    x = np.linspace(0.0, 1.0, 400)
    y_top = 4.2

    # regions, painted by classifying the plane rather than by hand-drawn
    # polygons -- so the picture cannot disagree with `transition_loads`
    gx, gy = np.meshgrid(np.linspace(0, 1.0, 600), np.linspace(0, y_top, 600))
    loads = transition_loads(gx.ravel(), gy.ravel())
    code = np.zeros(gx.size)
    for i, key in enumerate(("E", "S", "AP", "R")):
        name = {"E": "elastic", "S": "shakedown", "AP": "alternating",
                "R": "ratchetting"}[key]
        code[loads.regime_at(1.0) == name] = i
    ax.contourf(gx, gy, code.reshape(gx.shape), levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=[REGIME_COLOUR[k] for k in ("E", "S", "AP", "R")],
                alpha=0.16, zorder=1)

    ax.plot(x, np.maximum(1.0 - x, 0), color=INK, lw=1.0, zorder=3)
    ax.plot(x, np.minimum(2.0, np.maximum(4.0 * (1.0 - x), 0)), color=INK,
            lw=1.0, zorder=3)
    # only for x <= 0.5: past the corner R1 governs and x y = 1 is not a
    # boundary at all, so drawing it there would show a line the regime map
    # does not have
    xa = np.linspace(1.0 / y_top, 0.5, 200)
    ax.plot(xa, 1.0 / xa, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=3)

    for name, (x_hat, y_hat) in RAYS.items():
        t = np.linspace(0.0, 1.0, 2)
        scale = min(1.0 / x_hat, y_top / y_hat)
        ax.plot(t * scale * x_hat, t * scale * y_hat, color=INK_2, lw=0.9,
                ls=(0, (1, 1.5)), zorder=4)
        loads_ray = transition_loads(x_hat, y_hat)
        for lam, marker in ((float(loads_ray.elastic), "o"),
                            (float(loads_ray.shakedown), "s"),
                            (float(loads_ray.ratchet), "^")):
            if np.isfinite(lam) and lam * y_hat <= y_top:
                ax.plot(lam * x_hat, lam * y_hat, marker, ms=3.4, color=INK,
                        mfc="white", mew=0.9, zorder=5)
        ax.annotate(f"ray {name}", (0.82 * scale * x_hat, 0.82 * scale * y_hat),
                    textcoords="offset points", xytext=(5, -9),
                    fontsize=7.5, color=INK_2, zorder=6)

    for key, (tx, ty, text) in (
            ("E", (0.17, 0.16, "elastic")),
            ("S", (0.30, 1.30, "shakedown")),
            ("AP", (0.16, 2.55, "alternating\nplasticity")),
            ("R", (0.74, 3.45, "ratcheting"))):
        ax.text(tx, ty, text, fontsize=7.5, color=INK_2, ha="center",
                va="center", linespacing=1.3, zorder=6)

    ax.set_xlabel(r"$x = \sigma_P/\sigma_y$")
    ax.set_ylabel(r"$y = \sigma_T/\sigma_y$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, y_top)
    ax.legend(handles=[
        Line2D([], [], color=INK, lw=1.0, label="shakedown boundary"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2)),
               label=r"$xy=1$ (AP$\to$R)"),
        Line2D([], [], color=INK, marker="o", mfc="white", ls="none", ms=3.4,
               label=r"$\lambda_{\rm el}$"),
        Line2D([], [], color=INK, marker="s", mfc="white", ls="none", ms=3.4,
               label=r"$\lambda_{\rm sd}$"),
        Line2D([], [], color=INK, marker="^", mfc="white", ls="none", ms=3.4,
               label=r"$\lambda_{\rm r}$")],
        loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
        handlelength=1.6, labelspacing=0.3, columnspacing=1.4)
    _finish(ax, grid="both")
    if quick:
        ax.set_title("QUICK RUN", color="#eb6834", fontsize=7)
    _save(fig, out, "bree_diagram",
          "deterministic Bree diagram at mean properties")


# ---------------------------------------------------------------------------
# Fig. 2 -- regime probabilities against the reference
# ---------------------------------------------------------------------------

def figure_regime_probabilities(out: Path, quick: bool, n_points: int,
                                n_mc: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.34), sharey=True)

    for ax, (name, (x_hat, y_hat)) in zip(axes, RAYS.items()):
        dist = distribution(*ray_means(x_hat, y_hat))
        nominal = transition_loads(x_hat, y_hat)
        top = (2.2 * float(nominal.ratchet) if np.isfinite(nominal.ratchet)
               else 3.5 * float(nominal.shakedown))
        lam = np.linspace(0.2, top, 241)

        reference, _, _ = monte_carlo(dist, n_mc, lam)
        curves, _ = regime_curves(gf_points(dist, n_points), lam)

        bottom = np.zeros_like(lam)
        for key in ("E", "S", "AP", "R"):
            ax.fill_between(lam, bottom, bottom + curves[key],
                            facecolor=REGIME_COLOUR[key], alpha=0.75,
                            lw=0.6, edgecolor="white", zorder=2)
            band = curves[key]
            # direct label where the band is thick enough to carry one, so
            # identity does not rest on colour alone
            if band.max() > 0.22:
                j = int(np.argmax(band))
                ax.text(lam[j], bottom[j] + 0.5 * band[j], key, ha="center",
                        va="center", fontsize=7.5, color="white", zorder=4)
            bottom = bottom + band

        cumulative = np.zeros_like(lam)
        for key in ("E", "S", "AP"):
            cumulative = cumulative + reference[key]
            ax.plot(lam, cumulative, color=INK, lw=0.9, zorder=3)

        ax.set_xlim(lam[0], lam[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"load factor $\lambda$")
        ax.set_title(f"ray {name}  ($\\hat y/\\hat x = "
                     f"{y_hat / x_hat:.0f}$)", color=INK_2)
        _finish(ax, grid="y")

    axes[0].set_ylabel("regime probability")
    axes[0].legend(handles=[Patch(facecolor=REGIME_COLOUR[k], alpha=0.75,
                                  label=REGIME_LABEL[k])
                            for k in ("E", "S", "AP", "R")]
                   + [Line2D([], [], color=INK, lw=0.9,
                             label=f"reference ({n_mc:,} MC)")],
                   loc="upper center", bbox_to_anchor=(1.05, -0.26), ncol=5,
                   handlelength=1.4, columnspacing=1.2)
    if quick:
        axes[0].text(0.02, 1.04, "QUICK RUN", transform=axes[0].transAxes,
                     color="#eb6834", fontsize=7)
    _save(fig, out, "bree_regime_probabilities",
          f"DPIM-H with N={n_points} GF points against {n_mc} Monte Carlo "
          f"samples through the same closed form")


# ---------------------------------------------------------------------------
# Fig. 3 -- the density of the shakedown factor, and the bandwidth
# ---------------------------------------------------------------------------

def _shakedown_factors(points):
    x_hat, y_hat = ray_coordinates(points.z)
    return transition_loads(x_hat, y_hat).shakedown


def figure_shakedown_density(out: Path, quick: bool, n_points: int,
                             n_mc: int) -> None:
    from scion.uq.density import silverman_bandwidth, solve_dpim

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.34))
    shades = {0.5: "#9ec4ef", 1.0: "#2a78d6", 2.0: "#16437a"}

    for ax, (name, (x_hat, y_hat)) in zip(axes, RAYS.items()):
        dist = distribution(*ray_means(x_hat, y_hat))
        rng = np.random.default_rng(3)
        reference = transition_loads(*ray_coordinates(
            dist.sample(n_mc, rng=rng))).shakedown

        ax.hist(reference, bins=120, density=True, histtype="stepfilled",
                facecolor=GRID, edgecolor="none", zorder=1,
                label=f"reference ({n_mc:,} MC)")

        points = gf_points(dist, n_points)
        values = _shakedown_factors(points)
        # a Silverman-family width, not the spacing rule: these responses come
        # from a nonlinear map of five inputs and land like a sample, so their
        # spacing falls like N^-1/2 while the right bandwidth falls like
        # N^-1/5. `solve_dpim` warns about exactly this if left to default.
        for scale, colour in shades.items():
            density = solve_dpim(
                points, values, n_cells=400, margin=0.25,
                sigma=silverman_bandwidth(points, values, factor=scale))
            ax.plot(density.x, density.pdf[0], color=colour, lw=1.3,
                    zorder=3, label=rf"DPIM-S, $h\times{scale:g}$")

        ax.set_xlim(values.min() * 0.75, values.max() * 1.25)
        ax.set_xlabel(r"shakedown factor $\lambda_{\rm sd}$")
        ax.set_title(f"ray {name}", color=INK_2)
        _finish(ax, grid="y")

    axes[0].set_ylabel("probability density")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.26), ncol=4,
                   handlelength=1.6, columnspacing=1.4)
    if quick:
        axes[0].text(0.02, 1.04, "QUICK RUN", transform=axes[0].transAxes,
                     color="#eb6834", fontsize=7)
    _save(fig, out, "bree_shakedown_density",
          f"DPIM-S at N={n_points}, bandwidth scaled by 0.5, 1 and 2, against "
          f"{n_mc} Monte Carlo samples")


# ---------------------------------------------------------------------------
# Fig. 5 -- error against point count
# ---------------------------------------------------------------------------

def figure_error_vs_n(out: Path, quick: bool, counts, n_mc: int,
                      n_seeds: int = 5) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.34))
    designs = {"gf": ("#2a78d6", "GF-discrepancy", gf_points),
               "sobol": ("#eb6834", "scrambled Sobol'", scrambled_sobol),
               "crude": (INK_2, "crude Monte Carlo", crude_points)}

    for ax, (name, (x_hat, y_hat)) in zip(axes, RAYS.items()):
        dist = distribution(*ray_means(x_hat, y_hat))
        nominal = transition_loads(x_hat, y_hat)
        top = (2.2 * float(nominal.ratchet) if np.isfinite(nominal.ratchet)
               else 3.5 * float(nominal.shakedown))
        lam = np.linspace(0.2, top, 241)
        reference, _, _ = monte_carlo(dist, n_mc, lam)
        f_ref = reference["AP"] + reference["R"]

        for key, (colour, label, build) in designs.items():
            mean, spread = [], []
            for n in counts:
                gaps = []
                for seed in range(n_seeds):
                    curves, _ = regime_curves(build(dist, n, seed=seed), lam)
                    gaps.append(float(np.abs(
                        curves["AP"] + curves["R"] - f_ref).max()))
                mean.append(np.mean(gaps))
                spread.append(np.std(gaps))
            ax.errorbar(counts, mean, yerr=spread, color=colour, lw=1.2,
                        marker="o", ms=3.5, capsize=2.5, elinewidth=0.8,
                        label=label, zorder=3)

        ref_rate = np.array(mean, dtype=float)[0] * (
            np.array(counts, dtype=float) / counts[0]) ** -0.5
        ax.plot(counts, ref_rate, color=INK, lw=0.8, ls=(0, (4, 2)),
                zorder=2, label=r"$N^{-1/2}$")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("representative points $N$")
        ax.set_title(f"ray {name}", color=INK_2)
        _finish(ax, grid="both")

    axes[0].set_ylabel(r"$\max_\lambda |F_{\rm sd} - F_{\rm sd}^{\rm ref}|$")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.26), ncol=4,
                   handlelength=1.8, columnspacing=1.4)
    if quick:
        axes[0].text(0.02, 1.04, "QUICK RUN", transform=axes[0].transAxes,
                     color="#eb6834", fontsize=7)
    _save(fig, out, "bree_error_vs_n",
          f"mean and standard deviation over {n_seeds} seeds, against "
          f"{n_mc} Monte Carlo samples")


# ---------------------------------------------------------------------------
# Fig. 4 -- the mixed distribution, atom and all
# ---------------------------------------------------------------------------

def figure_mixed_distribution(out: Path, quick: bool, n_points: int,
                              eta: float = 1.0e-6) -> None:
    from scion.uq.density import silverman_bandwidth

    x_hat, y_hat = RAYS["A"]
    dist = distribution(*ray_means(x_hat, y_hat))
    points = gf_points(dist, n_points)
    factors = _shakedown_factors(points)
    lam = 1.1 * float(points.probability @ factors)

    rays_x, rays_y = ray_coordinates(points.z)
    amplitude = np.array([
        rsdm_strip(rays_x[q], rays_y[q], lam, n_thickness=41, n_time=16,
                   max_iter=2000, tol=1e-10).cyclic_amplitude
        for q in range(points.n_points)])

    split = mixed_density(points, amplitude, bound=0.0, tol=eta)
    atom, grid = split.atom, split.grid

    fig, ax = plt.subplots(figsize=(COL1 * 1.45, COL1 * 0.82))

    # the comparator: one kernel over every point, atom included, which is
    # what smoothing the whole sum amounts to
    h = silverman_bandwidth(points, amplitude)
    naive = reflected_density(grid, amplitude, points.probability, h)
    ax.plot(grid, naive, color=INK_2, lw=1.2, ls=(0, (4, 2)), zorder=3,
            label="naive: smoothed over all points")

    positive = split.density
    ax.plot(grid, positive, color="#2a78d6", lw=1.4, zorder=4,
            label=r"split: continuous part $p_A^+$")

    top = max(float(naive.max()), float(positive.max())) * 1.22
    ax.vlines(0.0, 0.0, top * 0.92, color="#eb6834", lw=2.2, zorder=5)
    ax.plot([0.0], [top * 0.92], marker="o", ms=4.5, color="#eb6834",
            zorder=5)
    ax.annotate(rf"atom $\pi_0 = {atom:.3f}$" "\n(probability, not density)",
                (0.0, top * 0.92), textcoords="offset points",
                xytext=(9, -10), fontsize=8, color="#eb6834",
                linespacing=1.3)

    ax.set_xlim(-0.02 * grid[-1], grid[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel(r"cyclic amplitude $A$")
    ax.set_ylabel("probability density")
    ax.set_title(rf"ray A at $\lambda = 1.1\,\mathbb{{E}}[\lambda_{{\rm sd}}]"
                 rf" = {lam:.3f}$", color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              handlelength=1.8, columnspacing=1.4)
    _finish(ax, grid="y")
    if quick:
        ax.text(0.02, 1.04, "QUICK RUN", transform=ax.transAxes,
                color="#eb6834", fontsize=7)
    _save(fig, out, "bree_mixed_distribution",
          f"A from RSDM on the strip at N={n_points} representative points, "
          f"atom {atom:.4f}")


# ---------------------------------------------------------------------------
# Fig. 6 -- the probabilistic map
# ---------------------------------------------------------------------------

def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return np.interp(q, cdf, values[order])


def figure_probabilistic_map(out: Path, quick: bool, n_points: int,
                             n_rays: int = 15) -> None:
    """Iso-probability contours over the Bree plane.

    One point set serves every ray. Both ``x_hat`` and ``y_hat`` are
    *linear* in the means of ``sigma_P`` and ``DT`` -- each is that mean times
    a fixed random factor -- so a ray at angle ``theta`` is the reference set
    scaled by ``cos theta`` and ``sin theta``. Exact, and it turns 15
    five-dimensional GF constructions into one.
    """
    reference = gf_points(distribution(*ray_means(1.0, 1.0)), n_points)
    x_ref, y_ref = ray_coordinates(reference.z)
    weights = reference.probability

    angles = np.deg2rad(np.linspace(5.0, 85.0, n_rays))
    levels = (1.0e-3, 0.5, 0.999)
    shades = {1.0e-3: "#9ec4ef", 0.5: "#2a78d6", 0.999: "#16437a"}

    fig, ax = plt.subplots(figsize=(COL1 * 1.35, COL1 * 0.95))
    y_top = 4.2
    gx, gy = np.meshgrid(np.linspace(0, 1.0, 500), np.linspace(0, y_top, 500))
    code = np.zeros(gx.size)
    for i, name in enumerate(("elastic", "shakedown", "alternating",
                              "ratchetting")):
        code[transition_loads(gx.ravel(), gy.ravel()).regime_at(1.0) == name] = i
    ax.contourf(gx, gy, code.reshape(gx.shape),
                levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=[REGIME_COLOUR[k] for k in ("E", "S", "AP", "R")],
                alpha=0.13, zorder=1)
    xs = np.linspace(0.0, 1.0, 300)
    ax.plot(xs, np.minimum(2.0, np.maximum(4.0 * (1.0 - xs), 0)), color=INK,
            lw=0.9, zorder=3, label="deterministic boundary")

    for level in levels:
        pts = []
        for theta in angles:
            cx, cy = np.cos(theta), np.sin(theta)
            factors = transition_loads(cx * x_ref, cy * y_ref).shakedown
            lam = _weighted_quantile(factors, weights, level)
            pts.append((lam * cx, lam * cy))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=shades[level], lw=1.4,
                marker="o", ms=2.6, zorder=4,
                label=rf"$\Pr[\lambda_{{\rm sd}}<\lambda] = {level:g}$")

    ax.set_xlabel(r"$x = \sigma_P/\sigma_y$")
    ax.set_ylabel(r"$y = \sigma_T/\sigma_y$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              handlelength=1.8, columnspacing=1.2)
    _finish(ax, grid="both")
    if quick:
        ax.text(0.02, 1.04, "QUICK RUN", transform=ax.transAxes,
                color="#eb6834", fontsize=7)
    _save(fig, out, "bree_probabilistic_map",
          f"iso-probability contours from DPIM-H at N={n_points}, "
          f"{n_rays} rays")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="_incoming/figures")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out.resolve()}")
    t0 = time.time()
    n_points = 64 if args.quick else 256
    n_mc = 50_000 if args.quick else 1_000_000

    figure_bree_diagram(out, args.quick)
    figure_regime_probabilities(out, args.quick, n_points, n_mc)
    figure_shakedown_density(out, args.quick, n_points, n_mc)
    counts = (64, 256) if args.quick else (64, 128, 256, 512, 1024)
    figure_error_vs_n(out, args.quick, counts, n_mc,
                      n_seeds=2 if args.quick else 5)
    figure_mixed_distribution(out, args.quick, n_points)
    figure_probabilistic_map(out, args.quick, 256 if args.quick else 1024,
                             n_rays=7 if args.quick else 15)
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
