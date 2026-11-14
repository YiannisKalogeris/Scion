#!/usr/bin/env python3
r"""
Figures for Example 2, the plate with a hole and a random yield field.

    python -m examples.shakedown.plate_figures --out _incoming/figures
    python -m examples.shakedown.plate_figures --out _incoming/figures --quick

Reads ``results/plate.npz`` (or ``plate_quick.npz``), which
:mod:`examples.shakedown.probabilistic_plate` writes. Generated rather than
drawn, for the same reason as the Bree figures: a figure produced by the code
cannot drift from it, and each PDF carries a provenance line in its metadata.

Colour
------
The regime curves reuse the categorical palette of the Bree figures in the
same fixed order, so a regime keeps its colour between the two examples --
colour follows the entity, never its rank, and a reader who has learnt the
key in Fig. 2 does not relearn it here.

The field and activity maps are *magnitudes*, so they take a single-hue
sequential ramp, light to dark, and not the categorical set. The activity map
additionally reaches zero over most of the plate, which a diverging map would
misrepresent as a midpoint.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.tri import Triangulation

from scion.uq.density import RepresentativePoints

from examples.shakedown.bree_figures import (COL1, COL2, GRID, INK, INK_2,
                                             REGIME_COLOUR, REGIME_LABEL,
                                             _finish, _save)
from examples.shakedown.plate_model import L_HALF, R_HOLE, quarter_plate

SEQUENTIAL = "viridis"
RESULTS = Path(__file__).parent / "results"


@lru_cache(maxsize=4)
def _mesh(mesh: int):
    """The quarter-plate mesh and the operators that read a field off it."""
    from scion.discretization.fem.tables import QuadratureTables

    domain = quarter_plate(mesh, mesh)
    return domain, QuadratureTables(domain, thickness=1.0)


def _field(mesh: int, xq: np.ndarray, values: np.ndarray):
    """A quadrature field as a triangulation over the plate's own boundary.

    ``tricontourf`` on the quadrature points alone paints their convex
    *hull*, and with a mesh graded towards the hole that hull is not the
    plate: the inset is half an element, 0.6% of L next to the hole against
    4.8% at the outer edges, so the hull leans from one to the other --
    measured, its left edge runs from (0.006, 0.110) to (0.054, 0.952) --
    and the outer corner is cut by a chord between (0.866, 0.951) and
    (0.951, 0.866). The drawn domain then disagrees with the mesh in
    Fig. 7(a) and stops 5% short of the two loaded edges.

    The fix is to add the mesh *nodes* as data sites, which puts the corners
    and the hole boundary on the hull, and to carry the quadrature values
    themselves unchanged -- recovering onto the nodes alone would be the
    standard thing and loses the peak the map exists to show, since the
    recovery is an average: the activity map's maximum falls from 1.000 to
    0.976, i.e. from "yields in every realization" to not.

    A node takes the weighted average of the quadrature values around it,
    ``u_I = sum_q w_q N_qI v_q / sum_q w_q N_qI``, the lumped form of the L2
    projection. It is a convex combination, so a node cannot leave the range
    of the data -- the consistent projection can, and would put the activity
    map outside [0, 1] where its front is sharp, which a fixed set of contour
    levels then leaves unpainted. Triangles across the hole are masked by
    their centroid, as before; nothing has to be done at the outer boundary.
    """
    domain, tables = _mesh(mesh)
    nodes = domain.coordinates
    # the archive carries the points but not the mesh they came from; this is
    # the check that the two are the same object
    here = np.stack([tables.interpolate(nodes[:, 0]),
                     tables.interpolate(nodes[:, 1])], axis=1)
    if here.shape != xq.shape or not np.allclose(here, xq, rtol=0, atol=1e-12):
        raise SystemExit("the archive's quadrature points are not this mesh's")

    v = np.asarray(values, dtype=float).reshape(tables.w.shape)
    n_nodes = nodes.shape[0]
    nodal = tables.load(v, n_nodes) / tables.load(np.ones_like(v), n_nodes)

    points = np.vstack([nodes, np.asarray(xq, dtype=float)])
    tri = Triangulation(points[:, 0], points[:, 1])
    centroids = points[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(centroids[:, 0], centroids[:, 1]) < R_HOLE)
    return tri, np.concatenate([nodal, v.ravel()])


def _load(quick: bool) -> dict:
    path = RESULTS / ("plate_quick.npz" if quick else "plate.npz")
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run\n"
            f"    python -m examples.shakedown.probabilistic_plate"
            f"{' --quick' if quick else ''}\nfirst.")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _tags(archive: dict):
    """The correlation lengths present, in increasing order."""
    out = []
    for key in archive:
        if key.endswith("_modes"):
            out.append(key[:-len("_modes")])
    return sorted(out, key=lambda t: float(t[2:]))


def _counts(archive: dict, tag: str):
    seen = set()
    for key in archive:
        if key.startswith(f"{tag}_n") and key.endswith("_sd"):
            seen.add(int(key[len(tag) + 2:-3]))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Fig. 7 -- the model and two realizations of the field
# ---------------------------------------------------------------------------

def figure_model(out: Path, archive: dict, quick: bool) -> None:
    from scion.uq.forward import KarhunenLoeve, LogNormalField
    from examples.shakedown.plate_model import YIELD
    from examples.shakedown.probabilistic_plate import YIELD_COV

    xq, w = archive["xq"], archive["weights"]
    mesh = int(archive["mesh"])
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35))

    # (a) the mesh and the two tractions
    ax = axes[0]
    domain = quarter_plate(mesh, mesh)
    nodes = domain.coordinates
    for block in domain.mesh.element_blocks:
        for element in block.connectivity:
            loop = np.append(element, element[0])
            ax.plot(nodes[loop, 0], nodes[loop, 1], color=INK_2, lw=0.35,
                    zorder=2)
    for y in np.linspace(0.08, 0.92, 6):
        ax.annotate("", xy=(1.17, y), xytext=(1.02, y),
                    arrowprops=dict(arrowstyle="->", color=REGIME_COLOUR["E"],
                                    lw=0.9))
    for x in np.linspace(0.08, 0.92, 6):
        ax.annotate("", xy=(x, 1.17), xytext=(x, 1.02),
                    arrowprops=dict(arrowstyle="->", color=REGIME_COLOUR["AP"],
                                    lw=0.9))
    ax.text(1.22, 0.5, r"$\lambda p_1$", color=REGIME_COLOUR["E"],
            rotation=90, va="center", ha="left", fontsize=8)
    ax.text(0.5, 1.22, r"$\lambda p_2$", color=REGIME_COLOUR["AP"],
            va="bottom", ha="center", fontsize=8)
    ax.add_patch(plt.Circle((0, 0), R_HOLE, facecolor="white", zorder=3,
                            edgecolor=INK, lw=0.8))
    ax.plot([0, L_HALF, L_HALF], [L_HALF, L_HALF, 0], color=INK, lw=0.9,
            zorder=4)
    ax.set_xlim(-0.08, 1.45); ax.set_ylim(-0.08, 1.45)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"(a) quarter model, {int(archive['points'])} points",
                 fontsize=8)

    # (b), (c) two realizations at the shortest and longest correlation length
    tags = _tags(archive)
    show = [tags[0], tags[-1]] if len(tags) > 1 else [tags[0], tags[0]]
    rng = np.random.default_rng(3)
    for ax, tag, label in zip(axes[1:], show, "bc"):
        ell = float(tag[2:])
        kl = KarhunenLoeve.from_points(xq, w, ell, n_modes=int(archive[f"{tag}_modes"]))
        field = LogNormalField(kl, mean=YIELD, cov=YIELD_COV)
        values = field.realize(rng.standard_normal(kl.n_modes))
        tri, nodal = _field(mesh, xq, values)
        tp = ax.tricontourf(tri, nodal, levels=14, cmap=SEQUENTIAL)
        ax.add_patch(plt.Circle((0, 0), R_HOLE, facecolor="white", zorder=3,
                                edgecolor=INK_2, lw=0.6))
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({label}) $\\sigma_y$, $\\ell_c/L = {ell:g}$, "
                     f"{kl.n_modes} modes", fontsize=8)
        bar = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.02)
        bar.ax.tick_params(labelsize=7, color=INK_2)
        bar.outline.set_edgecolor(INK_2); bar.outline.set_linewidth(0.5)
        bar.set_label("MPa", fontsize=7)

    fig.tight_layout()
    _save(fig, out, "plate_model",
          f"mesh {mesh}x{mesh}, {int(archive['points'])} quadrature points"
          + (" -- QUICK RUN" if quick else ""))


# ---------------------------------------------------------------------------
# Fig. 8 -- regime probabilities against load factor
# ---------------------------------------------------------------------------

def figure_regimes(out: Path, archive: dict, quick: bool) -> None:
    tags = _tags(archive)
    fig, axes = plt.subplots(1, len(tags), figsize=(COL2, 2.5), sharey=True,
                             squeeze=False)
    bands = ("E", "S", "AP")
    label = {"E": REGIME_LABEL["E"], "S": REGIME_LABEL["S"],
             "AP": "not shaking down"}
    for ax, tag in zip(axes[0], tags):
        lam = archive[f"{tag}_lam"]
        n = _counts(archive, tag)[-1]
        # the third band is the complement: lambda_rr is not computed per
        # realization here, so alternating plasticity and ratcheting are
        # reported together rather than split on an instrument this example
        # does not run. See the module docstring of probabilistic_plate.
        dpim = {"E": archive[f"{tag}_n{n}_E"], "S": archive[f"{tag}_n{n}_S"]}
        dpim["AP"] = 1.0 - dpim["E"] - dpim["S"]
        ref_n = int(archive[f"{tag}_ref_sd"].size)
        ref = {"E": archive[f"{tag}_mc_E"], "S": archive[f"{tag}_mc_S"]}
        ref["AP"] = 1.0 - ref["E"] - ref["S"]
        for key in bands:
            ax.plot(lam, ref[key], color=REGIME_COLOUR[key], lw=2.0,
                    alpha=0.28, zorder=2, solid_capstyle="round")
            ax.plot(lam, dpim[key], color=REGIME_COLOUR[key], lw=1.3,
                    zorder=3, solid_capstyle="round")
        ax.set_xlabel(r"load factor $\lambda$")
        ax.set_title(f"$\\ell_c/L = {float(tag[2:]):g}$", fontsize=8)
        ax.set_xlim(lam[0], lam[-1]); ax.set_ylim(-0.03, 1.03)
        _finish(ax)
    axes[0][0].set_ylabel("probability")
    handles = [Patch(facecolor=REGIME_COLOUR[k], label=label[k]) for k in bands]
    handles.append(plt.Line2D([], [], color=INK_2, lw=1.3,
                              label=f"DPIM-H, $N = {n}$"))
    handles.append(plt.Line2D([], [], color=INK_2, lw=2.0, alpha=0.28,
                              label=f"Monte Carlo, $n = {ref_n}$"))
    # one row beneath the panels: in the right-hand panel it sat on the curves
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, out, "plate_regime_probabilities",
          f"DPIM-H at N = {n} against the reference"
          + (" -- QUICK RUN" if quick else ""))


# ---------------------------------------------------------------------------
# Fig. 9 -- the density of the shakedown factor
# ---------------------------------------------------------------------------

def figure_density(out: Path, archive: dict, quick: bool) -> None:
    from scion.uq.density import silverman_bandwidth

    tags = _tags(archive)
    fig, axes = plt.subplots(1, len(tags), figsize=(COL2, 2.4), sharey=True,
                             squeeze=False)
    for ax, tag in zip(axes[0], tags):
        n = _counts(archive, tag)[-1]
        values = archive[f"{tag}_n{n}_sd"]
        weights = archive[f"{tag}_n{n}_weights"]
        ref = archive[f"{tag}_ref_sd"]
        grid = np.linspace(0.9 * min(values.min(), ref.min()),
                           1.1 * max(values.max(), ref.max()), 400)
        # silverman_bandwidth reads only the assigned probabilities and the
        # response, so the input coordinates are not needed to reconstruct it
        h = silverman_bandwidth(
            RepresentativePoints(np.zeros((weights.size, 1)), weights), values)
        z = (grid[:, None] - values[None, :]) / h
        density = (np.exp(-0.5 * z ** 2) @ weights) / (h * np.sqrt(2 * np.pi))
        ax.hist(ref, bins=24, density=True, color=GRID, edgecolor=INK_2,
                linewidth=0.4, zorder=2,
                label=f"Monte Carlo, $n = {ref.size}$")
        ax.plot(grid, density, color=REGIME_COLOUR["S"], lw=1.4, zorder=3,
                label=f"DPIM-S, $N = {n}$")
        ax.set_xlabel(r"$\lambda_{\mathrm{sd}}$")
        ax.set_title(f"$\\ell_c/L = {float(tag[2:]):g}$", fontsize=8)
        _finish(ax)
    axes[0][0].set_ylabel("density")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, out, "plate_shakedown_density",
          "DPIM-S against a Monte Carlo histogram"
          + (" -- QUICK RUN" if quick else ""))


# ---------------------------------------------------------------------------
# Fig. 10 -- where the plate yields, as a probability
# ---------------------------------------------------------------------------

def figure_activity(out: Path, archive: dict, quick: bool) -> None:
    if "activity" not in archive:
        print("  (no activity map in the archive; skipped)")
        return
    xq = archive["xq"]
    prob = archive["activity"]
    w = archive["weights"]
    lam = float(archive["activity_lambda"])
    area = float(w[prob > 0].sum() / w.sum())
    tri, nodal = _field(int(archive["mesh"]), xq, prob)

    # the plastic zone is a few percent of the area and sits entirely at the
    # hole, so the whole quarter plate is almost all zero -- the second panel
    # is where the figure's content actually is
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.6))
    levels = np.linspace(0.0, 1.0, 11)
    zoom = 4.5 * R_HOLE
    for ax, limit, label in ((axes[0], None, "a"), (axes[1], zoom, "b")):
        tp = ax.tricontourf(tri, nodal, levels=levels,
                            cmap=SEQUENTIAL, vmin=0.0, vmax=1.0)
        ax.add_patch(plt.Circle((0, 0), R_HOLE, facecolor="white", zorder=3,
                                edgecolor=INK_2, lw=0.6))
        if limit is None:
            ax.add_patch(Rectangle((0, 0), zoom, zoom, fill=False,
                                   edgecolor=INK, lw=0.7, zorder=4))
            ax.set_title("(a) the quarter plate", fontsize=8)
        else:
            ax.set_xlim(0, limit); ax.set_ylim(0, limit)
            ax.set_title("(b) at the hole", fontsize=8)
        ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    bar = fig.colorbar(tp, ax=axes, fraction=0.046, pad=0.02)
    bar.ax.tick_params(labelsize=7, color=INK_2)
    bar.outline.set_edgecolor(INK_2); bar.outline.set_linewidth(0.5)
    bar.set_label("probability of plastic activity", fontsize=7)
    # after the colorbar, which re-lays out both axes and had been undoing the
    # zoom set inside the loop
    axes[1].set_xlim(0.0, zoom); axes[1].set_ylim(0.0, zoom)
    fig.suptitle(r"$\lambda = %.3f$, %d realizations, %.1f%% of the area ever "
                 r"active" % (lam, int(archive["activity_counted"]),
                              100.0 * area), fontsize=8, y=1.02)
    _save(fig, out, "plate_activity_map",
          f"plastic activity at lambda = {lam:.4f}"
          + (" -- QUICK RUN" if quick else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("_incoming/figures"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)
    archive = _load(args.quick)
    if args.quick:
        print("  QUICK RUN -- do not paste these into the manuscript")
    figure_model(args.out, archive, args.quick)
    figure_regimes(args.out, archive, args.quick)
    figure_density(args.out, archive, args.quick)
    figure_activity(args.out, archive, args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
