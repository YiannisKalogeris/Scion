r"""Discretisation figures for the two examples.

    python -m examples.micropolar_phase_field.setup_figures

Draws the material points against the background grid for each problem, with
an inset at true scale showing the particles within one cell. These are the
*discretisation* figures; the geometry, loading and boundary conditions are
better drawn as a schematic, and :func:`print_specification` writes out
everything such a drawing needs.

Nothing here solves anything -- the particle layout and the grid are fixed by
``build`` before the first load step, so the figures can be produced in
seconds and are guaranteed to show the discretisation the runs actually used.
"""
from __future__ import annotations

import numpy as np

from . import notched_beam as nb
from . import notched_tension as nt
from ._common import PAD, ZPAD, NZ, RESULTS


def _grid_lines(ax, dx, x0, x1, y0, y1, **kw):
    for x in np.arange(x0, x1 + 0.5 * dx, dx):
        ax.plot([x, x], [y0, y1], **kw)
    for y in np.arange(y0, y1 + 0.5 * dx, dx):
        ax.plot([x0, x1], [y, y], **kw)


def figure(which, ax_main, ax_inset):
    """One panel: particles over the grid, plus a one-cell inset."""
    if which == "tension":
        pf, _, parts, _ = nt.build(0.0, ell_pf=nt.ELL_PF, dx=nt.DX)
        dx, Lx, Ly, title = nt.DX, nt.SIDE, nt.SIDE, "Notched tension"
    else:
        pf, _, parts, _ = nb.build(0.0, ell_pf=nb.ELL_PF, dx=nb.DX)
        dx, Lx, Ly, title = nb.DX, nb.SPAN, nb.DEPTH, "Notched beam"

    x = parts.x[:, :2]
    gx0, gx1 = -PAD * dx, Lx + PAD * dx
    gy0, gy1 = -PAD * dx, Ly + PAD * dx
    _grid_lines(ax_main, dx, gx0, gx1, gy0, gy1, color="0.85", lw=0.3,
                zorder=0)
    ax_main.plot(x[:, 0], x[:, 1], ".", ms=0.6, color="tab:blue", zorder=2)
    ax_main.add_patch(__import__("matplotlib").patches.Rectangle(
        (0, 0), Lx, Ly, fill=False, ec="k", lw=1.2, zorder=3))
    ax_main.set_aspect("equal")
    ax_main.set_xticks([]); ax_main.set_yticks([])
    ax_main.set_title(f"{title}: {parts.n:,} particles, "
                      f"{pf.n_nodes:,} grid nodes\n"
                      rf"$dx = {dx:g}$, {nt.PPC} particles per cell "
                      f"per direction", fontsize=8)

    # inset: four cells at true scale, so the particle spacing is legible
    cx, cy = 0.5 * Lx, 0.5 * Ly
    w = 2.0 * dx
    _grid_lines(ax_inset, dx, cx - w, cx + w, cy - w, cy + w,
                color="0.6", lw=0.8, zorder=0)
    m = (np.abs(x[:, 0] - cx) <= w) & (np.abs(x[:, 1] - cy) <= w)
    ax_inset.plot(x[m, 0], x[m, 1], "o", ms=4, color="tab:blue",
                  mec="k", mew=0.4, zorder=2)
    ax_inset.set_xlim(cx - w, cx + w); ax_inset.set_ylim(cy - w, cy + w)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    ax_inset.set_title(f"detail: {nt.PPC}x{nt.PPC} per cell", fontsize=8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    for which, ratio in (("tension", 1.0), ("beam", 0.45)):
        fig = plt.figure(figsize=(8.4, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.15)
        figure(which, fig.add_subplot(gs[0]), fig.add_subplot(gs[1]))
        p = RESULTS / f"{which}_discretisation.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
