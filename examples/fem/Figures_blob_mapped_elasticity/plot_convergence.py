#!/usr/bin/env python3
"""
Mesh-convergence figure for the pullback finite element solution.

Standalone: edit the data block below and run. Nothing is imported from the
solver, so the figure can be regenerated without rerunning any analysis.

    python plot_convergence.py

What the figure is meant to show
--------------------------------
The pullback formulation is an exact reformulation of the boundary value
problem, so its error splits into a finite element part that converges under
mesh refinement and a mapping part that does not. On a log-log plot that
appears as an initial decrease followed by a plateau: once the discretisation
error drops below the error of the learned diffeomorphism, refining the
reference mesh buys nothing. The O(h^2) guide line makes the departure from
optimal convergence visible, and the plateau level is the mapping error.

Reporting the plateau is the honest way to present this. A single error at one
mesh size invites the question of what happens under refinement; the curve
answers it, and turns a limitation into a measured quantity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ====================================================================
# DATA  --  edit here
# ====================================================================
# Elements per side of the structured reference mesh, and the relative
# L2 error of the pullback solution measured against the reference solution.
N_SQUARE = np.array([10, 40, 80, 120, 160, 200])
REL_L2 = np.array([1.48e-1, 5.96e-2, 2.07e-2, 1.57e-2, 1.41e-2, 1.56e-2])

LABEL = "Pullback FEM"
TITLE = "Mesh convergence of the pullback solution"

#: Optional second series for comparison, e.g. the direct FEM at matched cost.
#: Set to None to omit.
N_SQUARE_2 = None
REL_L2_2 = None
LABEL_2 = "Direct FEM"

#: Number of trailing points averaged to estimate the plateau. Set to 0 to
#: omit the plateau line -- do that if the curve has not flattened.
N_PLATEAU = 3

#: Reference domain side length, used only to convert N to a mesh size h.
DOMAIN_SIDE = 2.0

OUTDIR = Path(__file__).resolve().parent / "Figures_convergence"
STEM = "pullback_convergence"


# ====================================================================
def observed_rates(n, e):
    """Local convergence rate between consecutive refinements.

    r = log(e_i / e_{i+1}) / log(N_{i+1} / N_i), so r = 2 is optimal
    second-order behaviour for bilinear elements and r ~ 0 is a plateau.
    """
    return np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])


def report(n, e, label):
    h = DOMAIN_SIDE / n
    r = observed_rates(n, e)
    print(f"\n{label}")
    print(f"  {'N':>5} {'h':>9} {'rel L2':>11} {'rate':>7}")
    for i in range(len(n)):
        rate = f"{r[i-1]:7.2f}" if i else " " * 7
        print(f"  {n[i]:5d} {h[i]:9.4f} {e[i]:11.3e} {rate}")
    if N_PLATEAU > 0:
        tail = e[-N_PLATEAU:]
        print(f"  plateau (mean of last {N_PLATEAU}): {tail.mean():.3e} "
              f"+/- {tail.std():.1e}")
        print(f"  best point: {e.min():.3e} at N = {n[int(np.argmin(e))]}")
    return r


# ====================================================================
def main():
    r = report(N_SQUARE, REL_L2, LABEL)
    if N_SQUARE_2 is not None:
        report(np.asarray(N_SQUARE_2), np.asarray(REL_L2_2), LABEL_2)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.loglog(N_SQUARE, REL_L2, "o-", color="#1f4e79", lw=1.8, ms=7,
              label=LABEL, zorder=3)

    if N_SQUARE_2 is not None:
        ax.loglog(N_SQUARE_2, REL_L2_2, "s--", color="#b8332a", lw=1.6, ms=6,
                  label=LABEL_2, zorder=3)

    # O(h^2) guide, anchored at the first point so it is a reference slope
    # rather than a fit -- it shows what optimal convergence would look like
    # from the same starting error.
    n_ref = np.array([N_SQUARE[0], N_SQUARE[-1]], dtype=float)
    e_ref = REL_L2[0] * (n_ref / N_SQUARE[0]) ** -2.0
    ax.loglog(n_ref, e_ref, ":", color="0.45", lw=1.5,
              label=r"$\mathcal{O}(h^{2})$", zorder=2)

    if N_PLATEAU > 0:
        floor = REL_L2[-N_PLATEAU:].mean()
        ax.axhline(floor, color="#2e7d32", ls="-.", lw=1.4, zorder=2,
                   label=f"mapping-error floor $\\approx$ {floor:.2e}")

    ax.set_xlabel(r"elements per side of the reference mesh, $N$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(N_SQUARE)
    ax.set_xticklabels([str(int(v)) for v in N_SQUARE])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9)

    # a secondary axis in h, since the paper's error estimates are stated in h
    sec = ax.secondary_xaxis(
        "top", functions=(lambda n: DOMAIN_SIDE / np.maximum(n, 1e-12),
                          lambda h: DOMAIN_SIDE / np.maximum(h, 1e-12)))
    sec.set_xlabel(r"mesh size $h = 2/N$")

    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUTDIR / f"{STEM}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"\nSaved {path}")
    plt.show()


if __name__ == "__main__":
    main()
