r"""Size-dependent strength of a notched beam in bending.

    python -m examples.micropolar_phase_field.notched_beam
    python -m examples.micropolar_phase_field.notched_beam --quick

A notched cantilever of depth ``h``, clamped at one end -- translations *and*
microrotation -- and loaded by a prescribed transverse displacement at the free
end. The internal length ``l_b`` is swept with the geometry, the mesh, the
phase-field length and ``G_c`` all held fixed, so any change in the predicted
response is attributable to ``l_b`` alone: a classical phase-field model
applied to this sequence returns a single curve, because it has no length with
which to distinguish the cases.

Bending is chosen deliberately. Every null result on micropolar fracture
recorded so far came from a stretch-dominated problem, where the microrotation
field is nearly uniform and the wryness is small however ``l_b`` compares with
the specimen. Bending is the mode that produces a systematic rotation gradient
through the depth, and a beam a few internal lengths deep is the regime in
which micropolar size effects are observed experimentally.

Companion to ``notched_tension``, which is the same machinery on a
stretch-dominated problem and returns the null.
"""
from __future__ import annotations

import sys

import numpy as np

from scion.discretization.mpm import GridDirichlet

from ._common import (RESULTS, build_grid, check_resolution, make_problem,
                      plane_strain_bc, run_load_path, seed_notch,
                      seed_particles)

# --- material ---------------------------------------------------------------
E_MOD, NU, N_COUPLING, GC = 1.0e7, 0.25, 0.5, 5.0

# --- geometry ---------------------------------------------------------------
SPAN, DEPTH = 1.0, 0.4
NOTCH_X, NOTCH_DEPTH = 0.35, 0.12
TIP_DISPLACEMENT = 6.0e-3

# --- discretisation ---------------------------------------------------------
# l must be small against the *section depth*, not merely resolved by the grid.
# At l = 0.2 on a beam of depth 0.4 the phase-field length is half the section:
# the notch is smeared away entirely and the crack forms at the clamped end
# instead, where the bending moment is largest. Measured, at l_b/h = 0.25:
#
#     l = 0.20, dx = 0.025    d_clamp 0.999   d_notch 0.869   -> CLAMP
#     l = 0.08, dx = 0.020    d_clamp 0.379   d_notch 0.999   -> notch
#     l = 0.04, dx = 0.010    d_clamp 0.260   d_notch 0.999   -> notch
#
# The size-effect trend is monotone and clean in *both* regimes, so the table
# alone cannot distinguish them -- only the damage field can. l/h = 0.1 here.
ELL_PF, DX, PPC, STEPS = 0.04, 0.010, 2, 12
SWEEP = (0.0, 0.25, 0.375, 0.50)                 # l_b / h

QUICK_DX, QUICK_ELL, QUICK_STEPS = 0.020, 0.08, 6
QUICK_SWEEP = (0.0, 0.50)


def build(l_b, ell_pf=ELL_PF, dx=DX, ppc=PPC):
    check_resolution(dx, ell_pf, l_b)
    grid, nx, ny = build_grid((SPAN, DEPTH), dx)
    parts = seed_particles((SPAN, DEPTH), dx, ppc)
    nan = np.nan
    bcs = [
        # clamped: the microrotation is restrained too, which is what makes
        # this a built-in end for a Cosserat solid rather than a pin
        GridDirichlet(lambda c: c[:, 0] < 1e-9, np.zeros(6)),
        GridDirichlet(lambda c: np.abs(c[:, 0] - SPAN) < 1e-9,
                      np.array([nan, TIP_DISPLACEMENT, nan, nan, nan, nan])),
        plane_strain_bc(),
    ]
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING,
                               l_b=l_b, gc=GC, ell_pf=ell_pf, bcs=bcs)
    seed_notch(pf, parts, dx, ell_pf, GC,
               tip=np.array([NOTCH_X, NOTCH_DEPTH]), along="y")
    xy = np.asarray(grid.node_coordinates())
    tip_nodes = np.flatnonzero(np.abs(xy[:, 0] - SPAN) < 1e-9)
    return pf, stepper, parts, tip_nodes


def run(l_b_over_h, steps=STEPS, **kw):
    l_b = l_b_over_h * DEPTH
    pf, stepper, parts, tip_nodes = build(l_b, **kw)
    out = run_load_path(pf, stepper, parts, tip_nodes, 1,
                        steps, progress=f'l_b/h={l_b_over_h:g}')
    peak = int(np.argmax(np.abs(out["load"])))
    d, x = out["fields"][-1], parts.x[:, :2]
    out.update(d_clamp=float(d[x[:, 0] < 0.15].max()),
               d_notch=float(d[np.abs(x[:, 0] - NOTCH_X) < 0.10].max()),
               l_b=l_b, l_b_over_h=l_b_over_h,
               peak_load=abs(out["load"][peak]),
               peak_step=peak,
               shares_at_peak=out["shares"][peak],
               x=parts.x[:, :2].copy())
    return out


# ---------------------------------------------------------------------------

def table(rows):
    base = rows[0]["peak_load"]
    print(f"\n  {'l_b/h':>7} {'l_b':>8} {'l_b/dx':>7} {'peak load':>11} "
          f"{'normalised':>11} {'W_curv':>9} {'d_notch':>8} {'d_clamp':>8} "
          f"{'reached':>8}")
    print("  " + "-" * 88)
    for r in rows:
        s = r["shares_at_peak"]
        cells = "--" if r["l_b"] == 0 else f"{r['l_b'] / DX:.1f}"
        print(f"  {r['l_b_over_h']:7.3f} {r['l_b']:8.3f} {cells:>7} "
              f"{r['peak_load']:11.4g} {r['peak_load'] / base:11.4f} "
              f"{s['curvature']:9.3g} {r['d_notch']:8.3f} "
              f"{r['d_clamp']:8.3f} {r['reached']:8.3f}")


def make_figures(rows, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figures")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for r in rows:
        ax.plot(r["disp"] * TIP_DISPLACEMENT, np.abs(r["load"]), "o-",
                ms=3.5, lw=1.4, label=rf"$\ell_b/h = {r['l_b_over_h']:g}$")
    ax.set_xlabel("tip displacement")
    ax.set_ylabel("reaction")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Notched beam: load path at fixed geometry and mesh",
                 fontsize=9)
    p = RESULTS / f"beam_load{tag}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\n  wrote {p}")

    lam_c = min(r["reached"] for r in rows)
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.4), squeeze=False)
    for a, r in zip(axes[0], rows):
        k = int(np.argmin(np.abs(r["disp"] - lam_c)))
        s = a.scatter(r["x"][:, 0], r["x"][:, 1], c=r["fields"][k], s=3,
                      cmap="inferno", vmin=0.0, vmax=1.0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(rf"$\ell_b/h = {r['l_b_over_h']:g}$", fontsize=9)
    fig.suptitle(rf"damage at a common load level $\lambda = {lam_c:.3f}$",
                 fontsize=9, y=1.03)
    fig.colorbar(s, ax=axes[0], shrink=0.85, label="$d$")
    p = RESULTS / f"beam_damage{tag}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def main(quick=False):
    dx, ell, steps, sweep = DX, ELL_PF, STEPS, SWEEP
    if quick:
        dx, ell, steps, sweep = QUICK_DX, QUICK_ELL, QUICK_STEPS, QUICK_SWEEP

    print("Micropolar phase-field fracture: notched beam in bending")
    print(f"  span {SPAN} x depth {DEPTH}, notch at x = {NOTCH_X} of depth "
          f"{NOTCH_DEPTH}")
    print(f"  E = {E_MOD:g}, nu = {NU}, N = {N_COUPLING}, G_c = {GC}")
    print(f"  l = {ell:g} ({ell / dx:.1f} cells), dx = {dx:g}, "
          f"{PPC} particles per cell per direction, {steps} load steps")
    if quick:
        print("  --quick: coarse grid, longer phase-field length and fewer "
              "load steps.\n  The trend survives; the numbers are not "
              "converged and must not be quoted.")

    rows = [run(r, ell_pf=ell, dx=dx, steps=steps) for r in sweep]
    table(rows)

    base, top = rows[0], rows[-1]
    print(f"\n    Geometry, mesh, l and G_c are identical across every row, so "
          f"a classical model\n    -- having no length of its own -- returns "
          f"one curve for all of them. Here the peak\n    load changes by "
          f"{top['peak_load'] / base['peak_load']:.2f}x, and the curvature "
          f"share of the damage driving force\n    rises from "
          f"{base['shares_at_peak']['curvature']:.1e} to "
          f"{top['shares_at_peak']['curvature']:.3f}: the couple stress "
          f"carries load the\n    classical beam has to shed into damage.")
    monotone = all(a["peak_load"] < b["peak_load"]
                   for a, b in zip(rows, rows[1:]))
    print(f"    Monotone in l_b: {monotone}. Two points could be a coincidence "
          f"of parameters;\n    a monotone sweep is the statement that l_b is "
          f"what is doing the work.")

    at_clamp = [r for r in rows if r["d_clamp"] > r["d_notch"]]
    if at_clamp:
        print(f"\n    WARNING: {len(at_clamp)} of {len(rows)} rows crack at "
              f"the CLAMP rather than the notch.\n    The trend above is then "
              f"a property of the support, not of the notched section, and\n"
              f"    must not be reported. Reduce l relative to the depth.")

    make_figures(rows, tag="_quick" if quick else "")
    return rows


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
