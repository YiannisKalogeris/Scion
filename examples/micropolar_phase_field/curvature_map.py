r"""Where the curvature energy lives at the peak load: bending against tension.

    python -m examples.micropolar_phase_field.curvature_map          # run + plot
    python -m examples.micropolar_phase_field.curvature_map --plot   # plot only

The paper's central claim is that the internal length reaches fracture through
rotation *gradients*, which bending produces and stretching barely does. The
sweep tables state it as totals -- a curvature share of 0.446 in the beam
against 2.1e-3 in the plate. This draws it as a field: the curvature energy
density at each specimen's own peak load, for the largest internal length of
each sweep (``l_b/h = 0.5``, ``l_b/l = 4``).

**Plotted quantity.** ``W_curv / max W+``, the curvature energy density scaled
by the largest crack-driving energy density in the *same* specimen outside the
seeded band, on a common
logarithmic colour scale. Scaling each specimen by its own maximum removes the
two problems' different units and load levels; the common log scale keeps the
difference in *magnitude* visible, which a separate linear scale per panel
would hide.

**The seeded notch band is masked, and the reason is measured.** The band where
the notch is imposed (distance to the notch segment within the seeding radius
``r_0``) is already broken, ``d ~ 1``, and the strain smeared across it is
large: it holds 0.6% (beam) and 1.3% (plate) of the particles and **92% and 97%
of the whole specimen's W_iso+**. Normalising by a maximum that lives there
would set the colour scale from pre-broken material. The map therefore greys
the band out and scales by the largest ``W+`` *outside* it, and `regions` prints
the curvature share by region -- which is where the paper's contrast should be
read: ahead of the notch tip it is 0.585 in the beam against 0.0053 in the plate.

**Self-check.** Each run repeats the load steps of its sweep exactly, stopping
once past the peak, so the shares recomputed here from the densities must
reproduce the sweep's tables. They are printed next to the table values; a
mismatch would mean the map does not describe the calculation in the paper.

Data go to ``results/curvature_map_{beam,tension}.npz``; the figure to
``results/paper/curvature_map.png``.
"""
from __future__ import annotations

import sys

import numpy as np

from ._common import RESULTS, run_load_path, seed_particles

TABLE = {"beam": 0.4499, "tension": 2.736e-3}     # W_curv shares, post-fix sweeps


def _densities(pf, stepper, parts):
    return np.stack(pf.driving_densities())        # (3, n): iso+, rel, curv


def run_beam():
    from . import notched_beam as nb
    pf, stepper, parts, tip = nb.build(0.5 * nb.DEPTH)
    X0 = seed_particles((nb.SPAN, nb.DEPTH), nb.DX, nb.PPC).x
    out = run_load_path(pf, stepper, parts, tip, 1, nb.STEPS,
                        progress="beam l_b/h=0.5", stop_past_peak=0.9,
                        on_step=_densities)
    return X0, out


def run_tension():
    from . import notched_tension as nt
    u_top = nt.OVERSHOOT * nt.critical_displacement(nt.ELL_PF)
    pf, stepper, parts, top = nt.build(4.0 * nt.ELL_PF, u_top=u_top)
    X0 = seed_particles((nt.SIDE, nt.SIDE), nt.DX, nt.PPC).x
    out = run_load_path(pf, stepper, parts, top, 1, nt.STEPS,
                        progress="tension l_b/l=4", stop_past_peak=0.9,
                        on_step=_densities)
    return X0, out


def solve(name, runner):
    X0, out = runner()
    k = int(np.argmax(np.abs(out["load"])))
    dens = out["extra"][k]
    share = dens[2].sum() / dens.sum()
    print(f"  {name}: peak at step {k}, t = {out['disp'][k]:.3f}; "
          f"W_curv share recomputed {share:.4g} against the table's "
          f"{TABLE[name]:.4g} ({100 * (share / TABLE[name] - 1):+.2f}%)",
          flush=True)
    np.savez_compressed(RESULTS / f"curvature_map_{name}.npz", X0=X0,
                        dens=dens, t=out["disp"][k], share=share)


def band_and_zone(name, X):
    """The seeded notch band (as `seed_notch` defines it) and the zone ahead of it."""
    if name == "beam":
        from . import notched_beam as m
        xn, an = m.NOTCH_X, m.NOTCH_DEPTH
        r0 = max(0.25 * m.ELL_PF, m.DX)
        behind = X[:, 1] <= an
        dist = np.where(behind, np.abs(X[:, 0] - xn),
                        np.hypot(X[:, 0] - xn, X[:, 1] - an))
        zone = (np.abs(X[:, 0] - xn) <= 0.08) & (X[:, 1] > an) & (X[:, 1] <= an + 0.13)
    else:
        from . import notched_tension as m
        xn, yn = m.NOTCH_X, m.NOTCH_Y
        r0 = max(0.25 * m.ELL_PF, m.DX)
        behind = X[:, 0] <= xn
        dist = np.where(behind, np.abs(X[:, 1] - yn),
                        np.hypot(X[:, 0] - xn, X[:, 1] - yn))
        zone = np.hypot(X[:, 0] - xn, X[:, 1] - yn) <= 0.1
    band = dist <= r0 + 1e-9
    return band, zone & ~band


def regions():
    print(f"  {'':8s} {'region':12s} {'particles':>9s} {'W_iso+ %':>9s} "
          f"{'W_curv/W+':>10s}")
    for name in ("tension", "beam"):
        d = np.load(RESULTS / f"curvature_map_{name}.npz")
        X0, dens = d["X0"], d["dens"]
        band, zone = band_and_zone(name, X0)
        for lab, m in (("seeded band", band), ("tip zone", zone),
                       ("rest", ~(band | zone)), ("all but band", ~band),
                       ("whole", np.ones(len(X0), bool))):
            sub = dens[:, m].sum(axis=1)
            print(f"  {name:8s} {lab:12s} {int(m.sum()):9d} "
                  f"{100 * sub[0] / dens[0].sum():9.1f} "
                  f"{sub[2] / sub.sum():10.4g}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})

    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.5, 0.08], wspace=0.18)
    norm = LogNorm(vmin=1e-5, vmax=1.0)
    panels = (("tension", r"(a) notched tension, $\ell_b/\ell = 4$"),
              ("beam", r"(b) notched beam, $\ell_b/h = 0.5$"))
    for col, (name, title) in enumerate(panels):
        d = np.load(RESULTS / f"curvature_map_{name}.npz")
        X0, dens = d["X0"], d["dens"]
        layer = X0[:, 2] < X0[:, 2].min() + 1e-9   # one layer of the slab
        band, _ = band_and_zone(name, X0)
        Wp = dens.sum(axis=0)
        val = np.clip(dens[2] / Wp[~band].max(), 1e-5, None)
        a = fig.add_subplot(gs[0, col])
        m = layer & ~band
        s = a.scatter(X0[m, 0], X0[m, 1], c=val[m], s=0.5,
                      cmap="viridis", norm=norm, rasterized=True)
        a.scatter(X0[layer & band, 0], X0[layer & band, 1], c="0.75", s=0.5,
                  rasterized=True)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(title + "\n" + rf"$t = {float(d['t']):.3f}$, "
                    "seeded notch band in grey",
                    fontsize=8)
    cb = fig.colorbar(s, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label(r"$W_{\mathrm{curv}} \,/\, \max W^{+}$ outside the band")
    out = RESULTS / "paper" / "curvature_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        for name, runner in (("beam", run_beam), ("tension", run_tension)):
            if not (RESULTS / f"curvature_map_{name}.npz").exists():
                solve(name, runner)
    regions()
    plot()
