r"""Publication versions of the example figures, drawn from the saved runs.

    python -m examples.micropolar_phase_field.paper_figures [--copy DIR]

The figures the examples write while running are diagnostics: they carry titles
stating conclusions ("the curves lie on top of one another") and axes without
units. A journal figure carries neither -- the caption states the claim and the
axes state the quantity. These are redrawn from the ``.npz`` files, so nothing
is re-solved, with one exception: the MPM-against-FEM load paths were never
persisted, so that comparison is re-computed once (a few minutes, 2-D) and
cached in ``tension_classical_paths.npz``.

Units follow the examples' own systems. Notched tension is in kN and mm with
reactions per unit thickness; the beam is in SI with reactions for the modelled
slab of thickness ``h_cell``, which is how the beam's table reports them.

``--copy DIR`` also copies the six files into ``DIR`` under the names the
manuscript includes.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from ._common import RESULTS

STYLE = {"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
         "axes.labelsize": 9, "legend.fontsize": 8,
         "legend.frameon": False}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def tension_fem(plt):
    from .notched_tension import DX, ELL_PF, STEPS, classical_limit
    cache = RESULTS / "tension_classical_paths.npz"
    if not cache.exists():
        print("  recomputing the MPM-against-FEM paths (not saved by the run)")
        _, _, _, raw_m, raw_f = classical_limit(ELL_PF, DX, STEPS)
        np.savez_compressed(cache, mpm=raw_m, fem=raw_f)
    c = np.load(cache)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(c["fem"][:, 0], c["fem"][:, 1], "k-", lw=1.4,
            label="phase-field FEM")
    ax.plot(c["mpm"][:, 0], c["mpm"][:, 1], "o", ms=3.5, mfc="none",
            color="C3", label="present MPM")
    ax.set_xlabel(r"applied displacement $\bar u$ [mm]")
    ax.set_ylabel("reaction [kN/mm]")
    ax.legend()
    return fig, "sent_fem.png"


def tension_load(plt):
    from .notched_tension import ELL_PF, OVERSHOOT, critical_displacement
    d = np.load(RESULTS / "tension_data.npz")
    u_bar = OVERSHOOT * critical_displacement(ELL_PF)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    rows = [("classical", "classical", dict(color="k", ls="--", lw=1.4))]
    rows += [(f"lb{r}", rf"$\ell_b/\ell = {r}$", dict(marker="o", ms=2.5, lw=1.1))
             for r in (0, 1, 2, 4)]
    for tag, lab, kw in rows:
        ax.plot(d[f"{tag}_disp"] * u_bar, np.abs(d[f"{tag}_load"]), label=lab,
                **kw)
    ax.set_xlabel(r"applied displacement $\bar u$ [mm]")
    ax.set_ylabel("reaction [kN/mm]")
    ax.legend()
    return fig, "sent_load.png"


def tension_damage(plt):
    d = np.load(RESULTS / "tension_data.npz")
    panels = [("classical", "classical"), ("lb0", r"$\ell_b/\ell = 0$"),
              ("lb1", r"$\ell_b/\ell = 1$"), ("lb2", r"$\ell_b/\ell = 2$"),
              ("lb4", r"$\ell_b/\ell = 4$")]
    fig = plt.figure(figsize=(6.6, 4.6))
    gs = fig.add_gridspec(2, 6, hspace=0.28, wspace=0.25)
    slots = [gs[0, 1:3], gs[0, 3:5], gs[1, 0:2], gs[1, 2:4], gs[1, 4:6]]
    axes = []
    for slot, (tag, lab) in zip(slots, panels):
        a = fig.add_subplot(slot)
        lam = float(d[f"{tag}_reached"].ravel()[0])
        s = a.scatter(d[f"{tag}_x"][:, 0], d[f"{tag}_x"][:, 1],
                      c=d[f"{tag}_fields"][-1], s=0.6, cmap="inferno",
                      vmin=0.0, vmax=1.0, rasterized=True)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{lab},  " + rf"$t = {lam:.3f}$", fontsize=8)
        axes.append(a)
    fig.colorbar(s, ax=axes, shrink=0.6, label="$d$")
    return fig, "sent_damage.png"


def beam_load(plt):
    from .notched_beam import TIP_DISPLACEMENT
    d = np.load(RESULTS / "beam_data.npz")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, r in enumerate(d["l_b_over_h"]):
        ax.plot(d[f"row{i}_disp"] * TIP_DISPLACEMENT * 1e3,
                np.abs(d[f"row{i}_load"]), "o-", ms=2.5, lw=1.1,
                label=rf"$\ell_b/h = {r:g}$")
    ax.set_xlabel(r"tip displacement $\bar v$ [mm]")
    ax.set_ylabel("reaction [N]")
    ax.legend()
    return fig, "beam_load.png"


def beam_damage(plt):
    d = np.load(RESULTS / "beam_data.npz")
    i = len(d["l_b_over_h"]) - 1
    F, x = d[f"row{i}_fields"], d[f"row{i}_x"]
    u, L = d[f"row{i}_disp"], np.abs(d[f"row{i}_load"])
    k_pk, last = int(d[f"row{i}_peak_step"][0]), len(F) - 1
    ks = np.unique([0, k_pk, min(k_pk + 1, last), last])
    # one row, because the manuscript includes it at \textwidth
    fig, axes = plt.subplots(1, len(ks), figsize=(7.0, 1.85), squeeze=False)
    for a, k in zip(axes[0], ks):
        s = a.scatter(x[:, 0], x[:, 1], c=F[k], s=0.8, cmap="inferno",
                      vmin=0.0, vmax=1.0, rasterized=True)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(rf"$t = {u[k]:.3f}$" "\n" rf"$F = {L[k]:.3g}$ N",
                    fontsize=8)
    fig.colorbar(s, ax=axes[0], shrink=0.8, label="$d$")
    return fig, "beam_damage.png"


def beam_size(plt):
    """Size series: whichever ``beam_size_h*.npz`` exist, so it grows as sizes land."""
    from .beam_size_effect import L_B, load, nominal_strength
    hs = sorted(float(f.stem.split("_h")[1]) for f in
                RESULTS.glob("beam_size_h*.npz"))
    sr = np.array([nominal_strength(h, load(h)["ref"]["load"].max()) for h in hs])
    sm = np.array([nominal_strength(h, load(h)["mp"]["load"].max()) for h in hs])
    hs = np.array(hs)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    a.loglog(hs, sr / 1e3, "ko--", ms=4, lw=1.1, label=r"$\ell_b = 0$")
    a.loglog(hs, sm / 1e3, "s-", color="C3", ms=4, lw=1.1,
             label=rf"$\ell_b = {L_B:g}$ m")
    a.set_xlabel("beam depth $h$ [m]")
    a.set_ylabel(r"nominal strength $\sigma_N$ [kPa]")
    a.legend()
    a.set_title("(a)", loc="left", fontsize=9)
    x = L_B / hs
    ex = sm / sr - 1.0
    b.loglog(x, ex, "s-", color="C3", ms=4, lw=1.1)
    # reference slopes through the largest l_b/h point -- NOT a fit. A slope-1
    # fit through the three smaller sizes predicted 1.77 at h = 0.6 and the
    # computed ratio was 1.64, so drawing a fitted line would show a law the
    # data refuted.
    k = int(np.argmax(x))
    xx = np.array([x.min() * 0.85, x.max()])
    b.loglog(xx, ex[k] * (xx / x[k]), "k--", lw=0.9, label="slope 1")
    b.loglog(xx, ex[k] * (xx / x[k]) ** 2, "k:", lw=1.1, label="slope 2")
    b.set_xlabel(r"$\ell_b / h$")
    b.set_ylabel(r"$\sigma_N / \sigma_N^{(\ell_b = 0)} - 1$")
    b.legend()
    b.set_title("(b)", loc="left", fontsize=9)
    # plain tick labels: a log axis spanning less than a decade otherwise
    # labels itself 2x10^-1, 3x10^-1, which reads worse than the numbers
    from matplotlib.ticker import FuncFormatter, NullFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax in (a, b):
        ax.xaxis.set_major_formatter(plain); ax.xaxis.set_minor_formatter(plain)
        ax.yaxis.set_major_formatter(plain); ax.yaxis.set_minor_formatter(plain)
    from matplotlib.ticker import FixedLocator
    a.yaxis.set_major_locator(FixedLocator([5, 7, 10, 15, 20, 30]))
    a.yaxis.set_minor_formatter(NullFormatter())
    b.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.7, 1, 1.5, 2, 3]))
    b.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    return fig, "beam_size.png"


def double_notch(plt):
    """Crack patterns at a common load level, and the load paths.

    Every row of the sweep runs the same increments and stops at the same
    ``t``, so the panels are a like-for-like comparison rather than each row at
    its own final state -- the trap recorded for the beam, where a stiffer
    specimen reaches its peak at a smaller displacement and a common-``t``
    snapshot would compare different stages. Here the stopping state *is*
    common, so the panels can be read directly against one another.
    """
    from .double_notch import N_SWEEP, U_MAX
    files = [(n, RESULTS / f"double_notch_N{n:g}_dx1.npz") for n in N_SWEEP]
    files = [(n, f) for n, f in files if f.exists()]
    m = len(files)
    # twelve columns so the load curve can be centred under an even number of
    # square panels; spanning whole panel columns cannot centre it
    fig = plt.figure(figsize=(7.0, 3.75))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.12, 1.0],
                          hspace=0.04, wspace=0.6)
    w = 12 // m
    axes = []
    for j, (n, f) in enumerate(files):
        d = np.load(f)
        a = fig.add_subplot(gs[0, j * w:(j + 1) * w])
        s = a.scatter(d["x"][:, 0], d["x"][:, 1], c=d["fields"][-1], s=0.25,
                      cmap="inferno", vmin=0.0, vmax=1.0, rasterized=True)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(rf"$N = {n:g}$", fontsize=9)
        axes.append(a)
    fig.colorbar(s, ax=axes, shrink=0.62, pad=0.015, label="$d$")

    ax = fig.add_subplot(gs[1, 3:9])
    for n, f in files:
        d = np.load(f)
        # the resultant of the two equal prescribed components
        ax.plot(d["disp"] * U_MAX * np.sqrt(2.0) * 1e3, np.abs(d["load45"]),
                "o-", ms=2.5, lw=1.1, label=rf"$N = {n:g}$")
    ax.set_xlabel(r"prescribed displacement $\sqrt{2}\,\bar u$ [$\mu$m]")
    ax.set_ylabel("reaction [N]")
    ax.legend(loc="upper left")
    return fig, "double_notch.png"


def main():
    plt = _plt()
    out = RESULTS / "paper"
    out.mkdir(parents=True, exist_ok=True)
    copy = None
    if "--copy" in sys.argv:
        copy = Path(sys.argv[sys.argv.index("--copy") + 1])
    for make in (tension_fem, tension_load, tension_damage, beam_load,
                 beam_damage, beam_size, double_notch):
        fig, name = make(plt)
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")
        if copy is not None:
            shutil.copy(path, copy / name)
    if copy is not None:
        print(f"  copied into {copy}")


if __name__ == "__main__":
    main()
