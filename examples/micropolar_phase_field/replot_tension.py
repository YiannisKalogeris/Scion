"""Redraw the notched-tension damage panels from a completed run.

Each panel is drawn at its OWN final state rather than at a load level every
row reached. That is the honest picture when one row stops early -- a common
level throws away most of the load path for the four rows that completed --
but it means the panels are no longer directly comparable, so every title
carries the lambda it was drawn at and the truncated row is flagged.

Reads results/tension_data.npz, written by notched_tension.save_results.
"""
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
TAGS = ["classical", "lb0", "lb1", "lb2", "lb4"]
TITLES = {
    "classical": "classical (no microrotation)",
    "lb0": r"micropolar, $\ell_b/\ell = 0$",
    "lb1": r"micropolar, $\ell_b/\ell = 1$",
    "lb2": r"micropolar, $\ell_b/\ell = 2$",
    "lb4": r"micropolar, $\ell_b/\ell = 4$",
}
TOP = ["classical", "lb0"]


def main(tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(RESULTS / f"tension_data{tag}.npz", allow_pickle=True)
    have = [t for t in TAGS if f"{t}_fields" in d.files]
    top = [t for t in have if t in TOP]
    bot = [t for t in have if t not in TOP]
    wide = max(len(top), len(bot))

    fig = plt.figure(figsize=(2.9 * wide, 5.8))
    # every panel spans two columns so all are the same size; the shorter row
    # is centred by an offset rather than stretched, which would make the
    # control panels larger than the sweep they are meant to be compared with
    gs = fig.add_gridspec(2, 2 * wide, hspace=0.34, wspace=0.30)
    axes, order = [], []
    for row, group in ((0, top), (1, bot)):
        off = (2 * wide - 2 * len(group)) // 2
        for i, t in enumerate(group):
            axes.append(fig.add_subplot(gs[row, off + 2 * i: off + 2 * i + 2]))
            order.append(t)

    lam_full = max(float(d[f"{t}_reached"].ravel()[0]) for t in have)
    truncated = []
    for a, t in zip(axes, order):
        x, F = d[f"{t}_x"], d[f"{t}_fields"]
        lam = float(d[f"{t}_reached"].ravel()[0])
        s = a.scatter(x[:, 0], x[:, 1], c=F[-1], s=2, cmap="inferno",
                      vmin=0.0, vmax=1.0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        short = lam < lam_full - 1e-9
        if short:
            truncated.append((TITLES[t], lam))
        a.set_title(rf"{TITLES[t]}" "\n"
                    rf"$\lambda = {lam:.3f}$" + (" (stalled)" if short else ""),
                    fontsize=9, color="crimson" if short else "black")

    fig.colorbar(s, ax=axes, shrink=0.7, label="$d$")
    fig.suptitle("damage at the end of each load path", fontsize=10, y=0.98)
    p = RESULTS / f"tension_damage{tag}_final.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")
    for name, lam in truncated:
        print(f"  note: {name} stopped at lambda = {lam:.3f}, "
              f"not {lam_full:.3f} -- its panel is not a completed crack")


if __name__ == "__main__":
    import sys
    main("_quick" if "--quick" in sys.argv else "")
