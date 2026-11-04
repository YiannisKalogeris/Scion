"""Redraw the notched-beam crack-evolution sequence from a completed run.

Same purpose as ``replot_tension``: ``notched_beam.make_figures`` takes the
in-memory sweep, so a wording or styling change to the figure would otherwise
cost the whole solve again. Reads results/beam_data.npz, written by
``notched_beam.save_results``, and redraws only the sequence panel -- the one
the paper carries.
"""
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def main(tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(RESULTS / f"beam_data{tag}.npz", allow_pickle=True)
    i = len(np.asarray(d["l_b_over_h"]).ravel()) - 1      # the largest l_b
    disp = np.asarray(d[f"row{i}_disp"]).ravel()
    load = np.asarray(d[f"row{i}_load"]).ravel()
    fields = np.asarray(d[f"row{i}_fields"])
    x = np.asarray(d[f"row{i}_x"])
    l_b_over_h = float(np.asarray(d["l_b_over_h"]).ravel()[i])

    # the first step, the peak, the first step past it and the end -- the same
    # four ``notched_beam`` picks, and not evenly spaced for the same reason:
    # the crack runs inside a single increment
    last = len(fields) - 1
    k_pk = int(np.asarray(d[f"row{i}_peak_step"]).ravel()[0])
    ks = np.unique([0, k_pk, min(k_pk + 1, last), last])

    fig, axes = plt.subplots(1, len(ks), figsize=(3.1 * len(ks), 2.4),
                             squeeze=False)
    for a, k in zip(axes[0], ks):
        s = a.scatter(x[:, 0], x[:, 1], c=fields[k], s=3, cmap="inferno",
                      vmin=0.0, vmax=1.0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(rf"$t = {disp[k]:.3f}$, $F = {abs(load[k]):.3g}$",
                    fontsize=8)
    fig.suptitle(rf"crack evolution at $\ell_b/h = {l_b_over_h:g}$",
                 fontsize=9, y=1.06)
    fig.colorbar(s, ax=axes[0], shrink=0.85, label="$d$")
    p = RESULTS / f"beam_damage{tag}_sequence.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    import sys
    main("_quick" if "--quick" in sys.argv else "")
