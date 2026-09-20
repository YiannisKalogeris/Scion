"""Pre-fix against fixed: does the wryness-update fix move any paper number?"""
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent

def beam():
    a, b = np.load(R / "beam_data_prefix.npz"), np.load(R / "beam_data.npz")
    print("BEAM (production)")
    print(f"  {'l_b/h':>6} {'peak pre':>9} {'peak post':>9} {'dpeak %':>9} "
          f"{'W_rel pre':>10} {'post':>9} {'W_curv pre':>10} {'post':>7} "
          f"{'curv 1st step pre/post':>23}")
    for i, r in enumerate(a["l_b_over_h"]):
        la, lb = np.abs(a[f"row{i}_load"]), np.abs(b[f"row{i}_load"])
        ka, kb = int(np.argmax(la)), int(np.argmax(lb))
        sa, sb = a[f"row{i}_shares"], b[f"row{i}_shares"]
        print(f"  {r:6.3f} {la.max():9.4f} {lb.max():9.4f} "
              f"{100 * (lb.max() / la.max() - 1):+9.4f} "
              f"{sa[ka, 1]:10.4g} {sb[kb, 1]:9.4g} {sa[ka, 2]:10.4f} "
              f"{sb[kb, 2]:7.4f} {sa[0, 2]:11.4f}/{sb[0, 2]:.4f}"
              + ("" if ka == kb else f"   PEAK STEP MOVED {ka}->{kb}"))

def tension(tag="_quick_peak"):
    a = np.load(R / f"tension_data{tag}_prefix.npz")
    b = np.load(R / f"tension_data{tag}.npz")
    print(f"TENSION ({tag})")
    for t in sorted({k[:-7] for k in a.files if k.endswith("_shares")}):
        la, lb = np.abs(a[f"{t}_load"]), np.abs(b[f"{t}_load"])
        ka, kb = int(np.argmax(la)), int(np.argmax(lb))
        sa, sb = a[f"{t}_shares"][ka], b[f"{t}_shares"][kb]
        print(f"  {t:6s} peak {la.max():.6g} -> {lb.max():.6g} "
              f"({100 * (lb.max() / la.max() - 1):+.4f}%)   "
              f"W_rel {sa[1]:.4g} -> {sb[1]:.4g}   W_curv {sa[2]:.4g} -> {sb[2]:.4g}")

if __name__ == "__main__":
    import sys
    if "tension" in sys.argv: tension()
    if "tension_prod" in sys.argv: tension("_peak")
    if "beam" in sys.argv: beam()
