r"""Mesh refinement of the notched-beam result: does 3.56x survive a finer grid?

    python -m examples.micropolar_phase_field.beam_refinement
    python -m examples.micropolar_phase_field.beam_refinement --dx 0.005,0.00625

Every fracture result in the paper is at one resolution, ``l/dx = 4``. This
repeats the two ends of the beam sweep -- ``l_b = 0`` and ``l_b = h/2`` -- on
finer grids with *everything else* fixed, including ``l``, so ``l/dx`` rises to
6.4 and 8. What is compared is the peak load of each and their ratio.

Four things are held fixed deliberately, each because letting it move would
change the answer for a reason unrelated to the mesh:

* **``l`` and ``l_b``**: refining at fixed lengths is a convergence study;
  refining ``l`` with the mesh would change the model.
* **the seeded notch**: its radius is ``max(l/4, dx)``, which is 0.01 m on every
  grid used here, so the notch is physically the same;
* **the load steps**: the same 12 increments of the same tip displacement, so
  the peak is sampled at the same displacements;
* **the thickness of comparison**: the plane-strain slab is one cell thick, so
  the raw reaction scales with ``dx``. Loads are divided by the slab thickness
  before any comparison -- the trap already recorded for the shear-layer study,
  where the reaction halved with each refinement, exactly.

``dx`` must divide both the span and the depth, or the boundary-condition node
selectors find no nodes; 0.0075 is therefore not available, 0.00625 is.

Each run stops once past its peak. Results go to
``results/beam_refine_dx{dx}.npz``, one file per grid, skipped when present.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ._common import NZ, RESULTS, run_load_path
from .notched_beam import DEPTH, DX, ELL_PF, STEPS, build

GRIDS = (0.005, 0.00625)
L_B = 0.5 * DEPTH


def path(dx):
    return RESULTS / f"beam_refine_dx{dx:g}.npz"


def solve(dx):
    rec = {"dx": dx}
    for tag, l_b in (("ref", 0.0), ("mp", L_B)):
        t0 = time.time()
        pf, stepper, parts, tip = build(l_b, dx=dx)
        out = run_load_path(pf, stepper, parts, tip, 1, STEPS,
                            progress=f"dx={dx:g} l_b={l_b:g}",
                            stop_past_peak=0.9)
        load = np.abs(out["load"]) / (NZ * dx)          # per unit thickness
        k = int(np.argmax(load))
        rec[f"{tag}_disp"], rec[f"{tag}_load"] = out["disp"], load
        rec[f"{tag}_peak"], rec[f"{tag}_peak_step"] = load[k], k
        rec[f"{tag}_curv"] = out["shares"][k]["curvature"]
        rec[f"{tag}_minutes"] = (time.time() - t0) / 60.0
        rec[f"{tag}_particles"] = parts.n
    np.savez_compressed(path(dx), **rec)
    return rec


def base():
    """The production sweep at dx = 0.01, from beam_data.npz."""
    d = np.load(RESULTS / "beam_data.npz")
    r = list(np.round(d["l_b_over_h"] * DEPTH, 12))
    rec = {"dx": DX}
    for tag, l_b in (("ref", 0.0), ("mp", L_B)):
        i = r.index(round(l_b, 12))
        load = np.abs(d[f"row{i}_load"]) / (NZ * DX)
        k = int(np.argmax(load))
        rec[f"{tag}_disp"], rec[f"{tag}_load"] = d[f"row{i}_disp"], load
        rec[f"{tag}_peak"], rec[f"{tag}_peak_step"] = load[k], k
        rec[f"{tag}_curv"] = float(d[f"row{i}_shares"][k, 2])
        rec[f"{tag}_minutes"], rec[f"{tag}_particles"] = np.nan, 0
    return rec


def main():
    grids = GRIDS
    if "--dx" in sys.argv:
        grids = tuple(float(v) for v in
                      sys.argv[sys.argv.index("--dx") + 1].split(","))
    rows = [base()]
    for dx in grids:
        if path(dx).exists():
            d = np.load(path(dx))
            rows.append({k: d[k] for k in d.files})
        else:
            rows.append(solve(dx))
        rows.sort(key=lambda r: -float(r["dx"]))
        table(rows)


def table(rows):
    b = rows[0]
    print(f"\n  {'dx':>8} {'l/dx':>5} {'peak l_b=0':>11} {'peak l_b=h/2':>13} "
          f"{'ratio':>7} {'W_curv':>7} {'pk steps':>9} {'minutes':>9}")
    print("  " + "-" * 78)
    for r in rows:
        ratio = float(r["mp_peak"]) / float(r["ref_peak"])
        print(f"  {float(r['dx']):8.5f} {ELL_PF / float(r['dx']):5.1f} "
              f"{float(r['ref_peak']):11.1f} {float(r['mp_peak']):13.1f} "
              f"{ratio:7.4f} {float(r['mp_curv']):7.4f} "
              f"{int(r['ref_peak_step']):4d}/{int(r['mp_peak_step']):<4d} "
              f"{float(r['ref_minutes']):4.0f}/{float(r['mp_minutes']):<4.0f}")
    print("\n  Loads per unit thickness. Differences relative to dx = 0.01:")
    for r in rows[1:]:
        dr = float(r["ref_peak"]) / float(b["ref_peak"]) - 1
        dm = float(r["mp_peak"]) / float(b["mp_peak"]) - 1
        dq = (float(r["mp_peak"]) / float(r["ref_peak"])) / \
             (float(b["mp_peak"]) / float(b["ref_peak"])) - 1
        print(f"    dx = {float(r['dx']):g}: l_b=0 {100 * dr:+.2f}%, "
              f"l_b=h/2 {100 * dm:+.2f}%, ratio {100 * dq:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
