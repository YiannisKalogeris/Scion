r"""What the exact wryness update buys, in a structural solve.

    python -m examples.mpm.micropolar_wryness_update

The wryness accumulates as ``Gamma += Q^T H(dphi) grad(dphi) F``, where ``H``
is the tangent map of the exponential. Setting ``H = I`` is correct to first
order in the increment. Measured *kinematically*, on an imposed rotation field
with an exact reference (``tests/mpm/test_micropolar.py``, class
``TestWryness``): the exact map is exact at any number of steps, ``H = I`` is off
by 18.7 / 9.3 / 4.6 / 2.3 % at 1 / 2 / 4 / 8 steps, and ``H^T`` by twice that.

This asks the question that test cannot: **how much does it matter in a real
solve**, where the increments come from Newton rather than being imposed, and
where the solution carries a second, independent load-step dependence -- the
rotations are interpolated with Lagrange shape functions and each increment
re-embeds that error (recorded for the manufactured solution: 75% growth in the
rotation error from 1 to 8 steps with the exact map).

**The problem must have a moving rotation axis.** If the axis is fixed, every
increment is parallel to its own gradient and ``H`` acts as the identity on
exactly the vectors it touches, so every plane-strain problem -- including all
of the phase-field examples -- is blind to the choice. The 45-degree curved
cantilever is the one problem here whose axis moves (rotations to 1.40 rad, axes
spanning 13 degrees), so it is used, on its resolved 1 m mesh.

Two separations are reported, both at the end of the load path:

* **update error** -- ``|Gamma(map) - Gamma(exact)| / |Gamma(exact)|`` at the
  *same* number of steps, so interpolation error cancels and what remains is
  the choice of map;
* **residual load-step dependence of the exact scheme** -- ``Gamma(exact, n)``
  against the finest run, which is what the exact map does *not* remove.

Each run is saved as ``results/wryness_update/{map}_n{steps}.npz`` as it
finishes and skipped when present, so the sweep can be interrupted and resumed.
Particle order is deterministic, so fields are compared particle by particle.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .micropolar_curved_cantilever import invariants, run

DX = 1.0
STEPS = (8, 15, 30, 60)
MAPS = ("exact", "identity", "transpose")
OUT = Path(__file__).resolve().parent / "results" / "wryness_update"


def one(mode, n):
    path = OUT / f"{mode}_n{n}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    print(f"  running {mode:9s} n = {n:3d} ...", flush=True)
    try:
        u, stepper, _, _, particles = run(DX, n_steps=n, wryness_map=mode)
        rec = dict(ok=True, u=u, wryness=particles.wryness.copy(),
                   quat=particles.quat.copy(), x=particles.x.copy())
    except Exception as exc:                # a failed run is a result too
        print(f"    failed: {type(exc).__name__}: {exc}", flush=True)
        rec = dict(ok=False, error=str(exc))
    rec["minutes"] = (time.time() - t0) / 60.0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rec)
    print(f"    done in {rec['minutes']:.1f} min", flush=True)
    return rec


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    steps = STEPS
    if "--steps" in sys.argv:
        steps = tuple(int(v) for v in
                      sys.argv[sys.argv.index("--steps") + 1].split(","))
    print("Wryness update on the curved cantilever, 1 m mesh")
    print(f"  maps {MAPS}, load steps {steps}\n")
    runs = {(m, n): one(m, n) for n in steps for m in MAPS}

    print(f"\n  {'n':>4} {'map':>10} {'along load':>11} {'in-plane':>9} "
          f"{'|Gam - Gam_exact|':>18} {'min':>6}")
    print("  " + "-" * 64)
    for n in steps:
        ex = runs[("exact", n)]
        for m in MAPS:
            r = runs[(m, n)]
            if not bool(r["ok"]):
                print(f"  {n:4d} {m:>10}   failed: {r['error']}")
                continue
            a, p, _ = invariants(r["u"])
            d = rel(r["wryness"], ex["wryness"]) if bool(ex["ok"]) else np.nan
            print(f"  {n:4d} {m:>10} {a:11.4f} {p:9.4f} {d:18.3e} "
                  f"{float(r['minutes']):6.1f}")

    fine = max(n for n in steps if bool(runs[("exact", n)]["ok"]))
    ref = runs[("exact", fine)]
    print(f"\n  exact map against its own n = {fine} run "
          f"(the load-step dependence it does not remove):")
    for n in steps:
        r = runs[("exact", n)]
        if bool(r["ok"]) and n != fine:
            a0, p0, _ = invariants(ref["u"])
            a, p, _ = invariants(r["u"])
            print(f"    n = {n:3d}   |Gam - Gam_fine| {rel(r['wryness'], ref['wryness']):.3e}"
                  f"   tip along load {100 * (a / a0 - 1):+.3f}%"
                  f"   in-plane {100 * (p / p0 - 1):+.3f}%")


if __name__ == "__main__":
    main()
