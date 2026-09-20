r"""Size effect in the notched beam: fixed material, geometrically similar specimens.

    python -m examples.micropolar_phase_field.beam_size_effect
    python -m examples.micropolar_phase_field.beam_size_effect --sizes 0.2
    python -m examples.micropolar_phase_field.beam_size_effect --sizes 0.2,0.4,0.8

``notched_beam`` varies the internal length at a fixed specimen, which isolates
the micropolar contribution but is not a size-effect experiment. This is: the
material is held fixed -- ``E``, ``nu``, ``N``, ``G_c``, ``l``, ``l_b`` -- and
every dimension of the beam scales with its depth ``h``. At each size the beam is
solved twice, with the internal length and without it, so the curve of nominal
strength against size can be compared with the classical phase-field one, which
has a size effect of its own through ``l/h`` and ``G_c/(E h)``.

**What to expect, stated before any run.** At ``h = 0.4`` the micropolar beam is
2.15x the classical one (row ``l_b/h = 0.25`` of the sweep). The ratio should be
larger at ``h = 0.2``, where ``l_b/h = 0.5``, and approach one as ``h >> l_b``.

**Why the range stops at h = 0.2.** ``l`` is a material constant here, so a
smaller beam has a larger ``l/h``. At ``l/h = 0.5`` the notch is smeared out and
the beam cracks at the clamp instead (measured, recorded in ``notched_beam``);
``l/h = 0.2`` still cracks at the notch. ``d_notch`` and ``d_clamp`` are
reported at every size so that is checked rather than assumed.

**Similarity rules.** Span ``2.5 h``, notch at ``0.875 h`` of depth ``0.3 h``,
tip displacement ``6 mm x h/0.4`` over 12 steps -- which makes ``h = 0.4``
*identical* to the existing sweep, so that pair is imported from
``beam_data.npz`` rather than recomputed. The damage-measurement windows scale
too; fixed windows would put the notch inside the "clamp" window on the small
beam. Nominal strength is ``sigma_N = 6 P (L - x_n) / (t h^2)``, the bending
stress at the notched section for a slab of thickness ``t``.

Each size is written to ``beam_size_h{h}.npz`` as soon as it finishes and skipped
if already present, so a long run can be interrupted and resumed. The largest
size has four times the particles of the current beam; time a small one first.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ._common import NZ, RESULTS, run_load_path
from .notched_beam import (DEPTH, DX, ELL_PF, NOTCH_DEPTH, NOTCH_X, SPAN,
                           STEPS, TIP_DISPLACEMENT, build)

L_B = 0.10                      # the material's internal length, fixed
H_REF = DEPTH                   # the size the existing sweep was run at
SIZES = (0.2, 0.4, 0.8)


def geometry(h):
    s = h / H_REF
    return dict(span=SPAN * s, depth=h, notch_x=NOTCH_X * s,
                notch_depth=NOTCH_DEPTH * s, tip=TIP_DISPLACEMENT * s)


def windows(x, g):
    """Damage at the notch and at the clamp, in windows that scale with the beam."""
    s = g["depth"] / H_REF
    clamp = x[:, 0] < 0.15 * s
    notch = np.abs(x[:, 0] - g["notch_x"]) < 0.10 * s
    return notch, clamp


def solve(h, l_b, extend=1):
    """``extend`` lengthens the path and multiplies the steps by the same factor.

    The physical displacement increment is therefore unchanged, so the first
    ``STEPS`` steps are bit-identical to the unextended run: extending a row
    that ended before its peak *continues* it rather than re-solving a
    different problem. An extended run stops once past its peak, since only the
    peak is wanted from it.
    """
    g = geometry(h)
    g["tip"] *= extend
    pf, stepper, parts, tip_nodes = build(l_b, ell_pf=ELL_PF, dx=DX, **g)
    t0 = time.time()
    out = run_load_path(pf, stepper, parts, tip_nodes, 1, STEPS * extend,
                        progress=f"h={h:g} l_b={l_b:g} x{extend}",
                        stop_past_peak=0.9 if extend > 1 else None)
    x = parts.x[:, :2].copy()
    notch, clamp = windows(x, g)
    d = out["fields"][-1]
    keys = ("symmetric", "relative_rotation", "curvature")
    return dict(disp=out["disp"], load=np.abs(out["load"]),
                reached=out["reached"],
                peak_step=int(np.argmax(np.abs(out["load"]))),
                shares=np.array([[sh[k] for k in keys] for sh in out["shares"]]),
                fields=np.asarray(out["fields"]), x=x,
                d_notch=float(d[notch].max()), d_clamp=float(d[clamp].max()),
                minutes=(time.time() - t0) / 60.0, extend=extend)


def from_sweep(h):
    """The h = 0.4 pair, taken from the completed sweep instead of re-solved."""
    src = RESULTS / "beam_data.npz"
    if abs(h - H_REF) > 1e-12 or not src.exists():
        return None
    d = np.load(src)
    ratios = list(np.round(d["l_b_over_h"] * H_REF, 12))
    if 0.0 not in ratios or L_B not in ratios:
        return None
    g = geometry(h)
    rows = {}
    for tag, l_b in (("ref", 0.0), ("mp", L_B)):
        i = ratios.index(l_b)
        x, F = d[f"row{i}_x"], d[f"row{i}_fields"]
        notch, clamp = windows(x, g)
        rows[tag] = dict(disp=d[f"row{i}_disp"], load=np.abs(d[f"row{i}_load"]),
                         reached=float(d[f"row{i}_reached"][0]),
                         peak_step=int(d[f"row{i}_peak_step"][0]),
                         shares=d[f"row{i}_shares"], fields=F, x=x,
                         d_notch=float(F[-1][notch].max()),
                         d_clamp=float(F[-1][clamp].max()), minutes=np.nan)
    return rows


def not_a_peak(row):
    load = np.asarray(row["load"])
    return int(np.argmax(load)) == len(load) - 1


def path(h):
    return RESULTS / f"beam_size_h{h:.2f}.npz"


def save(h, rows):
    out = {"h": h}
    for tag, r in rows.items():
        for k, v in r.items():
            out[f"{tag}_{k}"] = v
    np.savez_compressed(path(h), **out)


def load(h):
    d = np.load(path(h))
    rows = {}
    for tag in ("ref", "mp"):
        rows[tag] = {k[len(tag) + 1:]: d[k] for k in d.files
                     if k.startswith(tag + "_")}
    return rows


def nominal_strength(h, P):
    g = geometry(h)
    return 6.0 * P * (g["span"] - g["notch_x"]) / (NZ * DX * h ** 2)


def main():
    sizes = SIZES
    if "--sizes" in sys.argv:
        sizes = tuple(float(v) for v in
                      sys.argv[sys.argv.index("--sizes") + 1].split(","))
    print("Notched beam size series: fixed material, similar geometry")
    print(f"  l_b = {L_B}, l = {ELL_PF}, dx = {DX}; sizes h = {sizes}\n")

    results = {}
    for h in sizes:
        # rows loaded from disk go through the peak check below as well, so a
        # saved run that ended before its peak is extended, not trusted
        rows = load(h) if path(h).exists() else from_sweep(h)
        if path(h).exists():
            print(f"  h = {h:g}: already on disk, loading")
        elif rows is not None:
            print(f"  h = {h:g}: identical to the existing sweep, imported")
        else:
            if ELL_PF / h > 0.2 + 1e-9:
                print(f"  WARNING h = {h:g}: l/h = {ELL_PF / h:.2f} > 0.2, "
                      f"the notch may be smeared out -- read d_clamp")
            rows = {"ref": solve(h, 0.0), "mp": solve(h, L_B)}
        # A peak on the last step is not a peak: the load was still rising when
        # the path ended. Measured on the first h = 0.2 run -- the classical beam
        # read 1.333, 1.416, 1.479 over its last three steps, and the table
        # reported that last value as a peak, overstating the ratio. Peak
        # displacement grows more slowly than h, so small beams need a longer
        # path than the similarity rule gives them.
        for tag, l_b in (("ref", 0.0), ("mp", L_B)):
            ext = int(rows[tag].get("extend", 1))
            while not_a_peak(rows[tag]):
                if ext >= 8:
                    print(f"  h = {h:g} {tag}: still rising at 8x the path; "
                          f"giving up on this row")
                    break
                ext *= 2
                print(f"  h = {h:g} {tag}: load still rising at the end of "
                      f"the path; extending x{ext}")
                rows[tag] = solve(h, l_b, extend=ext)
        save(h, rows)
        results[h] = rows

    print(f"\n  {'h':>5} {'l/h':>5} {'l_b/h':>6} {'P ref':>9} {'P mp':>9} "
          f"{'ratio':>6} {'sN ref':>9} {'sN mp':>9} {'pk step':>8} "
          f"{'notch/clamp ref':>16} {'notch/clamp mp':>15} {'min':>9}")
    print("  " + "-" * 118)
    for h in sorted(results):
        r, m = results[h]["ref"], results[h]["mp"]
        pr, pm = float(np.max(r["load"])), float(np.max(m["load"]))
        mins = [float(v) for v in (r["minutes"], m["minutes"])]
        print(f"  {h:5.2f} {ELL_PF / h:5.2f} {L_B / h:6.3f} {pr:9.4g} {pm:9.4g} "
              f"{pm / pr:6.3f} {nominal_strength(h, pr):9.4g} "
              f"{nominal_strength(h, pm):9.4g} "
              f"{int(r['peak_step']):>3d}/{int(m['peak_step']):<4d} "
              f"{float(r['d_notch']):7.3f}/{float(r['d_clamp']):<8.3f} "
              f"{float(m['d_notch']):7.3f}/{float(m['d_clamp']):<7.3f} "
              f"{mins[0]:4.0f}/{mins[1]:<4.0f}"
              + ("   NOT A PEAK" if not_a_peak(r) or not_a_peak(m) else ""))
    print("\n  A peak step below ~4 means the peak is under-sampled; a clamp "
          "damage close to\n  the notch damage means the beam did not crack "
          "where the size series assumes.")

    if len(results) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hs = np.array(sorted(results))
        sr = [nominal_strength(h, float(np.max(results[h]["ref"]["load"])))
              for h in hs]
        sm = [nominal_strength(h, float(np.max(results[h]["mp"]["load"])))
              for h in hs]
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.loglog(hs, sr, "ko--", ms=4, label=r"$\ell_b = 0$")
        ax.loglog(hs, sm, "C3s-", ms=4,
                  label=rf"$\ell_b = {L_B:g}$ m")
        ax.set_xlabel("beam depth $h$ [m]")
        ax.set_ylabel(r"nominal strength $\sigma_N$ [Pa]")
        ax.legend(frameon=False, fontsize=8)
        p = RESULTS / "beam_size_effect.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
