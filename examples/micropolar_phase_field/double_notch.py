r"""Double-notched plate under inclined loading: the coupling number, not the length.

    python -m examples.micropolar_phase_field.double_notch
    python -m examples.micropolar_phase_field.double_notch --dx 0.625 --N 0,0.5

The configuration of Behera, Das & Rahaman (2024), Section 4.3: a square plate
with two collinear notches entering from opposite edges at mid-height, its base
restrained and its top edge displaced at 45 degrees, so that both cracks grow in
mixed mode, kink, and may join. Their geometry, material and internal length are
kept; what is adapted is the phase-field length, for the reason below.

**Why this case.** The other examples here load in tension or in bending, sweep
``l_b`` at fixed coupling number, and have a crack path fixed by the notch. Here
the loading is mixed-mode, the path is an outcome rather than an input, and the
sweep is over the **coupling number** ``N`` at fixed ``l_b`` -- the one material
parameter the rest of the paper never varies. ``N = 0`` is well posed in this
sweep, unlike the ``l_b = 0`` row of the tension example, because the curvature
moduli remain and the rotational equation keeps its stiffness.

**What is adapted, and what that costs.** Their ``l_s = 0.525`` mm on a 100 mm
plate would need some 4.6 million material points at the four-cells-per-length
floor used throughout this work. The phase-field length is therefore raised to
``l = 4 dx``, which keeps the floor and the notch geometry (25 mm >> l) while
making the sweep affordable. The comparison with their Figs. 14 and 15 is then
qualitative -- whether the cracks kink, in which sense, and whether they join --
which is what they report; the peak loads are not comparable, since in AT2 the
peak stress scales as ``1/sqrt(l)``.

Their damage variable runs from 1 (intact) to 0 (broken), the opposite of the
convention here.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from scion.discretization.mpm import GridDirichlet

from ._common import (RESULTS, build_grid, check_resolution, make_problem,
                      plane_strain_bc, run_load_path, seed_particles)

# --- their geometry and material (mm, N) ------------------------------------
SIDE = 100.0                     # square plate
NOTCH = 25.0                     # each notch, entering at mid-height
G_MOD, NU, GC = 12.5e3, 0.3, 0.1          # N/mm^2, -, N/mm
E_MOD = 2.0 * G_MOD * (1.0 + NU)          # 32.5 GPa
L_B = 15.0                                # their bending length
N_SWEEP = (0.0, 0.3, 0.5, 0.7)

# --- our discretisation ------------------------------------------------------
DX, PPC, STEPS = 1.0, 2, 24
#: stop once the load has fallen to this fraction of its own peak. The crack
#: pattern -- which is what this example reports -- is complete within two or
#: three increments of the peak; the long tail after it costs more than the
#: rest of the path together (measured at dx = 0.8: 4 h to the peak, then 1.5 h
#: per post-peak increment) and adds nothing to the figure.
STOP_PAST_PEAK = 0.3
ELL_OVER_DX = 4.0
U_MAX = 0.06                     # per component; the 45 degree load is (u, u)
SEED_AMPLITUDE, SEED_WIDTH = 1.0e3, 0.25


def seed_two_notches(pf, parts, dx, ell):
    """Both notches at once: ``seed_notch`` writes ``H`` and would overwrite one.

    Each notch is a segment along ``y = SIDE/2``; the field is a half-plane
    behind the tip and radial ahead of it, as in the single-notch case, and the
    two are combined by taking the larger.
    """
    x = parts.x[:, :2]
    r0 = max(SEED_WIDTH * ell, dx)
    H0 = np.zeros(parts.n)
    for tip, behind in ((np.array([NOTCH, 0.5 * SIDE]), x[:, 0] <= NOTCH),
                        (np.array([SIDE - NOTCH, 0.5 * SIDE]),
                         x[:, 0] >= SIDE - NOTCH)):
        dist = np.where(behind, np.abs(x[:, 1] - tip[1]),
                        np.hypot(x[:, 0] - tip[0], x[:, 1] - tip[1]))
        inside = dist <= r0
        H0[inside] = np.maximum(
            H0[inside],
            SEED_AMPLITUDE * GC / (2.0 * ell) * (1.0 - dist[inside] / r0))
    pf.H = H0
    return pf


def build(n_coupling, dx=DX, ell=None):
    ell = ELL_OVER_DX * dx if ell is None else ell
    check_resolution(dx, ell, L_B)
    grid, _, _ = build_grid((SIDE, SIDE), dx)
    parts = seed_particles((SIDE, SIDE), dx, PPC)
    nan = np.nan
    bcs = [
        # base restrained in plane; the drawing shows rollers, but with the top
        # edge driven in both directions a roller base leaves a rigid horizontal
        # mode, so both components are held
        GridDirichlet(lambda c: c[:, 1] < 1e-9,
                      np.array([0.0, 0.0, nan, nan, nan, nan])),
        # top edge displaced at 45 degrees: equal components, as they impose
        GridDirichlet(lambda c: np.abs(c[:, 1] - SIDE) < 1e-9,
                      np.array([U_MAX, U_MAX, nan, nan, nan, nan])),
        plane_strain_bc(),
    ]
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=n_coupling,
                               l_b=L_B, gc=GC, ell_pf=ell, bcs=bcs,
                               max_stagger=30)
    seed_two_notches(pf, parts, dx, ell)
    xy = np.asarray(grid.node_coordinates())
    top = np.flatnonzero(np.abs(xy[:, 1] - SIDE) < 1e-9)
    return pf, stepper, parts, top


def run(n_coupling, dx=DX, ell=None, steps=STEPS):
    ell = ELL_OVER_DX * dx if ell is None else ell
    pf, stepper, parts, top = build(n_coupling, dx, ell)
    X0 = parts.x.copy()
    t0 = time.time()

    def reaction(pf_, stepper_, parts_):
        """Both components on the driven edge; the load is at 45 degrees."""
        f = stepper_.internal_force(parts_)
        return np.array([f[top * 6 + 0].sum(), f[top * 6 + 1].sum()])

    out = run_load_path(pf, stepper, parts, top, 1, steps,
                        progress=f"N={n_coupling:g}", on_step=reaction,
                        stop_past_peak=STOP_PAST_PEAK)
    R = np.asarray(out["extra"])                       # (n_steps, 2)
    load45 = (R[:, 0] + R[:, 1]) / np.sqrt(2.0)        # along the load direction
    k = int(np.argmax(np.abs(load45)))
    np.savez_compressed(
        # dx belongs in the name: a coarse trial run and a production run are
        # different calculations, and a file that records only N would let the
        # sweep silently reuse one for the other
        RESULTS / f"double_notch_N{n_coupling:g}_dx{dx:g}.npz",
        N=n_coupling, dx=dx, ell=ell, X0=X0, disp=out["disp"],
        reaction=R, load45=load45, peak_step=k, reached=out["reached"],
        fields=np.asarray(out["fields"]), x=parts.x[:, :2].copy(),
        shares=np.array([[s[q] for q in ("symmetric", "relative_rotation",
                                         "curvature")] for s in out["shares"]]),
        minutes=(time.time() - t0) / 60.0)
    print(f"    N = {n_coupling:g}: peak {abs(load45[k]):.4g} N at "
          f"t = {out['disp'][k]:.3f}, reached {out['reached']:.3f}, "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)


def main():
    dx = DX
    sweep = N_SWEEP
    if "--dx" in sys.argv:
        dx = float(sys.argv[sys.argv.index("--dx") + 1])
    if "--N" in sys.argv:
        sweep = tuple(float(v) for v in
                      sys.argv[sys.argv.index("--N") + 1].split(","))
    ell = ELL_OVER_DX * dx
    n = int(round(SIDE / dx))
    print("Double-notched plate under inclined loading, after Behera et al. (2024)")
    print(f"  {SIDE} x {SIDE} mm, two {NOTCH} mm notches at mid-height, "
          f"E = {E_MOD / 1e3:g} GPa, nu = {NU}, G_c = {GC} N/mm")
    print(f"  l_b = {L_B} mm fixed, N swept over {sweep}")
    print(f"  dx = {dx:g} mm, l = {ell:g} mm (l/dx = {ELL_OVER_DX:g}), "
          f"{n * n * PPC * PPC * PPC:,} particles, {steps_note()}\n")
    for nc in sweep:
        path = RESULTS / f"double_notch_N{nc:g}_dx{dx:g}.npz"
        if path.exists():
            print(f"    N = {nc:g}: already on disk")
            continue
        run(nc, dx, ell)


def steps_note():
    return f"{STEPS} load steps to u = {U_MAX} mm per component"


if __name__ == "__main__":
    main()
