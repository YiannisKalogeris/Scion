r"""Three-dimensional single-edge notched tension: the rotation axis moves.

    python -m examples.micropolar_phase_field.notched_tension_3d --time
    python -m examples.micropolar_phase_field.notched_tension_3d --quick
    python -m examples.micropolar_phase_field.notched_tension_3d
    python -m examples.micropolar_phase_field.notched_tension_3d --reduction

Every other example here is the plane-strain reduction of the formulation, in
which ``phi_x = phi_y = 0`` and ``Q`` is a rotation about ``e_3`` at every
material point. That reduction is blind to two of the choices the wryness
update has to make: with a fixed axis every increment is parallel to its own
gradient, so ``H`` acts as the identity on the vectors it is given, and
rotations about a common axis commute, so the Lagrange interpolation of the
rotation field is exact. Both are inactive in plane strain *by construction*,
which is why a three-dimensional configuration is needed to exercise them at
all. Behera, Das & Rahaman (EFM 295, 2024, Section 4.6) and Prakash, Behera,
Rahaman & Roy (IJMS 303, 2025, Section 5.6) include one for the same reason;
the second states it outright -- "unlike the 2D cases, the axis of rotation in
3D is not fixed, which highlights the generic nonlinearity in rotation update".

The configuration is the plate of ``notched_tension`` given a real thickness
``B`` with **free lateral surfaces** and no plane-strain constraint. Nothing
else changes. The free surfaces are what does the work: the interior is nearly
plane strain while the surface is nearly plane stress, so the through-thickness
variation releases ``phi_x`` and ``phi_y`` and the microrotation axis tilts away
from ``e_3``. No torsion or out-of-plane loading is required. Worth stating
because the obvious alternative is wrong: a merely *thicker* plate with the
plane-strain constraint left on costs the same and verifies nothing, since the
axis is still fixed everywhere.

**What can and cannot be compared with Behera et al.** Their Section 4.6 runs
``E = 118 GPa``, ``nu = 0.26``, ``G_c = 2 N/mm`` and ``l_s = l_b = l_t =
0.02 mm`` with ``chi = 1`` on a 1 mm plate, sweeping ``N = 0.2, 0.5, 0.7`` on
27537 nodes and 15217 tetrahedra with the Vol-Dev split. At four cells across
``l``, the resolution used throughout this work, their ``l_s`` needs a cell size
of 0.005 mm and some 200 x 200 x n_z cells, which is out of reach here for the
same reason their double-notch length scale was. So the comparison is of the
*dimensionless* statement -- the trend of the peak load with the coupling
number -- and not of the loads, since the AT2 peak stress scales as
``l^{-1/2}`` and absolute loads at different ``l`` are not comparable. Their
finding is a slight increase with ``N``; theirs is also the Vol-Dev split,
which they report agrees with the spectral one under tension (their Section
4.1), so the comparison is not confounded by that difference either.

**Three measurements, in the order of what they establish.**

- the out-of-plane microrotation, as the ratio of the largest in-plane
  component to the largest drilling component over the body. It is identically
  zero in plane strain, so a nonzero value is the evidence that the
  three-dimensional rotation machinery is being exercised rather than carried.
  See ``rotation_out_of_plane`` for why the obvious statistic, the maximum tilt
  angle, measures round-off instead.
- the reduction (``--reduction``): re-imposing the plane-strain constraint on
  the *same* three-dimensional mesh must return the plane-strain answer. This is
  the control that says the setup is correct rather than merely different, and
  without it a wrong 3-D answer is indistinguishable from a 3-D effect.
- the load path and the trend with ``N``.

The reaction is reported **per unit thickness** throughout. The slab here is
many cells thick where the plane-strain runs are one, so raw forces differ by
the thickness ratio and comparing them directly is the easiest way to
manufacture an effect that is not there.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from scion.core.rotation import rotvec_from_quat
from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.discretization.mpm.micropolar import attach_micropolar_state

from ._common import (PAD, RESULTS, check_resolution, make_problem,
                      plane_strain_bc, run_load_path, seed_notch)
from .notched_tension import (E_MOD, GC, NOTCH_B, NOTCH_WIDTH_FACTOR, NOTCH_X,
                              NOTCH_Y, NU, OVERSHOOT, SIDE,
                              critical_displacement)

# --- discretisation ---------------------------------------------------------
# Coarser in-plane than the plane-strain sweep (dx = 0.0125, l = 0.05) because
# the cost of a fully 3-D step is superlinear in the number of z-layers: the
# sparse factorisation's fill-in, not the assembly, is what grows. Measured on
# the notched beam, one Newton solve went 33.6 s -> 267.4 s when the layer count
# doubled from 4 to 8 while the particle count only doubled. l/dx = 4 is
# preserved, which is the floor that matters.
ELL_PF, DX, PPC, STEPS = 0.125, 0.03125, 2, 16
NZ = 8                       # cells through the thickness
ZPAD = 2                     # z is a free direction here, so it needs padding
# Behera et al. Sec. 4.6 sweep N = 0.2, 0.5, 0.7; their 0, 0.3, 0.5, 0.7 is the
# *double-notch* list that Section 6.4 already follows. Theirs plus N = 0, which
# is kept because W_rel = mu a |E^skw|^2 with a = N^2/(1-N^2) vanishes
# identically there -- an exact null is worth more than a matched list.
SWEEP = (0.0, 0.2, 0.5, 0.7)                      # coupling number N
L_B_OVER_L = 2.0             # held fixed while N is swept, as in Behera 4.6

QUICK_DX, QUICK_ELL, QUICK_STEPS, QUICK_NZ = 0.0625, 0.25, 6, 4
QUICK_SWEEP = (0.0, 0.7)

THICKNESS = NZ * DX          # 0.25, a quarter of the in-plane side


def build_grid_3d(dx, nz, zpad=ZPAD):
    nx = ny = int(round(SIDE / dx))
    grid = CartesianGrid(shape=(nx + 2 * PAD, ny + 2 * PAD, nz + 2 * zpad),
                         dx=dx,
                         origin=np.array([-PAD * dx, -PAD * dx, -zpad * dx]),
                         basis=get_basis("linear"))
    return grid


def seed_particles_3d(dx, ppc, nz):
    sub = dx / ppc
    ax = [np.arange(sub / 2, e, sub) for e in (SIDE, SIDE, nz * dx)]
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")], axis=1)
    return attach_micropolar_state(ParticleSet(pts, vol0=sub ** 3,
                                               density=1.0))


def build(N, *, l_b, ell_pf=ELL_PF, dx=DX, nz=NZ, ppc=PPC, u_top,
          plane_strain=False):
    """The plate with a real thickness. ``plane_strain=True`` is the control."""
    check_resolution(dx, ell_pf, l_b)
    grid = build_grid_3d(dx, nz)
    parts = seed_particles_3d(dx, ppc, nz)
    nan = np.nan
    # The elongation is split between the two loaded edges for the reason given
    # in notched_tension: a body whose centre drifts relative to a fixed grid
    # has a weight operator that no longer commutes with the mirror symmetry.
    # The lateral faces z = 0 and z = B carry no condition at all -- they are
    # the free surfaces that make this problem three-dimensional.
    bcs = [
        GridDirichlet(lambda c: c[:, 1] < 1e-9,
                      np.array([0.0, -0.5 * u_top, nan, nan, nan, nan])),
        GridDirichlet(lambda c: np.abs(c[:, 1] - SIDE) < 1e-9,
                      np.array([0.0, 0.5 * u_top, nan, nan, nan, nan])),
    ]
    if plane_strain:
        bcs.append(plane_strain_bc())
    # The direct solver, deliberately. BiCGSTAB with ILU was measured on these
    # tangents and is 3.5x faster and wrong: on three captured solves the
    # solution differed from the direct one by 3.4%, 17.8% and 100%, the first
    # of those while reporting info = 0 and a relative residual of 7e-13. The
    # tangent is ill-conditioned enough that a small residual does not bound the
    # solution error, and scion.solvers.linear.IterativeSolver logs a breakdown
    # and returns the vector anyway. Use scion.solvers.bicgstab, which raises,
    # if this is ever revisited.
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N, l_b=l_b,
                               gc=GC, ell_pf=ell_pf, bcs=bcs, max_stagger=30)
    # The notch is a plane through the thickness: seed_notch measures distance
    # in x-y only, so it is z-independent already and needs no change.
    seed_notch(pf, parts, dx, ell_pf, GC,
               tip=np.array([NOTCH_X, NOTCH_Y]), along="x",
               amplitude=NOTCH_B, width_factor=NOTCH_WIDTH_FACTOR)
    xyz = np.asarray(grid.node_coordinates())
    top = np.flatnonzero(np.abs(xyz[:, 1] - SIDE) < 1e-9)
    return pf, stepper, parts, top


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def rotation_out_of_plane(parts):
    """How far the microrotation axis leaves ``e_3``, reported three ways.

    Identically zero under the plane-strain constraint, which is what makes this
    the instrument for the example -- and why the ``--reduction`` control has to
    be run: a nonzero reading is evidence of a three-dimensional effect only if
    the same code returns zero when the constraint is imposed.

    **The maximum tilt angle over the particles is the wrong statistic**, and it
    was the first thing tried. The axis of a nearly unrotated particle is the
    direction of round-off, so restricting to the largest rotations by any fixed
    fraction still admits particles at the threshold whose angle is noise: the
    reading came out 82 degrees at ``N = 0`` and 90 at ``N = 0.7``, i.e. the
    same answer a random field gives, for a body whose largest rotation is
    2e-3 rad and is almost entirely the drilling component. What is reported
    instead is a ratio of *magnitudes over the body*, which no single
    small-rotation point can inflate, together with the tilt at the single
    most-rotated particle and a mean weighted by the squared rotation.
    """
    w = rotvec_from_quat(parts.quat)
    inplane = np.hypot(w[:, 0], w[:, 1])
    drill = np.abs(w[:, 2])
    mag = np.linalg.norm(w, axis=1)
    ratio = float(inplane.max() / drill.max()) if drill.max() > 0.0 else 0.0
    i = int(np.argmax(mag))
    tilt_at_max = float(np.degrees(np.arctan2(inplane[i], drill[i])))
    wt = mag ** 2
    tilt = np.degrees(np.arctan2(inplane, drill))
    tilt_w = float((tilt * wt).sum() / wt.sum()) if wt.sum() > 0.0 else 0.0
    return ratio, tilt_at_max, tilt_w, float(mag.max())


def through_thickness(parts, d_p, X0, dx, nz):
    """Peak damage and peak drilling rotation in each layer of particles.

    The interior is nearly plane strain and the free surfaces nearly plane
    stress, so a variation here is the mechanism the example rests on. Reported
    against the mid-plane rather than in absolute terms, since only the contrast
    is meaningful.
    """
    w = rotvec_from_quat(parts.quat)
    z = X0[:, 2]
    edges = np.linspace(0.0, nz * dx, nz + 1)
    lay = np.clip(np.digitize(z, edges) - 1, 0, nz - 1)
    out = []
    for k in range(nz):
        m = lay == k
        out.append((float(d_p[m].max()), float(np.abs(w[m, 2]).max())))
    return out


# ---------------------------------------------------------------------------

def run(N, *, l_b_over_l=L_B_OVER_L, ell_pf=ELL_PF, dx=DX, nz=NZ,
        steps=STEPS, plane_strain=False, progress=None):
    l_b = l_b_over_l * ell_pf
    u_top = OVERSHOOT * critical_displacement(ell_pf)
    pf, stepper, parts, top = build(N, l_b=l_b, ell_pf=ell_pf, dx=dx, nz=nz,
                                    u_top=u_top, plane_strain=plane_strain)
    X0 = parts.x.copy()
    thickness = nz * dx

    def probe(pf_, stepper_, parts_):
        return np.array(rotation_out_of_plane(parts_))

    out = run_load_path(pf, stepper, parts, top, 1, steps,
                        progress=progress or f"3-D N={N:g}", on_step=probe)
    load = np.abs(np.asarray(out["load"])) / thickness      # per unit thickness
    peak = int(np.argmax(load))
    P = np.asarray(out["extra"])
    d_p = pf.tables.interpolate(pf.d)
    out.update(N=N, l_b=l_b, thickness=thickness, load_per_thickness=load,
               peak_load=float(load[peak]), peak_step=peak,
               shares_at_peak=out["shares"][peak], probe=P,
               ratio=float(P[-1, 0]), ratio_at_peak=float(P[peak, 0]),
               tilt_at_max=float(P[-1, 1]), tilt_weighted=float(P[-1, 2]),
               w_max=float(P[:, 3].max()),
               layers=through_thickness(parts, d_p, X0, dx, nz),
               X0=X0, x=parts.x.copy(), d=d_p)
    return out


def table(rows):
    base = rows[0]["peak_load"]
    print(f"\n  {'N':>5} {'peak/thick':>11} {'normalised':>11} "
          f"{'|w_ip|/|w_z|':>12} {'tilt@wmax':>10} {'tilt_wtd':>9} "
          f"{'|w|max':>9} {'W_rel':>9} {'W_curv':>9} {'reached':>8}")
    print("  " + "-" * 104)
    for r in rows:
        s = r["shares_at_peak"]
        print(f"  {r['N']:5.2f} {r['peak_load']:11.5g} "
              f"{r['peak_load'] / base:11.4f} {r['ratio']:12.4f} "
              f"{r['tilt_at_max']:9.3f}d {r['tilt_weighted']:8.3f}d "
              f"{r['w_max']:9.3g} "
              f"{s['relative_rotation']:9.3g} {s['curvature']:9.3g} "
              f"{r['reached']:8.3f}")
    print(f"\n  spread of the peak load over the sweep: "
          f"{100 * (max(r['peak_load'] for r in rows) / base - 1):.2f}%")
    print("  Behera et al. (EFM 295, 2024) Sec. 4.6 report a slight increase "
          "in peak\n  load with N on the same configuration; the trend, not "
          "the magnitude, is\n  what is comparable (their l_s = 0.02 mm "
          "against 0.125 here).")


def main(quick=False, reduction=False):
    dx, ell, steps, nz, sweep = ((QUICK_DX, QUICK_ELL, QUICK_STEPS, QUICK_NZ,
                                 QUICK_SWEEP) if quick else
                                (DX, ELL_PF, STEPS, NZ, SWEEP))
    tag = "_quick" if quick else ""
    t0 = time.time()
    print(f"Three-dimensional notched tension: {SIDE} x {SIDE} x {nz * dx:g}, "
          f"free lateral faces")
    print(f"  dx = {dx:g}, l = {ell:g} (l/dx = {ell / dx:.1f}), "
          f"{nz} cells through the thickness, {steps} load steps")
    print(f"  l_b/l = {L_B_OVER_L:g} fixed; N swept over {sweep}\n")

    if reduction:
        print("  reduction control: the same 3-D mesh with the plane-strain "
              "constraint re-imposed")
        a = run(sweep[-1], ell_pf=ell, dx=dx, nz=nz, steps=steps,
                plane_strain=True, progress="plane-strain control")
        b = run(sweep[-1], ell_pf=ell, dx=dx, nz=nz, steps=steps,
                progress="fully 3-D")
        print(f"\n  {'':22} {'peak/thick':>12} {'|w_ip|/|w_z|':>13} "
              f"{'tilt@wmax':>11} {'tilt_wtd':>10}")
        for nm, r in (("plane strain (control)", a), ("fully 3-D", b)):
            print(f"  {nm:22} {r['peak_load']:12.6g} {r['ratio']:13.3e} "
                  f"{r['tilt_at_max']:10.3e}d {r['tilt_weighted']:9.3e}d")
        print(f"\n  peak load differs by "
              f"{100 * abs(b['peak_load'] / a['peak_load'] - 1):.2f}%. "
              f"Every out-of-plane measure of the control must be zero: "
              f"ratio {a['ratio']:.1e}, tilt {a['tilt_at_max']:.1e} deg.")
        np.savez_compressed(RESULTS / f"tension3d_reduction{tag}.npz",
                            ps_load=a["load_per_thickness"], ps_peak=a["peak_load"],
                            d3_load=b["load_per_thickness"], d3_peak=b["peak_load"],
                            ps_ratio=a["ratio"], d3_ratio=b["ratio"],
                            ps_tilt=a["tilt_at_max"], d3_tilt=b["tilt_at_max"],
                            disp=a["disp"], minutes=(time.time() - t0) / 60)
        return

    rows = [run(N, ell_pf=ell, dx=dx, nz=nz, steps=steps,
                progress=f"3-D N={N:g}") for N in sweep]
    table(rows)
    print("\n  through-thickness profile at the largest N "
          "(layer: peak d, peak |phi_z|)")
    for k, (d, w) in enumerate(rows[-1]["layers"]):
        print(f"    layer {k:2d}  d {d:.4f}   |phi_z| {w:.4e}")
    np.savez_compressed(
        RESULTS / f"tension3d{tag}.npz",
        N=np.array([r["N"] for r in rows]),
        peak=np.array([r["peak_load"] for r in rows]),
        ratio=np.array([r["ratio"] for r in rows]),
        tilt=np.array([r["tilt_at_max"] for r in rows]),
        disp=rows[0]["disp"],
        load=np.array([r["load_per_thickness"] for r in rows]),
        layers=np.array([r["layers"] for r in rows]),
        X0=rows[-1]["X0"], x=rows[-1]["x"], d=rows[-1]["d"],
        dx=dx, ell=ell, nz=nz, thickness=nz * dx,
        minutes=(time.time() - t0) / 60)
    print(f"\n  {(time.time() - t0) / 60:.0f} min")


def timing():
    """One Newton solve at each setting, to price the path before running it."""
    u_top = OVERSHOOT * critical_displacement(ELL_PF)
    for nz, dx, ell in ((QUICK_NZ, QUICK_DX, QUICK_ELL), (NZ, DX, ELL_PF)):
        pf, stepper, parts, top = build(0.5, l_b=L_B_OVER_L * ell, ell_pf=ell,
                                        dx=dx, nz=nz, u_top=u_top)
        t0 = time.time()
        stepper.step(parts, 1.0 / STEPS, 1.0 / STEPS)
        print(f"  nz={nz:2d} dx={dx:<8g} nodes {stepper.grid.n_nodes:7d} "
              f"particles {parts.n:8d}  {time.time() - t0:7.1f} s / Newton "
              f"solve", flush=True)


if __name__ == "__main__":
    if "--time" in sys.argv:
        timing()
    else:
        main(quick="--quick" in sys.argv, reduction="--reduction" in sys.argv)
