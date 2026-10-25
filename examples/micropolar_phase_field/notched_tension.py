r"""Single-edge notched tension: the classical limit, and the absence of a
size effect.

    python -m examples.micropolar_phase_field.notched_tension
    python -m examples.micropolar_phase_field.notched_tension --quick
    python -m examples.micropolar_phase_field.notched_tension --no-fem
    python -m examples.micropolar_phase_field.notched_tension --to-peak

``--to-peak`` stops each row once its load has fallen to 90% of its own
maximum. Everything reported at the peak -- the peak load and the three
driving-force shares -- is unchanged, because the stop is tested after the
increment update and so alters no step before it; what is skipped is the
post-peak branch, which is most of the cost. Use it to recover the table from a
sweep whose figures are already in hand. It writes under its own ``_peak`` tag
so it cannot overwrite a completed run.

Two parts, in the order they are argued in the paper.

**The classical limit.** With the coupling switched off, the phase-field MPM
solve is compared against a phase-field *finite element* solution of the same
problem at identical parameters. This is the check that the MPM discretisation
of the damage problem is sound before the internal length is switched on;
Kakouris & Triantafyllou (IJNME 2017, Section 5.1) make the same comparison and
report that the two are practically indistinguishable.

**The absence of a size effect.** The internal length is then swept over a
fourfold range -- ``l_b/l`` of 1, 2 and 4, plus the limit ``l_b = 0`` -- with
the geometry, the mesh, the phase-field length and ``G_c`` held fixed. The result is a *null*, and it is the point of the
example rather than a disappointment: a notched plate in tension is
stretch-dominated, so the microrotation field is nearly uniform, the wryness is
small however ``l_b`` compares with the notch or the specimen, and the internal
length has almost no route into the damage evolution. Contrast
``notched_beam``, which is the same machinery on a bending problem and where
the peak load moves substantially.
"""
from __future__ import annotations

import sys

import numpy as np

from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.materials.fracture import PhaseFieldModel
from scion.multiphysics.phase_field_mpm import PhaseFieldMPMProblem

import examples.phase_field.single_edge_notched_tension as _fem

from ._common import (NZ, RESULTS, build_grid, check_resolution,
                      make_problem, plane_strain_bc, run_load_path,
                      seed_notch, seed_particles)

# --- material ---------------------------------------------------------------
# Taken from the finite element reference rather than restated, because part 1
# compares two discretisations of the *same* problem and a divergence here is
# invisible: the reference works in kN/mm^2, where E = 210, and writing the
# same modulus in MPa as 2.1e5 while keeping G_c = 2.7e-3 leaves a body a
# thousand times stiffer at identical fracture energy. Both solves still run,
# both still crack, and the load paths differ by a factor of ten.
E_MOD, NU, GC = _fem.E, _fem.NU, _fem.GC
N_COUPLING = 0.35

# --- geometry ---------------------------------------------------------------
SIDE = 1.0
NOTCH_X, NOTCH_Y = 0.5, 0.5
# The seeded notch, matched to examples.phase_field.single_edge_notched_tension
# so that part 1 compares two discretisations of the *same* problem. Both
# numbers are calibrated there against the exact surface energy of a crack of
# length 0.5, and changing either one alone changes the peak load by tens of
# percent while still looking entirely plausible.
NOTCH_B, NOTCH_WIDTH_FACTOR = 1.0e3, 0.25

# --- discretisation ---------------------------------------------------------
# dx = 0.0125 puts 4 cells across the phase-field length and at least 4 across
# every non-zero l_b in the sweep, which is the floor checked in _common.
ELL_PF, DX, PPC, STEPS = 0.05, 0.0125, 2, 16
SWEEP = (0.0, 1.0, 2.0, 4.0)                      # l_b / l

QUICK_DX, QUICK_ELL, QUICK_STEPS = 0.025, 0.10, 8
QUICK_SWEEP = (0.0, 2.0)

#: How far past the estimated peak to drive the load. The peak of a *notched*
#: specimen is below the homogeneous estimate, so this only has to be greater
#: than one; adaptive cut-back stops the path where it genuinely runs out.
OVERSHOOT = 1.2


def critical_displacement(ell_pf: float) -> float:
    """AT2 homogeneous-bar estimate of the displacement at peak load.

    ``sigma_c = sqrt(9 E G_c / (32 l))`` for AT2, so the strain at peak is
    ``sigma_c / E`` and the displacement is that times the height. Used to set
    the load range from the *material*: a fixed displacement chosen for one
    ``l`` is wrong for another, and being 30x past the peak looks like a
    converged run with an implausibly small reaction rather than an error.
    """
    return SIDE * np.sqrt(9.0 * GC / (32.0 * E_MOD * ell_pf))


# ---------------------------------------------------------------------------
# part 1: the classical limit, MPM against FEM
# ---------------------------------------------------------------------------

def build_classical_mpm(ell_pf, dx, ppc=PPC):
    """Phase-field MPM with no micropolar degrees of freedom at all.

    Deliberately the *classical* problem class, not the micropolar one with
    ``l_b = 0``: the comparison is against a classical finite element solve,
    so anything the micropolar solver adds would confound it.
    """
    sub, pad = dx / ppc, 2
    n = int(round(SIDE / dx))
    grid = CartesianGrid(shape=(n + 2 * pad,) * 2, dx=dx,
                         origin=np.array([-pad * dx, -pad * dx]),
                         basis=get_basis("linear"))
    ax = [np.arange(sub / 2, SIDE, sub)] * 2
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")], axis=1)
    parts = ParticleSet(pts, vol0=sub ** 2, density=1.0)
    model = PhaseFieldModel(E_MOD, NU, GC, ell_pf, split="spectral",
                            coupling="hybrid", dissipation="AT2")
    p = PhaseFieldMPMProblem(grid, parts, model, thickness=1.0,
                             stagger_tol=10.0, max_stagger=50)
    xy = np.asarray(grid.node_coordinates())[:, :2]
    bot = np.flatnonzero(np.abs(xy[:, 1]) < 1e-9)
    top = np.flatnonzero(np.abs(xy[:, 1] - SIDE) < 1e-9)
    if bot.size == 0 or top.size == 0:
        raise ValueError(
            f"dx = {dx:g} puts no grid node on y = 0 or y = {SIDE}, so the "
            f"boundary conditions select nothing and every reaction reads "
            f"zero. Choose dx that divides {SIDE}.")
    p.u.prescribe(bot, 0, 0.0)
    # The elongation is split symmetrically between the two edges, matching the
    # micropolar rows, so the whole study describes one boundary value problem.
    # Here it is a rigid translation and changes nothing -- this path is small strain and never advects its particles --
    # which is exactly what makes it safe to unify on.
    p.set_displacement_control(bot, 1, -0.5)
    p.set_displacement_control(top, 1, 0.5)     # factor = u in mm
    # The reported reaction is the force on the LOADED edge. Both edges are now
    # control dofs and their reactions cancel by equilibrium, so the control set
    # is narrowed back to the top before any step runs.
    p._control_dofs = np.array(
        sorted(p.u.dof_map.dof(int(n), 1) for n in top), dtype=np.int64)

    xq = np.einsum("eqa,ead->eqd", p.N,
                   p.domain.coordinates[p.conn]).reshape(-1, 2)
    behind = xq[:, 0] <= NOTCH_X
    dist = np.where(behind, np.abs(xq[:, 1] - NOTCH_Y),
                    np.hypot(xq[:, 0] - NOTCH_X, xq[:, 1] - NOTCH_Y))
    # identical to fem.notch_history: same amplitude and same width, or the
    # two solves are given different notches and the comparison is meaningless
    width = max(NOTCH_WIDTH_FACTOR * ell_pf, dx)
    H0 = np.zeros(xq.shape[0])
    inside = dist <= width
    H0[inside] = NOTCH_B * GC / (2.0 * ell_pf) * (1.0 - dist[inside] / width)
    p.state["H"] = H0
    p.state.commit()
    return p, top


def classical_reference(ell_pf, dx, steps):
    """The classical solve, as the baseline the micropolar rows are read against.

    Preferred over the micropolar run at ``l_b = 0`` for one structural reason:
    with every curvature modulus zero the term ``mu a |skew E|^2`` is the only
    thing tying ``Q`` to the continuum rotation, and it is degraded by ``g(d)``
    like everything else. In a fully broken band ``g -> k``, the constraint is
    switched off, the microrotation is released, and the tangent goes singular.
    The classical solver has no microrotation to lose and completes the
    post-peak branch.

    The two are not interchangeable in *units*: this path is two-dimensional
    with unit thickness, the micropolar path a slab one cell thick. Reactions
    are returned per unit thickness so the two are comparable -- measured, the
    ratio of the raw loads is 82 against a slab thickness of 1/80.
    """
    from scion.core.logging import quiet
    u_bar = OVERSHOOT * critical_displacement(ell_pf)
    problem, top = build_classical_mpm(ell_pf, dx)
    rec, fields = [], []

    def record(r, prob):
        if not r.converged:
            return
        rec.append((r.factor / u_bar, abs(r.reactions["control"])))
        # the damage at the particles, so this row can be drawn in the same
        # panel grid as the micropolar ones
        fields.append(prob.tables.interpolate(prob.d.values).copy())

    with quiet():
        problem.run(np.linspace(0.0, u_bar, steps + 1)[1:], callback=record,
                    adaptive=True, stop_on_failure=True)
    a = np.array(rec)
    return {"disp": a[:, 0], "load": a[:, 1], "peak_load": a[:, 1].max(),
            "reached": a[-1, 0], "cutbacks": 0, "l_b": 0.0,
            "l_b_over_l": 0.0, "classical": True, "u_top": u_bar,
            "fields": fields, "x": problem.particles.x[:, :2].copy(),
            "shares_at_peak": {"symmetric": 1.0, "relative_rotation": 0.0,
                               "curvature": 0.0}}


def classical_limit(ell_pf, dx, steps):
    """MPM and FEM load paths on the same problem at the same parameters.

    Both sides are driven by ``run`` with adaptive stepping, which is what
    brittle fracture needs -- past the peak the crack releases more energy in
    one increment than it can absorb and no number of staggered iterations
    helps. The two therefore land on *different* displacement samples, so the
    comparison is made by interpolating both onto a common grid rather than by
    pairing step indices, which would silently compare different load levels.
    """
    import examples.phase_field.single_edge_notched_tension as fem
    from scion.core.logging import quiet

    schedule = np.linspace(0.0, OVERSHOOT * critical_displacement(ell_pf),
                           steps + 1)[1:]

    def drive(problem):
        rec = []
        with quiet():
            problem.run(schedule, callback=lambda r, _: rec.append(
                (r.factor, abs(r.reactions["control"]))) if r.converged
                else None, adaptive=True, stop_on_failure=True)
        return np.array(rec) if rec else np.zeros((0, 2))

    mpm_problem, _ = build_classical_mpm(ell_pf, dx)
    mpm = drive(mpm_problem)

    # the finite element reference, at the same l and a comparable element
    # size so that neither discretisation is favoured by resolution
    _, _, fem_problem = fem.build(ell=ell_pf, h_fine=dx)
    fe = drive(fem_problem)

    if mpm.size == 0 or fe.size == 0:
        raise RuntimeError("one of the two solves produced no converged step")
    hi = min(mpm[-1, 0], fe[-1, 0])
    common = np.linspace(hi / steps, hi, steps)
    return (common,
            np.interp(common, mpm[:, 0], mpm[:, 1]),
            np.interp(common, fe[:, 0], fe[:, 1]),
            mpm, fe)


# ---------------------------------------------------------------------------
# part 2: the internal-length sweep
# ---------------------------------------------------------------------------

def build(l_b, ell_pf=ELL_PF, dx=DX, ppc=PPC, u_top=None):
    check_resolution(dx, ell_pf, l_b)
    u_top = OVERSHOOT * critical_displacement(ell_pf) if u_top is None \
        else u_top
    grid, nx, ny = build_grid((SIDE, SIDE), dx)
    parts = seed_particles((SIDE, SIDE), dx, ppc)
    nan = np.nan
    # Both edges are pinned in x, exactly as the finite element reference and
    # the classical solve of part 1 do. Pinning only one corner instead leaves
    # the specimen a valid problem right up until the crack separates it -- and
    # then the upper half has no x constraint at all, translates rigidly, and
    # the tangent goes singular. The symptom is a load step that stops
    # converging at the moment of failure, which reads exactly like a physical
    # limit and is not one. It also made parts 1 and 2 different boundary value
    # problems, so the classical comparison did not licence the sweep.
    #
    # The elongation is split symmetrically between the two edges rather than
    # clamping one and driving the other. In the continuum the two differ by a
    # rigid translation and nothing else; on a *fixed* background grid they do
    # not. This solve is updated-Lagrangian, so the particles advect and the
    # stencil is rebuilt against the grid every step -- and a body that drifts
    # relative to the grid has a weight operator that no longer commutes with
    # the mirror y -> SIDE - y. Measured with no solve at all: the commutation
    # error is 1.7e-14 at zero drift and grows linearly, reaching 0.42 at half
    # a cell. Clamped-and-driven, that seeds an asymmetry the marginally stable
    # mode-I crack path amplifies -- the damage field goes from 4e-11 at the
    # first step to 1.4e-01 once the crack runs, and the crack leaves the
    # symmetry plane by 6% of the height. Driven symmetrically the same elastic
    # solve is mirror symmetric to 1e-16.
    #
    # The classical comparison row does not need this: PhaseFieldMPMProblem is
    # small strain and never advects its particles, so its weights stay
    # mirror-exact for the whole run. That difference, not the micropolar
    # formulation, is why its crack was the only straight one.
    bcs = [
        GridDirichlet(lambda c: c[:, 1] < 1e-9,
                      np.array([0.0, -0.5 * u_top, nan, nan, nan, nan])),
        GridDirichlet(lambda c: np.abs(c[:, 1] - SIDE) < 1e-9,
                      np.array([nan, 0.5 * u_top, nan, nan, nan, nan])),
        plane_strain_bc(),
    ]
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING,
                               l_b=l_b, gc=GC, ell_pf=ell_pf, bcs=bcs,
                               max_stagger=30)
    seed_notch(pf, parts, dx, ell_pf, GC,
               tip=np.array([NOTCH_X, NOTCH_Y]), along="x",
               amplitude=NOTCH_B, width_factor=NOTCH_WIDTH_FACTOR)
    xy = np.asarray(grid.node_coordinates())
    top = np.flatnonzero(np.abs(xy[:, 1] - SIDE) < 1e-9)
    return pf, stepper, parts, top


def run(l_b_over_l, ell_pf=ELL_PF, steps=STEPS, to_peak=False, **kw):
    l_b = l_b_over_l * ell_pf
    u_top = OVERSHOOT * critical_displacement(ell_pf)
    pf, stepper, parts, top = build(l_b, ell_pf=ell_pf, u_top=u_top, **kw)
    out = run_load_path(pf, stepper, parts, top, 1, steps,
                        progress=f'l_b/l={l_b_over_l:g}',
                        stop_past_peak=0.9 if to_peak else None)
    peak = int(np.argmax(np.abs(out["load"])))
    out.update(l_b=l_b, l_b_over_l=l_b_over_l, u_top=u_top,
               peak_load=abs(out["load"][peak]), peak_step=peak,
               shares_at_peak=out["shares"][peak],
               x=parts.x[:, :2].copy())
    return out


# ---------------------------------------------------------------------------

def table(rows, dx):
    base = rows[0]["peak_load"]
    print("\n  (row 0 is the CLASSICAL solve, per unit thickness; the "
          "micropolar rows are\n   scaled to the same thickness so the "
          "peak loads are directly comparable)")
    print(f"\n  {'l_b/l':>7} {'l_b':>8} {'l_b/dx':>7} {'peak load':>11} "
          f"{'normalised':>11} {'W_rel':>9} {'W_curv':>9} {'reached':>8} "
          f"{'cuts':>5}")
    print("  " + "-" * 82)
    for r in rows:
        s = r["shares_at_peak"]
        cells = "--" if r["l_b"] == 0 else f"{r['l_b'] / dx:.1f}"
        tag = "class." if r.get("classical") else f"{r['l_b_over_l']:7.2f}"
        print(f"  {tag:>7} {r['l_b']:8.4f} {cells:>7} "
              f"{r['peak_load']:11.4g} {r['peak_load'] / base:11.4f} "
              f"{s['relative_rotation']:9.3g} {s['curvature']:9.3g} "
              f"{r['reached']:8.3f} {r['cutbacks']:5d}")


def save_results(rows, tag=""):
    """Write the raw sweep to a .npz so figures never require a re-solve.

    Learned the hard way: a four-hour run that only emits PNGs cannot be
    re-plotted, so any change to the figure costs the whole run again.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, r in enumerate(rows):
        tagi = f"classical" if r.get("classical") else f"lb{r['l_b_over_l']:g}"
        out[f"{tagi}_disp"] = r["disp"]
        out[f"{tagi}_load"] = r["load"]
        out[f"{tagi}_reached"] = np.array([r["reached"]])
        out[f"{tagi}_peak"] = np.array([r["peak_load"]])
        out[f"{tagi}_peak_step"] = np.array([r.get("peak_step", -1)])
        # The three driving-force shares, per load step and in a fixed order.
        # These are what the paper's sweep table reports, and leaving them out
        # once meant a completed run whose table existed only in the terminal
        # scrollback -- the figures could be redrawn from the .npz and the
        # table could not.
        if "shares" in r:
            keys = ("symmetric", "relative_rotation", "curvature")
            out[f"{tagi}_shares"] = np.array(
                [[sh[k] for k in keys] for sh in r["shares"]])
        if "fields" in r:
            out[f"{tagi}_fields"] = np.asarray(r["fields"])
            out[f"{tagi}_x"] = r["x"]
    p = RESULTS / f"tension_data{tag}.npz"
    np.savez_compressed(p, **out)
    print(f"\n  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")


def make_figures(rows, classical, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figures")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    if classical is not None:
        us, mpm, fem, raw_m, raw_f = classical
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.plot(raw_f[:, 0], raw_f[:, 1], "k-", lw=1.6,
                label="phase-field FEM")
        ax.plot(raw_m[:, 0], raw_m[:, 1], "o", ms=4, mfc="none",
                color="crimson", label="phase-field MPM")
        ax.set_xlabel("applied displacement")
        ax.set_ylabel("reaction")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Notched tension: MPM against FEM, identical parameters",
                     fontsize=9)
        p = RESULTS / f"tension_classical{tag}.png"
        fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
        print(f"\n  wrote {p}")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for r in rows:
        lab = "classical" if r.get("classical") else \
            rf"$\ell_b/\ell = {r['l_b_over_l']:g}$"
        st = "k--" if r.get("classical") else "o-"
        ax.plot(r["disp"] * r["u_top"], np.abs(r["load"]), st, ms=3.5,
                lw=1.6 if r.get("classical") else 1.4, label=lab)
    ax.set_xlabel("applied displacement")
    ax.set_ylabel("reaction")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Notched tension: the curves lie on top of one another",
                 fontsize=9)
    p = RESULTS / f"tension_load{tag}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")

    # a load level every row actually reached, so the panels are comparable
    panels = [q for q in rows if "fields" in q]
    lam_c = min(r["reached"] for r in panels)
    n = len(panels)

    # Two rows when the classical solve is present: the top row is the pair
    # WITHOUT an internal length -- classical, and micropolar at l_b = 0, which
    # should agree -- and the bottom row is the micropolar sweep. The grouping
    # is the argument: top row is the control, bottom row is the variable.
    n_top = sum(1 for q in panels if q.get("classical") or q["l_b"] == 0.0)
    two_row = n >= 5 and 0 < n_top < n
    if two_row:
        n_bot = n - n_top
        wide = max(n_top, n_bot)
        fig = plt.figure(figsize=(2.9 * wide, 5.8))
        # every panel spans two columns so all are the SAME size; the shorter
        # row is centred by an offset rather than stretched to fill the width,
        # which would make the control panels bigger than the sweep they are
        # meant to be compared against
        gs = fig.add_gridspec(2, 2 * wide, hspace=0.30, wspace=0.30)
        axes = []
        for row, count in ((0, n_top), (1, n_bot)):
            off = (2 * wide - 2 * count) // 2
            axes += [fig.add_subplot(gs[row, off + 2 * i: off + 2 * i + 2])
                     for i in range(count)]
    else:
        fig, ax = plt.subplots(1, n, figsize=(2.8 * n, 2.8), squeeze=False)
        axes = list(ax[0])

    for a, r in zip(axes, panels):
        k = int(np.argmin(np.abs(r["disp"] - lam_c)))
        s = a.scatter(r["x"][:, 0], r["x"][:, 1], c=r["fields"][k], s=2,
                      cmap="inferno", vmin=0.0, vmax=1.0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title("classical (no microrotation)" if r.get("classical")
                    else rf"micropolar, $\ell_b/\ell = {r['l_b_over_l']:g}$",
                    fontsize=9)
    fig.colorbar(s, ax=axes, shrink=0.7, label="$d$")
    fig.suptitle(rf"damage at a common load level $\lambda = {lam_c:.3f}$",
                 fontsize=10, y=0.97 if two_row else 1.02)
    p = RESULTS / f"tension_damage{tag}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def main(quick=False, skip_fem=False, to_peak=False):
    dx, ell, steps, sweep = DX, ELL_PF, STEPS, SWEEP
    if quick:
        dx, ell, steps, sweep = QUICK_DX, QUICK_ELL, QUICK_STEPS, QUICK_SWEEP

    print("Micropolar phase-field fracture: single-edge notched tension")
    print(f"  {SIDE} x {SIDE} plate, notch to x = {NOTCH_X} at mid-height")
    print(f"  E = {E_MOD:g}, nu = {NU}, N = {N_COUPLING}, G_c = {GC}")
    print(f"  l = {ell:g} ({ell / dx:.1f} cells), dx = {dx:g}, "
          f"{PPC} particles per cell per direction, {steps} load steps")
    if quick:
        print("  --quick: coarse grid, longer phase-field length and fewer "
              "load steps.\n  The null survives; the numbers are not "
              "converged and must not be quoted.")

    classical = None
    if not skip_fem:
        print("\n  Part 1: the classical limit, MPM against FEM")
        classical = classical_limit(ell, dx, steps)
        us, mpm, fem, raw_m, raw_f = classical
        pk_m, pk_f = raw_m[:, 1].max(), raw_f[:, 1].max()
        print(f"    peak load   MPM {pk_m:.6g}   FEM {pk_f:.6g}   "
              f"differ {100 * abs(pk_m - pk_f) / pk_f:.2f}%")
        # Only up to the peak, and only where the load is a real fraction of
        # it. A relative error past the peak is an artefact: the load has
        # collapsed to a percent or two of its maximum, so a negligible
        # absolute gap reads as tens of percent and says nothing about the two
        # discretisations agreeing. The honest global number is the largest
        # absolute deviation measured against the peak load, reported below.
        u_peak = us[int(np.argmax(fem))]
        rise = (us <= u_peak) & (fem > 0.1 * pk_f)
        if rise.any():
            rel = np.abs(mpm[rise] - fem[rise]) / fem[rise]
            print(f"    rising branch to peak: {100 * rel.max():.2f}% worst, "
                  f"{100 * rel.mean():.2f}% mean over {int(rise.sum())} points")
        gap = np.abs(mpm - fem).max() / pk_f
        print(f"    whole path: max deviation {100 * gap:.2f}% of the peak "
              f"load")

    print("\n  Part 2: the internal-length sweep, everything else fixed")
    ref = classical_reference(ell, dx, steps)
    print(f"    classical baseline: peak {ref['peak_load']:.5g} per unit "
          f"thickness, reached lambda = {ref['reached']:.3f}")
    thickness = NZ * dx
    rows = []
    for r in sweep:
        o = run(r, ell_pf=ell, dx=dx, steps=steps, to_peak=to_peak)
        o["load"] = o["load"] / thickness      # to unit thickness
        o["peak_load"] = o["peak_load"] / thickness
        rows.append(o)
    rows = [ref] + rows
    table(rows, dx)

    # A row that stopped because it was ASKED to, once past its own peak, is
    # not a truncated run: everything reported here is evaluated at the peak,
    # which it reached. A row that stopped because it stopped converging is a
    # different thing entirely, and the two must not print the same warning.
    short = [r for r in rows if r["reached"] < 0.999
             and not r.get("stopped_past_peak")]
    if to_peak:
        n = sum(1 for r in rows if r.get("stopped_past_peak"))
        print(f"\n    --to-peak: {n} of {len(rows)} rows stopped once the load "
              f"had fallen to 90% of\n    their own maximum. The peak load and "
              f"the three shares are evaluated at the\n    peak, so they are "
              f"unaffected; the post-peak branch and the damage figures\n"
              f"    are not available from this run.")
    if short:
        print(f"\n    WARNING: {len(short)} of {len(rows)} rows stopped before "
              f"the end of the load path\n    without reaching a peak. Their "
              f"'peak load' is the last converged point, not\n    a peak, and "
              f"the sweep below compares a truncated path against a complete\n"
              f"    one. Do not report it.")

    peaks = np.array([r["peak_load"] for r in rows])
    curv = max(r["shares_at_peak"]["curvature"] for r in rows)
    print(f"\n    Peak load varies by "
          f"{100 * (peaks.max() - peaks.min()) / peaks.min():.2f}% across the "
          f"whole sweep, and the curvature\n    share of the driving force "
          f"never exceeds {curv:.1e}. The internal length is not\n    reaching "
          f"the crack. This is not a deficiency of the formulation: the "
          f"wryness\n    measures the *gradient* of the microrotation, a "
          f"notched plate in tension is\n    stretch-dominated, so that "
          f"gradient is small however l_b compares with the\n    notch, the "
          f"specimen or l. Compare notched_beam, where the same machinery in\n"
          f"    bending moves the peak load substantially.")

    # A --to-peak run writes under its own tag. Without that it would
    # overwrite the .npz and the figures of a COMPLETED sweep with truncated
    # ones -- destroying, to save a few hours, exactly the artefacts those
    # hours were spent producing.
    tag = ("_quick" if quick else "") + ("_peak" if to_peak else "")
    save_results(rows, tag=tag)
    make_figures(rows, classical, tag=tag)
    # ...and the same panels at each row's own final state. The common-load-
    # level figure above is the honest side-by-side comparison; this one shows
    # each crack as it actually ended, which is what a reader wants to see when
    # one row stops early. Driven from the .npz rather than reimplemented, so
    # the two can never disagree about what was plotted.
    from .replot_tension import main as replot
    replot(tag)
    return rows, classical


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, skip_fem="--no-fem" in sys.argv,
         to_peak="--to-peak" in sys.argv)
