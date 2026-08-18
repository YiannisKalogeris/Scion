#!/usr/bin/env python3
r"""
The Sod shock tube: why conservation form is not a stylistic choice.

Problem
-------
A tube divided by a diaphragm, gas at rest on both sides, ``(rho, u, p)`` equal
to ``(1, 0, 1)`` on the left and ``(0.125, 0, 0.1)`` on the right. Remove the
diaphragm at ``t = 0`` and integrate the Euler equations to ``t = 0.2``.

Sod (1978) is the standard first test for a compressible solver, and it is
standard because its solution contains one of each wave -- a rarefaction fan
moving left, a contact discontinuity moving right at ``u* = 0.92745``, and a
shock moving right faster still at ``1.75216``. A scheme cannot get a passable
answer by being good at only one of them, and the three fail differently:
the fan tests accuracy, the contact tests the Riemann solver, the shock tests
whether the scheme is in conservation form at all.

The shock is the point
----------------------
Across a shock the solution is discontinuous, so the differential form of the
Euler equations does not hold there; only the integral form does, and the
Rankine-Hugoniot conditions that fix the shock *speed* come out of it. The
Lax-Wendroff theorem then says a scheme in conservation form converges to a
correct weak solution -- and says nothing at all about one that is not.

The failure mode is specific and worth recognising: a non-conservative scheme
does not produce a smeared or wobbly shock. It produces a perfectly sharp shock
in the wrong place, and refining the mesh gives a sharper shock still in the
wrong place. Error norms fall, the picture looks better, and the answer stays
wrong. So the number this example prints is the shock *position* against the
exact 0.850431, and it is the only number here that could not be improved by a
better limiter.

What each knob does
-------------------
**Riemann solver.** All three get the shock right; they differ on the contact.
The measurement is unambiguous on an isolated stationary contact -- HLLC holds
it exactly, HLL and Rusanov smear it over twenty cells -- and much less
dramatic on Sod, where reconstruction is doing most of the work. Worth knowing
before optimising the wrong end.

**Limiter.** The bigger lever once the solver is sane, and the source of the
oscillations if it is switched off. ``none`` is included to show what the
limiters are preventing.

**Resolution.** The L1 error falls at a rate slightly below one and stays
there. That is not a defect: the solution is discontinuous, so the handful of
cells straddling the shock and the contact each carry an ``O(1)`` error, and no
amount of accuracy in the smooth regions can beat that. A second-order rate on
a shock problem would mean the reference solution was wrong.

Run
---
    python -m examples.fvm.sod_shock_tube
    python -m examples.fvm.sod_shock_tube --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simulacrum.discretization.fvm import (
    SHOCK_TUBES,
    SOD,
    EulerEquations,
    EulerSolver1D,
    star_pressure,
)

RESULTS = Path(__file__).resolve().parent / "results"

#: Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, table 4.1
TORO_P_STAR, TORO_U_STAR = 0.30313, 0.92745
SHOCK_SPEED = 1.752157
SHOCK_X = 0.5 + SHOCK_SPEED * 0.2

RESOLUTIONS = [100, 200, 400, 800, 1600]
QUICK = [100, 200, 400]

EOS = EulerEquations(1.4)


def run(n, riemann="hllc", limiter="minmod"):
    solver = EulerSolver1D(n, riemann=riemann, limiter=limiter)
    result = solver.run_tube(SOD, cfl=0.4, stages=2)
    return solver, result, EOS.primitive(result.U)


def l1_density(result, W):
    return float(np.abs(W[:, 0] - SOD.exact(result.x, result.t)[:, 0]).mean())


def captured_shock(x, rho):
    """Where the largest density drop in the right half of the tube sits."""
    right = x[:-1] > 0.75
    i = int(np.argmax(np.abs(np.diff(rho)) * right))
    return 0.5 * (x[i] + x[i + 1])


def contact_width(x, rho):
    """Cells spanning the middle 80% of the density jump at the contact."""
    window = (x > 0.60) & (x < 0.78)
    lo, hi = rho[window].min(), rho[window].max()
    return int(np.count_nonzero((rho[window] > lo + 0.1 * (hi - lo))
                                & (rho[window] < lo + 0.9 * (hi - lo))))


# ---------------------------------------------------------------------------

def star_state():
    print("\n  The exact star state, against Toro table 4.1")
    p, u = star_pressure(SOD.left, SOD.right, 1.4)
    left = SOD.exact(np.array([0.5 + (u - 1e-9) * 0.2]), 0.2)[0]
    right = SOD.exact(np.array([0.5 + (u + 1e-9) * 0.2]), 0.2)[0]
    print(f"    p*      {p:.6f}   (published {TORO_P_STAR})")
    print(f"    u*      {u:.6f}   (published {TORO_U_STAR})")
    print(f"    rho*_L  {left[0]:.6f}   rho*_R  {right[0]:.6f}   "
          f"(density is the only thing that jumps across the contact)")


def resolution_study(resolutions):
    print("\n  Refinement, HLLC + minmod")
    print(f"\n    {'n':>6}{'L1(rho)':>10}{'rate':>7}{'shock x':>11}"
          f"{'error':>10}{'h':>10}")
    errors, rows = [], []
    for n in resolutions:
        solver, result, W = run(n)
        err = l1_density(result, W)
        errors.append(err)
        shock = captured_shock(result.x, W[:, 0])
        rate = (np.log2(errors[-2] / errors[-1]) if len(errors) > 1
                else float("nan"))
        print(f"    {n:>6}{err:10.5f}{rate:7.2f}{shock:11.5f}"
              f"{abs(shock - SHOCK_X):10.5f}{solver.h:10.5f}")
        rows.append((n, err, shock, solver.h))

    print(f"\n    Exact shock at x = {SHOCK_X:.6f}. The position error tracks")
    print("    h, not h^2: the shock is located to within the cell it")
    print("    occupies and no better. That it tracks h at all is the")
    print("    conservation property; a non-conservative scheme converges to")
    print("    a fixed offset instead.")
    return rows, errors


def riemann_comparison():
    print("\n  Riemann solvers")
    print("\n    (a) an isolated stationary contact, 100 cells, t = 0.4")
    print("        a density jump at rest under uniform pressure. The exact")
    print("        solution is that nothing happens.\n")
    for name in ("rusanov", "hll", "hllc"):
        solver = EulerSolver1D(100, riemann=name, limiter="first-order")
        rho0 = np.where(solver.x < 0.5, 1.0, 0.125)
        U0 = EOS.conserved(rho0, np.zeros(100), np.ones(100))
        W = EOS.primitive(solver.solve(U0, 0.4, cfl=0.4).U)
        print(f"      {name:<9} max density error {np.abs(W[:, 0] - rho0).max():.3e}")
    print("\n      HLLC is exact to the bit. Its star state reduces")
    print("      algebraically to the cell state when u and p are uniform,")
    print("      so there is nothing left to dissipate. HLL's single")
    print("      intermediate state cannot hold a density jump and leaves a")
    print("      mass flux proportional to S_L S_R, which is negative.")

    print("\n    (b) the Sod contact, 800 cells: width in cells")
    print(f"\n      {'limiter':<14}{'rusanov':>9}{'hll':>7}{'hllc':>7}")
    for limiter in ("first-order", "minmod", "van-leer"):
        widths = []
        for name in ("rusanov", "hll", "hllc"):
            _, result, W = run(800, riemann=name, limiter=limiter)
            widths.append(contact_width(result.x, W[:, 0]))
        print(f"      {limiter:<14}" + "".join(f"{w:>9}" if i == 0
                                               else f"{w:>7}"
                                               for i, w in enumerate(widths)))
    print("\n      The ordering holds throughout, but the gap closes: with")
    print("      van Leer all three land within a cell of each other. Once")
    print("      MUSCL is on, the limiter is the lever, not the solver.")


def limiter_comparison():
    """What the limiter is actually preventing, measured three ways."""
    print("\n  Limiters, 200 cells, HLLC")
    print(f"\n    {'limiter':<14}{'L1(rho)':>10}{'rho max':>10}{'p max':>9}"
          f"{'u max':>9}")
    fields = {}
    for name in ("first-order", "minmod", "van-leer", "mc", "superbee",
                 "none"):
        _, result, W = run(200, limiter=name)
        print(f"    {name:<14}{l1_density(result, W):10.5f}"
              f"{W[:, 0].max():10.5f}{W[:, 2].max():9.5f}{W[:, 1].max():9.5f}")
        fields[name] = (result.x, W)
    print("\n    Bounds: density and pressure start at 1.0 and cannot exceed")
    print("    it; the exact velocity peaks at u* = 0.92745. Every limited")
    print("    scheme holds rho and p to 1.00000 exactly. All of them,")
    print("    first order included, overshoot the velocity by about 0.3%,")
    print("    which is the honest state of the art: MUSCL on a system is")
    print("    not TVD, because total variation is not defined for a vector,")
    print("    and no theorem here promises otherwise.")

    print("\n    Unlimited is a different matter: on Sod it breaks the")
    print("    density and pressure bounds by 3-5% and the velocity by 50%.")
    print("    On the stronger Lax tube it does not merely oscillate:")
    tube = SHOCK_TUBES["lax"]
    eos = EulerEquations(tube.gamma)
    solver = EulerSolver1D(200, riemann="hllc", limiter="none",
                           gamma=tube.gamma, length=tube.length)
    result = solver.run_tube(tube)
    W = eos.primitive(result.U)
    print(f"      min density  {W[:, 0].min():.4g}")
    print(f"      min pressure {W[:, 2].min():.4g}")
    print(f"      positivity had to be forced {result.repairs} times")
    print("\n    That is why 'repairs' is reported rather than swallowed. A")
    print("    solver that silently floors its own state turns this into a")
    print("    plausible-looking picture.")
    return fields


def other_tubes():
    print("\n  The other tubes, 400 cells, HLLC + minmod")
    print(f"\n    {'tube':<8}{'steps':>7}{'L1(rho)':>10}{'mass drift':>13}"
          f"{'repairs':>9}  note")
    for name, tube in SHOCK_TUBES.items():
        solver = EulerSolver1D(400, riemann="hllc", limiter="minmod",
                               gamma=tube.gamma, length=tube.length)
        result = solver.run_tube(tube)
        eos = EulerEquations(tube.gamma)
        W = eos.primitive(result.U)
        err = float(np.abs(W[:, 0] - tube.exact(result.x, result.t)[:, 0]
                           ).mean())
        print(f"    {name:<8}{result.steps:>7}{err:10.5f}"
              f"{result.mass_drift:13.2e}{result.repairs:>9}  "
              f"{tube.note.split(':')[0]}")
    print("\n    The nonzero mass drifts are correct, not leaks: those tubes")
    print("    drive fluid out through the transmissive ends. On Sod nothing")
    print("    reaches a boundary by t = 0.2 and the drift is exactly zero.")


# ---------------------------------------------------------------------------

def plot(rows, errors, fields, resolutions):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))

    # (0,0) the three primitive variables at moderate resolution
    _, result, W = run(200, limiter="van-leer")
    fine = np.linspace(0.0, 1.0, 4000)
    exact = SOD.exact(fine, 0.2)
    for k, (label, colour) in enumerate((("density", "C0"),
                                         ("velocity", "C1"),
                                         ("pressure", "C2"))):
        ax[0, 0].plot(fine, exact[:, k], "-", c=colour, lw=1.3)
        ax[0, 0].plot(result.x, W[:, k], "o", c=colour, ms=2.2, label=label)
    ax[0, 0].set(xlabel="x", ylabel="value",
                 title="Sod at t = 0.2, 200 cells, HLLC + van Leer")
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[0, 0].grid(alpha=0.3)

    # (0,1) refinement of the density profile near the shock
    for n in resolutions:
        _, result, W = run(n)
        ax[0, 1].plot(result.x, W[:, 0], "o-", ms=2.0, lw=0.9, label=f"n={n}")
    ax[0, 1].plot(fine, exact[:, 0], "k-", lw=1.3, label="exact")
    ax[0, 1].axvline(SHOCK_X, c="crimson", ls="--", lw=1.0)
    ax[0, 1].set(xlim=(0.80, 0.90), xlabel="x", ylabel="density",
                 title="The shock, refined (dashed: exact position)")
    ax[0, 1].legend(fontsize=7, frameon=False)
    ax[0, 1].grid(alpha=0.3)

    # (1,0) convergence
    h = 1.0 / np.array(resolutions, dtype=float)
    ax[1, 0].loglog(h, errors, "o-", lw=1.3, ms=4, label="L1 in density")
    ax[1, 0].loglog(h, [abs(r[2] - SHOCK_X) for r in rows], "s-", lw=1.3,
                    ms=4, label="shock position error")
    ax[1, 0].loglog(h, errors[0] * (h / h[0]), "k--", lw=1.0, label="$h^1$")
    ax[1, 0].set(xlabel="cell size $h$", ylabel="error",
                 title="Both errors track $h$, and that is correct")
    ax[1, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].grid(alpha=0.3, which="both")

    # (1,1) limiters near the contact
    ax[1, 1].plot(fine, exact[:, 0], "k-", lw=1.4, label="exact")
    for name in ("first-order", "minmod", "superbee", "none"):
        x, W = fields[name]
        ax[1, 1].plot(x, W[:, 0], "o-", ms=2.0, lw=0.9, label=name)
    ax[1, 1].set(xlim=(0.55, 0.90), xlabel="x", ylabel="density",
                 title="Contact and shock, 200 cells, four limiters")
    ax[1, 1].legend(fontsize=7, frameon=False)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "sod_shock_tube.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    resolutions = QUICK if "--quick" in sys.argv else RESOLUTIONS

    print("The Sod shock tube")
    print("  (rho, u, p) = (1, 0, 1) | (0.125, 0, 0.1),  gamma = 1.4,  "
          "t = 0.2")

    star_state()
    rows, errors = resolution_study(resolutions)
    riemann_comparison()
    fields = limiter_comparison()
    other_tubes()
    plot(rows, errors, fields, resolutions)
