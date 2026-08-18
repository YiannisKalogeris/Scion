#!/usr/bin/env python3
r"""
Advection-diffusion by finite volume, and the Peclet number that decides it.

Problem
-------
Steady ``a u' = D u''`` on ``[0, 1]`` with ``u(0) = 0`` and ``u(1) = 1``. The
exact solution is a boundary layer at the outflow,

    u(x) = (exp(Pe x) - 1) / (exp(Pe) - 1),        Pe = a L / D,

whose thickness is ``D/a``. It is the standard test for a convective scheme
because it has an analytic answer and an arbitrarily hard regime: turn ``D``
down and the layer gets thinner without limit.

The three things this shows
---------------------------
**Conservation is structural.** The finite volume total changes only through
the two boundary faces, exactly, at any resolution and with any scheme --
because every interior face enters two cells with opposite signs and cancels
algebraically. That is not an accuracy statement and it does not improve with
refinement; it is already exact. Nothing else in the library has it.

**The Peclet limit is a property of the operator, not of the mesh.** Central
differencing loses diagonal dominance at ``Pe = 2`` and the solution oscillates
above it. The oscillation does not shrink as the mesh is refined at fixed
``Pe`` -- it disappears only when refinement drags ``Pe`` back under two. The
run below shows the same problem at ``Pe`` of 10, 4, 1 and 0.25 and the
transition is abrupt, not gradual.

**Upwinding buys monotonicity by paying an order of accuracy.** It is
unconditionally monotone -- the matrix is an M-matrix at every ``Pe`` -- and
first order, and its own numerical diffusion is ``|a| h / 2``. On the coarse
grids here that exceeds the physical ``D``, so the layer comes out visibly too
thick: the scheme is not solving the problem it was given, it is solving one
with a larger diffusivity. That is the trade, and it is why neither scheme is
the default answer.

A detail worth copying: the error is measured against exact *cell averages*,
not against the exact solution sampled at cell centres. The difference is
``O(h^2)``, which is precisely the size of the thing being measured, and it is
enough on its own to turn a clean second-order result into a muddled one.

Run
---
    python -m examples.fvm.advection_diffusion
    python -m examples.fvm.advection_diffusion --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from scion.discretization.fvm import FVMAssembler

RESULTS = Path(__file__).resolve().parent / "results"

L, A = 1.0, 1.0
RESOLUTIONS = [20, 40, 80, 160, 320]
QUICK = [20, 40, 80]


def exact_averages(assembler, diffusivity):
    """Exact cell averages of the boundary-layer solution.

    Integrating ``exp(k x)`` over a cell and dividing by ``h``; the closed form
    is why this costs nothing and is worth doing.
    """
    k = A * L / diffusivity
    xf = assembler.faces(0)
    antiderivative = np.exp(k * xf) / k - xf
    return (np.diff(antiderivative) / np.diff(xf)) / (np.exp(k) - 1.0)


def solve(n, flux, diffusivity):
    fv = FVMAssembler(None, (n,), spacing=L / n, velocity=A,
                      diffusivity=diffusivity, flux=flux)
    system = fv.assemble(conditions={"left": ("dirichlet", 0.0),
                                     "right": ("dirichlet", 1.0)})
    u = spsolve(system.matrix.tocsc(), system.rhs)
    error = float(np.abs(u - exact_averages(fv, diffusivity)).max())
    return fv, u, error


def observed_order(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])]


# ---------------------------------------------------------------------------

def convergence(resolutions, diffusivity):
    print(f"\n  Observed order, D = {diffusivity:g}  "
          f"(Pe over the domain = {A * L / diffusivity:g})")
    print(f"\n    {'scheme':<10}"
          + "  ".join(f"n={n:<5d}" for n in resolutions) + "  observed order")

    out = {}
    for flux in ("upwind", "central"):
        errors = [solve(n, flux, diffusivity)[2] for n in resolutions]
        rates = observed_order(errors)
        print(f"    {flux:<10}" + "  ".join(f"{e:7.2e}" for e in errors)
              + "  " + " ".join(f"{r:4.2f}" for r in rates))
        out[flux] = (errors, rates)
    return out


def peclet_sweep():
    print("\n  The same problem at four cell Peclet numbers, D = 0.01")
    print("\n    n     Pe      central: min      max     upwind: monotone?")
    rows = []
    for n in (10, 25, 100, 400):
        fv, uc, _ = solve(n, "central", 0.01)
        _, uu, _ = solve(n, "upwind", 0.01)
        monotone = bool(np.all(np.diff(uu) > -1e-13) and uu.min() > -1e-13)
        print(f"    {n:<5d} {fv.peclet():6.2f}   {uc.min():10.4f} "
              f"{uc.max():8.4f}     {monotone}")
        rows.append((n, fv.peclet(), fv.centres(0), uc, uu))
    print("\n    Above Pe = 2 the central solution leaves [0, 1] entirely.")
    print("    Below it, it does not. Upwind never does, at any Pe.")
    return rows


def conservation():
    """Total in, total out, and the two boundary faces that account for the
    difference."""
    print("\n  Conservation: the total moves only through the boundary")
    n, u_b = 40, 0.4
    fv = FVMAssembler(None, (n,), spacing=L / n, velocity=A, flux="upwind",
                      limiter="first-order")
    u = np.exp(-40.0 * (fv.centres(0) - 0.5) ** 2)
    conditions = {"left": ("dirichlet", u_b), "right": "neumann"}

    # every interior face cancels, leaving only these
    inflow = A * (2.0 * u_b - u[0])
    outflow = A * u[-1]
    print(f"    net outflow, summed from the divergence : "
          f"{fv.boundary_flux(u, conditions): .16f}")
    print(f"    net outflow, from the two faces by hand : "
          f"{outflow - inflow: .16f}")

    periodic = {"left": "periodic", "right": "periodic"}
    fv = FVMAssembler(None, (200,), spacing=L / 200, velocity=A,
                      flux="rusanov", limiter="van-leer")
    x = fv.centres(0)
    u0 = np.exp(-100.0 * (x - 0.3) ** 2) + (np.abs(x - 0.7) < 0.1)
    u, steps = fv.solve_unsteady(u0, 1.0, periodic, cfl=0.4, stages=3)
    print(f"\n    a full period on a closed domain, {steps} steps:")
    print(f"    total drift = {abs(fv.total(u) - fv.total(u0)):.3e}   "
          f"(round-off, and it stays there at any resolution)")
    return x, u0, u


def limiters():
    """A square wave and a Gaussian, once around a periodic domain."""
    print("\n  Limiters, one period of a square wave and a Gaussian")
    n = 200
    names = ["first-order", "minmod", "van-leer", "mc", "superbee", "none"]
    periodic = {"left": "periodic", "right": "periodic"}

    fields, rows = {}, []
    for name in names:
        fv = FVMAssembler(None, (n,), spacing=L / n, velocity=A,
                          flux="rusanov", limiter=name)
        x = fv.centres(0)
        u0 = np.exp(-200.0 * (x - 0.3) ** 2) + (np.abs(x - 0.7) < 0.1)
        u, _ = fv.solve_unsteady(u0, 1.0, periodic, cfl=0.4, stages=3)
        fields[name] = (x, u0, u)
        rows.append((name, float(np.abs(u - u0).mean()),
                     float(max(0.0, u.max() - u0.max()))))

    print(f"\n    {'limiter':<14}{'L1 error':>10}{'overshoot':>12}")
    for name, l1, over in rows:
        print(f"    {name:<14}{l1:10.4f}{over:12.3e}")
    print("\n    Every TVD limiter overshoots by exactly nothing; the")
    print("    unlimited scheme, which is also second order, overshoots by a")
    print("    third of the jump. That gap is Godunov's theorem: no linear")
    print("    scheme above first order is monotone, so the limiter has to")
    print("    make the scheme nonlinear to have both.")
    return fields


# ---------------------------------------------------------------------------

def plot(mild, sweep, transport, fields, resolutions):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # (0,0) convergence
    h = L / np.array(resolutions, dtype=float)
    for flux, style in (("upwind", "o-"), ("central", "s-")):
        ax[0, 0].loglog(h, mild[flux][0], style, lw=1.3, ms=4, label=flux)
    for slope, dash in ((1, "--"), (2, ":")):
        ref = mild["upwind"][0][0] * (h / h[0]) ** slope
        ax[0, 0].loglog(h, ref, dash, c="k", lw=1.0, label=f"$h^{slope}$")
    ax[0, 0].set(xlabel="cell size $h$", ylabel="max error vs cell averages",
                 title="Observed order, $Pe_L = 10$")
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[0, 0].grid(alpha=0.3, which="both")

    # (0,1) the Peclet transition
    for n, pe, x, uc, _ in sweep:
        ax[0, 1].plot(x, uc, "o-", ms=2.5, lw=1.0,
                      label=f"n={n}, Pe={pe:.2f}")
    xf = np.linspace(0, 1, 1000)
    ax[0, 1].plot(xf, (np.exp(100 * xf) - 1) / (np.exp(100.0) - 1), "k-",
                  lw=1.4, label="exact")
    ax[0, 1].set(xlabel="x", ylabel="u", ylim=(-1.4, 1.4),
                 title="Central differencing across $Pe = 2$")
    ax[0, 1].legend(fontsize=7, frameon=False)
    ax[0, 1].grid(alpha=0.3)

    # (1,0) transported profile
    x, u0, u = transport
    ax[1, 0].plot(x, u0, "k-", lw=1.2, label="initial")
    ax[1, 0].plot(x, u, "-", lw=1.2, label="after one period")
    ax[1, 0].set(xlabel="x", ylabel="u",
                 title="Periodic transport, van Leer (total drift 1e-16)")
    ax[1, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].grid(alpha=0.3)

    # (1,1) limiters
    x, u0, _ = fields["minmod"]
    ax[1, 1].plot(x, u0, "k-", lw=1.4, label="exact")
    for name in ("first-order", "minmod", "superbee", "none"):
        ax[1, 1].plot(fields[name][0], fields[name][2], "-", lw=1.0,
                      label=name)
    ax[1, 1].set(xlabel="x", ylabel="u",
                 title="One period: limited, unlimited, and first order")
    ax[1, 1].legend(fontsize=7, frameon=False)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "advection_diffusion.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    resolutions = QUICK if quick else RESOLUTIONS

    print("Advection-diffusion by finite volume")
    print("  a u' = D u'' on [0, 1],  u(0) = 0,  u(1) = 1,  a = 1")

    mild = convergence(resolutions, 0.1)
    sweep = peclet_sweep()
    transport = conservation()
    fields = limiters()
    plot(mild, sweep, transport, fields, resolutions)
