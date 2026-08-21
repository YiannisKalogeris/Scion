#!/usr/bin/env python3
r"""
The micropolar size effect: a cantilever whose stiffness depends on its size.

Reproduces Section 4.2 of O'Hare, Gourgiotis, Coombs & Augarde, CMAME 419
(2024) 116668. A 10 m x 1 m x 1 m beam in plane strain carries a 50 kN end
load, applied in 50 equal steps. Everything about the two runs is identical
except the bending characteristic length ``l_b``, which is 0.1 m in one and
1.0 m in the other.

What the example is for
-----------------------
A classical continuum has no length of its own, so scaling a structure and its
load together leaves the strain unchanged -- a beam behaves the same whatever
its size. Real materials do not: once the structure approaches the scale of its
own microstructure it stiffens, which is the Hall-Petch mechanism and is why
MEMS and nanoscale devices are stiffer than the classical prediction. A
micropolar continuum reproduces that because the couple-stress moduli carry a
length,

    l_b = sqrt((beta + gam) / (4 mu)) ,

and the response depends on ``l_b / H``, the microstructural scale relative to
the beam depth. This is the cheapest problem that makes the effect unmistakable.

Measured here: the tip deflects **-6.4 m at l_b = 0.1 m and -0.85 m at
l_b = 1.0 m**, a factor of 7.5 stiffer for the beam whose microstructure is
comparable to its own depth. The reference shows the same in its Fig. 7 as a
load-displacement curve.

Two things worth watching in the numbers the run prints
--------------------------------------------------------
**The flexible beam stops being a beam.** At ``l_b = 0.1`` the tip also draws in
by about 3.7 m along the axis, and the microrotation reaches ~1.1 rad (65
degrees). The load-displacement curve stiffens towards the end not because the
material stiffens but because the deformation mode changes from bending to
axial tension -- the reference makes the same remark. Any "effective modulus"
read off the late part of that curve is measuring geometry.

**The size effect is monotone in ``l_b``, not a two-point comparison.** The
sweep at the end holds everything else fixed and varies ``l_b`` alone, which is
the statement that ``l_b`` is doing the work rather than some coincidence of the
two parameter sets.

**No Newton order is reported here, deliberately.** Every step converges to
~1e-12 in four iterations, but the load path is strongly nonlinear -- the
residual typically *rises* on the first iteration (1.00 to 2.54) before
collapsing 2.54 -> 1.1e-4 -> 3.0e-11 -- so the first ratio is pre-asymptotic and
an observed order fitted through it reads ~1.5 on a sequence that is plainly
better than quadratic. The tangent's order is measured where it can be measured
cleanly, in ``micropolar_manufactured_solution.py``.

Boundary conditions, which are not the obvious ones
----------------------------------------------------
The reference uses a *roller* at the root -- horizontal translation fixed,
vertical free -- plus a single pin at the root's mid-height, rather than a fully
clamped end. That leaves the root cross-section free to breathe while still
removing the rigid modes, and avoids the artificial stress concentration a full
clamp would impose on a micropolar solid, where the rotation is separately
constrained. All rotations are fixed at the root. Plane strain is imposed by
constraining out-of-plane translation and the two in-plane rotation components,
which is how a 3-D SO(3) formulation expresses a 2-D Cosserat problem.

The load is a set of point forces shared among the material points nearest the
tip -- a "pseudo-Neumann" condition. MPM has no surface to integrate a traction
over, and this is the standard substitute.

Run
---
    python -m examples.mpm.micropolar_cantilever
    python -m examples.mpm.micropolar_cantilever --quick
"""

from __future__ import annotations

import sys
import time

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
    ParticleSet,
    get_basis,
)
from scion.discretization.mpm.micropolar import (
    MicropolarMPMStepper,
    attach_micropolar_state,
)
from scion.materials.micropolar import MicropolarHooke

# --- reference Section 4.2 -------------------------------------------------
LENGTH, DEPTH, WIDTH = 10.0, 1.0, 1.0
END_LOAD = 50.0e3
LAM, MU, COUPLING = 3.333e6, 5.0e6, 1.0 / 3.0

LOAD_STEPS = 50
QUICK_STEPS = 10
SWEEP = [0.05, 0.1, 0.25, 0.5, 1.0]
QUICK_SWEEP = [0.1, 0.5, 1.0]


def build(l_b: float, dx: float = 0.5, ppc: int = 3):
    """Grid, particles, tip-load array and material for one ``l_b``."""
    # beta = gam is the reference's choice, and then l_b fixes both
    beta = gam = 2.0 * MU * l_b ** 2
    material = MicropolarHooke(lam=LAM, mu=MU, a=COUPLING, alpha=0.0,
                               beta=beta, gam=gam)

    # the grid has to contain the *deflected* beam, which at l_b = 0.1 drops
    # more than six metres -- a body that leaves the grid finds no nodes and
    # the solve fails rather than silently truncating
    origin = np.array([-1.0, -8.0, -0.5])
    shape = (int(round((LENGTH + 2.0) / dx)), int(round(10.0 / dx)),
             int(round(2.0 / dx)))
    grid = CartesianGrid(shape=shape, dx=dx, origin=origin,
                         basis=get_basis("linear"))

    sub = dx / ppc
    axes = [np.arange(sub / 2, extent, sub)
            for extent in (LENGTH, DEPTH, WIDTH)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    particles = attach_micropolar_state(
        ParticleSet(X.copy(), vol0=sub ** 3, density=1.0))

    tip = X[:, 0] > LENGTH - sub
    force = np.zeros((particles.n, 3))
    force[tip, 1] = -END_LOAD / tip.sum()
    return grid, particles, X, force, material, tip


def boundary_conditions():
    nan = np.nan
    root = lambda c: c[:, 0] < 1.0e-9
    pin = lambda c: (c[:, 0] < 1.0e-9) & (np.abs(c[:, 1] - DEPTH / 2) < 1.0e-9)
    everywhere = lambda c: np.ones(len(c), dtype=bool)
    return [
        # roller at the root: horizontal fixed, vertical free, no rotation
        GridDirichlet(root, np.array([0.0, nan, nan, 0.0, 0.0, 0.0])),
        # one pin at mid-height removes the remaining vertical rigid mode
        GridDirichlet(pin, np.array([0.0, 0.0, nan, 0.0, 0.0, 0.0])),
        # plane strain: no out-of-plane translation, no in-plane microrotation
        GridDirichlet(everywhere, np.array([nan, nan, 0.0, 0.0, 0.0, nan])),
    ]


def run(l_b: float, n_steps: int = LOAD_STEPS, dx: float = 0.5, ppc: int = 3):
    grid, particles, X, force, material, tip = build(l_b, dx, ppc)
    stepper = MicropolarMPMStepper(grid, material,
                                   dirichlet=boundary_conditions(),
                                   particle_force=force, max_iter=25)
    t0 = time.time()
    with quiet():
        stepper.run(particles, n_steps=n_steps)
    elapsed = time.time() - t0

    u = particles.x - X
    sub = dx / ppc
    bottom_tip = tip & (X[:, 1] < sub)
    phi = stepper.microrotation_vector(particles)
    return {
        "l_b": l_b,
        "deflection": float(u[bottom_tip, 1].mean()),
        "draw_in": float(u[bottom_tip, 0].mean()),
        "max_rotation": float(np.abs(phi[:, 2]).max()),
        "steps": len(stepper.history),
        "iterations": sum(h.iterations for h in stepper.history),
        "final_residual": max(h.residuals[-1] for h in stepper.history),
        "seconds": elapsed,
        "particles": particles.n,
    }


def table(rows, title: str) -> None:
    print(f"\n  {title}")
    print(f"    {'l_b (m)':>8} {'l_b/H':>7} {'tip u_y (m)':>12} "
          f"{'draw-in (m)':>12} {'max phi_z':>10} {'steps':>6} "
          f"{'its':>5} {'worst res':>10} {'sec':>6}")
    for r in rows:
        print(f"    {r['l_b']:8.3g} {r['l_b'] / DEPTH:7.3g} "
              f"{r['deflection']:12.4f} {r['draw_in']:12.4f} "
              f"{r['max_rotation']:10.4f} {r['steps']:6d} "
              f"{r['iterations']:5d} {r['final_residual']:10.1e} "
              f"{r['seconds']:6.1f}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_steps = QUICK_STEPS if quick else LOAD_STEPS
    sweep = QUICK_SWEEP if quick else SWEEP

    print("Micropolar cantilever: the size effect")
    print(f"  {LENGTH:g} x {DEPTH:g} x {WIDTH:g} m beam, plane strain, "
          f"{END_LOAD / 1e3:g} kN end load in {n_steps} steps")
    print(f"  lam = {LAM / 1e6:g} MPa, mu = {MU / 1e6:g} MPa, "
          f"a = {COUPLING:.4g} (N = 0.5)")
    if quick:
        print("  --quick: fewer load steps and a shorter sweep. The load path "
              "is nonlinear,\n  so the deflections below are not the "
              "converged ones.")

    cases = [run(l_b, n_steps) for l_b in (0.1, 1.0)]
    table(cases, "the reference's two cases")

    flexible, stiff = cases
    ratio = flexible["deflection"] / stiff["deflection"]
    print(f"\n    The beam with l_b = 1.0 m is {ratio:.1f}x stiffer than the "
          f"one with l_b = 0.1 m,")
    print("    from the same load, geometry and Lame parameters. That is the "
          "size effect: a\n    classical continuum has no length of its own "
          "and cannot produce it at all.")
    print(f"    The flexible beam also draws in by "
          f"{-flexible['draw_in']:.2f} m and reaches "
          f"{flexible['max_rotation']:.2f} rad")
    print("    of microrotation, so its late stiffening is the mode changing "
          "from bending to\n    axial tension, not the material.")

    rows = [run(l_b, n_steps) for l_b in sweep]
    table(rows, "l_b alone, everything else held fixed")
    monotone = all(a["deflection"] < b["deflection"]
                   for a, b in zip(rows, rows[1:]))
    print(f"\n    Monotone in l_b: {monotone}. Two points could be a "
          f"coincidence of parameter sets;\n    a monotone sweep is the "
          f"statement that l_b is what is doing the work.")
