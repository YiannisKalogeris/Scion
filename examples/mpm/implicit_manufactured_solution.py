#!/usr/bin/env python3
r"""
Implicit MPM verified by the Method of Manufactured Solutions.

Why this and not a benchmark problem
------------------------------------
An error is a number and can be small for the wrong reasons; the *rate* it falls
at is a property of the scheme alone, and cannot be reached by tuning anything.
That makes an observed order of accuracy the strongest verification available,
and it is what the reference for this formulation uses -- O'Hare, Gourgiotis,
Coombs & Augarde, CMAME 419 (2024) 116668, Fig. 3(a). There is no closed-form
solution for a large-deformation neo-Hookean square, so instead a displacement
field is *chosen* and the body force that would produce it is derived from the
governing equations:

    u_1 = u_2 = A sin(2 pi X) sin(2 pi Y),        A = 1/30

on the unit square, vanishing on all four sides. Then

    B_i = -dP_iJ/dX_J = -A_iJkL u_k,LJ

with ``A = dP/dF`` the same tangent the solver uses. That last point is worth
being careful about: using the solver's own tangent to build the load could in
principle hide an error in it, but it cannot hide one here, because a wrong
``A`` would produce a load the *residual* does not equilibrate, and the residual
is assembled from ``P`` alone. The two would disagree and Newton would not
converge quadratically.

What the run reports, and what each line would catch
----------------------------------------------------
**Order of accuracy.** Linear basis, second order, measured 1.99/2.00/2.00.
Anything else means the code is consistently solving a *different* equation --
the failure mode that no single-mesh error test can see.

**Newton residual sequence.** Textbook quadratic (1e0, 3e-1, 1e-2, 3e-5, 2e-10).
A tangent that is merely close still converges, linearly, in two or three more
iterations. Nothing but the sequence distinguishes them.

**The quadratic B-spline drops to first order, and that is the boundary
condition, not the basis.** A B-spline is not interpolatory, so setting the
nodal coefficients outside the body to zero is not the same as setting the
*field* to zero on the boundary; the condition lands O(h) off. The measurement
that identifies it: restricting the error to the middle of the domain gives the
same first-order rate, so it is global pollution from the boundary rather than a
local layer. This matters beyond this example -- the smooth bases are the ones
the configurational-force work wants, and this is the price they charge on a
Dirichlet problem.

Run
---
    python -m examples.mpm.implicit_manufactured_solution
    python -m examples.mpm.implicit_manufactured_solution --quick
"""

from __future__ import annotations

import sys

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
    ImplicitMPMStepper,
    ParticleSet,
    get_basis,
)
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

AMPLITUDE = 1.0 / 30.0
TWO_PI = 2.0 * np.pi

RESOLUTIONS = [8, 16, 32, 64]
QUICK = [8, 16, 32]

#: the field is bounded by ``1 - 2 pi A`` away from inversion; at A = 1/30 that
#: is det F >= 0.79, so the manufactured state is comfortably admissible
E, NU = 1000.0, 0.30


# ----------------------------------------------------------------------
# the manufactured solution and the load that produces it
# ----------------------------------------------------------------------
def u_exact(X: np.ndarray) -> np.ndarray:
    s = AMPLITUDE * np.sin(TWO_PI * X[:, 0]) * np.sin(TWO_PI * X[:, 1])
    return np.stack([s, s], axis=1)


def grad_u(X: np.ndarray) -> np.ndarray:
    cs = TWO_PI * AMPLITUDE * np.cos(TWO_PI * X[:, 0]) * np.sin(TWO_PI * X[:, 1])
    sc = TWO_PI * AMPLITUDE * np.sin(TWO_PI * X[:, 0]) * np.cos(TWO_PI * X[:, 1])
    g = np.zeros((len(X), 2, 2))
    g[:, 0, 0] = g[:, 1, 0] = cs
    g[:, 0, 1] = g[:, 1, 1] = sc
    return g


def hess_u(X: np.ndarray) -> np.ndarray:
    """``h[p, k, L, J] = d^2 u_k / dX_L dX_J``."""
    ss = -(TWO_PI ** 2) * AMPLITUDE * np.sin(TWO_PI * X[:, 0]) * np.sin(
        TWO_PI * X[:, 1])
    cc = (TWO_PI ** 2) * AMPLITUDE * np.cos(TWO_PI * X[:, 0]) * np.cos(
        TWO_PI * X[:, 1])
    h = np.zeros((len(X), 2, 2, 2))
    for k in (0, 1):
        h[:, k, 0, 0] = h[:, k, 1, 1] = ss
        h[:, k, 0, 1] = h[:, k, 1, 0] = cc
    return h


def body_force(X: np.ndarray, material) -> np.ndarray:
    """``B_i = -A_iJkL u_k,LJ``, per unit *reference* volume."""
    _, _, A = material.pk1_and_tangent(np.eye(2) + grad_u(X))
    return -np.einsum("piJkL,pkLJ->pi", A, hess_u(X))


# ----------------------------------------------------------------------
def solve(n_cells: int, ppc: int = 2, basis: str = "linear", pad: int = 3):
    """One mesh. Returns ``(e_d, e_core, stepper, particles, X)``."""
    dx = 1.0 / n_cells
    grid = CartesianGrid(shape=(n_cells + 2 * pad,) * 2, dx=dx,
                         origin=np.full(2, -pad * dx), basis=get_basis(basis))
    sub = dx / ppc
    axis = np.arange(sub / 2, 1.0, sub)
    mesh = np.meshgrid(axis, axis, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)

    particles = ParticleSet(X.copy(), vol0=sub ** 2, density=1.0)
    material = NeoHookeanCompressible.from_E_nu(E, NU)

    outside = lambda c: ~((c[:, 0] > 1e-12) & (c[:, 0] < 1.0 - 1e-12)
                          & (c[:, 1] > 1e-12) & (c[:, 1] < 1.0 - 1e-12))
    stepper = ImplicitMPMStepper(
        grid, material,
        particle_force=body_force(X, material) * sub ** 2,
        dirichlet=[GridDirichlet(outside, np.zeros(2))],
        max_iter=30)
    with quiet():
        stepper.run(particles, n_steps=1)

    err = np.linalg.norm((particles.x - X) - u_exact(X), axis=1)
    core = (X > 0.3).all(axis=1) & (X < 0.7).all(axis=1)
    return (float((sub ** 2 * err).sum()),
            float((sub ** 2 * err[core]).sum()),
            stepper, particles, X)


def convergence(resolutions, basis: str) -> list:
    rows = []
    prev_all = prev_core = None
    for n in resolutions:
        e_all, e_core, stepper, _, _ = solve(n, basis=basis)
        rate = (np.log(prev_all / e_all) / np.log(2.0)
                if prev_all else float("nan"))
        rate_core = (np.log(prev_core / e_core) / np.log(2.0)
                     if prev_core else float("nan"))
        rows.append((n, e_all, rate, e_core, rate_core, stepper.history[-1]))
        prev_all, prev_core = e_all, e_core
    return rows


def report(basis: str, rows) -> None:
    print(f"\n  {basis} basis")
    print(f"    {'cells':>6} {'h':>9} {'e_d':>11} {'order':>7} "
          f"{'e_d (core)':>11} {'order':>7} {'Newton':>7}")
    for n, e_all, rate, e_core, rate_core, res in rows:
        print(f"    {n:6d} {1.0 / n:9.5f} {e_all:11.4e} {rate:7.3f} "
              f"{e_core:11.4e} {rate_core:7.3f} {res.iterations:7d}")
    last = rows[-1][-1]
    print("    finest-mesh residuals: "
          + "  ".join(f"{r:.2e}" for r in last.residuals))
    print(f"    observed Newton order: {last.convergence_rate:.2f} "
          f"(2 if the tangent is consistent)")


if __name__ == "__main__":
    resolutions = QUICK if "--quick" in sys.argv else RESOLUTIONS

    print("Implicit MPM: method of manufactured solutions on the unit square")
    print(f"  u_1 = u_2 = {AMPLITUDE:.4f} sin(2 pi X) sin(2 pi Y), "
          f"neo-Hookean E = {E:g}, nu = {NU}")
    print("  one load step, so this is an FE solve on an MPM discretisation "
          "-- integration\n  error from partly filled cells is deliberately "
          "removed")

    linear = convergence(resolutions, "linear")
    report("linear hat", linear)
    quadratic = convergence(resolutions, "quadratic")
    report("quadratic B-spline", quadratic)

    print("\n  Reading it")
    print(f"    linear reaches order {linear[-1][2]:.2f} against a theoretical "
          f"2 for a linear basis.")
    print(f"    quadratic reaches only {quadratic[-1][2]:.2f}, and the core "
          f"error falls at {quadratic[-1][4]:.2f} too --")
    print("    the same rate, so it is not a boundary layer but global "
          "pollution. A B-spline")
    print("    is not interpolatory, so zeroing the nodal coefficients outside "
          "the body puts")
    print("    the Dirichlet condition O(h) away from the boundary. The basis "
          "is not at fault;")
    print("    the way the condition is imposed on it is.")
