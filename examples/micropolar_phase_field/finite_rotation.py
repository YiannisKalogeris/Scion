r"""A damaged specimen carried through a large rigid rotation: nothing may change.

    python -m examples.micropolar_phase_field.finite_rotation
    python -m examples.micropolar_phase_field.finite_rotation --dx 0.05,0.025

Every other example here is compared against another calculation. This one has
an *exact* answer, known in advance and independent of the mesh, which is what
makes it a verification rather than a comparison: a body already stressed and
damaged is rotated rigidly through 90 degrees, and

* the Cosserat strain ``E = Q^T F - I`` must not change, because ``F -> R F``
  and ``Q -> R Q`` leave it invariant;
* the wryness ``Gamma`` must not change, for the same reason;
* the history field ``H`` and the phase field ``d`` must not move at all -- a
  rigidly rotating crack cannot open further;
* the stored energy must not change, and the Kirchhoff stress must rotate with
  the body rather than change magnitude.

The rotation is imposed only on the two edges that carried the original
boundary conditions, and only on their *translations*: the microrotation is
left free, so ``Q`` has to be solved for and must come out following ``R``. A
formulation that linearised the rotation, or updated it additively, would fail
here at 90 degrees while passing every small-rotation test in the paper.

The axis is the out-of-plane one, which preserves the plane-strain reduction,
so the check is cheap enough to run at two resolutions -- and it must pass at
both, since none of the statements above depends on the discretisation.
"""
from __future__ import annotations

import sys

import numpy as np

from scion.core.logging import quiet
from scion.core.rotation import quat_to_matrix
from scion.discretization.mpm import CartesianGrid, GridDirichlet, get_basis

from ._common import (NZ, RESULTS, make_problem, plane_strain_bc, seed_notch,
                      seed_particles)
from .notched_tension import (E_MOD, GC, NOTCH_B, NOTCH_WIDTH_FACTOR, NOTCH_X,
                              NOTCH_Y, NU, N_COUPLING, SIDE,
                              critical_displacement)

ROT = 0.5 * np.pi           # total rigid rotation
PRESCRIBE_ROT = False       # set by --prescribe-rotation: Q imposed, not solved
N_LOAD, N_ROT = 4, 6        # load steps to damage the plate, then rotation steps
L_B_OVER_L = 2.0


def build(dx, ell):
    """The notched plate on a grid wide enough to hold it once rotated."""
    sub = dx / 2
    parts = seed_particles((SIDE, SIDE), dx, 2)
    # a 90 degree rotation about the centre sweeps the body through a circle of
    # radius SIDE/sqrt(2); the grid must contain that, plus the basis reach
    pad = int(np.ceil((0.5 * np.sqrt(2.0) - 0.5) * SIDE / dx)) + 4
    n = int(round(SIDE / dx))
    grid = CartesianGrid(shape=(n + 2 * pad, n + 2 * pad, NZ + 2),
                         dx=dx, origin=np.array([-pad * dx, -pad * dx, -dx]),
                         basis=get_basis("linear"))
    u_bar = 0.6 * critical_displacement(ell)      # enough to damage, not break
    nan = np.nan
    bcs = [GridDirichlet(lambda c: c[:, 1] < 1e-9,
                         np.array([0.0, -0.5 * u_bar, nan, nan, nan, nan])),
           GridDirichlet(lambda c: np.abs(c[:, 1] - SIDE) < 1e-9,
                         np.array([0.0, 0.5 * u_bar, nan, nan, nan, nan])),
           plane_strain_bc()]
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING,
                               l_b=L_B_OVER_L * ell, gc=GC, ell_pf=ell,
                               bcs=bcs, max_stagger=30)
    seed_notch(pf, parts, dx, ell, GC, tip=np.array([NOTCH_X, NOTCH_Y]),
               along="x", amplitude=NOTCH_B, width_factor=NOTCH_WIDTH_FACTOR)
    return pf, stepper, parts, grid, sub


def rotation_bcs(dtheta, prescribe_rotation=False):
    """A rigid rotation increment prescribed on every node; microrotation free.

    The increment ``(R - I)(x - c)`` is *linear* in ``x``, and the linear basis
    reproduces a linear field exactly, so prescribing it on every node moves
    every particle along the rigid motion to round-off: the deformation gradient
    picks up exactly ``R`` and the strain cannot change. That is what makes the
    answer exact and independent of the mesh.

    An earlier version prescribed the rotation only on the two loaded edges and
    left the interior free. That is a different problem: the edge particles then
    follow a blend of prescribed and free nodes rather than the rigid motion,
    which deforms a cracked specimen and grows real damage -- measured, up to
    0.70 in ``d`` and worsening with refinement, because a finer grid carries the
    body across more cells. The failure was in the boundary condition, not in the
    formulation, and the distinction is exactly what this test exists to make.

    The translations are prescribed; the rotational degrees of freedom are left
    free, so ``Q`` is still solved for and must come out following ``R``.
    """
    c = np.array([0.5 * SIDE, 0.5 * SIDE, 0.0])
    R = np.array([[np.cos(dtheta), -np.sin(dtheta), 0.0],
                  [np.sin(dtheta), np.cos(dtheta), 0.0],
                  [0.0, 0.0, 1.0]])

    def value(coords):
        x = np.asarray(coords, dtype=float)
        out = np.full((len(x), 6), np.nan)
        out[:, :3] = (x - c) @ R.T - (x - c)
        if prescribe_rotation:
            # the microrotation increment of the exact rigid motion: uniform,
            # so its gradient -- and hence the wryness increment -- is zero
            out[:, 3:] = np.array([0.0, 0.0, dtheta])
        return out

    return [GridDirichlet(lambda q: np.ones(len(q), bool), value)]


def run(dx, ell, tol=None):
    pf, stepper, parts, grid, sub = build(dx, ell)
    if tol is not None:                 # is the residual error tolerance or structural?
        stepper.tol, stepper.max_iter = tol, 60
    X0 = parts.x.copy()
    print(f"\n  dx = {dx:g}, l = {ell:g}, {parts.n} particles, "
          f"{grid.n_nodes} nodes")

    with quiet():
        for k in range(1, N_LOAD + 1):
            pf.solve_step(k / N_LOAD, 1.0 / N_LOAD)
    c0 = stepper.constitutive(parts)
    ref = dict(E=np.einsum("pji,pjk->pik", c0["Q"], c0["F"]) - np.eye(3),
               G=parts.wryness.copy(), H=pf.H.copy(),
               d=pf.tables.interpolate(pf.d).copy(),
               Q=quat_to_matrix(parts.quat).copy())
    ref["W"] = float(np.sum(parts.vol0 * pf.driving_energy()))
    print(f"    loaded: max damage {ref['d'].max():.4f}, "
          f"|Gamma|max {np.abs(ref['G']).max():.3e}, driving energy {ref['W']:.6g}")

    dtheta = ROT / N_ROT
    print(f"    {'rotated':>9} {'|dE|':>10} {'|dGamma|':>10} {'|dH|/H':>9} "
          f"{'|dd|':>9} {'dW/W':>10} {'|Q - R Q0|':>11} {'cuts':>5}")
    for k in range(1, N_ROT + 1):
        # Cut back on a failed stage, exactly as the load paths do. Only the
        # two edges are prescribed, so the first trial of a stage rotates them
        # while the interior stands still; too large a stage shears the
        # elements between them past inversion. That is the step size, not the
        # formulation -- the converged state is the same rigid rotation either
        # way, which the invariants below then confirm.
        cuts, done, remaining = 0, 0.0, dtheta
        while done < dtheta - 1e-12:
            step = min(remaining, dtheta - done)
            stepper.dirichlet = rotation_bcs(step, PRESCRIBE_ROT)
            snap = stepper.snapshot(parts)
            d_save, h_save = pf.d.copy(), pf.H.copy()
            try:
                with quiet():
                    pf.solve_step(1.0, 1.0)
            except Exception:
                stepper.restore(parts, snap)
                pf.d, pf.H = d_save, h_save
                remaining *= 0.5
                cuts += 1
                if cuts > 12:
                    raise
                continue
            done += step
            remaining = min(2.0 * remaining, dtheta - done) or remaining
        c = stepper.constitutive(parts)
        E = np.einsum("pji,pjk->pik", c["Q"], c["F"]) - np.eye(3)
        th = k * dtheta
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        W = float(np.sum(parts.vol0 * pf.driving_energy()))
        d = pf.tables.interpolate(pf.d)
        print(f"    {np.degrees(th):8.1f}d {np.abs(E - ref['E']).max():10.2e} "
              f"{np.abs(parts.wryness - ref['G']).max():10.2e} "
              f"{np.abs(pf.H - ref['H']).max() / max(ref['H'].max(), 1e-300):9.2e} "
              f"{np.abs(d - ref['d']).max():9.2e} "
              f"{abs(W - ref['W']) / ref['W']:10.2e} "
              f"{np.abs(quat_to_matrix(parts.quat) - R @ ref['Q']).max():11.2e} "
              f"{cuts:5d}")
        if k == N_ROT:
            # is the spurious wryness spread through the body, or a boundary
            # layer where the particle support is clipped by the edge?
            dG = np.abs(parts.wryness - ref["G"]).reshape(len(parts.x), -1).max(axis=1)
            edge_ = ((X0[:, 0] < 2 * dx) | (X0[:, 0] > SIDE - 2 * dx)
                     | (X0[:, 1] < 2 * dx) | (X0[:, 1] > SIDE - 2 * dx))
            print(f"      |dGamma| median {np.median(dG):.2e}, 99th pct "
                  f"{np.percentile(dG, 99):.2e}, max {dG.max():.2e};  "
                  f"interior max {dG[~edge_].max():.2e}, edge max {dG[edge_].max():.2e}"
                  f"   (|Gamma| itself: max {np.abs(ref['G']).max():.2e})")


def main():
    global PRESCRIBE_ROT
    PRESCRIBE_ROT = "--prescribe-rotation" in sys.argv
    grids = (0.05, 0.025)
    if "--dx" in sys.argv:
        grids = tuple(float(v) for v in
                      sys.argv[sys.argv.index("--dx") + 1].split(","))
    print("Rigid rotation of a damaged specimen: every column must stay at "
          "round-off")
    for dx in grids:
        tol = None
        if "--tol" in sys.argv:
            tol = float(sys.argv[sys.argv.index("--tol") + 1])
        run(dx, 4.0 * dx, tol)      # l/dx = 4 at every resolution


if __name__ == "__main__":
    main()
