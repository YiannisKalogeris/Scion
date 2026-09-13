"""Why the micropolar crack left the symmetry plane and the classical one did not.

    python -m examples.micropolar_phase_field.diagnose_symmetry

The notched-tension problem is mirror symmetric about ``y = SIDE/2``, so a
mode-I crack must run straight along it. In the production run it did for the
classical solve and did not for any micropolar row: every one of them left the
plane by ~6% of the height, in the same direction, by the same amount at every
``l_b``. Same direction rules out a bifurcation seeded by round-off, and
``l_b``-independence rules out an internal-length effect.

The cause is not in the micropolar formulation. It is that the micropolar path
is **finite-strain updated-Lagrangian**: the particles advect and the stencil is
rebuilt against a *fixed* background grid every step. Clamping one edge and
driving the other makes the body drift relative to that grid, and a drifted body
has a weight operator that no longer commutes with the mirror. The classical
comparison never shows it because ``PhaseFieldMPMProblem`` is small strain and
never moves its particles at all.

Three tests, cheapest first. Only the last one needs a solve.

  A  the weight operator against the mirror, as a function of drift. No solve,
     no material, no phase field -- pure kinematics of the basis.
  B  the elastic micropolar solve, clamped-and-driven against driven
     symmetrically. Same total stretch, same physics, different alignment.
  C  the staggered run, asymmetry of the damage field per load step.

The lesson generalises past this example: **a symmetry argument about the
continuum says nothing until the discrete operator is checked against the same
symmetry**, and on a fixed-grid method that check has to include where the body
sits on the grid.
"""
from __future__ import annotations

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import GridDirichlet
from scion.discretization.mpm.micropolar import MicropolarMPMStepper

from ._common import (build_grid, make_problem, plane_strain_bc,
                      seed_particles)
from .notched_tension import (E_MOD, GC, N_COUPLING, NU, OVERSHOOT, SIDE,
                              build, critical_displacement)

DX, ELL, PPC = 0.025, 0.10, 2
LB = 2.0 * ELL              # l_b/l = 2, and l_b/dx = 8, well past the floor


def mirror_maps(X0, nodes=None):
    """Exact mirror permutations of the particle lattice and the grid nodes.

    A permutation, never a nearest-neighbour lookup: interpolating would smear
    the very asymmetry being measured. Both are checked to be involutions,
    which is what makes a failure here a broken lattice rather than a bug in
    the test.
    """
    def perm(P, flip):
        k = {tuple(r): i for i, r in enumerate(np.round(P, 9))}
        m = np.array([k[tuple(r)] for r in np.round(flip(P), 9)])
        assert (m[m] == np.arange(len(P))).all()
        return m

    def flip(P):
        return np.c_[P[:, 0], SIDE - P[:, 1], P[:, 2]]

    return perm(X0, flip), (None if nodes is None else perm(nodes, flip))


def test_a():
    print("A  does the weight operator commute with the mirror?")
    print("     w(p, I) against w(mirror p, mirror I), as the body drifts")
    grid, _, _ = build_grid((SIDE, SIDE), DX)
    X0 = seed_particles((SIDE, SIDE), DX, PPC).x.copy()
    nc = np.asarray(grid.node_coordinates())
    mir, sig = mirror_maps(X0, nc)
    print(f"     {'drift':>9} {'drift/dx':>9} {'max |w - w_mirrored|':>22}")
    for frac in (0.0, 0.02, 0.05, 0.125, 0.25, 0.5):
        dy = frac * DX
        st = grid.build_stencil(X0 + np.array([0.0, dy, 0.0]))
        err = 0.0
        for p in range(0, len(X0), 37):        # a lattice; sampling is enough
            a = dict(zip(sig[st.node_id[p]], st.w[p]))
            b = dict(zip(st.node_id[mir[p]], st.w[mir[p]]))
            err = max(err, max(abs(a.get(q, 0.0) - b.get(q, 0.0))
                               for q in set(a) | set(b)))
        print(f"     {dy:9.5f} {frac:9.3f} {err:22.3e}")
    print("     exact only at zero drift, and linear in it thereafter.\n")


def _elastic(ubot, utop, nsteps=6):
    grid, _, _ = build_grid((SIDE, SIDE), DX)
    parts = seed_particles((SIDE, SIDE), DX, PPC)
    X0 = parts.x.copy()
    nan = np.nan
    bcs = [GridDirichlet(lambda c: c[:, 1] < 1e-9,
                         np.array([0.0, ubot, nan, nan, nan, nan])),
           GridDirichlet(lambda c: np.abs(c[:, 1] - SIDE) < 1e-9,
                         np.array([0.0, utop, nan, nan, nan, nan])),
           plane_strain_bc()]
    _, st = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING, l_b=LB,
                         gc=GC, ell_pf=ELL, bcs=bcs, max_stagger=30)
    with quiet():
        for k in range(1, nsteps + 1):
            st.step(parts, k / nsteps, 1.0 / nsteps)
    phi = MicropolarMPMStepper.microrotation_vector(parts)[:, 2]
    return X0, parts.x - X0, phi


def test_b():
    uc = critical_displacement(ELL)
    print("B  the elastic solve, same stretch, two ways of applying it")
    for lbl, ubot, utop in (("clamped / driven", 0.0, 1.2 * uc),
                            ("symmetric drive ", -0.6 * uc, 0.6 * uc)):
        X0, u, phi = _elastic(ubot, utop)
        mir, _ = mirror_maps(X0)
        # u_y maps to (ubot + utop) - u_y; phi_z is an axial vector in the
        # mirror plane, so it changes sign
        ax = np.abs(u[mir, 0] - u[:, 0]).max()
        ay = np.abs((ubot + utop - u[mir, 1]) - u[:, 1]).max()
        az = np.abs(phi[mir] + phi).max()
        print(f"     {lbl}  drift {0.5 * (ubot + utop) / DX:+.3f} dx   "
              f"asym u_x {ax:.2e}  u_y {ay:.2e}  phi_z {az:.2e}   "
              f"|phi_z|max {np.abs(phi).max():.3e}")
    print("     same |phi_z|, so the same physics -- only the alignment "
          "differs.\n")


def test_c(nsteps=10):
    print("C  the staggered run, damage-field asymmetry per load step")
    X0 = seed_particles((SIDE, SIDE), DX, PPC).x
    mir, _ = mirror_maps(X0)
    u_top = OVERSHOOT * critical_displacement(ELL)
    pf, stepper, parts, top = build(LB, ell_pf=ELL, dx=DX, ppc=PPC,
                                    u_top=u_top)
    for k in range(1, nsteps + 1):
        with quiet():
            pf.solve_step(k / nsteps, 1.0 / nsteps)
        d = pf.tables.interpolate(pf.d)
        f = stepper.internal_force(parts)
        a = np.abs(d - d[mir])
        drift = float((parts.x[:, 1] - X0[:, 1]).mean())
        print(f"     lam {k / nsteps:.3f}  load {abs(f[top * 6 + 1].sum()):9.5f}"
              f"  d_max {d.max():.4f}  drift {drift / DX:+.4f} dx"
              f"  asym rms {np.sqrt((a ** 2).mean()):.2e} max {a.max():.2e}")


def main():
    print(f"Mirror symmetry about y = {SIDE / 2}, at dx = {DX}, l = {ELL}, "
          f"l_b = {LB} (l_b/l = {LB / ELL:g})\n")
    test_a()
    test_b()
    test_c()


if __name__ == "__main__":
    main()
