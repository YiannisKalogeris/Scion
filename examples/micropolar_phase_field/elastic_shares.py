"""Energy composition of an unnotched elastic cantilever, as the cross-check
for the notched-beam sweep.

    python -m examples.micropolar_phase_field.elastic_shares

The notched-beam example reports the share of the *damage driving force*
carried by each of the three terms of the exact decomposition

    W = W_iso(sym E) + mu a |skew E|^2 + W_curv(Gamma).

This script computes the composition of the *stored energy* of the same beam
with no notch and no phase field, over the same ``l_b/h`` sweep.

**It does not agree with the fracture run, and it is not supposed to.**
Measured: unnotched 0.604 / 0.650 / 0.612 against 0.226 / 0.362 / 0.447 at the
fracture run's peak. The notched run's own first load step -- essentially
elastic, damage confined to the seeded notch -- already gives 0.233 / 0.365 /
0.447, so the gap is the *notch*, which concentrates tensile symmetric energy
at its tip, and not the coupling. The check on the coupling is the fracture
run's share history (``beam_data.npz``, ``row{i}_shares``): flat to within 3%
from the first step to the peak. This script supplies the contrast that shows
the unnotched beam is the wrong reference.

An earlier configuration *did* agree (0.737 against 0.714), and for a reason
worth knowing: there ``l = 0.20`` on a depth of 0.4 had smeared the notch away,
so both sides were effectively unnotched.

**What the comparison is and is not.** It uses the same
``driving_shares`` routine, so it is not an independent implementation of the
decomposition; what differs is the *state* it is evaluated on -- undamaged and
unnotched here, mid-crack there. So it tests the coupling, not the algebra.
The algebra is tested separately, by the decomposition being exact to 2.6e-16.
"""
from __future__ import annotations

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import GridDirichlet

from ._common import (build_grid, check_resolution, make_problem,
                      plane_strain_bc, seed_particles)
from .notched_beam import (DEPTH, DX, ELL_PF, E_MOD, GC, NU, N_COUPLING, PPC,
                           SPAN, SWEEP, TIP_DISPLACEMENT)


def elastic_shares(l_b, ell_pf=ELL_PF, dx=DX, ppc=PPC, steps=4):
    """Load the *unnotched* beam elastically and decompose its stored energy."""
    check_resolution(dx, ell_pf, l_b)
    grid, _, _ = build_grid((SPAN, DEPTH), dx)
    parts = seed_particles((SPAN, DEPTH), dx, ppc)
    nan = np.nan
    bcs = [GridDirichlet(lambda c: c[:, 0] < 1e-9, np.zeros(6)),
           GridDirichlet(lambda c: np.abs(c[:, 0] - SPAN) < 1e-9,
                         np.array([nan, TIP_DISPLACEMENT, nan, nan, nan, nan])),
           plane_strain_bc()]
    # The phase-field object is built only for its energy decomposition; no
    # damage is ever solved, so g(d) = 1 everywhere and the state stays elastic.
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING,
                               l_b=l_b, gc=GC, ell_pf=ell_pf, bcs=bcs)
    with quiet():
        for k in range(1, steps + 1):
            stepper.step(parts, k / steps, 1.0 / steps)
    return pf.driving_shares()


def main():
    print("Unnotched elastic cantilever: composition of the stored energy")
    print(f"  span {SPAN} x depth {DEPTH}, E = {E_MOD:g}, nu = {NU}, "
          f"N = {N_COUPLING}")
    print(f"  dx = {DX:g}, l = {ELL_PF:g} (phase field built but never solved)")
    print(f"\n  {'l_b/h':>7} {'l_b':>8} {'W_iso':>9} {'W_rel':>10} "
          f"{'W_curv':>10}")
    print("  " + "-" * 48)
    for r in SWEEP:
        l_b = r * DEPTH
        s = elastic_shares(l_b)
        print(f"  {r:7.3f} {l_b:8.3f} {s['symmetric']:9.4f} "
              f"{s['relative_rotation']:10.4g} {s['curvature']:10.4g}")
    print("\n  Compare the last column against the W_curv column of "
          "notched_beam's\n  table. They are the same decomposition on "
          "different states: undamaged\n  and unnotched here, mid-crack "
          "there.")


if __name__ == "__main__":
    main()
