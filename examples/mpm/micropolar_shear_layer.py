#!/usr/bin/env python3
r"""
The shear layer: does the band have a width of its own?

Section 4.1 of O'Hare, Coombs, Gourgiotis & Augarde, CMAME 449 (2026) 118516,
and the test the whole micropolar chain was built for.

An infinitely wide layer is sheared at its top and pinned at its base (Couette
flow), with one row of cells given a **10% lower yield strength** to seed
localisation. The same problem is run on three grids, each a 2x refinement of
the last, with two materials that differ only in their internal length:

* ``l_b = 5 mm`` -- micropolar, the couple stress carries a length;
* ``l_b -> 0``   -- no internal length, i.e. classical behaviour reached
  through the same code path rather than through a second implementation.

What is being tested, and why it cannot be tuned
------------------------------------------------
A classical continuum has no length of its own, so nothing sets the width of a
shear band: it collapses to **one element**, whatever the element is. Refining
the mesh makes the band thinner, the post-peak response more brittle, and the
plastic strain inside it larger without limit -- the dissipation concentrates
into a zone of zero thickness, which is not a physical failure mechanism. This
library already documents the same pathology for local damage
(``materials/damage/``), where the band collapses to one element on three
meshes in a row.

A micropolar continuum has ``l_b``. The couple stress resists sharp gradients of
microrotation, the band acquires a width set by the *material* rather than the
mesh, and the answer stops moving under refinement. The claim is therefore
**mesh invariance** -- band width in millimetres constant while the cell size
halves, so ``band / h`` doubles -- which is a property of the formulation, not a
number that can be reached by adjusting a tolerance.

A reconstruction, and where it is uncertain
--------------------------------------------
The reference gives the slice as "100 mm by 10 mm" on grids of 9x5, 19x10 and
39x20, which cannot all be square cells, and its figure is not in the text. What
the physics needs is unambiguous -- the problem is one-dimensional in ``y`` and
the reference says so -- so this example fixes:

    layer thickness  H = 100 mm  (the direction the band forms across)
    cross-section    one cell in x and z, at every resolution
    cells            square, h = H/n with n = 10, 20, 40

Three choices here were got wrong first time and each produced a *plausible*
wrong answer, so they are worth stating:

* **``l_b`` must be resolved at every mesh.** At ``l_b = 5 mm`` the internal
  length was half a cell on the coarsest grid, which cannot regularise
  anything; at 20 mm it is 2, 4 and 8 cells.
* **The imperfection must have a fixed physical thickness.** Held at one *cell*
  row it shrinks 10 -> 5 -> 2.5 mm under refinement, so the trigger weakens, the
  band spreads, and the measured width *grows* 30 -> 35 -> 60 mm -- which reads
  as a failure of regularisation and is a shrinking defect.
* **The cross-section must not be tied to the cell count.** With the slice one
  cell thick, the body's volume halves with every refinement; the base reaction
  then halves too (measured ratios 1.985 and 1.991, i.e. exactly two) and is not
  comparable between meshes. Reported as a stress instead.

The applied shear is 5% of the thickness, several times the shear yield strain
of ~1.4%.

Everything else is the reference's: ``lambda = mu = 4 GPa``, ``k = 100 MPa``,
``H_hardening = 0``, ``phi = 8.6 deg``, ``psi = 2 deg``, ``kappa = mu/10``,
and the remaining curvature moduli from the principle of bounded stiffness
(``l_t = 2 l_b``, ``psi_polar = 3/2``).

**Softening comes from the non-associated flow rule alone.** ``psi < phi`` with
``H_hardening = 0``: no softening law is needed to localise, which is the
opposite situation to ``materials/damage/`` and worth keeping in view.

Run
---
    python -m examples.mpm.micropolar_shear_layer
    python -m examples.mpm.micropolar_shear_layer --quick
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
from scion.materials.micropolar_plasticity import MicropolarDruckerPrager

THICKNESS = 0.100                        # metres: y is across the layer
E_MODULUS, POISSON = 10.0e9, 0.25        # equivalent to lam = mu = 4 GPa
COUPLING = 0.3015                        # N, from kappa = mu/10 -> a = 0.1
YIELD = 100.0e6
FRICTION, DILATION = 8.6, 2.0
SHEAR = 0.005                            # 5% of the thickness
WEAK = 0.9                               # imperfect row, 10% weaker

#: l_b must be resolved at *every* mesh or the coarsest one has no
#: regularisation to demonstrate. At 20 mm it is 2, 4 and 8 cells across the
#: three grids below.
LENGTH_MICROPOLAR = 0.020
LENGTH_CLASSICAL = 2.0e-5                # far below any cell: no regularisation

#: physical half-thickness of the weakened zone, fixed in millimetres rather
#: than at one cell row -- see the module docstring
IMPERFECTION = 0.005

RESOLUTIONS = [10, 20, 40]
QUICK = [10, 20]
LOAD_STEPS = 120


def build_material(l_b: float, strength_scale: np.ndarray):
    """Bounded stiffness leaves only the coupling number and one length free."""
    elastic = MicropolarHooke.from_technical(
        E=E_MODULUS, nu=POISSON, N=COUPLING,
        l_t=2.0 * l_b, l_b=l_b, psi=1.5)
    phi = np.radians(FRICTION)
    cohesion = YIELD * (3.0 - np.sin(phi)) / (6.0 * np.cos(phi))
    return MicropolarDruckerPrager(
        elastic, cohesion=cohesion * strength_scale, friction=FRICTION,
        dilation=DILATION, plastic_length=max(l_b, 1.0e-4))


def run(n_cells: int, l_b: float, n_steps: int = LOAD_STEPS, ppc: int = 2):
    h = THICKNESS / n_cells
    pad = 2
    # One cell across in x and z at every resolution. The problem is
    # one-dimensional in y, so the cross-section is arbitrary -- but it must not
    # be tied to the *number* of cells, or refining changes the body as well as
    # the mesh. The reaction is reported as a stress for the same reason.
    grid = CartesianGrid(
        shape=(1 + 2 * pad, n_cells + 2 * pad, 1 + 2 * pad), dx=h,
        origin=np.full(3, -pad * h), basis=get_basis("linear"))

    sub = h / ppc
    axes = [np.arange(sub / 2, extent, sub) for extent in (h, THICKNESS, h)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)

    # The weakened zone is a fixed *physical* thickness. Held at one cell row
    # instead it shrinks with the mesh, so the trigger weakens under refinement
    # and the band spreads -- measured 30 -> 35 -> 60 mm, which reads as a
    # failure of regularisation and is a shrinking defect.
    weak = np.abs(X[:, 1] - THICKNESS / 2) < IMPERFECTION
    material = build_material(l_b, np.where(weak, WEAK, 1.0))

    particles = attach_micropolar_state(
        ParticleSet(X.copy(), vol0=sub ** 3, density=1.0), plastic=True)

    nan = np.nan
    everywhere = lambda c: np.ones(len(c), dtype=bool)
    bcs = [
        GridDirichlet(lambda c: c[:, 1] < 1e-9, np.zeros(6)),          # pinned
        GridDirichlet(lambda c: c[:, 1] > THICKNESS - 1e-9,
                      np.array([SHEAR, 0.0, 0.0, nan, nan, nan])),
        # no vertical motion, plane strain, and no microrotation condition --
        # the reference deliberately leaves the rotation free at both faces
        GridDirichlet(everywhere,
                      np.array([nan, 0.0, 0.0, 0.0, 0.0, nan])),
    ]
    stepper = MicropolarMPMStepper(grid, material, dirichlet=bcs, max_iter=25)

    base = np.flatnonzero(grid.node_coordinates()[:, 1] < 1e-9) * 6
    curve = []

    def record(step, result, p):
        # normalised by the cross-section, so it is a stress and comparable
        # between meshes whose cross-section differs
        curve.append((result.load_factor * SHEAR,
                      float(stepper.internal_force(p)[base].sum()) / h ** 2))

    t0 = time.time()
    with quiet():
        stepper.run(particles, n_steps=n_steps, callback=record)
    return particles, stepper, X, h, np.array(curve), time.time() - t0


def band_profile(particles, X, h):
    """Cell-averaged plastic multiplier across the layer, and its width at half
    maximum. Half-maximum rather than a threshold on ``chi`` itself, because the
    absolute level differs between meshes and materials and the *width* is the
    quantity being compared."""
    y, chi = X[:, 1], particles.hardening
    edges = np.arange(0.0, THICKNESS + h / 2, h)
    prof = np.array([chi[(y >= a) & (y < b)].mean()
                     if ((y >= a) & (y < b)).any() else 0.0
                     for a, b in zip(edges[:-1], edges[1:])])
    if prof.max() <= 0.0:
        return prof, 0.0
    return prof, float((prof > 0.5 * prof.max()).sum() * h)


def study(resolutions, l_b, label):
    print(f"\n  {label}   (l_b = {l_b * 1e3:g} mm)")
    print(f"    {'cells':>6} {'h (mm)':>8} {'band (mm)':>10} {'band/h':>7} "
          f"{'chi_max':>10} {'reaction':>11} {'steps':>6} {'sec':>7}")
    rows = []
    for n in resolutions:
        particles, stepper, X, h, curve, dt = run(n, l_b)
        prof, width = band_profile(particles, X, h)
        rows.append((n, h, width, particles.hardening.max(), curve, prof))
        print(f"    {n:6d} {h * 1e3:8.2f} {width * 1e3:10.2f} {width / h:7.2f} "
              f"{particles.hardening.max():10.3e} {curve[-1, 1]:11.3e} "
              f"{len(stepper.history):6d} {dt:7.0f}")
    return rows


if __name__ == "__main__":
    resolutions = QUICK if "--quick" in sys.argv else RESOLUTIONS

    print("Shear layer: is the band width a property of the material or of "
          "the mesh?")
    print(f"  {THICKNESS * 1e3:g} mm layer, {SHEAR * 1e3:g} mm of shear in "
          f"{LOAD_STEPS} steps, one row {100 * (1 - WEAK):g}% weaker")
    print(f"  k = {YIELD / 1e6:g} MPa, phi = {FRICTION:g} deg, "
          f"psi = {DILATION:g} deg, no hardening -- softening comes from the "
          f"non-associated\n  flow rule alone")

    micropolar = study(resolutions, LENGTH_MICROPOLAR, "micropolar")
    classical = study(resolutions, LENGTH_CLASSICAL, "no internal length")

    print("\n  Reading it")
    mw = [r[2] * 1e3 for r in micropolar]
    cw = [r[2] * 1e3 for r in classical]
    mr = [r[2] / r[1] for r in micropolar]
    cr = [r[2] / r[1] for r in classical]
    print(f"    micropolar band width (mm): "
          + " -> ".join(f"{v:.1f}" for v in mw)
          + f"   spread {max(mw) / max(min(mw), 1e-9):.2f}x")
    print(f"    no internal length    (mm): "
          + " -> ".join(f"{v:.1f}" for v in cw)
          + f"   spread {max(cw) / max(min(cw), 1e-9):.2f}x")
    print(f"    in cells: micropolar "
          + "/".join(f"{v:.1f}" for v in mr)
          + f"   classical " + "/".join(f"{v:.1f}" for v in cr))
    print("    A width that holds in millimetres while the cell size halves is "
          "the material\n    setting the scale. A width that holds in *cells* "
          "is the mesh setting it, and\n    that is the failure micropolar "
          "regularisation exists to remove.")
