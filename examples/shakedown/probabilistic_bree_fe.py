#!/usr/bin/env python3
r"""
Example 1b: the probabilistic Bree problem on a finite element model.

Example 1a pushes the five random variables of Table 2 through Bree's closed
form. This does the same through a finite element model, so that the only
thing changed is the mechanics -- the random inputs, the representative point
sets, the Heaviside estimator and the Monte Carlo reference are the ones
``probabilistic_bree.py`` already uses, imported rather than reimplemented.

Why a cylinder rather than a strip
----------------------------------
Bree's wall is restrained against bending. In a flat strip that has to be
imposed as a constraint tying the end section to remain plane, which this
library has no facility for, and a strip with both ends simply fixed is a
*different* problem: its residual field may then carry net force, so the
admissible set is larger and the shakedown factor rises. In a cylinder the
restraint is structural -- hoop continuity supplies it -- so the model is a
thin plane-strain annulus at ``b/a = 1.02``, loaded by internal pressure and a
radial temperature gradient. Measured on that model, the thermal hoop stress
carries essentially no net force -- its thickness mean is 1.6e-4 of the face
value, falling to 4.0e-5 as the radial mesh is refined, so a quadrature
residual rather than an exact zero -- which is the restrained-curvature state,
and its face value is within 0.43% of ``E alpha DT / [2(1 - nu)]``.

Why the finite element work is done once
----------------------------------------
``lambda_sd`` and ``lambda_rr`` are positively homogeneous of degree -1 in the
load: the yield function is degree one, the elastic field is linear in the
load, and the admissible residual fields form a cone. So a ray *direction*
fixes the whole ray, and the mechanics reduces exactly to three functions of
one angle. Verified rather than assumed -- ``c * lambda_sd`` is 2.500000 at
``c`` = 0.5, 1, 1.7 and 3.

That is what makes a Monte Carlo reference affordable here at all. The map is
tabulated once by the finite element sweep (``--map``), and every realization
then costs an interpolation in the angle and a division by the magnitude. The
only approximation is the interpolation, and its size is reported.

Run
---
    python -m examples.shakedown.probabilistic_bree_fe            # 1e6 MC
    python -m examples.shakedown.probabilistic_bree_fe --quick
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from scion.shakedown import transition_loads
from scion.shakedown.bree import TransitionLoads
from scion.uq.density import gf_points

from .probabilistic_bree import (RAYS, distribution, monte_carlo, ray_means,
                                 regime_curves)

#: tabulated by ``fe_bree_map.py``; see the module docstring
MAP = Path(__file__).with_name("results") / "bree_fe_map.npz"


class FEMap:
    """The finite element transition loads as a function of the ray.

    Interpolates in the ray angle and scales by the magnitude, which is exact
    rather than an approximation -- see the module docstring on homogeneity.
    Angles outside the tabulated range are clamped, and how often that happens
    is counted rather than ignored, because a clamped realization is one the
    map does not actually cover.
    """

    def __init__(self, path=MAP):
        d = np.load(path)
        self.deg = d["deg"]
        self.sd, self.rr, self.el = d["fe_sd"], d["fe_rr"], d["fe_el"]
        self.clamped = 0
        self.total = 0

    def __call__(self, x_hat, y_hat) -> TransitionLoads:
        x = np.asarray(x_hat, dtype=float)
        y = np.asarray(y_hat, dtype=float)
        rho = np.hypot(x, y)
        phi = np.degrees(np.arctan2(y, np.maximum(x, 1e-300)))
        self.total += phi.size
        self.clamped += int(((phi < self.deg[0]) | (phi > self.deg[-1])).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(rho > 0.0, 1.0 / rho, np.inf)
        return TransitionLoads(
            elastic=np.interp(phi, self.deg, self.el) * inv,
            shakedown=np.interp(phi, self.deg, self.sd) * inv,
            ratchet=np.interp(phi, self.deg, self.rr) * inv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if not MAP.exists():
        print(f"the finite element map is missing: {MAP}\n"
              f"generate it with examples/shakedown/fe_bree_map.py first")
        return 1
    fe = FEMap()
    n_mc = 20_000 if quick else 1_000_000
    counts = (64, 256) if quick else (64, 128, 256, 512, 1024)

    print("Probabilistic Bree problem on a finite element model (Example 1b)")
    print("reference: Monte Carlo through the same finite element map\n")
    print(f"map: {len(fe.deg)} ray directions, "
          f"{fe.deg[0]:.0f} to {fe.deg[-1]:.0f} degrees")

    for name, (x_hat, y_hat) in RAYS.items():
        dT, sP = ray_means(x_hat, y_hat)
        dist = distribution(dT, sP)
        nominal_fe = fe(np.array([x_hat]), np.array([y_hat]))
        nominal_cf = transition_loads(x_hat, y_hat)
        print(f"\nRay {name}: (x_hat, y_hat) = ({x_hat}, {y_hat})")
        print(f"  closed form      lambda_el {np.ravel(nominal_cf.elastic)[0]:.4f}  "
              f"lambda_sd {np.ravel(nominal_cf.shakedown)[0]:.4f}  "
              f"lambda_rr {np.ravel(nominal_cf.ratchet)[0]:.4f}")
        print(f"  finite element   lambda_el {nominal_fe.elastic.item():.4f}  "
              f"lambda_sd {nominal_fe.shakedown.item():.4f}  "
              f"lambda_rr {nominal_fe.ratchet.item():.4f}")

        top = 2.2 * nominal_fe.ratchet.item()
        if not np.isfinite(top):
            top = 2.2 * nominal_fe.shakedown.item()
        lam = np.linspace(0.2, top, 60)

        t0 = time.time()
        ref, _, negative = monte_carlo(dist, n_mc, lam, loads_of=fe)
        print(f"  Monte Carlo: {n_mc} samples in {time.time()-t0:.1f} s"
              + (f"  ({negative} rejected as negative)" if negative else ""))

        print(f"  {'N':>6} {'sum-1':>9} {'max |dF_sd|':>12} {'P_AP max':>9}"
              f" {'clamped':>8}")
        for n in counts:
            before = fe.clamped
            pts = gf_points(dist, n)
            curves, _ = regime_curves(pts, lam, loads_of=fe)
            total = sum(curves.values()) - 1.0
            fsd = curves["AP"] + curves["R"]
            ref_sd = ref["AP"] + ref["R"]
            print(f"  {n:>6} {np.abs(total).max():>9.1e} "
                  f"{np.abs(fsd - ref_sd).max():>12.5f} "
                  f"{curves['AP'].max():>9.5f} {fe.clamped - before:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
