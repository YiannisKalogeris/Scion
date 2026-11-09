#!/usr/bin/env python3
r"""
The finite element Bree map: transition loads over ray directions.

Tabulates ``lambda_el``, ``lambda_sd`` and ``lambda_rr`` for the thin
plane-strain annulus of Example 1b, as functions of the ray *direction* alone.
That reduction is exact, not an approximation: the yield function is
positively homogeneous of degree one, the elastic field is linear in the load
and the admissible residual fields form a cone, so

    lambda(c x_hat, c y_hat) = lambda(x_hat, y_hat) / c

for every transition load. Verified rather than assumed -- ``c lambda_sd`` is
2.500000 at ``c`` = 0.5, 1, 1.7 and 3.

Why a cylinder rather than a strip
----------------------------------
Bree's wall is restrained against bending. A flat strip needs a multi-point
constraint to hold its end section plane, and a strip with both ends simply
fixed is a different problem -- its residual field may carry net force, so the
admissible set is larger. A cylinder supplies the restraint structurally
through hoop continuity. Measured on this model: the thermal hoop stress
carries essentially no net force, its thickness mean being 1.6e-4 of the face
value at ``n_r = 6`` and 4.0e-5 at ``n_r = 12`` -- a quadrature residual that
converges with the *radial* mesh, not an exact zero -- and its face value is
within 0.43% of ``E alpha DT / [2 (1 - nu)]`` at ``b/a = 1.02``.

Two things the model does not inherit from Bree
-----------------------------------------------
Plane strain is not uniaxial, and the two load cases are not equally
multiaxial: ``sigma_eq / |sigma_theta|`` at the face is 0.90370 under pressure
and 0.92676 under the thermal gradient. So the finite element map is not
Bree's with a rescaled yield stress, and neither reciprocal reproduces the
measured offsets. What the offsets *are* is one constant per branch -- see the
table this script prints.

``lambda_rr`` is bisected only where the alternating band is non-empty, which
is ``y_hat >= 4 x_hat``, i.e. ``phi >= arctan 4 = 75.96`` degrees. Below that
the band is empty and ``lambda_rr = lambda_sd``, which is checked rather than
assumed: at ``1.05 lambda_sd`` on those rays ``Delta eps^p`` and ``r`` come out
*identically equal*, the signature of one-way ratcheting with no reversed
plasticity.

Run
---
    python -m examples.shakedown.fe_bree_map          # ~20 min
    python -m examples.shakedown.fe_bree_map --quick  # fewer directions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import linalg as spla

from scion.core.domain import Domain
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.linear_elastic import isotropic_stiffness_mandel
from scion.materials.notation import (mandel_to_voigt_stress,
                                      mandel_to_voigt_tangent,
                                      voigt_strain_to_mandel)
from scion.mesh.mesh import ElementBlock, Mesh
from scion.shakedown import cyclic_steady_state, transition_loads
from scion.shakedown.bree import _waveform
from scion.shakedown.rsdm import _effective

YIELD = 300.0
E_REF, NU, ALPHA_REF = 200.0e3, 0.3, 1.2e-5
SAFE = ("elastic", "shakedown")
OUT = Path(__file__).with_name("results") / "bree_fe_map.npz"


def annulus(a, b, n_r, n_theta) -> Domain:
    r = np.linspace(a, b, n_r + 1)
    th = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    nodes = np.array([[ri * np.cos(tj), ri * np.sin(tj)]
                      for tj in th for ri in r])
    conn = np.array([[j * (n_r + 1) + i, j * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i + 1, (j + 1) * (n_r + 1) + i]
                     for j in range(n_theta) for i in range(n_r)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)


def _bore_forces(domain, dof_map, a):
    nodes = domain.coordinates
    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    bore = np.where(np.abs(rad - a) < 1e-9 * a)[0]
    bore = bore[np.argsort(np.arctan2(nodes[bore, 1], nodes[bore, 0]))]
    f = np.zeros(dof_map.n_total)
    for p, q in zip(bore[:-1], bore[1:]):
        xp, xq = nodes[p], nodes[q]
        length = np.linalg.norm(xq - xp)
        n = xp / np.linalg.norm(xp) + xq / np.linalg.norm(xq)
        n /= np.linalg.norm(n)
        for nd in (p, q):
            f[dof_map.dof(int(nd), 0)] += 0.5 * length * n[0]
            f[dof_map.dof(int(nd), 1)] += 0.5 * length * n[1]
    return f


def build(a=100.0, b=102.0, n_r=6, n_theta=8):
    """The b-bar annulus and its two elastic load cases.

    ``bbar=True`` because the constrained cyclic solve is displacement-based
    with exactly incompressible plastic flow, so it locks otherwise -- by 17%
    on a thick cylinder, with no symptom in the run.
    """
    domain = annulus(a, b, n_r, n_theta)
    tables = QuadratureTables(domain, thickness=1.0, bbar=True)
    dm, nodes = tables.vector_dofs, domain.coordinates
    on_x = np.where(np.abs(nodes[:, 1]) < 1e-9 * b)[0]
    on_y = np.where(np.abs(nodes[:, 0]) < 1e-9 * b)[0]
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x)[1::2],
                                      dm.element_dofs(on_y)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)
    C = isotropic_stiffness_mandel(E_REF, NU)
    nv, nd = tables.n_voigt, dm.n_total

    xq = np.stack([tables.interpolate(nodes[:, 0]),
                   tables.interpolate(nodes[:, 1])], axis=1)
    rq = np.hypot(xq[:, 0], xq[:, 1])

    K, _ = tables.stiffness_and_force(
        np.zeros((tables.n_points, nv)),
        np.broadcast_to(mandel_to_voigt_tangent(C[None], 2)[0],
                        (tables.n_points, nv, nv)), nd)
    solve = spla.factorized(K.tocsr()[free][:, free].tocsc())

    def stress(u, eps_th):
        eps = voigt_strain_to_mandel(tables.strains_voigt(u), 2)
        return (eps - eps_th) @ C.T

    f1 = _bore_forces(domain, dm, a)
    u1 = np.zeros(nd); u1[free] = solve(f1[free])
    pressure = stress(u1, np.zeros((tables.n_points, 6)))

    # a temperature linear in r about mid-wall: antisymmetric, so it induces
    # no membrane force and what is left is the restrained-curvature state
    T = (rq - 0.5 * (a + b)) / (b - a)
    eps_th = np.zeros((tables.n_points, 6))
    eps_th[:, :3] = (ALPHA_REF * T)[:, None]
    _, f2 = tables.stiffness_and_force(
        mandel_to_voigt_stress(eps_th @ C.T, 2),
        np.zeros((tables.n_points, nv, nv)), nd)
    u2 = np.zeros(nd); u2[free] = solve(f2[free])
    thermal = stress(u2, eps_th)
    return dict(tables=tables, free=free, C=C, xq=xq, a=a, b=b,
                pressure=pressure, thermal=thermal)


def hoop(stress_field, xq):
    """The hoop component of a Mandel stress field at the quadrature points."""
    th = np.arctan2(xq[:, 1], xq[:, 0])
    t = np.stack([-np.sin(th), np.cos(th)], axis=1)
    sxx, syy = stress_field[:, 0], stress_field[:, 1]
    sxy = stress_field[:, 5] / np.sqrt(2.0)
    return (t[:, 0] ** 2 * sxx + t[:, 1] ** 2 * syy
            + 2.0 * t[:, 0] * t[:, 1] * sxy)


def history(m, x_hat, y_hat, n_time=16):
    """The cyclic elastic stress at ``lambda = 1``, normalized on the model.

    Scaled by the model's *own* membrane and face hoop stresses rather than by
    the thin-shell formulae, which takes the wall-thickness and hoop-mesh
    discretization errors out of the comparison with the closed form.
    """
    hp = hoop(m["pressure"], m["xq"]).mean()
    ht = np.abs(hoop(m["thermal"], m["xq"])).max()
    g = _waveform("hump", n_time)
    return (x_hat * YIELD / hp * m["pressure"][None]
            + g[:, None, None] * (y_hat * YIELD / ht) * m["thermal"][None])


def _bisect(pred, lo, hi, steps=12):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    m = build()
    deg = (np.array([30.0, 54.0, 78.0, 84.0]) if quick else
           np.array([24., 30., 36., 42., 48., 54., 60., 66., 72., 76.,
                     78., 80., 82., 84., 86., 88.]))
    print(f"thin plane-strain annulus b/a = {m['b']/m['a']:.2f}, "
          f"{m['tables'].n_points} quadrature points")
    print(f"  membrane hoop per unit pressure "
          f"{hoop(m['pressure'], m['xq']).mean():.5f}; face thermal hoop "
          f"{np.abs(hoop(m['thermal'], m['xq'])).max():.5f} "
          f"(thin shell {E_REF*ALPHA_REF/(2*(1-NU)):.5f})")
    _ht = hoop(m["thermal"], m["xq"])
    print(f"  thermal hoop thickness mean {_ht.mean():.2e}, "
          f"{abs(_ht.mean())/np.abs(_ht).max():.1e} of the face value: "
          f"the restrained-curvature state, to quadrature\n")
    print("   phi   lam_el    lam_sd    lam_rr   sd/Bree  rr/Bree  branch")

    rows = []
    for d in deg:
        phi = np.radians(d)
        x, y = float(np.cos(phi)), float(np.sin(phi))
        tl = transition_loads(x, y)
        bsd, brr = float(tl.shakedown), float(tl.ratchet)
        h = history(m, x, y)
        lam_el = YIELD / _effective(history(m, x, y, 24).reshape(-1, 6)).max()

        def state(lam, mc=150):
            try:
                c = cyclic_steady_state(lam * h, m["tables"], m["free"],
                                        m["C"], YIELD, max_cycles=mc)
                return c.regime(), c
            except Exception as exc:
                return ("collapse" if "equilibrate" in str(exc)
                        else "slow"), None

        t0 = time.time()
        sd = _bisect(lambda L: state(L)[0] not in SAFE, 0.6 * bsd, 1.5 * bsd)
        if y < 4.0 * x:
            rr, branch = sd, "R1"
        else:
            def ratchets(lam):
                reg, c = state(lam)
                if c is None:
                    return reg == "collapse"
                return c.ratchet_rate > 1e-6 * c.plastic_strain_scale
            hi = min(1.5 * brr, 0.98 / (x * 0.904))
            rr = _bisect(ratchets, 1.005 * sd, hi, 10) if ratchets(hi) else np.nan
            branch = "AP"
        rows.append((d, x, y, lam_el, sd, bsd, rr, brr))
        print(f"  {d:5.1f} {lam_el:8.5f} {sd:9.5f} {rr:9.5f}  {sd/bsd:7.4f}  "
              f"{rr/brr:7.4f}  {branch}   ({time.time()-t0:.0f} s)", flush=True)

    a = np.array(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, deg=a[:, 0], x=a[:, 1], y=a[:, 2], fe_el=a[:, 3],
             fe_sd=a[:, 4], bree_sd=a[:, 5], fe_rr=a[:, 6], bree_rr=a[:, 7])
    ok = bool(np.all(a[:, 3] <= a[:, 4] + 1e-12)
              and np.all(a[:, 4] <= a[:, 6] + 1e-12))
    print(f"\n  ordering lambda_el <= lambda_sd <= lambda_rr: {ok}")
    print(f"  saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
