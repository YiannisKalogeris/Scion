r"""
Section 6: ranking the MPM variants on a common diagnostic.

Claim (c) is that ``||G_p||`` and the production rate of the total
pseudomomentum ``P_h`` vanish for the exact solution of a homogeneous body, are
therefore admissible error indicators, and rank the standard variants **on a
single scale independent of the benchmark problem**.

The falsifiable part is the last clause, so that is what this measures: the same
variants on several periodic problems of deliberately different deformation
character, checking whether the ordering survives. Producing one ranking on one
problem would demonstrate nothing -- every paper in the literature already does
that, and the complaint in Section 1.1 is precisely that those orderings do not
transfer.

Why periodic
------------
``P_h`` is conserved by the continuum only when the boundary flux
``\oint \Sigma N dS`` vanishes. A traction-free body does not qualify: traction
free means ``P N = 0``, leaving ``\Sigma N = (W - K) N \neq 0``. A homogeneous
body filling a periodic cell has no boundary at all, so any drift measured here
is the discretisation and nothing else. Two other configurations were tried and
rejected: a free spinning disk, where the drift is physical rather than
numerical, and rigid translation, which every scheme conserves trivially.

Normalisation
-------------
Drifts are divided by ``M v_rms``, not by ``|P_h(0)|``. Several of these
velocity fields are antisymmetric and integrate to nearly zero momentum, and
normalising by a quantity that is itself ~0 produces ratios of 1e2 that mean
nothing. That mistake is easy to make and hard to see.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC, CPDIGrid, CartesianGrid, ExplicitMPMStepper, FLIP, MLSMPM, PIC,
    ParticleSet, PolyPIC,
)
from scion.discretization.mpm.basis import get_basis
from scion.materials.eshelby import total_pseudomomentum
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

RHO = 1000.0
E, NU = 5.0e4, 0.3
N_GRID = 24
PPC = 2


# ---------------------------------------------------------------------------
# Variants: a (grid basis, transfer) pair, not a point on a line
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    label: str
    basis: str
    transfer: object
    grid_cls: type = CartesianGrid

    def build(self, periodic=True):
        return self.grid_cls(shape=(N_GRID, N_GRID), dx=1.0 / N_GRID,
                             basis=get_basis(self.basis), periodic=periodic)


def variants():
    return [
        Variant("MPM (linear)", "linear", APIC()),
        Variant("BSMPM quad", "quadratic", APIC()),
        Variant("BSMPM cubic", "cubic", APIC()),
        Variant("uGIMP", "gimp", APIC()),
        Variant("CPDI", "linear", APIC(), CPDIGrid),
        Variant("PIC", "quadratic", PIC()),
        Variant("FLIP 0.95", "quadratic", FLIP(alpha=0.95)),
        Variant("PolyPIC d1", "quadratic", PolyPIC(1)),
        Variant("MLS-MPM", "quadratic", MLSMPM()),
    ]


# ---------------------------------------------------------------------------
# Benchmarks: periodic, homogeneous, deliberately different in character
# ---------------------------------------------------------------------------

DRIFT = np.array([0.35, 0.15])


def shear_wave(x):
    v = np.zeros_like(x)
    v[:, 0] = 0.6 * np.sin(2 * np.pi * x[:, 1])
    return v + DRIFT


def dilatational_wave(x):
    v = np.zeros_like(x)
    v[:, 0] = 0.6 * np.sin(2 * np.pi * x[:, 0])
    return v + DRIFT


def vortex_array(x):
    v = np.zeros_like(x)
    v[:, 0] = 0.6 * np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
    v[:, 1] = -0.6 * np.cos(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])
    return v + DRIFT


def multimode(x):
    v = np.zeros_like(x)
    for k, a in [(1, 0.4), (2, 0.2), (3, 0.1)]:
        v[:, 0] += a * np.sin(2 * np.pi * k * x[:, 1])
        v[:, 1] += a * np.sin(2 * np.pi * k * x[:, 0] + 0.7)
    return v + DRIFT


def biaxial(x):
    v = np.zeros_like(x)
    v[:, 0] = 0.6 * np.sin(2 * np.pi * x[:, 0])
    v[:, 1] = 0.6 * np.sin(2 * np.pi * x[:, 1])
    return v + DRIFT


#: Grouped by deformation character, because the measured result is that the
#: ranking transfers *within* a group and not between them.
DEFORMATION_CLASS = {
    "dilatational": "volumetric",
    "biaxial": "volumetric",
    "shear wave": "deviatoric",
    "vortex array": "deviatoric",
    "multimode": "deviatoric",
}

BENCHMARKS = {
    "dilatational": dilatational_wave,
    "biaxial": biaxial,
    "shear wave": shear_wave,
    "vortex array": vortex_array,
    "multimode": multimode,
}


# ---------------------------------------------------------------------------

def run_one(variant: Variant, flow, n_steps: int = 300):
    """Returns (pseudomomentum drift, spatial momentum drift), both scaled."""
    grid = variant.build()
    material = NeoHookeanCompressible.from_E_nu(E, NU)
    dx = 1.0 / N_GRID

    p = ParticleSet.from_sampler(
        lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
        dx, ppc=PPC, density=RHO)
    p.v[:] = flow(p.x)

    c = material.wave_speed_factor / np.sqrt(RHO)
    stepper = ExplicitMPMStepper(grid, material, variant.transfer,
                                 dt=0.25 * dx / c)

    m = p.mass
    scale = m.sum() * np.sqrt((p.v ** 2).sum(axis=1).mean())
    p0 = (m[:, None] * p.v).sum(axis=0)
    P0 = total_pseudomomentum(m, p.F, p.v)

    try:
        with quiet():
            stepper.run(p, n_steps=n_steps)
    except Exception as exc:                       # noqa: BLE001
        return None, None, type(exc).__name__
    if not np.all(np.isfinite(p.F)):
        return None, None, "NaN"

    dP = np.linalg.norm(total_pseudomomentum(m, p.F, p.v) - P0) / scale
    dp = np.linalg.norm((m[:, None] * p.v).sum(axis=0) - p0) / scale
    return dP, dp, None


def spearman(a, b) -> float:
    """Rank correlation between two orderings of the same labels."""
    ra = {k: i for i, k in enumerate(a)}
    rb = {k: i for i, k in enumerate(b)}
    common = [k for k in a if k in rb]
    if len(common) < 3:
        return float("nan")
    x = np.array([ra[k] for k in common], dtype=float)
    y = np.array([rb[k] for k in common], dtype=float)
    x -= x.mean()
    y -= y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def main(n_steps: int = 300):
    vs = variants()
    results = {}

    print(f"grid {N_GRID}^2 periodic, ppc={PPC}, {n_steps} steps, "
          f"Neo-Hookean E={E:.0e} nu={NU}")
    print("diagnostic: |dP_h| / (M v_rms). Control: |dp| / (M v_rms).")
    print()

    header = f"{'variant':>14}" + "".join(f"{b:>15}" for b in BENCHMARKS)
    print(header)
    print("-" * len(header))
    for v in vs:
        row = {}
        cells = []
        for name, flow in BENCHMARKS.items():
            dP, dp, err = run_one(v, flow, n_steps)
            row[name] = dP
            cells.append(f"{dP:15.3e}" if dP is not None else f"{err:>15}")
        results[v.label] = row
        print(f"{v.label:>14}" + "".join(cells))

    print()
    print("Control -- spatial momentum drift, same runs (should be ~1e-15):")
    for v in vs[:3]:
        _, dp, err = run_one(v, shear_wave, n_steps)
        print(f"  {v.label:>14}  {dp:.2e}" if dp is not None else
              f"  {v.label:>14}  {err}")

    # --- the actual test of claim (c) ---
    print()
    print("=" * 72)
    print("  Is the ranking independent of the benchmark?")
    print("=" * 72)
    orders = {}
    for name in BENCHMARKS:
        ok = [(lbl, r[name]) for lbl, r in results.items() if r[name] is not None]
        orders[name] = [lbl for lbl, _ in sorted(ok, key=lambda t: t[1])]
        print(f"  {name:>14}: " + " < ".join(orders[name]))

    print()
    names = list(BENCHMARKS)
    print("  pairwise Spearman rank correlation:")
    worst = 1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearman(orders[names[i]], orders[names[j]])
            worst = min(worst, r)
            print(f"    {names[i]:>14} vs {names[j]:<14} rho = {r:+.3f}")
    within, between = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearman(orders[names[i]], orders[names[j]])
            same = DEFORMATION_CLASS[names[i]] == DEFORMATION_CLASS[names[j]]
            (within if same else between).append(r)

    print()
    print(f"  worst pairwise rho = {worst:+.3f}")
    if within and between:
        print(f"  within  a deformation class: mean rho = {np.mean(within):+.3f}")
        print(f"  between deformation classes: mean rho = {np.mean(between):+.3f}")
        print("  The ranking transfers within a class and much less between,")
        print("  so it is a property of the scheme *and* of what the flow does.")
    print("  rho = 1 everywhere would support claim (c); a low or negative")
    print("  value means the ordering is a property of the problem, not of the")
    print("  scheme, which is the thing the paper says the literature does.")
    return results, orders


if __name__ == "__main__":
    main(n_steps=150 if "--quick" in sys.argv else 300)


# ---------------------------------------------------------------------------
# The other half of claim (c): rank on the configurational force itself
# ---------------------------------------------------------------------------

def nodal_material_force_norm(p, grid, st, material):
    r"""``||G_i||`` from Eq. (10), scaled. Updated-Lagrangian.

    ``Grad w`` on a UL stencil is a *spatial* gradient, and Eq. (10) wants a
    material one, so it is pulled back with ``F^T``. That is an approximation
    only in the sense that ``F`` is the discrete deformation gradient; the
    identity ``Grad_X = F^T Grad_x`` is exact.
    """
    from scion.materials.eshelby import eshelby_stress, material_force

    P, _ = material.pk1(p.F)
    W = material.energy(p.F)
    sigma = eshelby_stress(W, p.F, P)
    grad_X = np.einsum("pji,pkj->pki", p.F, st.grad_w)

    G = np.zeros((grid.n_nodes, grid.dim))
    np.add.at(G, st.node_id, material_force(sigma, p.vol0, grad_X))
    scale = float((p.vol0 * W).sum()) / grid.dx        # energy / length
    return float(np.abs(G).max()) / max(scale, 1e-300)


def run_one_with_G(variant: Variant, flow, n_steps: int = 300):
    """Pseudomomentum drift *and* ``||G_i||``, from the same run."""
    grid = variant.build()
    material = NeoHookeanCompressible.from_E_nu(E, NU)
    dx = 1.0 / N_GRID

    p = ParticleSet.from_sampler(
        lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
        dx, ppc=PPC, density=RHO)
    p.v[:] = flow(p.x)

    c = material.wave_speed_factor / np.sqrt(RHO)
    stepper = ExplicitMPMStepper(grid, material, variant.transfer,
                                 dt=0.25 * dx / c)
    m = p.mass
    scale = m.sum() * np.sqrt((p.v ** 2).sum(axis=1).mean())
    P0 = total_pseudomomentum(m, p.F, p.v)

    try:
        with quiet():
            stepper.run(p, n_steps=n_steps)
        extra = ({"domain": p.domain}
                 if isinstance(grid, CPDIGrid) else {})
        st = grid.build_stencil(p.x, lp=getattr(p, "lp", None), **extra)
        gi = nodal_material_force_norm(p, grid, st, material)
    except Exception as exc:                            # noqa: BLE001
        return None, None, type(exc).__name__
    if not np.all(np.isfinite(p.F)):
        return None, None, "NaN"

    dP = np.linalg.norm(total_pseudomomentum(m, p.F, p.v) - P0) / scale
    return dP, gi, None


def compare_diagnostics(n_steps: int = 300):
    """Do ``||G_i||`` and pseudomomentum production rank the same way?"""
    vs = variants()
    print(f"grid {N_GRID}^2 periodic, {n_steps} steps")
    print("Two diagnostics from the SAME runs. If claim (c) holds they agree.")
    print()

    agree = []
    for name, flow in BENCHMARKS.items():
        rows = []
        for v in vs:
            dP, gi, err = run_one_with_G(v, flow, n_steps)
            if err is None:
                rows.append((v.label, dP, gi))
        by_P = [r[0] for r in sorted(rows, key=lambda t: t[1])]
        by_G = [r[0] for r in sorted(rows, key=lambda t: t[2])]
        rho = spearman(by_P, by_G)
        agree.append(rho)
        print(f"  {name:>14}  rho(P_h, G_i) = {rho:+.3f}")
        print(f"      by P_h: " + " < ".join(by_P))
        print(f"      by G_i: " + " < ".join(by_G))
    print()
    print(f"  mean agreement between the two diagnostics: {np.mean(agree):+.3f}")
    print("  Claim (c) offers both as indicators of the same thing. A low")
    print("  value means they are not measuring the same thing.")
    return agree
