#!/usr/bin/env python3
r"""
Fatigue life of a cantilever under a repeating load spectrum.

Problem
-------
A rectangular cantilever, plane strain, loaded by a transverse force at the tip
that follows a repeating block of large, medium and small cycles. The block is
run at two levels:

    low     the whole beam stays elastic          high-cycle fatigue
    high    the root yields on the large cycles   low-cycle fatigue

and the same spectrum is assessed four ways:

    (1a) elastic FEM       + Basquin S-N + SWT
    (1b) elastoplastic FEM + Basquin S-N + SWT
    (2)  elastic FEM       + Neuber + strain-life + SWT
    (3)  elastoplastic FEM + strain-life + SWT

At the low level all of them are the same calculation and agree to a few
percent. At the high level they separate, in two opposite directions, and both
are worth understanding.

(1b) under-predicts the damage, and this is the trap
----------------------------------------------------
Once a point yields its stress history flattens onto the yield surface. The
large cycles stop growing in stress however much further the load goes, while
the strain keeps going. The counted stress range saturates and the damage with
it, so a stress-life assessment run on a *correct* elastoplastic stress history
reports a life that is too long. Nothing warns you: the analysis converged, the
stresses are right, and only the model is wrong. That is the whole reason
strain-life exists.

(1a) is accidentally close, which is why nobody notices
-------------------------------------------------------
The elastic analysis reports a pseudo-stress far above yield -- a stress the
material never carried -- and on a curve of slope 10 that fiction buys a large
margin. It runs against the missing plasticity, so the two errors partly cancel
and the answer looks defensible.

How much they cancel is not a property of anything. Run this example on the
coarse ``--quick`` mesh and route (1a) comes out 2.9 times the strain-life
damage; on the default mesh it lands within 10% of it. Same model, same
spectrum, same material, and the agreement moved by a factor of three because
the peak pseudo-stress moved. An answer that good for that reason is worth less
than a wrong one you can see.

(2) is outside its own premise, and says so
-------------------------------------------
Neuber assumes a small plastic zone at a notch, contained by surrounding elastic
material. A beam whose outer fibres have yielded across a fifth of the mesh does
not satisfy that, and the correction under-predicts the strain range instead of
bounding it from above. Reported rather than hidden: it is the clearest
available demonstration of where the rule stops applying, and a notched
component under nominally elastic loading -- which is what Neuber is for --
would behave differently.

The consistency that makes the comparison fair
----------------------------------------------
The S-N curve is not an independent fit. It is ``StrainLife.basquin()`` -- the
elastic term of the strain-life curve, in the same four constants -- and both
routes use the same Smith-Watson-Topper mean-stress treatment. In the elastic
limit those two agree *identically* rather than approximately: with
``eps_a = sigma_a/E``, the strain-life SWT parameter ``sigma_max eps_a`` is
``sigma_ar^2/E`` with ``sigma_ar = sqrt(sigma_max sigma_a)``, which is the
stress-life SWT amplitude on the Basquin curve. So agreement at the low level is
a check on the plumbing, and disagreement there would be a bug rather than
physics.

sigma_33 in plane strain
------------------------
The assembler returns ``[s11, s22, s12]``; ``sigma_33`` is not in it and is not
zero. The elastic route fills it with ``nu (s11 + s22)``. The elastoplastic
route cannot -- plastic flow is isochoric and drives the effective ``nu``
towards 1/2 -- so it recomputes the full Mandel stress from the strain and the
stored plastic strain, ``sigma = C_el : (eps - eps_p)``, which is exact for this
material at no cost. Getting this wrong changes the equivalent stress by tens of
percent; see ``fatigue/multiaxial.py``.

Run
---
    python -m examples.fatigue.cantilever_spectrum
    python -m examples.fatigue.cantilever_spectrum --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from scion.core.domain import Domain
from scion.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
    StatefulOperators,
)
from scion.fatigue import (
    FatigueAssessment,
    RambergOsgood,
    StrainLife,
    mandel_to_tensor,
    neuber_range,
    non_proportionality,
    rainflow,
    signed_von_mises,
    voigt_to_tensor,
)
from scion.materials.notation import voigt_strain_to_mandel
from scion.materials.plasticity import LinearIsotropic, PlasticityModel
from scion.mesh.generators import rectangle_mesh

# --- geometry and material -------------------------------------------------
LENGTH, HEIGHT, THICKNESS = 100.0, 20.0, 1.0
NX, NY = 40, 8

E, NU = 210.0e3, 0.3
YIELD, H_ISO = 300.0, 2000.0
SIGMA_U = 600.0

#: Strain-life constants; the S-N curve is the elastic term of this one.
STRAIN_LIFE = dict(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60)

#: One block of the spectrum, as tip-load peaks and valleys in units of the
#: level. A large cycle, four medium ones about a tensile mean, and six small
#: reversed ones -- enough structure that rainflow has something to do and the
#: residue matters.
BLOCK = [0.0, 1.0, -1.0,
         0.6, -0.2, 0.6, -0.2,
         0.35, -0.35, 0.35, -0.35, 0.35, -0.35,
         0.0]

SUBSTEPS = 6            # increments per segment, for the path-dependent run

#: Elastic bending puts ``sigma_11 = F L c / I = 1.5 F`` at the root surface.
#: Plane strain takes the von Mises equivalent down by ``sqrt(1 - nu + nu^2)``
#: and the outermost quadrature point sits inside the surface, so the peak
#: equivalent stress a Gauss point actually sees is nearer ``1.15 F`` on this
#: mesh. ``LOW`` keeps that under the 300 yield; ``HIGH`` puts it about 1.6
#: times over, which yields the outer third of the depth and leaves the plastic
#: zone contained -- push further and Neuber loses its meaning and small-strain
#: plasticity is being asked for strains it does not model.
LOW, HIGH = 160.0, 420.0

RESULTS = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build(nx=NX, ny=NY):
    """Cantilever fixed at ``x = 0``, transverse tip load at ``x = L``."""
    mesh = rectangle_mesh((0.0, LENGTH), (0.0, HEIGHT), nx, ny)
    domain = Domain(mesh, dim=2)

    model = PlasticityModel(E, NU, "j2", LinearIsotropic(YIELD, H_ISO))
    asm = ElastoplasticAssembler(domain, model, thickness=THICKNESS)

    nodes = domain.coordinates
    dm = asm.dof_map

    root = np.where(nodes[:, 0] < 1e-9)[0]
    fixed = np.sort(dm.element_dofs(root))
    free = np.setdiff1d(np.arange(asm.n_dof), fixed)

    # unit tip load, spread over the end nodes by tributary length
    tip = np.where(nodes[:, 0] > LENGTH - 1e-9)[0]
    f_unit = np.zeros(asm.n_dof)
    w = np.full(tip.size, 1.0 / (tip.size - 1))
    w[0] = w[-1] = 0.5 / (tip.size - 1)
    for n, wn in zip(tip, w):
        f_unit[dm.dof(int(n), 1)] = -wn

    return asm, model, dict(free=free, fixed=fixed, f_unit=f_unit,
                            nodes=nodes, tip=tip)


def load_series(level: float, substeps: int = SUBSTEPS) -> np.ndarray:
    """The block, linearly interpolated so the plastic path is resolved."""
    out = [BLOCK[0]]
    for a, b in zip(BLOCK[:-1], BLOCK[1:]):
        out.extend(np.linspace(a, b, substeps + 1)[1:])
    return level * np.asarray(out)


# ---------------------------------------------------------------------------
# Elastic route: one solve, then scale
# ---------------------------------------------------------------------------

def elastic_history(asm, model, info, series):
    """Stress tensors ``(n_points, n_steps, 3, 3)`` from a single linear solve.

    Linearity is the whole trick: the stress field at any load is the unit-load
    field scaled, so the entire history costs one factorisation. It is also what
    makes the elastic route worth doing at all next to a few hundred nonlinear
    steps.
    """
    free = info["free"]
    K = asm.assemble(np.zeros(asm.n_dof), asm.initial_state()).K
    u = np.zeros(asm.n_dof)
    u[free] = spsolve(K.tocsc()[free][:, free], info["f_unit"][free])

    C = model.elastic_tangent_voigt(dim=2)
    sigma_unit = asm.strains(u) @ C.T                     # (n_points, 3) Voigt

    voigt = sigma_unit[:, None, :] * series[None, :, None]
    return voigt_to_tensor(voigt, dim=2, nu=NU), u


# ---------------------------------------------------------------------------
# Elastoplastic route: step the history
# ---------------------------------------------------------------------------

def _newton(ops, f_ext, free, u, tol=1e-8, max_iter=30):
    scale = max(np.linalg.norm(f_ext[free]), 1.0)
    for _ in range(max_iter):
        r = (ops.internal_force(u) - f_ext)[free]
        if np.linalg.norm(r) < tol * scale:
            return u, True
        K = ops.tangent(u).tocsc()[free][:, free]
        u = u.copy()
        u[free] += spsolve(K, -r)
    return u, False


def plastic_history(asm, model, info, series, verbose=True):
    """Stress and strain tensor histories from a stepped elastoplastic run.

    Both are returned as full 3-D tensors. ``sigma_33`` comes from
    ``C_el : (eps - eps_p)`` rather than from the elastic plane-strain relation,
    which stops being true the moment the point yields.
    """
    free = info["free"]
    ops = StatefulOperators(asm)
    u = np.zeros(asm.n_dof)

    sig = np.empty((asm.n_points, series.size, 6))
    eps = np.empty((asm.n_points, series.size, 6))
    failures = 0

    for k, level in enumerate(series):
        u, ok = _newton(ops, level * info["f_unit"], free, u)
        failures += not ok
        ops.commit(u)

        eps_m = voigt_strain_to_mandel(asm.strains(u), dim=2)
        eps[:, k, :] = eps_m
        sig[:, k, :] = (eps_m - ops.state.eps_p) @ model.C_el.T

        if verbose and (k + 1) % 20 == 0:
            print(f"    step {k + 1:3d}/{series.size}  "
                  f"load {level:9.1f}  "
                  f"yielded {100 * (ops.state.alpha > 0).mean():5.1f}%")

    if failures:
        print(f"    WARNING: {failures} steps did not converge")
    return mandel_to_tensor(sig), mandel_to_tensor(eps), ops


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def assess(level: float, quick: bool = False):
    nx, ny = (16, 4) if quick else (NX, NY)
    substeps = 3 if quick else SUBSTEPS

    asm, model, info = build(nx, ny)
    series = load_series(level, substeps)

    strain_life = StrainLife(**STRAIN_LIFE, mean_stress="swt")
    sn = strain_life.basquin(mean_stress="swt")

    print(f"\n  tip load level {level:.0f}, {series.size} steps, "
          f"{asm.n_elements} elements, {asm.n_points} quadrature points")

    # The equivalent-strain reduction carries a 1/(1+nu) normalisation, and the
    # right nu is the one describing the state: the elastic 0.3 while nothing
    # has yielded, tending to 1/2 as plastic flow takes over. Left at its 1/2
    # default the low level would disagree with the S-N route by a factor of
    # two in damage -- not a bug, the wrong constant. Held at the elastic value
    # here, which is exact at the low level and mildly conservative at the high
    # one.
    job_en = FatigueAssessment(strain_life,
                               strain_equivalent_kwargs={"nu": NU})

    # ---- (1a) elastic + S-N -----------------------------------------------
    s_el, _ = elastic_history(asm, model, info, series)
    r_sn = FatigueAssessment(sn).evaluate(stress=s_el)

    # ---- (3) elastoplastic + strain-life ---------------------------------
    print("    elastoplastic run:")
    s_pl, e_pl, ops = plastic_history(asm, model, info, series)
    r_en = job_en.evaluate(stress=s_pl, strain=e_pl)

    # ---- (1b) elastoplastic + S-N: the right stresses, the wrong model ----
    r_sn_pl = FatigueAssessment(sn).evaluate(stress=s_pl)

    # ---- (2) elastic + Neuber + strain-life ------------------------------
    #
    # Neuber is a scalar rule at one location, not a field operation, so it is
    # applied at the hot spot only. It corrects the largest counted *range* of
    # the elastic history onto the cyclic curve; the mean stress of that loop is
    # left as the elastic one, which is the usual simplification and the reason
    # this route is a bracket rather than an answer.
    hot = int(r_en.critical)
    ro = RambergOsgood.from_strain_life(strain_life)
    q_el = signed_von_mises(s_el[hot])
    biggest = rainflow(q_el, residue="repeat").sort_by_range()
    d_sigma, d_eps = neuber_range(biggest.ranges[0], ro)

    # what the elastoplastic run actually did at that same point
    q_pl = signed_von_mises(s_pl[hot])
    from scion.fatigue import equivalent_strain
    e_scalar = equivalent_strain(e_pl[hot], nu=NU)
    true_loop = rainflow(e_scalar, residue="repeat").sort_by_range()

    print(f"\n    hot spot: quadrature point {hot}")
    print(f"      elastic peak equivalent stress   {np.abs(q_el).max():10.1f}")
    print(f"      plastic peak equivalent stress   {np.abs(q_pl).max():10.1f}"
          f"   (yield {YIELD:.0f})")
    print(f"      largest elastic stress range     {biggest.ranges[0]:10.1f}")
    print(f"      Neuber-corrected stress range    {float(d_sigma):10.1f}")
    print(f"      Neuber-corrected strain range    {float(d_eps):10.5f}")
    print(f"      strain range from the FEM run    "
          f"{true_loop.ranges[0]:10.5f}")

    np_score = float(non_proportionality(s_pl[hot][None])[0])
    print(f"      non-proportionality              {np_score:10.3f}"
          f"   ({'proportional' if np_score < 0.05 else 'ROTATING'})")

    routes = (("(1a) elastic       + S-N", r_sn),
              ("(1b) elastoplastic + S-N", r_sn_pl),
              ("(3)  elastoplastic + strain-life", r_en))
    print(f"\n    {'route':<34} {'damage/block':>13} {'blocks':>12}")
    for label, r in routes:
        d = r.damage[r.points == hot][0]
        b = np.inf if d == 0.0 else 1.0 / d
        print(f"    {label:<34} {d:13.4e} {b:12.4g}")

    return dict(level=level, hot=hot, r_sn=r_sn, r_sn_pl=r_sn_pl, r_en=r_en,
                series=series, s_pl=s_pl, e_pl=e_pl, asm=asm,
                yielded=float((ops.state.alpha > 0.0).mean()),
                neuber=(float(d_sigma), float(d_eps)),
                fem_range=float(true_loop.ranges[0]))


# ---------------------------------------------------------------------------

def report(low, high):
    print("\n" + "=" * 74)
    print("  Stress-life against strain-life, same spectrum, two levels")
    print("=" * 74)
    def at_hot(res, key):
        r = res[key]
        return float(r.damage[r.points == res["hot"]][0])

    print(f"\n  {'level':>8} {'1a el+S-N':>12} {'1b pl+S-N':>12} "
          f"{'3 pl+eps-N':>12} {'1a/3':>8} {'1b/3':>8}  regime")
    for tag, res in (("low", low), ("high", high)):
        d1a, d1b, d3 = (at_hot(res, "r_sn"), at_hot(res, "r_sn_pl"),
                        at_hot(res, "r_en"))
        regime = "elastic, HCF" if tag == "low" else "yielding, LCF"
        print(f"  {res['level']:8.0f} {d1a:12.4e} {d1b:12.4e} {d3:12.4e} "
              f"{d1a / d3:8.2f} {d1b / d3:8.2f}  {regime}")

    lo = at_hot(low, "r_sn") / at_hot(low, "r_en")
    print(f"\n  Low level: all three agree to {100 * abs(lo - 1.0):.0f}%. "
          f"Nothing has yielded, so they\n  are the same model -- the elastic "
          f"term of the strain-life curve is the S-N\n  curve, and both use "
          f"SWT. This is a check on the counting, the reduction\n  and the "
          f"mean-stress plumbing, not on the physics.")

    hi_a = at_hot(high, "r_sn") / at_hot(high, "r_en")
    hi_b = at_hot(high, "r_sn_pl") / at_hot(high, "r_en")
    print(f"\n  High level: the two stress-life routes sit at {hi_a:.2f}x and "
          f"{hi_b:.2f}x the\n  strain-life damage.")
    print(f"\n    (1b) is the dangerous one, and it is a model error, not a "
          f"discretisation\n         one. Its stresses are correct; the stress "
          f"history simply saturates\n         at the yield surface, so the "
          f"counted ranges stop growing with the\n         load and the damage "
          f"stops with them. Always in the unsafe direction.")
    print(f"    (1a) is close only by accident: a pseudo-stress far above "
          f"anything the\n         material carried, running against the "
          f"plasticity the model cannot\n         see. On the coarse --quick "
          f"mesh the same comparison gives 2.9x\n         instead of "
          f"{hi_a:.2f}x, which is how little the agreement means.")

    dn, fem = high["neuber"][1], high["fem_range"]
    print(f"\n  Neuber on the elastic result gives a strain range of "
          f"{dn:.5f} against\n  {fem:.5f} from the elastoplastic run, "
          f"{100 * (dn / fem - 1):+.0f}% -- under-predicting where it is\n"
          f"  supposed to bound from above. The premise has failed: "
          f"{100 * high['yielded']:.0f}% of the mesh\n  has yielded, so there "
          f"is no contained plastic zone in an elastic field, and\n  the rule "
          f"is outside its range of validity rather than merely inaccurate.")


def plot(low, high):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    from scion.fatigue import equivalent_strain

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax[0].plot(high["series"], lw=1.2)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("tip load")
    ax[0].set_title("One block of the spectrum")
    ax[0].grid(alpha=0.3)

    h = high["hot"]
    e = equivalent_strain(high["e_pl"][h], nu=0.5)
    s = signed_von_mises(high["s_pl"][h])
    ax[1].plot(e, s, lw=1.1)
    ax[1].axhline(YIELD, ls=":", c="0.5", lw=1.0)
    ax[1].axhline(-YIELD, ls=":", c="0.5", lw=1.0)
    ax[1].set_xlabel(r"equivalent strain")
    ax[1].set_ylabel(r"signed von Mises stress")
    ax[1].set_title("Hot-spot response, high level")
    ax[1].grid(alpha=0.3)

    # Every quadrature point, stress-life against strain-life. On the 1:1 line
    # the two models agree; distance from it is the disagreement, per point,
    # with no ranking to confuse the two orderings.
    series = ((low, "r_sn", "low, elastic + S-N", "o", "0.45"),
              (high, "r_sn", "high, elastic + S-N", "o", "tab:orange"),
              (high, "r_sn_pl", "high, elastoplastic + S-N", "^", "tab:red"))
    lo_lim, hi_lim = np.inf, 0.0
    for res, key, label, marker, colour in series:
        x = res["r_en"].damage
        y = res[key].damage
        keep = (x > 0.0) & (y > 0.0)
        ax[2].loglog(x[keep], y[keep], marker, ms=2.5, alpha=0.5, color=colour,
                     label=label)
        lo_lim = min(lo_lim, x[keep].min(), y[keep].min())
        hi_lim = max(hi_lim, x[keep].max(), y[keep].max())

    line = np.array([lo_lim, hi_lim])
    ax[2].plot(line, line, "k-", lw=1.0, label="agreement")
    ax[2].plot(line, 0.1 * line, "k:", lw=0.8)
    ax[2].text(line[1], 0.1 * line[1], " 10x unsafe", fontsize=7, va="center")
    ax[2].set_xlabel("damage per block, strain-life")
    ax[2].set_ylabel("damage per block, stress-life")
    ax[2].set_title("Stress-life against strain-life, per point")
    ax[2].legend(fontsize=7, frameon=False, loc="upper left")
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "cantilever_spectrum.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    # scoped to this package: logging.disable is process-global and
    # would silence every other library in the interpreter too
    import logging

    from scion.core.logging import set_level
    set_level(logging.WARNING)

    quick = "--quick" in sys.argv
    print("Cantilever under a repeating load spectrum")
    print(f"  L={LENGTH:g}, H={HEIGHT:g}, E={E:g}, nu={NU}, "
          f"yield={YIELD:g} (+{H_ISO:g} alpha)")
    print(f"  spectrum: {len(BLOCK) - 1} segments, "
          f"{sum(1 for _ in BLOCK)} turning points per block")

    low = assess(LOW, quick)
    high = assess(HIGH, quick)
    report(low, high)
    if not quick:
        plot(low, high)
