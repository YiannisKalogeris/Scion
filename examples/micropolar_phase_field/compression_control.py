r"""Does the unconditional micropolar driving force crack a specimen in compression?

    python -m examples.micropolar_phase_field.compression_control

The formulation assigns ``W_rel`` and ``W_curv`` to the crack-driving energy in
full, on the grounds that neither has a compressive branch to protect: a
relative rotation and a bending of the microstructure cost energy whatever
their sign, and neither can represent interpenetration. That argument is about
the *terms*; it says nothing about whether the resulting law can nucleate damage
where there is no tensile opening at all. Relative rotation and wryness are
perfectly capable of existing under compression, so the question is quantitative
and has to be measured rather than asserted.

The control is the notched plate of Section 7.1 driven the *other way*, to the
same magnitude of prescribed displacement as the tensile path reaches, at the
largest internal length of that sweep -- ``l_b/l = 4``, which carries the most
curvature energy and is therefore the least favourable case for the null.

**Two things the first version of this control got wrong, both measured.**
It assumed ``W_iso+`` would collapse in compression, leaving only the
micropolar part. It does not: a uniaxially compressed body carries *tensile*
principal strains by Poisson expansion, and the spectral split feeds those into
``W+``, so ``W_iso+`` came out at **65%** of the driving force. The micropolar
terms therefore cannot be read off a compressive run in isolation, and
`classical_compression` supplies the comparator that separates them -- what the
micropolar terms add on top of a split the classical model shares.

It also reported the absolute damage outside the seeded notch band, which reads
**0.90 at the first load step**, before compression has done anything: the
phase field smears the seeded notch over ``l``, so the band mask leaks and a
threshold on ``d`` measures the seed rather than the result. The quantity that
answers the question is the *growth* ``d(end) - d(first step)``, which is what
`report` prints.

A null here is a margin, not a zero: AT2 has no elastic stage, so any positive
driving force produces some damage, and the claim worth making is comparative
-- that the micropolar terms add little to what the split does on its own.

Note what this does *not* model: no contact is enforced across the notch faces,
so in compression they interpenetrate and the spectral split handles the
closure energetically rather than kinematically. That is a limitation of the
control as much as of the formulation, and it biases the test towards finding
*more* spurious damage, not less -- which is the safe direction for a null.
"""
from __future__ import annotations

import time

import numpy as np

from ._common import RESULTS, run_load_path
from .notched_tension import (ELL_PF, GC, NOTCH_WIDTH_FACTOR, NOTCH_X, NOTCH_Y,
                              OVERSHOOT, STEPS, DX, build,
                              critical_displacement)

L_B_OVER_L = 4.0


def notch_band(X, ell, dx):
    """The particles whose damage was seeded, as ``seed_notch`` defines them."""
    behind = X[:, 0] <= NOTCH_X
    dist = np.where(behind, np.abs(X[:, 1] - NOTCH_Y),
                    np.hypot(X[:, 0] - NOTCH_X, X[:, 1] - NOTCH_Y))
    return dist <= max(NOTCH_WIDTH_FACTOR * ell, dx) + 1e-12


def main():
    ell = ELL_PF
    u_max = OVERSHOOT * critical_displacement(ell)
    t0 = time.time()
    print("Compression control: the notched plate of Section 7.1, driven in "
          "compression")
    print(f"  l_b/l = {L_B_OVER_L:g}, |u| = {u_max:.6g} (the tensile path's "
          f"own maximum), {STEPS} steps\n")

    pf, stepper, parts, top = build(L_B_OVER_L * ell, ell_pf=ell,
                                    u_top=-u_max)
    X0 = parts.x.copy()
    band = notch_band(X0, ell, DX)
    crit = GC / (2.0 * ell)                 # the AT2 localisation scale

    def probe(pf_, stepper_, parts_):
        d = pf_.tables.interpolate(pf_.d)
        s = pf_.driving_shares()
        return np.array([d[~band].max(), d.max(), pf_.H[~band].max() / crit,
                         s["symmetric"], s["relative_rotation"],
                         s["curvature"]])

    out = run_load_path(pf, stepper, parts, top, 1, STEPS,
                        progress=f"compression l_b/l={L_B_OVER_L:g}",
                        on_step=probe)
    P = np.asarray(out["extra"])
    print(f"\n  {'t':>6} {'d_max off-notch':>16} {'d_max all':>10} "
          f"{'H_max/(Gc/2l)':>14} {'W_iso+':>9} {'W_rel':>9} {'W_curv':>9}")
    for t, r in zip(out["disp"], P):
        print(f"  {t:6.3f} {r[0]:16.3e} {r[1]:10.4f} {r[2]:14.3e} "
              f"{r[3]:9.4f} {r[4]:9.3e} {r[5]:9.3e}")
    np.savez_compressed(RESULTS / "compression_control.npz",
                        disp=out["disp"], load=out["load"], probe=P,
                        X0=X0, band=band, l_b_over_l=L_B_OVER_L, ell=ell,
                        u_max=u_max, crit=crit, reached=out["reached"],
                        fields=np.asarray(out["fields"]),
                        x=parts.x[:, :2].copy(),
                        minutes=(time.time() - t0) / 60.0)
    print(f"\n  reached t = {out['reached']:.3f}, "
          f"peak off-notch damage {P[:, 0].max():.3e}, "
          f"peak H/(Gc/2l) {P[:, 2].max():.3e}, "
          f"{(time.time() - t0) / 60:.0f} min")


def classical_compression(ell, u_max, steps):
    """The same compression with no micropolar degrees of freedom at all.

    This is the comparator the first version of this control lacked, and
    without it the test cannot answer the question it was written for. Under
    the spectral split a uniaxially compressed body still carries *tensile*
    principal strains, by Poisson expansion, so ``W_iso+`` does not vanish in
    compression -- measured at 65% of the driving force in the micropolar run.
    Whatever damage compression produces is therefore mostly a property of the
    split, which the classical model shares, and the micropolar question is
    only what the two extra terms add on top of it.
    """
    from scion.core.logging import quiet
    from .notched_tension import build_classical_mpm
    problem, top = build_classical_mpm(ell, DX)
    fields = []

    def record(r, prob):
        if r.converged:
            fields.append(prob.tables.interpolate(prob.d.values).copy())

    with quiet():
        problem.run(np.linspace(0.0, -u_max, steps + 1)[1:], callback=record,
                    adaptive=True, stop_on_failure=True)
    return np.asarray(fields), problem.particles.x[:, :2].copy()


def report():
    """Growth of damage during the path, which is the only honest instrument.

    The absolute damage just outside the seeded band is ~0.9 at the first step
    already, because the phase field smears the seeded notch over ``l``; a
    threshold on ``d`` therefore measures the seed, not the result. The
    quantity that answers the question is ``d(end) - d(first step)``.
    """
    m = np.load(RESULTS / "compression_control.npz")
    F = m["fields"]
    rows = [("micropolar, l_b/l = 4", F[-1] - F[0], m["probe"][-1, 3:])]
    cpath = RESULTS / "compression_classical.npz"
    if cpath.exists():
        c = np.load(cpath)["fields"]
        rows.append(("classical (no micropolar)", c[-1] - c[0], None))
    print(f"\n  Growth of damage over the compressive path, "
          f"dd = d(end) - d(first step)\n")
    print(f"  {'':26} {'max dd':>9} {'dd>0.01':>9} {'dd>0.05':>9} {'dd>0.10':>9}")
    for tag, g, _ in rows:
        print(f"  {tag:26} {g.max():9.4f} "
              + " ".join(f"{100*(g > t).mean():8.2f}%"
                         for t in (0.01, 0.05, 0.10)))
    for tag, g, sh in rows:
        if sh is not None:
            print(f"\n  {tag}: driving force at the end is "
                  f"W_iso+ {sh[0]:.3f}, W_rel {sh[1]:.3f}, W_curv {sh[2]:.3f}")


def main_classical():
    ell = ELL_PF
    u_max = OVERSHOOT * critical_displacement(ell)
    t0 = time.time()
    print(f"Classical comparator: same plate, same |u| = {u_max:.6g}, "
          f"no micropolar terms")
    F, x = classical_compression(ell, u_max, STEPS)
    np.savez_compressed(RESULTS / "compression_classical.npz", fields=F, x=x,
                        minutes=(time.time() - t0) / 60.0)
    print(f"  {F.shape[0]} steps recorded, "
          f"max damage growth {(F[-1] - F[0]).max():.4f}, "
          f"{(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    import sys
    if "--classical" in sys.argv:
        main_classical()
    elif "--report" in sys.argv:
        report()
    else:
        main()
