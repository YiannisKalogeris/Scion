r"""Is the notched beam's strength gain more than its stiffness gain?

    python -m examples.micropolar_phase_field.beam_stiffness_control
    python -m examples.micropolar_phase_field.beam_stiffness_control --quick

The sweep in ``notched_beam`` raises the peak load 3.56x between ``l_b = 0``
and ``l_b = h/2``, but the beam also becomes 6.06x stiffer, and in AT2 phase
field a stiffer body is stronger. This is the control that separates the two: the
``l_b = 0`` beam with its Young's modulus raised until its stiffness matches the
stiffest micropolar row, and nothing else changed.

**Its answer is known in advance, which is what makes it a check and not a
fishing trip.** With ``l_b = 0`` every modulus is proportional to ``E`` -- the
coupling modulus is ``mu a`` with ``a`` a function of ``N`` alone -- while
``G_c`` and ``l`` do not involve it. Substituting ``u = u'/sqrt(E)`` makes the
elastic energy independent of ``E``, and the fracture energy never depended on
it, so the problem maps onto itself and the reaction scales as exactly
``sqrt(E)``. The seeded notch ``H_0 ~ G_c/l`` is ``E``-independent too and maps
consistently. So the predicted peak ratio is ``sqrt(k)`` for a stiffness ratio
``k``; finite strain breaks the scaling slightly, a bug would break it a lot.

**The tip displacement is scaled by ``1/sqrt(k)`` as well**, so that every load
step of the control is the exact image of a step of the ``l_b = 0`` row. Keeping
the sweep's 6 mm over 12 steps would put the control's peak between steps 3 and
4 and under-sample it -- comparing a coarsely sampled peak against a well
sampled one would manufacture a strength difference out of the load stepping.

Reads the stiffness ratio from ``beam_data.npz`` (the completed sweep) rather
than restating it, and writes ``beam_control.npz``.
"""
from __future__ import annotations

import sys

import numpy as np

from ._common import RESULTS, run_load_path
from .notched_beam import (DX, E_MOD, ELL_PF, QUICK_DX, QUICK_ELL,
                           QUICK_STEPS, STEPS, TIP_DISPLACEMENT, build)


def main(quick=False):
    tag = "_quick" if quick else ""
    dx, ell, steps = (QUICK_DX, QUICK_ELL, QUICK_STEPS) if quick else \
        (DX, ELL_PF, STEPS)
    src = RESULTS / f"beam_data{tag}.npz"
    if not src.exists():
        sys.exit(f"{src} not found: run notched_beam{' --quick' if quick else ''} "
                 f"first, the control is defined against it")
    d = np.load(src)
    last = len(d["l_b_over_h"]) - 1
    f0, fN = np.abs(d["row0_load"]), np.abs(d[f"row{last}_load"])
    # both rows share the same first displacement, so the load ratio there is
    # the stiffness ratio -- measured on the notched beam actually being compared
    k = float(fN[0] / f0[0])
    pk0, pkN = f0.max(), fN.max()
    s = np.sqrt(k)

    print("Stiffness-matched control for the notched beam")
    print(f"  from {src.name}: l_b/h = {d['l_b_over_h'][last]:g} is {k:.4f}x "
          f"stiffer than l_b = 0 and {pkN / pk0:.4f}x stronger")
    print(f"  control: l_b = 0, E = {k:.4f} E0, tip displacement / {s:.4f}")
    print(f"  prediction (exact sqrt(E) scaling): peak ratio {s:.4f}\n")

    pf, stepper, parts, tip_nodes = build(0.0, ell_pf=ell, dx=dx,
                                          e_mod=k * E_MOD,
                                          tip=TIP_DISPLACEMENT / s)
    out = run_load_path(pf, stepper, parts, tip_nodes, 1, steps,
                        progress="control")
    fc = np.abs(out["load"])
    pkc = float(fc.max())
    k_meas = float(fc[0] * s / f0[0])        # its stiffness, against l_b = 0

    print(f"\n  control stiffness   {k_meas:.4f}x   (target {k:.4f}x)")
    print(f"  control peak        {pkc / pk0:.4f}x   (predicted {s:.4f}x, "
          f"off by {100 * (pkc / pk0 / s - 1):+.2f}%)")
    print(f"  micropolar peak     {pkN / pk0:.4f}x")
    print(f"\n  strength beyond stiffness: {pkN / pkc:.4f}x -- the part of the "
          f"micropolar\n  gain a classical beam of the same stiffness does not "
          f"reproduce.")
    np.savez_compressed(RESULTS / f"beam_control{tag}.npz",
                        k=k, disp=out["disp"], load=out["load"],
                        reached=out["reached"], fields=np.asarray(out["fields"]),
                        peak_ratio=pkc / pk0, micropolar_ratio=pkN / pk0)
    print(f"  wrote {RESULTS / f'beam_control{tag}.npz'}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
