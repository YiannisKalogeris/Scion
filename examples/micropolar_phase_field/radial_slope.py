"""The radial microrotation slope ``k`` in ``omega_r = k r``, and its fit rule.

The paper quotes ``k`` for the elastic torsion bar. Nothing else computed it:
``torsion_3d.fields`` only saves the rotation vectors and
``paper_figures.torsion`` only plots the binned profile, so the number had no
provenance and an earlier recorded value (``+0.38 tau``) turned out to be a
*free-intercept* fit -- which does not describe ``omega_r = k r``, whose line
passes through the origin.

Hence this prints every plausible rule side by side rather than one number.
The reason that matters is in the last two columns: at ``l_b = a/2`` the
profile is not linear, the quadratic term exceeds the linear one at the
surface, and the rules then disagree by 25%. At ``l_b = 0`` they agree to 3%,
which is what makes the classical check meaningful and the micropolar ``k`` a
summary rather than a fit.

    python -m examples.micropolar_phase_field.radial_slope

Reads results/torsion3d_fields_{0,0.5}.npz, written by
``torsion_3d.py --fields``.
"""
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
NBIN = 12          # as paper_figures.torsion bins it, for the secant rows


def section(frac):
    """Radius and radial microrotation on the mid-length section."""
    d = np.load(RESULTS / f"torsion3d_fields_{frac:g}.npz")
    X0, w, c = d["X0"], d["w"], d["centre"]
    L, a, dx = float(d["L"]), float(d["a"]), float(d["dx"])
    tau = float(d["theta"]) / L           # twist per unit length
    mid = np.abs(X0[:, 2] - 0.5 * L) < 0.6 * dx
    px, py = X0[mid, 0] - c[0], X0[mid, 1] - c[1]
    r = np.hypot(px, py)
    # the radial component, via the unit radial vector at each particle
    ur = np.stack([px, py], 1) / np.maximum(r, 1e-30)[:, None]
    return r, w[mid, 0] * ur[:, 0] + w[mid, 1] * ur[:, 1], tau, a


def through_origin(r, y):
    """The fit the model actually states: ``y = k r``, no intercept."""
    return float((r * y).sum() / (r * r).sum())


def binned(r, y, a, nbin=NBIN):
    edges = np.linspace(0.0, a, nbin + 1)
    j = np.clip(np.digitize(r, edges) - 1, 0, nbin - 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([y[j == b].mean() if (j == b).any() else np.nan
                     for b in range(nbin)])
    ok = np.isfinite(prof)
    return rc[ok], prof[ok]


def report(frac):
    r, y, tau, a = section(frac)
    rc, prof = binned(r, y, a)
    k0 = through_origin(r, y)
    inner = r <= 0.9 * a
    s_free, c_free = np.polyfit(r, y, 1)
    # y = k r + q r^2, the cheapest statement of how curved the profile is
    kq = np.linalg.lstsq(np.stack([r, r ** 2], 1), y, rcond=None)[0]
    nonlin = np.abs(prof - k0 * rc).max() / abs(k0 * a)

    print(f"\n  l_b/a = {frac:g}   (tau = {tau:g} /m, a = {a:g} m)")
    print(f"    through origin, all r          k/tau = {k0 / tau:+.4f}")
    print(f"    through origin, r/a <= 0.9     k/tau = "
          f"{through_origin(r[inner], y[inner]) / tau:+.4f}")
    print(f"    secant to the outermost bin    k/tau = "
          f"{prof[-1] / rc[-1] / tau:+.4f}")
    print(f"    free-intercept straight line   k/tau = {s_free / tau:+.4f}"
          f"   (intercept {c_free / (tau * a):+.4f} a)")
    print(f"    quadratic k r + q r^2          k/tau = {kq[0] / tau:+.4f},"
          f"  q a/tau = {kq[1] * a / tau:+.4f}")
    print(f"    departure from k r             {100 * nonlin:.1f}% of |k a|")


def main():
    print("Radial microrotation slope, omega_r = k r, on the mid-length "
          "section.\nThe classical field is k = -tau/2.")
    for frac in (0.0, 0.5):
        if (RESULTS / f"torsion3d_fields_{frac:g}.npz").exists():
            report(frac)
        else:
            print(f"\n  l_b/a = {frac:g}: no field file; run "
                  f"torsion_3d.py --fields {frac:g}")


if __name__ == "__main__":
    main()
