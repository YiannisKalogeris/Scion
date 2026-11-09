#!/usr/bin/env python3
r"""
Probabilistic Bree problem: regime probabilities by DPIM-H.

Problem
-------
The uniaxial Bree idealization -- a thin cylinder wall under a constant
primary membrane stress ``sigma_P`` and a cyclic through-thickness thermal
stress of surface magnitude ``sigma_T`` -- with five random material and load
properties. Every realization sits on a ray through the origin of the Bree
plane, ``(x, y) = lambda (x_hat, y_hat)``, and walks outwards as the load
factor grows, crossing from elastic to shakedown to alternating plasticity to
ratchetting.

Why this is the calibration problem
-----------------------------------
The transition loads are known in closed form (:mod:`scion.shakedown.bree`),
so the *reference* here is Monte Carlo through the **same** formula. Nothing
in the comparison depends on a finite element model or on RSDM: the only error
measured is the representative point set's quadrature error, which is exactly
what a point-selection study should isolate. Mixing in a structural solve
would confound the two.

Reading the numbers
-------------------
``P_E + P_S + P_AP + P_R = 1`` at every load factor, exactly, because the four
bands partition the probability -- that is an identity rather than a check on
accuracy, and it holds at 64 points as well as at 1024.

What varies with the point count is the *placement* of each curve, and the
honest error measure is the one the planned Fig. 5 uses:
``max_lambda |F_sd - F_sd_ref|``, the worst vertical gap in the shakedown
factor's CDF, plus its value at small design probabilities, where a relative
error matters more than an absolute one.

Two rays
--------
``A`` has ``y_hat/x_hat = 8``, so ``y_hat > 4 x_hat`` and it leaves the
shakedown region through ``y = 2`` into alternating plasticity, reaching
ratchetting later at ``x y = 1``: all four regimes appear. ``B`` has
``y_hat/x_hat = 1``, leaves through ``y = 4(1 - x)`` and ratchets at once, so
its alternating-plasticity band is empty. A method that reported a nonzero
``P_AP`` on ray B would be wrong in a way no error norm on ray A could see.

Run
---
    python -m examples.shakedown.probabilistic_bree
    python -m examples.shakedown.probabilistic_bree --quick
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from scipy.stats import qmc

from scion.shakedown import transition_loads
from scion.uq.density import (
    RepresentativePoints,
    band_probability,
    exceedance_probability,
    gf_points,
)
from scion.uq.forward import JointIndependent, LogNormal, Normal

RESULTS = Path(__file__).parent / "results"

NU = 0.3                       # deterministic, as the formulation states

#: Table of random variables: mean and coefficient of variation.
YIELD_MEAN, YIELD_COV = 300.0, 0.08            # MPa
E_MEAN, E_COV = 200.0e3, 0.05                  # MPa
ALPHA_MEAN, ALPHA_COV = 1.2e-5, 0.05           # 1/K
DT_COV = 0.10
SIGMA_P_COV = 0.10

#: The rays, as ``y_hat / x_hat`` at mean properties, with ``x_hat`` chosen so
#: the interesting load factors land near one.
RAYS = {"A": (0.10, 0.80), "B": (0.50, 0.50)}


def thermal_stress(E, alpha, delta_T):
    """``sigma_T = E alpha DT / [2 (1 - nu)]``, the *surface* value.

    Not the peak-to-peak variation through the thickness, which is twice it.
    Every Bree boundary moves by a factor of two under the other convention;
    :mod:`scion.shakedown.bree` records the measurement that settles it.
    """
    return E * alpha * delta_T / (2.0 * (1.0 - NU))


def distribution(delta_T_mean: float, sigma_p_mean: float) -> JointIndependent:
    """The five random inputs, in the order used throughout this script."""
    return JointIndependent([
        LogNormal.from_moments(YIELD_MEAN, YIELD_COV * YIELD_MEAN),
        LogNormal.from_moments(E_MEAN, E_COV * E_MEAN),
        LogNormal.from_moments(ALPHA_MEAN, ALPHA_COV * ALPHA_MEAN),
        Normal(delta_T_mean, DT_COV * delta_T_mean),
        Normal(sigma_p_mean, SIGMA_P_COV * sigma_p_mean),
    ])


def ray_means(x_hat: float, y_hat: float) -> tuple[float, float]:
    """The ``DT`` and ``sigma_P`` means that put the ray at ``(x_hat, y_hat)``
    when every property takes its mean value.

    Defined at mean properties rather than through ``E[x_hat]``, because that
    is what the deterministic Bree diagram the rays are drawn on shows, and
    because ``E[sigma_P/sigma_y]`` is not ``E[sigma_P]/E[sigma_y]`` for a
    lognormal denominator -- a distinction worth keeping visible rather than
    hiding in a solve.
    """
    sigma_p_mean = x_hat * YIELD_MEAN
    delta_T_mean = (y_hat * YIELD_MEAN * 2.0 * (1.0 - NU)
                    / (E_MEAN * ALPHA_MEAN))
    return delta_T_mean, sigma_p_mean


def ray_coordinates(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(x_hat, y_hat)`` for each row of ``theta``.

    The five inputs reach the answer only through these two combinations, so
    the response is two-dimensional however many variables are randomised.
    """
    sigma_y, E, alpha, delta_T, sigma_p = (theta[:, i] for i in range(5))
    return sigma_p / sigma_y, thermal_stress(E, alpha, delta_T) / sigma_y


def regime_bands(points, lam, loads):
    """The four regime probabilities of Eq. (regimes) from transition loads.

    Each band is closed on its upper side, which is what makes the four sum
    to exactly one; see :func:`~scion.uq.density.band_probability`.

    Split out from :func:`regime_curves` so that a model supplying the three
    transition loads by any means -- the closed form, the tabulated finite
    element map of Example 1b, or a solve per realization as in Example 2 --
    reaches the bands through one implementation. This is the one place the
    Heaviside conventions have to agree, and a second copy of it is how they
    would stop agreeing.
    """
    elastic = exceedance_probability(points, loads.elastic, lam, strict=False)
    return {
        "E": elastic,
        "S": band_probability(points, loads.elastic, loads.shakedown, lam),
        "AP": band_probability(points, loads.shakedown, loads.ratchet, lam),
        "R": 1.0 - exceedance_probability(points, loads.ratchet, lam,
                                          strict=False),
    }


def regime_curves(points, lam, loads_of=transition_loads):
    """:func:`regime_bands` on the Bree rays, by DPIM-H.

    ``loads_of`` maps ``(x_hat, y_hat)`` to the three transition loads and
    defaults to the closed form, so the finite element map of Example 1b can
    be substituted without a second copy of anything.
    """
    x_hat, y_hat = ray_coordinates(points.z)
    loads = loads_of(x_hat, y_hat)
    return regime_bands(points, lam, loads), loads


def scrambled_sobol(dist, n: int, seed: int) -> RepresentativePoints:
    """A scrambled Sobol' set with equal weights, as the comparator.

    Equal weights rather than Voronoi ones: that is what makes it the *raw*
    low-discrepancy set GF-discrepancy selection is supposed to improve on.
    Scrambling is what gives it a distribution to put error bars on -- an
    unscrambled Sobol' set is deterministic and would give one number with no
    honest spread, the same trap `MonteCarloResult.std_error` refuses for.
    """
    u = qmc.Sobol(dist.dim, scramble=True, seed=seed).random(n)
    return RepresentativePoints(dist.ppf(u), np.full(n, 1.0 / n))


def crude_points(dist, n: int, seed: int) -> RepresentativePoints:
    """``n`` plain iid samples with equal weights.

    The comparator that decides whether representative points are worth
    anything: the case for DPIM over Monte Carlo is fewer model evaluations
    for the same accuracy, and that claim is only testable against Monte Carlo
    *at the same n*, not against a reference run at a thousand times it.
    """
    rng = np.random.default_rng(10_000 + seed)
    return RepresentativePoints(dist.sample(n, rng=rng), np.full(n, 1.0 / n))


def monte_carlo(dist, n: int, lam, seed: int = 0,
                loads_of=transition_loads):
    """Reference curves, through the identical closed form.

    Crude Monte Carlo, so the error falls as ``1/sqrt(n)`` and the standard
    error of a probability near 0.5 at ``n = 1e6`` is 5e-4 -- which is the
    floor on every comparison below and is reported with them.
    """
    rng = np.random.default_rng(seed)
    theta = dist.sample(n, rng=rng)
    x_hat, y_hat = ray_coordinates(theta)
    negative = int((x_hat < 0.0).sum() + (y_hat < 0.0).sum())
    loads = loads_of(np.maximum(x_hat, 0.0), np.maximum(y_hat, 0.0))

    lam = np.asarray(lam, dtype=float).reshape(-1, 1)
    curves = {
        "E": (loads.elastic >= lam).mean(axis=1),
        "S": ((loads.shakedown >= lam) & (loads.elastic < lam)).mean(axis=1),
        "AP": ((loads.ratchet >= lam) & (loads.shakedown < lam)).mean(axis=1),
        "R": (loads.ratchet < lam).mean(axis=1),
    }
    return curves, loads, negative


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv

    counts = (64, 256) if quick else (64, 128, 256, 512, 1024)
    n_mc = 100_000 if quick else 1_000_000
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Probabilistic Bree problem, regime probabilities by DPIM-H")
    print(f"reference: {n_mc:,} Monte Carlo samples through the same closed "
          f"form\n")

    summary = {}
    for name, (x_hat, y_hat) in RAYS.items():
        delta_T_mean, sigma_p_mean = ray_means(x_hat, y_hat)
        dist = distribution(delta_T_mean, sigma_p_mean)
        nominal = transition_loads(x_hat, y_hat)

        print(f"Ray {name}: y_hat/x_hat = {y_hat / x_hat:.0f} at mean "
              f"properties, (x_hat, y_hat) = ({x_hat}, {y_hat})")
        print(f"  DT = {delta_T_mean:.2f} K, sigma_P = {sigma_p_mean:.1f} MPa")
        print(f"  nominal transition loads: lambda_el = "
              f"{float(nominal.elastic):.4f}, lambda_sd = "
              f"{float(nominal.shakedown):.4f}, lambda_rr = "
              f"{float(nominal.ratchet):.4f}")
        print(f"  alternating plasticity band is "
              f"{'empty' if nominal.alternating_is_empty else 'present'}")

        lam = np.linspace(0.2, 2.5 * float(nominal.ratchet)
                          if np.isfinite(nominal.ratchet)
                          else 4.0 * float(nominal.shakedown), 241)

        t0 = time.time()
        ref, ref_loads, negative = monte_carlo(dist, n_mc, lam)
        mc_seconds = time.time() - t0
        if negative:
            print(f"  WARNING: {negative} Monte Carlo samples had a negative "
                  f"x_hat or y_hat and were clipped")

        f_sd_ref = ref["AP"] + ref["R"]
        print(f"  reference: E[lambda_sd] = "
              f"{ref_loads.shakedown.mean():.4f}, "
              f"sd = {ref_loads.shakedown.std():.4f}  "
              f"({mc_seconds:.1f} s)")
        print(f"  {'N':>6s} {'sum-1':>9s} {'max |dF_sd|':>12s} "
              f"{'P_AP max':>9s} {'floor':>9s} {'F_sd at 1e-2':>21s} "
              f"{'F_sd at 1e-3':>21s} {'pts below':>10s}")

        # design load factors where the reference CDF crosses 1e-2 and 1e-3
        targets = {}
        for level in (1e-2, 1e-3):
            idx = int(np.argmin(np.abs(f_sd_ref - level)))
            targets[level] = (lam[idx], f_sd_ref[idx])

        rows = {}
        for n in counts:
            points = gf_points(dist, n)
            curves, _ = regime_curves(points, lam)
            total = curves["E"] + curves["S"] + curves["AP"] + curves["R"]
            f_sd = curves["AP"] + curves["R"]
            gap = float(np.abs(f_sd - f_sd_ref).max())

            cells = []
            for level, (lam_d, ref_value) in targets.items():
                got = float(np.interp(lam_d, lam, f_sd))
                cells.append(f"{got:.2e} vs {ref_value:.2e}")

            # the smallest nonzero probability a counted estimator can report,
            # and how many points actually sit below the tighter design load
            floor = float(points.probability.min())
            x_hat_q, y_hat_q = ray_coordinates(points.z)
            lam_sd_q = transition_loads(x_hat_q, y_hat_q).shakedown
            below = int((lam_sd_q < targets[1e-3][0]).sum())
            print(f"  {n:>6d} {float(np.abs(total - 1).max()):9.1e} "
                  f"{gap:12.5f} {float(curves['AP'].max()):9.5f} "
                  f"{floor:9.1e} " + " ".join(f"{c:>21s}" for c in cells)
                  + f" {below:>10d}")
            rows[n] = (gap, curves, f_sd)

        # Fig. 5's actual question: does GF-discrepancy selection beat a raw
        # scrambled Sobol' set once the inputs are mapped to a scalar
        # transition load?
        print(f"  {'N':>6s} {'GF':>17s} {'Sobol':>17s} {'crude MC':>17s}"
              f" {'GF/MC':>7s}")
        for n in counts:
            gaps = {}
            for label, build in (("gf", gf_points),
                                 ("sobol", scrambled_sobol),
                                 ("crude", crude_points)):
                values = []
                for seed in range(5):
                    points = build(dist, n, seed=seed)
                    curves, _ = regime_curves(points, lam)
                    values.append(float(np.abs(
                        curves["AP"] + curves["R"] - f_sd_ref).max()))
                gaps[label] = (float(np.mean(values)), float(np.std(values)))
            cells = " ".join(
                f"{gaps[k][0]:11.5f} +-{gaps[k][1]:.4f}"
                for k in ("gf", "sobol", "crude"))
            print(f"  {n:>6d} {cells} {gaps['gf'][0] / gaps['crude'][0]:7.3f}")
        print("  mean +- sd over 5 seeds of max_lambda |F_sd - F_sd_ref|. "
              "GF/MC below 1 means\n  representative points beat iid "
              "sampling at the same evaluation count.\n")

        summary[name] = (lam, ref, rows, nominal)
        gaps = [rows[n][0] for n in counts]
        rate = (np.log(gaps[0] / max(gaps[-1], 1e-300))
                / np.log(counts[-1] / counts[0]))
        print(f"  max |dF_sd| falls {gaps[0]:.4f} -> {gaps[-1]:.4f} over "
              f"{counts[0]} -> {counts[-1]} points, "
              f"ratio {gaps[0] / max(gaps[-1], 1e-300):.1f}x, "
              f"observed rate N^-{rate:.2f}")
        print("  the 'floor' column is min_q P_q: a counted probability "
              "cannot report anything\n  smaller, so a design level at or "
              "below it is unresolvable however good the point set.\n")

    np.savez(RESULTS / "probabilistic_bree.npz",
             **{f"{name}_lambda": summary[name][0] for name in summary},
             **{f"{name}_ref_{k}": v for name in summary
                for k, v in summary[name][1].items()})
    print(f"saved {RESULTS / 'probabilistic_bree.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
