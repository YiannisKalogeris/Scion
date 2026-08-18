#!/usr/bin/env python3
r"""
Forward UQ: the rate you cannot beat, and the three ways round it.

Everything here is measured against a problem whose answer is known, because a
Monte Carlo estimate that is wrong looks exactly like one that is right.

Part 1 -- the rate
------------------
The error falls as ``1/sqrt(N)``, and the run below fits the slope rather than
asserting it: **-0.4999** against a theoretical -0.5 on the full run (``--quick``
uses too few repetitions to resolve it that closely). Two things follow, and
they set up everything else.

It does not depend on the dimension. That is the property no tensor-product
quadrature has, and the only reason Monte Carlo is used on a twenty-parameter
model at all.

It cannot be improved by averaging more cleverly. Halving the error costs four
times the samples, always. So the methods that follow do not attack the rate --
they attack the constant in front of it, or they change the question.

Part 2 -- better designs, and one that can backfire
---------------------------------------------------
Same budget, different placement of the points. On a smooth low-dimensional
integrand the quasi-random designs win by orders of magnitude. Antithetic
variates are free and are *not safe*: guaranteed only for a model monotone in
every input, and measurably harmful for one symmetric about the centre of the
input distribution. Both cases are run.

A related trap that the code refuses to paper over: a quasi-random sample has no
honest standard error, because its points are deterministic and correlated. The
usual ``s/sqrt(N)`` understates it, which is the dangerous direction, so
``std_error`` is ``None`` unless the design is replicated.

Part 3 -- rare events, where crude Monte Carlo simply stops working
-------------------------------------------------------------------
Estimating a failure probability by counting failures has a coefficient of
variation of ``1/sqrt(N P_f)``, so 10% accuracy needs ``N ~ 100/P_f``. For a
target reliability index of 4.75 -- an ordinary structural requirement -- that
is ``1e8`` model evaluations. And the failure is silent: too few samples returns
zero failures, which reads as a probability of zero rather than as ignorance.

FORM, importance sampling about the design point, and subset simulation each
get there in a few thousand. The linear limit state has ``P_f = Phi(-beta)``
exactly, so the comparison is against truth and not against each other.

Part 4 -- which inputs matter
-----------------------------
Sobol indices on the Ishigami function, whose indices are analytic. Its third
input is the reason it is the standard benchmark: ``S_3 = 0`` **exactly** and
``S_T3 = 0.244``. An input can have no effect of its own and still carry a
quarter of the output variance through an interaction, so only a small *total*
index licenses fixing an input at its mean.

Run
---
    python -m examples.uq.forward_propagation
    python -m examples.uq.forward_propagation --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scion.uq.forward import (
    DESIGNS,
    Ishigami,
    JointIndependent,
    LinearLimitState,
    LogNormal,
    Normal,
    Uniform,
    control_variates,
    form,
    importance_sampling_probability,
    monte_carlo_probability,
    propagate,
    required_samples,
    sobol_indices,
    subset_simulation,
)

RESULTS = Path(__file__).resolve().parent / "results"

FULL = dict(rate=(256, 1024, 4096, 16384, 65536), repeats=200, design=4096,
            design_repeats=32, sobol=8192, reliability=5000, level=3000)
QUICK = dict(rate=(256, 1024, 4096), repeats=60, design=1024,
             design_repeats=12, sobol=2048, reliability=2000, level=1500)

STANDARD_NORMAL = JointIndependent([Normal(0.0, 1.0)])
SQUARE = lambda x: x[:, 0] ** 2                 # symmetric about the mean
MONOTONE = lambda x: np.exp(0.7 * x[:, 0])      # monotone in the input


# ---------------------------------------------------------------------------

def the_rate(size):
    print("\n" + "=" * 76)
    print("  1. The rate, measured")
    print("=" * 76)
    print("\n  E[X^2] for X ~ N(0,1); the exact answer is 1.")
    print(f"\n    {'N':>8}{'rmse':>12}{'x reduction':>14}")

    errors = []
    for n in size["rate"]:
        rmse = float(np.sqrt(np.mean([
            (propagate(SQUARE, STANDARD_NORMAL, n, "crude", seed=s,
                       vectorised=True).mean() - 1.0) ** 2
            for s in range(size["repeats"])])))
        ratio = "" if not errors else f"{errors[-1] / rmse:14.2f}"
        errors.append(rmse)
        print(f"    {n:>8}{rmse:12.3e}{ratio}")

    slope = float(np.polyfit(np.log(size["rate"]), np.log(errors), 1)[0])
    print(f"\n    fitted slope {slope:.4f}   (theory -0.5)")
    print("    Each 4x in samples buys 2x in accuracy. No estimator does")
    print("    better; only a smaller variance does.")
    return size["rate"], errors


def dimension_independence(size):
    print("\n  The same rate in 1 and in 20 dimensions")
    print(f"\n    {'dim':>5}{'rmse':>12}")
    for dim in (1, 5, 20):
        joint = JointIndependent([Uniform(0.0, 1.0) for _ in range(dim)])
        model = lambda x: np.sum(x, axis=1) / dim
        rmse = float(np.sqrt(np.mean([
            (propagate(model, joint, 2048, "crude", seed=s,
                       vectorised=True).mean() - 0.5) ** 2
            for s in range(size["repeats"] // 2)])))
        print(f"    {dim:>5}{rmse:12.3e}")
    print("\n    A tensor-product quadrature would need m^20 points here.")


def designs(size):
    print("\n" + "=" * 76)
    print("  2. Where the points go")
    print("=" * 76)

    joint = JointIndependent([Uniform(0.0, 1.0) for _ in range(4)])
    model = lambda x: np.sum(x, axis=1) ** 2
    exact = 4 * (1.0 / 3.0) + 12 * 0.25
    n = size["design"]

    print(f"\n  Integral of (sum x_i)^2 over the unit 4-cube, exact {exact:g}")
    print(f"  {n} points, {size['design_repeats']} repetitions\n")
    print(f"    {'design':<13}{'rmse':>12}{'vs crude':>12}")

    rmse = {}
    for design in DESIGNS:
        means = [propagate(model, joint, n, design, seed=s,
                           vectorised=True).mean()
                 for s in range(size["design_repeats"])]
        rmse[design] = float(np.sqrt(np.mean((np.array(means) - exact) ** 2)))
    for design in DESIGNS:
        gain = rmse["crude"] / rmse[design]
        print(f"    {design:<13}{rmse[design]:12.3e}{gain:11.1f}x")

    print("\n  Antithetic is the one that depends on the model")
    print(f"\n    {'model':<24}{'crude':>12}{'antithetic':>13}{'ratio':>9}")
    for label, f in (("exp(0.7x)  monotone", MONOTONE),
                     ("x^2        symmetric", SQUARE)):
        variances = {}
        for design in ("crude", "antithetic"):
            means = [propagate(f, STANDARD_NORMAL, 2000, design, seed=s,
                               vectorised=True).mean()
                     for s in range(size["repeats"])]
            variances[design] = float(np.var(means))
        ratio = variances["antithetic"] / variances["crude"]
        print(f"    {label:<24}{variances['crude']:12.3e}"
              f"{variances['antithetic']:13.3e}{ratio:8.2f}x")
    print("\n    Free, and not safe. Stratification is the one that cannot")
    print("    backfire, because it removes the between-strata variance and")
    print("    touches nothing else.")

    print("\n  A quasi-random sample has no standard error unless replicated")
    plain = propagate(model, joint, 512, "sobol", seed=1, vectorised=True)
    rep = propagate(model, joint, 512, "sobol", n_replications=16, seed=1,
                    vectorised=True)
    print(f"\n    unreplicated : std_error = {plain.std_error}")
    print(f"    16 scrambles : std_error = {rep.std_error:.3e}   "
          f"({rep.n_evaluations} evaluations)")
    return rmse


def control_variate_demo(size):
    print("\n  Control variates: a cheap correlated model with a known mean")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000)
    result = control_variates(np.exp(x), 1.0 + x + 0.5 * x ** 2,
                              control_mean=1.5)

    print(f"\n    exact E[exp(X)]  {np.exp(0.5):.6f}")
    print(f"    crude            {result.crude_mean:.6f} +- "
          f"{result.crude_std_error:.6f}")
    print(f"    with control     {result.mean:.6f} +- {result.std_error:.6f}")
    print(f"    correlation      {result.correlation:.4f}")
    print(f"    variance ratio   {result.variance_ratio:.4f}  "
          f"(theory 1 - rho^2 = {result.theoretical_ratio:.4f})")
    print("\n    Unbiased for any control, so a surrogate that turns out to be")
    print("    poor costs the speed-up and never the answer. A polynomial")
    print("    chaos expansion is the natural one: its mean is the first")
    print("    coefficient, exactly and for free.")


def reliability(size):
    print("\n" + "=" * 76)
    print("  3. Rare events")
    print("=" * 76)
    print("\n  Linear limit state, 4 Gaussian inputs. Pf = Phi(-beta) exactly.")

    rows = []
    for beta in (2.0, 3.0, 4.0, 4.75):
        limit = LinearLimitState.with_beta(beta, dim=4)
        exact = limit.probability
        needed = required_samples(exact)
        print(f"\n  beta = {beta}:  exact Pf = {exact:.4e}")
        print(f"    crude Monte Carlo would need {needed:,} samples for a "
              f"10% CoV")
        print(f"\n    {'method':<22}{'Pf':>12}{'error':>9}{'CoV':>8}"
              f"{'evaluations':>13}")

        entry = {"beta": beta, "exact": exact, "needed": needed}

        f_result = form(limit.evaluate, limit.distribution, vectorised=True)
        print(f"    {'FORM':<22}{f_result.probability:12.4e}"
              f"{abs(f_result.probability / exact - 1) * 100:8.2f}%"
              f"{'-':>8}{f_result.n_evaluations:13d}")
        entry["form"] = f_result

        is_result = importance_sampling_probability(
            limit.evaluate, limit.distribution, size["reliability"], seed=1,
            vectorised=True)
        print(f"    {'importance sampling':<22}{is_result.probability:12.4e}"
              f"{abs(is_result.probability / exact - 1) * 100:8.2f}%"
              f"{is_result.cov:8.3f}{is_result.n_evaluations:13d}")
        entry["is"] = is_result

        ss = subset_simulation(limit.evaluate, limit.distribution,
                               n_per_level=size["level"], seed=1,
                               vectorised=True)
        print(f"    {'subset simulation':<22}{ss.probability:12.4e}"
              f"{abs(ss.probability / exact - 1) * 100:8.2f}%"
              f"{ss.cov:8.3f}{ss.n_evaluations:13d}"
              f"   {ss.info['n_levels']} levels")
        entry["subset"] = ss

        if needed < 400000:
            mc = monte_carlo_probability(limit.evaluate, limit.distribution,
                                         200000, seed=1, vectorised=True)
            print(f"    {'crude (200k)':<22}{mc.probability:12.4e}"
                  f"{abs(mc.probability / exact - 1) * 100:8.2f}%"
                  f"{mc.cov:8.3f}{mc.n_evaluations:13d}")
            entry["crude"] = mc
        else:
            starved = monte_carlo_probability(limit.evaluate,
                                              limit.distribution, 20000,
                                              seed=1, vectorised=True)
            print(f"    {'crude (20k, starved)':<22}"
                  f"{starved.probability:12.4e}  -- "
                  f"{starved.info.get('note', '')[:44]}")
        rows.append(entry)

    print("\n  FORM is exact here because the failure surface is a plane in")
    print("  standard-normal space, which is precisely what it linearises.")
    print("  That makes this a test of the implementation, not of the method:")
    print("  on a curved limit state it is wrong, in a direction set by the")
    print("  curvature and not reported anywhere in its output.")

    print("\n  Subset simulation's coefficient of variation is optimistic,")
    print("  and the table shows by how much. Its levels are not independent")
    print("  -- each is seeded from the last -- but the standard estimate")
    print("  assumes they are:")
    print(f"\n    {'beta':>6}{'reported CoV':>14}{'actual error':>14}"
          f"{'ratio':>8}")
    for row in rows:
        actual = abs(row["subset"].probability / row["exact"] - 1.0)
        ratio = actual / row["subset"].cov if row["subset"].cov else np.nan
        print(f"    {row['beta']:>6}{row['subset'].cov:14.3f}"
              f"{actual:13.1%}{ratio:8.1f}x")
    print("\n  Importance sampling's CoV, which needs no such assumption,")
    print("  tracks its actual error closely. Use subset simulation when the")
    print("  failure region's shape is unknown, and do not read its error")
    print("  estimate as a bound.")
    return rows


def sensitivity(size):
    print("\n" + "=" * 76)
    print("  4. Which inputs matter")
    print("=" * 76)

    ishigami = Ishigami()
    result = sobol_indices(ishigami.evaluate, ishigami.distribution,
                           n_base=size["sobol"], seed=1, vectorised=True,
                           n_bootstrap=200)

    print(f"\n  Ishigami function, {result.n_evaluations} model evaluations")
    print(f"\n    {'input':<8}{'S_i':>9}{'exact':>9}{'S_Ti':>9}{'exact':>9}"
          f"{'interaction':>14}")
    for i, name in enumerate(("x1", "x2", "x3")):
        print(f"    {name:<8}{result.first_order[i]:9.4f}"
              f"{ishigami.first_order[i]:9.4f}{result.total[i]:9.4f}"
              f"{ishigami.total[i]:9.4f}{result.interaction[i]:14.4f}")
    print(f"    {'sum':<8}{result.additive_fraction:9.4f}"
          f"{np.sum(ishigami.first_order):9.4f}"
          f"{np.sum(result.total):9.4f}{np.sum(ishigami.total):9.4f}")

    print(f"\n    x3 has a first-order index of "
          f"{result.first_order[2]:.4f} against an exact 0, and a total index")
    print(f"    of {result.total[2]:.4f}. It has no effect on its own and "
          f"carries a quarter")
    print("    of the output variance through its interaction with x1. Only a")
    print("    small *total* index means an input can be fixed and forgotten.")
    print(f"\n    The first-order indices sum to "
          f"{result.additive_fraction:.4f}, not 1. The shortfall is the")
    print("    interaction content -- a measurement, not an error.")
    return result


def a_worked_input(size):
    """The lognormal parameterisation, since it is where real studies go wrong."""
    print("\n" + "=" * 76)
    print("  Aside: specifying a lognormal input")
    print("=" * 76)
    print("\n    A stiffness with a 30% coefficient of variation, normalised:")
    correct = LogNormal.from_moments(1.0, 0.3)
    naive = LogNormal(1.0, 0.3)
    print(f"      from_moments(1.0, 0.3) -> mean {correct.mean:.4f}, "
          f"std {correct.std:.4f}")
    print(f"      LogNormal(1.0, 0.3)    -> mean {naive.mean:.4f}, "
          f"std {naive.std:.4f}")
    print("\n    That is the dangerous case: 2.84 instead of 1.00 is wrong by")
    print("    a factor of three and still looks like a number. In physical")
    print("    units the same mistake is merely obvious --")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # deliberately absurd
        absurd = LogNormal(210.0e3, 21.0e3).mean
    print(f"      LogNormal(210e3, 21e3) -> mean {absurd:.3g}")
    print("\n    The constructor takes the parameters of the *underlying*")
    print("    normal, which is the mathematically natural choice and never")
    print("    what a material datasheet gives you.")


# ---------------------------------------------------------------------------

def plot(rate, designs_rmse, reliability_rows, sobol_result, size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    counts, errors = rate
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))

    ax[0, 0].loglog(counts, errors, "o-", lw=1.4, ms=5, label="measured rmse")
    reference = errors[0] * (np.array(counts) / counts[0]) ** -0.5
    ax[0, 0].loglog(counts, reference, "k--", lw=1.0, label=r"$N^{-1/2}$")
    ax[0, 0].set(xlabel="samples $N$", ylabel="rmse of the mean",
                 title="The rate is fixed")
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[0, 0].grid(alpha=0.3, which="both")

    names = list(designs_rmse)
    ax[0, 1].bar(names, [designs_rmse[k] for k in names],
                 color=["C7"] + ["C0"] * (len(names) - 1))
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(ylabel="rmse", title="Same budget, different designs")
    ax[0, 1].tick_params(axis="x", labelsize=8)
    ax[0, 1].grid(alpha=0.3, axis="y", which="both")

    betas = [r["beta"] for r in reliability_rows]
    ax[1, 0].semilogy(betas, [r["needed"] for r in reliability_rows], "s--",
                      c="crimson", lw=1.3, label="crude, for 10% CoV")
    ax[1, 0].semilogy(betas, [r["is"].n_evaluations for r in reliability_rows],
                      "o-", lw=1.3, label="importance sampling")
    ax[1, 0].semilogy(betas,
                      [r["subset"].n_evaluations for r in reliability_rows],
                      "^-", lw=1.3, label="subset simulation")
    ax[1, 0].set(xlabel=r"reliability index $\beta$",
                 ylabel="model evaluations",
                 title="Cost of a failure probability")
    ax[1, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].grid(alpha=0.3, which="both")

    ishigami = Ishigami()
    x = np.arange(3)
    ax[1, 1].bar(x - 0.2, sobol_result.first_order, 0.4, label="$S_i$ measured")
    ax[1, 1].bar(x + 0.2, sobol_result.total, 0.4, label="$S_{Ti}$ measured")
    ax[1, 1].plot(x - 0.2, ishigami.first_order, "k_", ms=18, label="exact")
    ax[1, 1].plot(x + 0.2, ishigami.total, "k_", ms=18)
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(["x1", "x2", "x3"])
    ax[1, 1].set(ylabel="index",
                 title="Ishigami: $x_3$ acts only through an interaction")
    ax[1, 1].legend(fontsize=8, frameon=False)
    ax[1, 1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "forward_propagation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    size = QUICK if "--quick" in sys.argv else FULL

    print("Forward uncertainty quantification")

    rate = the_rate(size)
    dimension_independence(size)
    designs_rmse = designs(size)
    control_variate_demo(size)
    reliability_rows = reliability(size)
    sobol_result = sensitivity(size)
    a_worked_input(size)
    plot(rate, designs_rmse, reliability_rows, sobol_result, size)
