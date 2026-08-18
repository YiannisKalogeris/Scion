#!/usr/bin/env python3
r"""
Four posterior samplers against a posterior that is known exactly.

Problem
-------
The conjugate linear-Gaussian problem, in two sizes.

**Scalar.** ``theta ~ N(0, 2^2)``, ten measurements ``y_i ~ N(theta, 0.5^2)``.
The posterior is Gaussian with ``1/sigma_n^2 = 1/sigma_0^2 + n/sigma^2``, so
``sigma_n = 0.157622`` exactly, and the mean is the precision-weighted average
of prior and data.

**Two correlated parameters.** The same algebra with a design matrix whose
columns are nearly parallel, which makes the posterior a narrow tilted ellipse.
Nothing about the closed form gets harder; the *sampling* gets much harder, and
that is the point.

Why bother sampling something you can solve
-------------------------------------------
Because a wrong sampler does not look wrong. A mistuned proposal, an adaptation
that never stops, a tempering step that underflows -- none of these produce a
visibly broken histogram, and all of them move the posterior variance by a few
percent. The only way to see it is to compare against an answer obtained by
different means, and this is the largest problem where that answer exists.

Two things measured here cannot be checked any other way:

**The posterior standard deviation.** Location is forgiving -- the likelihood
dominates and anything that runs finds the right neighbourhood. The width is
where a subtly broken sampler shows.

**The evidence.** ``log p(y)`` is closed-form here too, and TMCMC is the only
one of the four that estimates it. It depends on the normalisation of every
density in the calculation, so it catches the constants that Metropolis ratios
let you drop.

The currency is model evaluations
---------------------------------
Not samples. On an FE forward model the count of solves *is* the cost of the
analysis, and the ranking by samples and the ranking by cost are different.
HMC's whole argument is that it buys more independent information per
evaluation -- true here, and true only because the gradient is analytic. Take
the Jacobian away and each gradient costs ``2 * dim`` extra solves, which turns
the cheapest-looking sampler into much the most expensive one.

Run
---
    python -m examples.uq.posterior_sampling
    python -m examples.uq.posterior_sampling --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simulacrum.uq.bayesian import (
    AdaptiveMetropolis,
    ConjugateGaussian,
    GaussianLikelihood,
    HamiltonianMonteCarlo,
    IndependentNormal,
    InferenceProblem,
    MetropolisHastings,
    TMCMC,
    r_hat,
)

RESULTS = Path(__file__).resolve().parent / "results"

TRUE_THETA = 2.5
FULL = dict(chain=40000, burn=4000, hmc=10000, particles=8000)
QUICK = dict(chain=8000, burn=1000, hmc=2500, particles=2000)


def scalar_reference():
    """Exactly the setup the incoming code was written against."""
    return ConjugateGaussian(prior_mean=[0.0], prior_std=[2.0], noise=0.5,
                             design=np.ones((10, 1))).generate([TRUE_THETA],
                                                               seed=42)


def planar_reference():
    """Two parameters made strongly dependent through the design matrix."""
    design = np.array([[1.0, 0.0], [1.0, 0.98], [1.0, 1.02], [0.0, 1.0]] * 5)
    return ConjugateGaussian([100.0, 100.0], [20.0, 20.0], 1.0,
                             design=design).generate([110.0, 95.0], seed=7)


def build(problem, size, step, seed=1):
    """The four samplers, each tuned only by its one obvious knob."""
    return {
        "MH": (MetropolisHastings(problem, proposal_std=step, seed=seed),
               dict(n_samples=size["chain"], n_burn=size["burn"])),
        "Adaptive MH": (AdaptiveMetropolis(problem, initial_std=step,
                                           seed=seed),
                        dict(n_samples=size["chain"], n_burn=size["burn"])),
        "TMCMC": (TMCMC(problem, n_particles=size["particles"], seed=seed),
                  dict()),
        "HMC": (HamiltonianMonteCarlo(problem, step_size=0.5 * step,
                                      n_leapfrog=15, seed=seed),
                dict(n_samples=size["hmc"], n_burn=500)),
    }


# ---------------------------------------------------------------------------

def study(reference, size, step, title, names):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

    exact_mean = reference.posterior_mean()
    exact_std = reference.posterior_std()
    print(f"\n  Exact posterior (closed form)")
    for i, name in enumerate(names):
        print(f"    {name:<12} mean {exact_mean[i]:11.6f}   "
              f"std {exact_std[i]:10.6f}")
    print(f"    {'log evidence':<12}      {reference.log_evidence():11.6f}")

    problem = reference.as_problem()
    print(f"\n  Analytic gradient available: "
          f"{problem.has_analytic_gradient}  (decides whether HMC is viable)")

    print(f"\n    {'sampler':<13}{'mean err':>10}{'std err':>9}"
          f"{'log Z err':>11}{'evals':>9}{'ESS':>8}{'ESS/1k evals':>14}")

    results = {}
    for label, (sampler, kwargs) in build(problem, size, step).items():
        problem.reset_counter()
        result = sampler.sample(**kwargs)
        error = reference.error(result)
        results[label] = result

        if result.ordered:
            ess = float(result.ess().min())
            ess_text = f"{ess:8.0f}"
            rate_text = f"{1000 * ess / result.n_evaluations:14.1f}"
        else:
            # a resampled population has no lag structure; the honest
            # equivalent is how many particles are still distinct
            ess_text = f"{result.unique_fraction * result.n_samples:8.0f}"
            rate_text = f"{'(particles)':>14}"

        evidence = (f"{'-':>11}" if result.log_evidence is None
                    else f"{error['log_evidence_error']:11.4f}")
        print(f"    {label:<13}{error['mean_error']:10.4f}"
              f"{error['std_error']:9.4f}{evidence}"
              f"{result.n_evaluations:9d}{ess_text}{rate_text}")

    print("\n    mean err is in posterior standard deviations; std err is")
    print("    relative. Both are Monte Carlo noise at these sizes -- rerun")
    print("    with another seed and they move, without settling to one side.")
    return results


def report(scalar_results, planar_results):
    print("\n" + "=" * 78)
    print("  What the table says")
    print("=" * 78)

    tmcmc = planar_results["TMCMC"]
    print(f"\n  TMCMC is the only one that returns an evidence, and it is the")
    print(f"  only quantity here sensitive to a dropped normalising constant.")
    print(f"  It found its own schedule: {tmcmc.info['n_stages']} stages at")
    print(f"  beta = {np.round(tmcmc.info['betas'], 4)}")
    print(f"  A fixed linear schedule of the same length would have spent")
    print(f"  most of its stages where the weights barely change.")

    hmc = planar_results["HMC"]
    walk = planar_results["MH"]
    gain = ((hmc.ess().min() / hmc.n_evaluations)
            / (walk.ess().min() / walk.n_evaluations))
    print(f"\n  On the correlated posterior HMC delivers {gain:.1f}x the")
    print(f"  independent samples per model evaluation that random-walk")
    print(f"  Metropolis does, and that ratio is the entire case for needing")
    print(f"  a gradient. The section below measures what happens without")
    print(f"  one: 2*dim = 4 extra solves per gradient, 15 gradients per")
    print(f"  sample, which is far more than a {gain:.1f}x advantage can absorb.")

    adaptive = planar_results["Adaptive MH"]
    print(f"\n  Adaptive Metropolis accepted {adaptive.acceptance_rate:.3f} "
          f"against {walk.acceptance_rate:.3f} for the fixed proposal,")
    print(f"  having learned the posterior's correlation during burn-in.")
    print(f"  It stops adapting when burn-in ends: a proposal that keeps")
    print(f"  depending on the chain's own history is not a Markov chain.")


def convergence_check(reference, size):
    """Four dispersed chains, and the diagnostic that needs more than one."""
    print("\n" + "=" * 78)
    print("  Convergence, which a single chain cannot establish")
    print("=" * 78)

    problem = reference.as_problem()
    chains = [MetropolisHastings(problem, proposal_std=0.4, seed=s).sample(
        size["chain"] // 4, n_burn=size["burn"]).samples for s in range(4)]

    print(f"\n    R-hat over 4 dispersed chains : "
          f"{float(r_hat(chains)[0]):.5f}   (converged below ~1.01)")

    stuck = [np.random.default_rng(s).normal(loc, 0.1, size=(2000, 1))
             for s, loc in enumerate((0.0, 5.0, 10.0, 15.0))]
    print(f"    R-hat over 4 chains that never met: "
          f"{float(r_hat(stuck)[0]):.5f}")
    print("\n    Each of those four is perfectly converged on its own, and")
    print("    every single-chain diagnostic says so. That is the failure")
    print("    R-hat exists for.")


def cost_of_a_numeric_gradient(reference, size):
    """What HMC costs when the forward model has no Jacobian."""
    print("\n" + "=" * 78)
    print("  HMC without an analytic gradient")
    print("=" * 78)

    blind = InferenceProblem(
        prior=IndependentNormal(reference.prior_mean, reference.prior_std),
        likelihood=GaussianLikelihood(reference.model, reference.data,
                                      reference.noise))          # no jacobian
    sighted = reference.as_problem()

    n = max(400, size["hmc"] // 10)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blind.reset_counter()
        HamiltonianMonteCarlo(blind, step_size=0.1, n_leapfrog=15,
                              seed=1).sample(n, n_burn=100)
        sighted.reset_counter()
        HamiltonianMonteCarlo(sighted, step_size=0.1, n_leapfrog=15,
                              seed=1).sample(n, n_burn=100)

    print(f"\n    {n} samples, {blind.dim} parameters, 15 leapfrog steps")
    print(f"      with an analytic jacobian : "
          f"{sighted.n_evaluations:8d} model evaluations")
    print(f"      by finite differences     : "
          f"{blind.n_evaluations:8d} model evaluations")
    print(f"      ratio                     : "
          f"{blind.n_evaluations / sighted.n_evaluations:8.1f}x")
    print("\n    Same chain, same answer, and on an FE model that factor is")
    print("    the difference between an afternoon and a fortnight. The")
    print("    library warns once rather than letting it be discovered from")
    print("    the runtime.")


# ---------------------------------------------------------------------------

def plot(scalar_ref, scalar_results, planar_ref, planar_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (0,0) scalar posteriors against the exact density
    mean, std = scalar_ref.posterior_mean()[0], scalar_ref.posterior_std()[0]
    grid = np.linspace(mean - 5 * std, mean + 5 * std, 600)
    exact = scalar_ref.posterior_pdf(grid.reshape(-1, 1))
    ax[0, 0].plot(grid, exact, "k-", lw=2.0, label="exact", zorder=5)
    for label, result in scalar_results.items():
        ax[0, 0].hist(result.samples[:, 0], bins=90, density=True,
                      histtype="step", lw=1.2, label=label)
    ax[0, 0].axvline(TRUE_THETA, c="crimson", ls=":", lw=1.2,
                     label=f"true = {TRUE_THETA}")
    ax[0, 0].set(xlabel=r"$\theta$", ylabel="density",
                 title="Scalar posterior, four samplers vs closed form")
    ax[0, 0].legend(fontsize=7, frameon=False)

    # (0,1) the error that matters, on a log scale
    labels = list(scalar_results)
    x = np.arange(len(labels))
    mean_err = [scalar_ref.error(r)["mean_error"] for r in
                scalar_results.values()]
    std_err = [scalar_ref.error(r)["std_error"] for r in
               scalar_results.values()]
    ax[0, 1].bar(x - 0.2, mean_err, 0.4, label="mean (posterior sds)")
    ax[0, 1].bar(x + 0.2, std_err, 0.4, label="std (relative)")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(labels, fontsize=8)
    ax[0, 1].set(ylabel="error", title="Error against the exact posterior")
    ax[0, 1].legend(fontsize=8, frameon=False)
    ax[0, 1].grid(alpha=0.3, axis="y")

    # (1,0) the 2-D posterior: exact contours, samples on top
    pm, ps = planar_ref.posterior_mean(), planar_ref.posterior_std()
    axes = [np.linspace(m - 4 * s, m + 4 * s, 200) for m, s in zip(pm, ps)]
    X, Y = np.meshgrid(*axes, indexing="ij")
    pdf = planar_ref.posterior_pdf(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    sub = planar_results["TMCMC"].samples
    ax[1, 0].plot(sub[:, 0], sub[:, 1], ".", ms=1.2, alpha=0.25, c="C0",
                  label="TMCMC particles")
    ax[1, 0].contour(X, Y, pdf, levels=6, colors="k", linewidths=1.0)
    ax[1, 0].set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
                 title="Correlated posterior: exact contours, sampled points")
    ax[1, 0].legend(fontsize=8, frameon=False)

    # (1,1) efficiency in the currency that matters
    chains = {k: v for k, v in planar_results.items() if v.ordered}
    names = list(chains)
    eff = [1000 * float(v.ess().min()) / v.n_evaluations
           for v in chains.values()]
    ax[1, 1].barh(names, eff, color=["C0", "C1", "C2"][:len(names)])
    ax[1, 1].set(xlabel="independent samples per 1000 model evaluations",
                 title="Cost, measured in forward solves")
    ax[1, 1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "posterior_sampling.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    size = QUICK if "--quick" in sys.argv else FULL

    print("Bayesian inference: four samplers against a closed-form posterior")

    scalar_ref = scalar_reference()
    scalar_results = study(scalar_ref, size, 0.4,
                           "Scalar: theta ~ N(0, 4), ten measurements",
                           ["theta"])

    planar_ref = planar_reference()
    planar_results = study(planar_ref, size, 1.0,
                           "Two correlated parameters, twenty measurements",
                           ["theta_1", "theta_2"])

    report(scalar_results, planar_results)
    convergence_check(scalar_ref, size)
    cost_of_a_numeric_gradient(planar_ref, size)
    plot(scalar_ref, scalar_results, planar_ref, planar_results)
