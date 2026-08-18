r"""
Transitional Markov Chain Monte Carlo (Ching and Chen, 2007).

The problem TMCMC solves is that the prior and the posterior can be almost
disjoint. For an FE model calibrated against good data the posterior occupies a
tiny fraction of the prior volume, and a Metropolis chain started from the prior
spends its whole budget walking towards that region -- if it finds it at all.

TMCMC replaces the single hard transition with a sequence of easy ones. It
carries a *population* of particles through the intermediate distributions

    p_k(theta)  proportional to  prior(theta) * likelihood(theta)^beta_k,
    0 = beta_0 < beta_1 < ... < beta_m = 1

Each stage reweights the population by the incremental likelihood, resamples
according to those weights, and then moves each survivor with a short Metropolis
chain. Stage 0 is the prior, which is easy to sample; stage ``m`` is the
posterior, which is not; and every step between them is a small one.

Three things this buys over a single chain
------------------------------------------
**No burn-in and no convergence question in the usual sense.** The population
starts distributed exactly according to ``p_0`` and each stage preserves that
property, so there is no transient to discard.

**Multimodality.** The population can hold several modes at once. A single
chain has to travel between them through a low-density region and typically
does not.

**The evidence, for free.** ``log p(data)`` is the sum of the log mean
incremental weights over the stages, and it comes out of quantities the method
already computed. This is the main reason to prefer TMCMC when models are being
compared, since Metropolis-Hastings and HMC work with an unnormalised posterior
and never see the constant at all.

The adaptive schedule
---------------------
The default. Choosing ``beta_k`` by hand is the method's one real difficulty:
too large a step and the weights concentrate on a handful of particles and the
population collapses; too small and stages are wasted. Ching and Chen pick each
increment so that the coefficient of variation of the incremental weights hits a
target, conventionally 1.0, found here by bisection. A fixed linear schedule is
available for teaching and for reproducing an old run, but it is the wrong
default: the same schedule that is fine for one dataset degenerates on another,
and nothing warns you.

Overflow
--------
Every weight is formed in log space and shifted by its maximum before
exponentiating. Writing ``np.exp(d_beta * log_L)`` directly, as is common, is a
guaranteed overflow for any likelihood with more than a handful of observations
-- log-likelihoods of -1e3 are ordinary, and ``exp`` of that is zero, so every
weight becomes zero and the resampling step divides by zero. The shift is
algebraically invisible because the weights are normalised immediately after,
which is exactly why the bug survives casual testing on toy problems.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from scion.uq.bayesian.chains import PosteriorSamples
from scion.uq.bayesian.problem import InferenceProblem

__all__ = ["TMCMC"]


class TMCMC:
    """Transitional MCMC with an adaptive or a prescribed tempering schedule.

    Parameters
    ----------
    problem : the :class:`~.problem.InferenceProblem`. The prior and likelihood
        must be separable -- that is the whole point, see
        :mod:`~.problem`.
    n_particles : population size, held constant across stages.
    schedule : ``"adaptive"`` (default) or an increasing sequence ending at 1.0.
    target_cv : target coefficient of variation of the incremental weights for
        the adaptive schedule. 1.0 is Ching and Chen's recommendation; smaller
        gives more, gentler stages.
    proposal_scale : the ``beta`` of the paper, scaling the proposal covariance
        against the weighted population covariance. 0.2 is the recommended
        value and gives acceptance rates in the useful range.
    n_mutation : Metropolis steps applied to each particle per stage. One is
        the classical choice; more decorrelates the duplicates that resampling
        creates, at proportional cost.
    max_stages : a guard, not a tuning parameter.
    """

    name = "tmcmc"

    def __init__(self, problem: InferenceProblem, n_particles: int = 1000,
                 schedule="adaptive", target_cv: float = 1.0,
                 proposal_scale: float = 0.2, n_mutation: int = 1,
                 max_stages: int = 100, seed: int | None = None):
        self.problem = problem
        self.n_particles = int(n_particles)
        if self.n_particles < 2:
            raise ValueError("TMCMC needs a population, not a single particle")

        if isinstance(schedule, str):
            if schedule != "adaptive":
                raise ValueError("schedule must be 'adaptive' or a sequence "
                                 "of beta values ending at 1.0")
            self.schedule = None
        else:
            betas = np.atleast_1d(np.asarray(schedule, dtype=float))
            if betas[0] == 0.0:
                betas = betas[1:]          # stage 0 is the prior, implicitly
            if np.any(np.diff(betas) <= 0.0) or not np.isclose(betas[-1], 1.0):
                raise ValueError("a prescribed schedule must increase and end "
                                 "at exactly 1.0")
            self.schedule = betas

        self.target_cv = float(target_cv)
        self.proposal_scale = float(proposal_scale)
        self.n_mutation = int(n_mutation)
        self.max_stages = int(max_stages)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _next_beta(self, beta: float, log_like: np.ndarray) -> float:
        """Largest increment whose weights keep a coefficient of variation at
        or below the target, by bisection on ``beta``.

        The CV is monotone in the increment -- a bigger jump always concentrates
        the weights further -- which is what makes bisection valid here.
        """
        def cv(candidate: float) -> float:
            log_w = (candidate - beta) * log_like
            log_w -= log_w.max()
            w = np.exp(log_w)
            mean = w.mean()
            if mean <= 0.0:
                return np.inf
            return float(w.std() / mean)

        if cv(1.0) <= self.target_cv:
            return 1.0

        low, high = beta, 1.0
        for _ in range(60):
            mid = 0.5 * (low + high)
            if cv(mid) > self.target_cv:
                high = mid
            else:
                low = mid
        # a guard against a stalled schedule on a pathological likelihood
        return max(low, beta + 1e-6)

    # ------------------------------------------------------------------
    def sample(self, n_samples: int | None = None) -> PosteriorSamples:
        """Run to ``beta = 1`` and return the final population.

        ``n_samples`` is accepted for interface symmetry with the chain
        samplers and must equal the population size if given -- TMCMC's output
        size is the population, and there is no way to ask it for more samples
        except by enlarging the population and rerunning.
        """
        problem = self.problem
        n = self.n_particles
        if n_samples is not None and int(n_samples) != n:
            raise ValueError(
                f"TMCMC returns its population, so n_samples must be "
                f"{n} (the population size) or be left out")

        particles = problem.sample_prior(n, self.rng)
        log_like = problem.log_likelihood_batch(particles)

        beta = 0.0
        log_evidence = 0.0
        betas, acceptances, unique = [0.0], [], []

        stage = 0
        while beta < 1.0:
            stage += 1
            if stage > self.max_stages:
                raise RuntimeError(
                    f"the tempering schedule stalled at beta = {beta:.6f} "
                    f"after {self.max_stages} stages; the likelihood is "
                    f"probably too sharp for this population size")

            beta_new = (self._next_beta(beta, log_like) if self.schedule is None
                        else float(self.schedule[min(stage - 1,
                                                     len(self.schedule) - 1)]))
            d_beta = beta_new - beta

            # log mean incremental weight: this stage's contribution to log Z.
            # logsumexp rather than exp-then-mean, for the reason in the module
            # docstring.
            log_w = d_beta * log_like
            log_evidence += float(logsumexp(log_w) - np.log(n))

            weights = np.exp(log_w - logsumexp(log_w))
            index = self.rng.choice(n, size=n, p=weights)
            particles = particles[index]
            log_like = log_like[index]

            # proposal covariance from the *weighted* population before
            # resampling, scaled by the paper's beta^2
            centred = particles - np.average(particles, axis=0)
            cov = (self.proposal_scale ** 2
                   * (centred.T @ centred) / max(1, n - 1))
            cov = np.atleast_2d(cov) + 1e-12 * np.eye(problem.dim)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # a fully collapsed population has no spread to learn from
                chol = 1e-6 * np.eye(problem.dim)

            particles, log_like, rate = self._mutate(particles, log_like,
                                                     beta_new, chol)
            beta = beta_new
            betas.append(beta)
            acceptances.append(rate)
            unique.append(len(np.unique(particles, axis=0)) / n)

        return PosteriorSamples(
            samples=particles, method=self.name,
            log_posterior=np.array([problem.log_prior(p) for p in particles])
            + log_like,
            acceptance_rate=float(np.mean(acceptances)),
            log_evidence=log_evidence,
            n_evaluations=problem.n_evaluations,
            ordered=False,
            info={"betas": np.array(betas),
                  "n_stages": len(betas) - 1,
                  "stage_acceptance": np.array(acceptances),
                  "stage_unique_fraction": np.array(unique)})

    # ------------------------------------------------------------------
    def _mutate(self, particles, log_like, beta, chol):
        """Short Metropolis chain per particle at the current temperature."""
        problem = self.problem
        n, dim = particles.shape
        log_prior = np.array([problem.log_prior(p) for p in particles])
        accepted = 0

        for _ in range(self.n_mutation):
            steps = self.rng.standard_normal((n, dim)) @ chol.T
            proposals = particles + steps
            u = np.log(self.rng.random(n))

            for i in range(n):
                lp = problem.log_prior(proposals[i])
                if not np.isfinite(lp):
                    continue                       # outside a bounded prior
                ll = problem.log_likelihood(proposals[i])
                current = log_prior[i] + beta * log_like[i]
                candidate = lp + beta * ll
                if u[i] < candidate - current:
                    particles[i] = proposals[i]
                    log_prior[i] = lp
                    log_like[i] = ll
                    accepted += 1

        return particles, log_like, accepted / (n * self.n_mutation)

    def __repr__(self) -> str:
        kind = "adaptive" if self.schedule is None else "fixed"
        return (f"TMCMC({self.n_particles} particles, {kind} schedule, "
                f"dim={self.problem.dim})")
