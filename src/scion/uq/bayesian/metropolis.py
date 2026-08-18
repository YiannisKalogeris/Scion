r"""
Random-walk Metropolis-Hastings, and its adaptive variant.

The baseline against which every other sampler has to justify its complexity.
Propose a perturbation, accept it with probability ``min(1, p_new / p_old)``,
and the chain's stationary distribution is the target -- with no gradient, no
tuning of anything but one step size, and no assumption about the target beyond
being able to evaluate it up to a constant.

Two implementation details that are not optional
------------------------------------------------
**Compare logs, never ratios.** The test is ``log u < log p_new - log p_old``
with ``u`` uniform, not ``u < min(1, exp(...))``. For any real likelihood the
exponential overflows to ``inf`` or underflows to ``0`` long before the
comparison is meaningless, and the failure is silent -- the chain keeps running
and returns samples from something that is not the posterior.

**Adaptation has to stop.** A chain whose proposal depends on its own history
is not Markov, and the convergence proof goes with it. Haario's adaptive
Metropolis is rescued by *diminishing* adaptation, but the cheap and certain fix
is to freeze the proposal at the end of burn-in and discard everything before
it, which is what :class:`AdaptiveMetropolis` does. An implementation that keeps
adapting to the last sample forever is sampling from no fixed distribution at
all.

The step size
-------------
The one thing to tune. Too small and the chain accepts everything and moves
nowhere; too large and it rejects everything and moves nowhere. The classical
result (Roberts, Gelman and Gilks) is an optimal acceptance rate of 0.234 as the
dimension grows, and about 0.44 in one dimension. :class:`AdaptiveMetropolis`
drives towards it automatically; for plain :class:`MetropolisHastings` the
acceptance rate on the result is what tells you whether the number you picked
was reasonable.
"""

from __future__ import annotations

import numpy as np

from scion.uq.bayesian.chains import PosteriorSamples
from scion.uq.bayesian.problem import InferenceProblem

__all__ = ["MetropolisHastings", "AdaptiveMetropolis"]


class MetropolisHastings:
    """Random-walk Metropolis with a fixed isotropic or diagonal proposal.

    Parameters
    ----------
    problem : the :class:`~.problem.InferenceProblem`.
    proposal_std : scalar or ``(dim,)`` proposal standard deviation.
    seed : passed to ``numpy.random.default_rng``.
    """

    name = "metropolis-hastings"

    def __init__(self, problem: InferenceProblem, proposal_std=0.5,
                 seed: int | None = None):
        self.problem = problem
        self.proposal_std = np.broadcast_to(
            np.atleast_1d(np.asarray(proposal_std, dtype=float)),
            (problem.dim,)).astype(float).copy()
        if np.any(self.proposal_std <= 0.0):
            raise ValueError("the proposal standard deviation must be positive")
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _initial_state(self, x0):
        if x0 is None:
            return self.problem.sample_prior(1, self.rng)[0]
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        if x0.size != self.problem.dim:
            raise ValueError(f"x0 has {x0.size} entries, expected "
                             f"{self.problem.dim}")
        return x0.copy()

    def sample(self, n_samples: int, x0=None, n_burn: int = 0,
               thin: int = 1) -> PosteriorSamples:
        """Run one chain of ``n_samples`` kept draws.

        ``n_burn`` is discarded *in addition* to ``n_samples``, so the chain
        length is ``n_burn + n_samples * thin`` and the result always holds the
        number of samples asked for. Folding burn-in into the total instead is
        a common convenience and it makes two runs with different burn-in
        incomparable.
        """
        problem = self.problem
        x = self._initial_state(x0)
        log_p = problem.log_posterior(x)
        if not np.isfinite(log_p):
            raise ValueError(
                "the starting point has zero posterior density; for a bounded "
                "prior make sure x0 is inside the box")

        total = n_burn + n_samples * thin
        kept = np.empty((n_samples, problem.dim))
        kept_log_p = np.empty(n_samples)
        n_accept, n_accept_kept = 0, 0

        for i in range(total):
            proposal = x + self.rng.normal(0.0, self.proposal_std)
            log_p_proposal = problem.log_posterior(proposal)

            # log-space test: exp() of the ratio overflows on any real problem
            if np.log(self.rng.random()) < log_p_proposal - log_p:
                x, log_p = proposal, log_p_proposal
                n_accept += 1
                if i >= n_burn:
                    n_accept_kept += 1

            if i >= n_burn and (i - n_burn) % thin == 0:
                index = (i - n_burn) // thin
                if index < n_samples:
                    kept[index] = x
                    kept_log_p[index] = log_p

        return PosteriorSamples(
            samples=kept, method=self.name, log_posterior=kept_log_p,
            acceptance_rate=n_accept / total,
            n_evaluations=problem.n_evaluations, ordered=True,
            info={"proposal_std": self.proposal_std.copy(),
                  "n_burn": n_burn, "thin": thin,
                  "acceptance_after_burn_in":
                      n_accept_kept / max(1, total - n_burn)})

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(dim={self.problem.dim}, "
                f"proposal_std={np.round(self.proposal_std, 4)})")


class AdaptiveMetropolis(MetropolisHastings):
    r"""Haario-style adaptive Metropolis, with the adaptation frozen after
    burn-in.

    Learns the proposal covariance from the chain's own history and scales it by
    the optimal factor ``2.38^2 / dim``, which makes the sampler affine
    invariant in practice: a posterior with a strong correlation between two
    parameters, or with components differing by orders of magnitude in scale,
    defeats an isotropic proposal completely and is handled here without
    reparameterising anything.

    Two safeguards, both load-bearing:

    * adaptation runs **only during burn-in** and every sample from that phase
      is discarded, so the returned chain is a genuine Markov chain with a fixed
      proposal
    * the empirical covariance is regularised by ``epsilon * I``, without which
      an early run of rejections gives a singular covariance and the Cholesky
      factorisation fails

    Parameters
    ----------
    initial_std : proposal used until the first adaptation.
    adapt_interval : recompute the covariance every this many burn-in steps.
    adapt_after : number of burn-in steps to collect before adapting at all.
    epsilon : the regularisation added to the diagonal.
    """

    name = "adaptive-metropolis"

    def __init__(self, problem: InferenceProblem, initial_std=0.5,
                 adapt_interval: int = 100, adapt_after: int = 200,
                 epsilon: float = 1e-8, seed: int | None = None):
        super().__init__(problem, initial_std, seed)
        self.adapt_interval = int(adapt_interval)
        self.adapt_after = int(adapt_after)
        self.epsilon = float(epsilon)

    def sample(self, n_samples: int, x0=None, n_burn: int = 1000,
               thin: int = 1) -> PosteriorSamples:
        problem = self.problem
        dim = problem.dim
        x = self._initial_state(x0)
        log_p = problem.log_posterior(x)
        if not np.isfinite(log_p):
            raise ValueError("the starting point has zero posterior density")

        if n_burn <= self.adapt_after:
            raise ValueError(
                f"n_burn={n_burn} leaves no room to adapt after "
                f"adapt_after={self.adapt_after}; the proposal would never be "
                f"learned and this would just be MetropolisHastings")

        scale = 2.38 ** 2 / dim
        chol = np.diag(self.proposal_std)
        history = np.empty((n_burn, dim))

        total = n_burn + n_samples * thin
        kept = np.empty((n_samples, dim))
        kept_log_p = np.empty(n_samples)
        n_accept = 0

        for i in range(total):
            proposal = x + chol @ self.rng.standard_normal(dim)
            log_p_proposal = problem.log_posterior(proposal)

            if np.log(self.rng.random()) < log_p_proposal - log_p:
                x, log_p = proposal, log_p_proposal
                n_accept += 1

            if i < n_burn:
                history[i] = x
                # adapt during burn-in only; a proposal that keeps depending on
                # the chain's own history is not a Markov chain
                if (i >= self.adapt_after
                        and (i + 1) % self.adapt_interval == 0):
                    # from adapt_after onwards, not from the start: the chain
                    # begins at a prior draw, which for a well-identified
                    # posterior is many posterior standard deviations away, and
                    # that approach path is not a sample of anything. Including
                    # it flattens the learned correlation badly -- measured at
                    # -0.39 against a true -0.67 on the module's own 2-D test.
                    window = history[self.adapt_after:i + 1]
                    cov = np.atleast_2d(np.cov(window, rowvar=False))
                    chol = np.linalg.cholesky(
                        scale * (cov + self.epsilon * np.eye(dim)))
            elif (i - n_burn) % thin == 0:
                index = (i - n_burn) // thin
                if index < n_samples:
                    kept[index] = x
                    kept_log_p[index] = log_p

        return PosteriorSamples(
            samples=kept, method=self.name, log_posterior=kept_log_p,
            acceptance_rate=n_accept / total,
            n_evaluations=problem.n_evaluations, ordered=True,
            info={"proposal_cholesky": chol,
                  "learned_cov": (chol @ chol.T) / scale,
                  "n_burn": n_burn, "thin": thin})
