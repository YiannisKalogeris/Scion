r"""
Hamiltonian Monte Carlo.

Metropolis proposes a random step and hopes. HMC gives the state a random
momentum, treats the negative log-posterior as a potential energy, and follows
the resulting Hamiltonian dynamics for a while before proposing where it lands.
Because the trajectory follows the geometry of the target instead of ignoring
it, the proposal can travel a long way and still be accepted, and the chain's
autocorrelation collapses -- the practical difference on a correlated posterior
is often two orders of magnitude in effective sample size per model evaluation.

    H(theta, p) = -log posterior(theta) + (1/2) p' M^-1 p

Two properties make the accept step work. The dynamics conserve ``H``, so an
exactly integrated trajectory would be accepted with probability one; and the
leapfrog integrator, while not exact, is *symplectic* and *reversible*, so the
error in ``H`` stays bounded rather than drifting, and the Metropolis correction
on ``exp(-Delta H)`` is valid. Reversibility is why the momentum is negated at
the end of the trajectory -- it is invisible in the acceptance probability
because the kinetic energy is even in ``p``, and omitting it breaks detailed
balance anyway.

The cost, stated plainly
------------------------
**HMC needs the gradient of the log-posterior.** With an analytic gradient it is
usually the best sampler here. Without one, :class:`~.problem.InferenceProblem`
falls back to central differences at ``2 * dim`` model evaluations per gradient,
and HMC takes ``n_leapfrog`` gradients per sample -- so a 4-parameter problem at
20 leapfrog steps costs **160 forward solves per sample** against one for
Metropolis-Hastings. On an FE forward model that is not a trade worth making,
and :meth:`HamiltonianMonteCarlo.sample` says so once rather than letting it be
discovered from the runtime.

Step size and trajectory length
-------------------------------
The two tuning parameters, and they interact. Too large a step size and the
leapfrog integration becomes unstable, ``H`` diverges, and everything is
rejected; too small and the trajectory does not go anywhere for the same cost.
Warm-up adapts the step size towards a target acceptance rate of 0.65 -- the
theoretical optimum for HMC, and notably *higher* than Metropolis's 0.234, which
is a common source of confusion when tuning by eye.

Trajectory length is left to the user. Choosing it automatically is what NUTS
does, and NUTS is not reimplemented here; the optional Pyro backend below
provides it for anyone who wants it.
"""

from __future__ import annotations

import numpy as np

from simulacrum.uq.bayesian.chains import PosteriorSamples
from simulacrum.uq.bayesian.problem import InferenceProblem

__all__ = ["HamiltonianMonteCarlo", "pyro_nuts", "PYRO_AVAILABLE"]

try:                                        # pragma: no cover - environment
    import pyro                             # noqa: F401
    import torch                            # noqa: F401
    PYRO_AVAILABLE = True
except ImportError:                         # pragma: no cover - environment
    PYRO_AVAILABLE = False


class HamiltonianMonteCarlo:
    """HMC with a leapfrog integrator and warm-up step-size adaptation.

    Parameters
    ----------
    problem : the :class:`~.problem.InferenceProblem`.
    step_size : initial leapfrog step. Adapted during warm-up unless
        ``adapt_step_size`` is False.
    n_leapfrog : steps per trajectory. The trajectory length is
        ``step_size * n_leapfrog``, and that product is what matters.
    mass : diagonal mass matrix, ``(dim,)``. The natural choice is the inverse
        of the posterior variances, which is why running a cheap Metropolis
        chain first to estimate them is a reasonable warm start.
    target_accept : warm-up target. 0.65 is the HMC optimum, not 0.234.
    jitter : fraction by which to randomise the trajectory length each
        iteration. A fixed length can resonate with a periodic direction of the
        target and return the chain to where it started; a little jitter
        removes the resonance and costs nothing.
    """

    name = "hmc"

    def __init__(self, problem: InferenceProblem, step_size: float = 0.1,
                 n_leapfrog: int = 20, mass=None, target_accept: float = 0.65,
                 adapt_step_size: bool = True, jitter: float = 0.2,
                 seed: int | None = None):
        self.problem = problem
        self.step_size = float(step_size)
        self.n_leapfrog = int(n_leapfrog)
        if self.n_leapfrog < 1:
            raise ValueError("a trajectory needs at least one leapfrog step")
        self.mass = (np.ones(problem.dim) if mass is None
                     else np.broadcast_to(np.atleast_1d(
                         np.asarray(mass, dtype=float)),
                         (problem.dim,)).astype(float).copy())
        if np.any(self.mass <= 0.0):
            raise ValueError("every mass entry must be positive")
        self.target_accept = float(target_accept)
        self.adapt_step_size = bool(adapt_step_size)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _potential(self, theta):
        """``-log posterior``; ``inf`` outside the support."""
        value = self.problem.log_posterior(theta)
        return np.inf if not np.isfinite(value) else -value

    def _grad_potential(self, theta):
        return -self.problem.grad_log_posterior(theta)

    def _kinetic(self, p):
        return 0.5 * float(np.sum(p * p / self.mass))

    def _leapfrog(self, theta, p, step, n_steps):
        """Half-kick, drift, full-kicks, half-kick.

        The half steps at the ends are what make the scheme symplectic and
        second order; a plain Euler update of the same equations is neither and
        the energy error grows without bound.
        """
        theta, p = theta.copy(), p.copy()
        grad = self._grad_potential(theta)
        if not np.all(np.isfinite(grad)):
            return theta, p, False

        p -= 0.5 * step * grad
        for i in range(n_steps):
            theta = theta + step * p / self.mass
            grad = self._grad_potential(theta)
            if not np.all(np.isfinite(grad)):
                return theta, p, False
            p -= step * grad if i < n_steps - 1 else 0.5 * step * grad
        # negate for reversibility; invisible in the acceptance test because
        # the kinetic energy is even in p, but detailed balance needs it
        return theta, -p, True

    # ------------------------------------------------------------------
    def sample(self, n_samples: int, x0=None, n_burn: int = 200,
               warn_on_numeric_gradient: bool = True) -> PosteriorSamples:
        """Run one chain.

        Warm-up (``n_burn`` iterations) adapts the step size and is discarded.
        """
        problem = self.problem
        if warn_on_numeric_gradient and not problem.has_analytic_gradient:
            import warnings
            warnings.warn(
                f"HMC on a problem with no analytic gradient: each of the "
                f"{self.n_leapfrog} leapfrog steps per sample costs "
                f"{2 * problem.dim} extra model evaluations, so this run will "
                f"make about {n_samples * self.n_leapfrog * 2 * problem.dim} "
                f"of them. Supply a jacobian, or use AdaptiveMetropolis.",
                RuntimeWarning, stacklevel=2)

        theta = (problem.sample_prior(1, self.rng)[0] if x0 is None
                 else np.atleast_1d(np.asarray(x0, dtype=float)).copy())
        potential = self._potential(theta)
        if not np.isfinite(potential):
            raise ValueError("the starting point has zero posterior density")

        step = self.step_size
        log_step_bar, h_bar, mu = np.log(step), 0.0, np.log(10.0 * step)

        total = n_burn + n_samples
        kept = np.empty((n_samples, problem.dim))
        kept_log_p = np.empty(n_samples)
        n_accept, n_divergent = 0, 0

        for i in range(total):
            p = self.rng.normal(0.0, np.sqrt(self.mass))
            hamiltonian = potential + self._kinetic(p)

            n_steps = self.n_leapfrog
            if self.jitter > 0.0:
                n_steps = max(1, int(round(
                    n_steps * (1.0 + self.jitter
                               * (2.0 * self.rng.random() - 1.0)))))

            new_theta, new_p, ok = self._leapfrog(theta, p, step, n_steps)
            if ok:
                new_potential = self._potential(new_theta)
                new_hamiltonian = new_potential + self._kinetic(new_p)
                delta = hamiltonian - new_hamiltonian
            else:
                new_potential, delta = np.inf, -np.inf

            # a large energy error means the integrator went unstable, not that
            # the proposal was merely unlikely; counting these separately is
            # what distinguishes a badly tuned step size from a hard posterior.
            # Only after warm-up: dual averaging deliberately overshoots the
            # step size while searching for it, so divergences there are part
            # of the search and say nothing about the returned samples.
            if (not np.isfinite(delta) or delta < -1000.0) and i >= n_burn:
                n_divergent += 1

            accept_prob = float(min(1.0, np.exp(min(0.0, delta))))
            if np.log(self.rng.random()) < delta:
                theta, potential = new_theta, new_potential
                n_accept += 1

            if i < n_burn:
                if self.adapt_step_size:
                    # Nesterov dual averaging (Hoffman and Gelman, 2014)
                    m = i + 1
                    h_bar = ((1.0 - 1.0 / (m + 10.0)) * h_bar
                             + (self.target_accept - accept_prob) / (m + 10.0))
                    log_step = mu - np.sqrt(m) / 0.05 * h_bar
                    weight = m ** -0.75
                    log_step_bar = (weight * log_step
                                    + (1.0 - weight) * log_step_bar)
                    step = float(np.exp(log_step))
                if i == n_burn - 1 and self.adapt_step_size:
                    step = float(np.exp(log_step_bar))   # freeze the average
            else:
                kept[i - n_burn] = theta
                kept_log_p[i - n_burn] = -potential

        return PosteriorSamples(
            samples=kept, method=self.name, log_posterior=kept_log_p,
            acceptance_rate=n_accept / total,
            n_evaluations=problem.n_evaluations, ordered=True,
            info={"step_size": step, "n_leapfrog": self.n_leapfrog,
                  "n_divergent": n_divergent, "n_burn": n_burn,
                  "analytic_gradient": problem.has_analytic_gradient})

    def __repr__(self) -> str:
        return (f"HamiltonianMonteCarlo(dim={self.problem.dim}, "
                f"step_size={self.step_size:g}, n_leapfrog={self.n_leapfrog})")


# ---------------------------------------------------------------------------
# Optional Pyro backend
# ---------------------------------------------------------------------------

def pyro_nuts(model, n_samples: int = 1000, n_burn: int = 500,
              seed: int | None = None, **kwargs) -> PosteriorSamples:
    """NUTS through Pyro, for anyone who wants the trajectory length chosen
    automatically.

    **This does not take an** :class:`~.problem.InferenceProblem`, and the
    reason is not an oversight. Pyro obtains gradients by automatic
    differentiation through a torch graph, and a NumPy log-posterior is opaque
    to it -- there is nothing to differentiate. So the argument here is a *Pyro
    model*, and the price of the automatic tuning is writing the model twice if
    you also want the NumPy samplers to run on it.

    Parameters
    ----------
    model : a Pyro model, taking no arguments and conditioning on its
        observations internally.
    kwargs : passed to ``pyro.infer.NUTS``.

    Raises
    ------
    ImportError
        If pyro and torch are not installed. They are deliberately not
        dependencies of Simulacrum: :class:`HamiltonianMonteCarlo` is the
        supported path and needs nothing beyond NumPy.
    """
    if not PYRO_AVAILABLE:                  # pragma: no cover - environment
        raise ImportError(
            "pyro-ppl and torch are needed for the NUTS backend and are not "
            "dependencies of Simulacrum. Install them explicitly, or use "
            "HamiltonianMonteCarlo, which is pure NumPy.")

    import torch                            # pragma: no cover - environment
    from pyro.infer import MCMC, NUTS       # pragma: no cover - environment

    if seed is not None:                    # pragma: no cover - environment
        torch.manual_seed(seed)

    mcmc = MCMC(NUTS(model, **kwargs), num_samples=n_samples,
                warmup_steps=n_burn)        # pragma: no cover - environment
    mcmc.run()                              # pragma: no cover - environment

    drawn = mcmc.get_samples()              # pragma: no cover - environment
    stacked = np.column_stack([             # pragma: no cover - environment
        np.asarray(v.detach().cpu().numpy()).reshape(n_samples, -1)
        for v in drawn.values()])
    return PosteriorSamples(                # pragma: no cover - environment
        samples=stacked, method="pyro-nuts", ordered=True,
        info={"site_names": list(drawn)})
