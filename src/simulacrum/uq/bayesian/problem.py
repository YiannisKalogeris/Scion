r"""
The inference problem: a prior, a likelihood, and the seam to a forward model.

Every sampler in this package consumes an :class:`InferenceProblem` rather than
a bare log-posterior callable, and the reason is structural rather than
stylistic: **tempering methods need the prior and the likelihood separately.**
TMCMC walks a sequence of distributions

    p_k(theta)  proportional to  prior(theta) * likelihood(theta)^beta_k

so a sampler handed only their product has no way to raise one factor to a
power. Metropolis-Hastings would be happy with the sum; TMCMC cannot be written
at all. Keeping the two apart from the start is what makes one interface serve
both.

The forward-model seam
----------------------
:class:`GaussianLikelihood` takes a plain callable ``model(theta) -> prediction``
and knows nothing else about it. That is the whole coupling to the rest of the
library: an analytic function, a surrogate, or a full nonlinear FE solve all
enter the same way, and the samplers stay as physics-free as
:mod:`simulacrum.solvers`.

    problem = InferenceProblem(
        prior=IndependentNormal(mean=[100.0, 100.0], std=[20.0, 20.0]),
        likelihood=GaussianLikelihood(model=run_pushover, data=measured,
                                      noise=0.5))

Counting model evaluations
--------------------------
:attr:`InferenceProblem.n_evaluations` counts calls to the likelihood, because
for an FE forward model that count *is* the cost of the analysis and it is the
only fair way to compare samplers. A method that needs a tenth of the samples
but evaluates the model three times as often is not cheaper. The counter lives
on the problem instance rather than in a module-level global, so two concurrent
studies cannot corrupt each other's tally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

__all__ = [
    "Prior",
    "IndependentNormal",
    "IndependentUniform",
    "Likelihood",
    "GaussianLikelihood",
    "CallableLikelihood",
    "InferenceProblem",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class Prior(ABC):
    """A prior that can be evaluated **and sampled**.

    Sampling is not optional. TMCMC starts from a prior population and every
    sampler here wants a sensible default start, so a prior that can only be
    evaluated is half a prior.
    """

    name: str = "prior"

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def log_pdf(self, theta: np.ndarray) -> float:
        """Log density at a single point, up to a constant."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, dim)`` independent draws."""

    def grad_log_pdf(self, theta: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`log_pdf`. Override where it is available."""
        raise NotImplementedError(
            f"{type(self).__name__} has no analytic gradient; "
            f"InferenceProblem will fall back to finite differences")

    def log_pdf_batch(self, thetas: np.ndarray) -> np.ndarray:
        """``(n, dim) -> (n,)``. Overridden where a vectorised form exists."""
        return np.array([self.log_pdf(t) for t in np.atleast_2d(thetas)])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"


class IndependentNormal(Prior):
    """Independent Gaussians, ``theta_i ~ N(mean_i, std_i^2)``."""

    name = "normal"

    def __init__(self, mean, std):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.std = np.atleast_1d(np.asarray(std, dtype=float))
        if self.std.size == 1 and self.mean.size > 1:
            self.std = np.full(self.mean.shape, float(self.std[0]))
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape")
        if np.any(self.std <= 0.0):
            raise ValueError("every prior standard deviation must be positive")
        self._const = float(-np.sum(np.log(self.std)) - 0.5 * self.dim * _LOG_2PI)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    def log_pdf(self, theta):
        z = (np.asarray(theta, dtype=float) - self.mean) / self.std
        return self._const - 0.5 * float(np.dot(z, z))

    def log_pdf_batch(self, thetas):
        z = (np.atleast_2d(thetas) - self.mean) / self.std
        return self._const - 0.5 * np.sum(z * z, axis=1)

    def grad_log_pdf(self, theta):
        return -(np.asarray(theta, dtype=float) - self.mean) / self.std ** 2

    def sample(self, n, rng):
        return rng.normal(self.mean, self.std, size=(n, self.dim))


class IndependentUniform(Prior):
    """Independent uniforms on ``[lower_i, upper_i]``.

    Returns ``-inf`` outside the box, which is the correct way to express a
    bounded prior and is exactly what a Metropolis acceptance test needs -- the
    proposal is rejected with probability one and no clipping or reflection is
    involved. Passing bounds into a *Gaussian* log-density instead, as
    ``(mean, std)``, silently evaluates a completely different prior; that is a
    real mistake found in the code this module was assembled from.
    """

    name = "uniform"

    def __init__(self, lower, upper):
        self.lower = np.atleast_1d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(upper, dtype=float))
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must exceed its lower bound")
        self._const = float(-np.sum(np.log(self.upper - self.lower)))

    @property
    def dim(self) -> int:
        return int(self.lower.size)

    def log_pdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        inside = bool(np.all(theta >= self.lower) and np.all(theta <= self.upper))
        return self._const if inside else -np.inf

    def log_pdf_batch(self, thetas):
        thetas = np.atleast_2d(thetas)
        inside = np.all((thetas >= self.lower) & (thetas <= self.upper), axis=1)
        return np.where(inside, self._const, -np.inf)

    def grad_log_pdf(self, theta):
        # flat inside the box, so the gradient is zero wherever it is defined.
        # HMC on a boxed prior is a bad idea for a different reason: the
        # trajectory leaves the support and the whole proposal is rejected.
        return np.zeros(self.dim)

    def sample(self, n, rng):
        return rng.uniform(self.lower, self.upper, size=(n, self.dim))


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------

class Likelihood(ABC):
    """``log p(data | theta)``, up to a constant that does not depend on theta."""

    name = "likelihood"

    @abstractmethod
    def log_pdf(self, theta: np.ndarray) -> float:
        ...

    def grad_log_pdf(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} has no analytic gradient; supply a "
            f"jacobian or let InferenceProblem use finite differences")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CallableLikelihood(Likelihood):
    """Wraps a bare ``log_likelihood(theta)`` callable."""

    name = "callable"

    def __init__(self, log_pdf: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray] | None = None):
        self._log_pdf = log_pdf
        self._grad = grad

    def log_pdf(self, theta):
        return float(self._log_pdf(np.asarray(theta, dtype=float)))

    def grad_log_pdf(self, theta):
        if self._grad is None:
            raise NotImplementedError("no gradient was supplied")
        return np.asarray(self._grad(np.asarray(theta, dtype=float)),
                          dtype=float)


class GaussianLikelihood(Likelihood):
    r"""Additive Gaussian measurement noise on a forward model.

        log p(y | theta) = -1/2 sum ((y - model(theta)) / sigma)^2
                           - sum log(sqrt(2 pi) sigma)

    Parameters
    ----------
    model : ``theta -> prediction``, any callable. The single point of contact
        with the physics; an FE solve goes here unchanged.
    data : the observations, broadcastable against the prediction.
    noise : scalar or per-observation standard deviation.
    jacobian : optional ``theta -> (n_data, dim)``. Supplying it is what makes
        HMC worth using -- without it the gradient costs ``2 * dim`` model
        evaluations per leapfrog step, and HMC takes tens of those per sample.

    The normalising constant is retained even though Metropolis ratios cancel
    it, because TMCMC's evidence estimate does not: ``log Z`` is only comparable
    between models if every likelihood is normalised the same way.
    """

    name = "gaussian"

    def __init__(self, model: Callable[[np.ndarray], np.ndarray], data,
                 noise, jacobian=None):
        self.model = model
        self.data = np.atleast_1d(np.asarray(data, dtype=float))
        self.noise = np.broadcast_to(
            np.atleast_1d(np.asarray(noise, dtype=float)),
            self.data.shape).astype(float)
        if np.any(self.noise <= 0.0):
            raise ValueError("the measurement noise must be positive")
        self.jacobian = jacobian
        self._const = float(-np.sum(np.log(self.noise))
                            - 0.5 * self.data.size * _LOG_2PI)

    def residual(self, theta) -> np.ndarray:
        prediction = np.atleast_1d(np.asarray(
            self.model(np.asarray(theta, dtype=float)), dtype=float))
        if prediction.shape != self.data.shape:
            raise ValueError(
                f"the model returned shape {prediction.shape} but the data has "
                f"shape {self.data.shape}")
        return (prediction - self.data) / self.noise

    def log_pdf(self, theta):
        r = self.residual(theta)
        return self._const - 0.5 * float(np.dot(r, r))

    def grad_log_pdf(self, theta):
        if self.jacobian is None:
            raise NotImplementedError("no jacobian was supplied")
        J = np.atleast_2d(np.asarray(
            self.jacobian(np.asarray(theta, dtype=float)), dtype=float))
        return -J.T @ (self.residual(theta) / self.noise)


# ---------------------------------------------------------------------------

class InferenceProblem:
    """Prior + likelihood, kept separable so tempering methods can work.

    Parameters
    ----------
    prior : a :class:`Prior`.
    likelihood : a :class:`Likelihood`, or a bare ``log_likelihood(theta)``
        callable, which is wrapped.
    fd_step : relative step for the finite-difference gradient fallback.
    """

    def __init__(self, prior: Prior, likelihood, fd_step: float = 1e-5):
        if not isinstance(prior, Prior):
            raise TypeError("prior must be a Prior; a bare callable cannot be "
                            "sampled from, and TMCMC needs to sample it")
        self.prior = prior
        self.likelihood = (likelihood if isinstance(likelihood, Likelihood)
                           else CallableLikelihood(likelihood))
        self.fd_step = float(fd_step)
        self.n_evaluations = 0

    @property
    def dim(self) -> int:
        return self.prior.dim

    # -- densities ------------------------------------------------------
    def log_prior(self, theta) -> float:
        return float(self.prior.log_pdf(theta))

    def log_likelihood(self, theta) -> float:
        """Evaluates the forward model, and counts it.

        Short-circuits to ``-inf`` outside the prior support. That is not an
        optimisation: an FE model handed a negative stiffness may not fail
        cleanly, and a bounded prior exists precisely to keep it from being
        asked.
        """
        lp = self.prior.log_pdf(theta)
        if not np.isfinite(lp):
            return -np.inf
        self.n_evaluations += 1
        return float(self.likelihood.log_pdf(theta))

    def log_posterior(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def tempered_log_posterior(self, theta, beta: float) -> float:
        """``log prior + beta * log likelihood``, the TMCMC intermediate."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if beta == 0.0:
            return lp
        return lp + beta * self.log_likelihood(theta)

    # -- gradients ------------------------------------------------------
    @property
    def has_analytic_gradient(self) -> bool:
        probe = self.prior.sample(1, np.random.default_rng(0))[0]
        try:
            self.prior.grad_log_pdf(probe)
            self.likelihood.grad_log_pdf(probe)
        except NotImplementedError:
            return False
        return True

    def grad_log_posterior(self, theta) -> np.ndarray:
        """Analytic where available, central differences otherwise.

        The fallback costs ``2 * dim`` model evaluations. HMC takes tens of
        leapfrog steps per sample, so on an expensive forward model this turns
        the cheapest-looking sampler into by far the most expensive one --
        check :attr:`has_analytic_gradient` before reaching for HMC.
        """
        theta = np.asarray(theta, dtype=float)
        try:
            return (np.asarray(self.prior.grad_log_pdf(theta), dtype=float)
                    + np.asarray(self.likelihood.grad_log_pdf(theta),
                                 dtype=float))
        except NotImplementedError:
            pass

        grad = np.zeros(self.dim)
        for i in range(self.dim):
            h = self.fd_step * max(1.0, abs(theta[i]))
            plus, minus = theta.copy(), theta.copy()
            plus[i] += h
            minus[i] -= h
            grad[i] = (self.log_posterior(plus)
                       - self.log_posterior(minus)) / (2.0 * h)
        return grad

    # -- helpers --------------------------------------------------------
    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.prior.sample(n, rng)

    def log_likelihood_batch(self, thetas) -> np.ndarray:
        """``(n, dim) -> (n,)``. TMCMC evaluates whole populations."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        return np.array([self.log_likelihood(t) for t in thetas])

    def reset_counter(self) -> None:
        self.n_evaluations = 0

    def __repr__(self) -> str:
        return (f"InferenceProblem(dim={self.dim}, prior={self.prior!r}, "
                f"likelihood={self.likelihood!r})")
