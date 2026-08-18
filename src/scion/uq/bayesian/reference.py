r"""
The conjugate linear-Gaussian problem, solved exactly.

Every sampler in this package is checked against this, and it is the only thing
here whose answer owes nothing to sampling. For

    theta ~ N(mu_0, Sigma_0),        y | theta ~ N(G theta, sigma^2 I)

the posterior is Gaussian and closed-form, by completing the square:

    Sigma_n = (Sigma_0^-1 + G' G / sigma^2)^-1
    mu_n    = Sigma_n (Sigma_0^-1 mu_0 + G' y / sigma^2)

and so, importantly, is the **evidence**, since marginalising a Gaussian
likelihood over a Gaussian prior leaves a Gaussian:

    y ~ N(G mu_0, sigma^2 I + G Sigma_0 G')

Why bother when the posterior is known
--------------------------------------
Because a sampler that is subtly wrong still returns plausible samples. A
mistuned proposal, a missing Jacobian factor, a tempering step that quietly
underflows -- none of these produce an obviously broken histogram, and all of
them shift the posterior variance by a few percent. Comparing against a closed
form catches that; comparing two samplers against each other does not, since
they can share a misconception.

The evidence is the sharper test of the two. Posterior *location* is forgiving:
the likelihood dominates and almost any sampler finds the right neighbourhood.
``log Z`` depends on the normalisation of every density in the calculation, so
it fails loudly when a constant has been dropped -- and dropping constants is
exactly what one does when only Metropolis ratios matter.

The scalar case
---------------
With ``G`` a column of ones this is the textbook "estimate a mean from ``n``
noisy measurements" problem: ``1/sigma_n^2 = 1/sigma_0^2 + n/sigma^2``. It is
the one-dimensional example these samplers were originally written against, and
it stays the first check any change has to pass.
"""

from __future__ import annotations

import numpy as np

from scion.uq.bayesian.problem import (
    GaussianLikelihood,
    IndependentNormal,
    InferenceProblem,
)

__all__ = ["ConjugateGaussian"]


class ConjugateGaussian:
    """Linear-Gaussian inference with an analytic posterior and evidence.

    Parameters
    ----------
    prior_mean, prior_std : ``(dim,)`` independent Gaussian prior.
    noise : scalar measurement standard deviation.
    design : ``(n_data, dim)`` observation operator ``G``. Defaults to a column
        of ones, which is the "repeated noisy measurements of a scalar" case
        and requires ``dim == 1``.
    data : the observations. If omitted, use :meth:`generate` to make some.
    """

    def __init__(self, prior_mean, prior_std, noise, design=None, data=None):
        self.prior_mean = np.atleast_1d(np.asarray(prior_mean, dtype=float))
        self.prior_std = np.broadcast_to(
            np.atleast_1d(np.asarray(prior_std, dtype=float)),
            self.prior_mean.shape).astype(float).copy()
        self.noise = float(noise)
        if self.noise <= 0.0:
            raise ValueError("the measurement noise must be positive")

        if design is None:
            if self.dim != 1:
                raise ValueError("a default design matrix only makes sense for "
                                 "a scalar parameter; pass `design` explicitly")
            design = np.ones((len(data) if data is not None else 1, 1))
        self.design = np.atleast_2d(np.asarray(design, dtype=float))
        if self.design.shape[1] != self.dim:
            raise ValueError(f"design has {self.design.shape[1]} columns but "
                             f"the parameter has {self.dim}")
        self.data = (None if data is None
                     else np.atleast_1d(np.asarray(data, dtype=float)))

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.prior_mean.size)

    @property
    def n_data(self) -> int:
        return int(self.design.shape[0])

    @property
    def prior_cov(self) -> np.ndarray:
        return np.diag(self.prior_std ** 2)

    def model(self, theta) -> np.ndarray:
        """``G theta`` -- the forward model, linear by construction."""
        return self.design @ np.atleast_1d(np.asarray(theta, dtype=float))

    def jacobian(self, theta) -> np.ndarray:
        """``G``, independent of theta. Supplying it is what lets HMC run at
        one model evaluation per gradient instead of ``2 * dim``."""
        return self.design

    def generate(self, true_theta, seed: int | None = None
                 ) -> "ConjugateGaussian":
        """Return a copy carrying synthetic data from ``true_theta``."""
        rng = np.random.default_rng(seed)
        true_theta = np.atleast_1d(np.asarray(true_theta, dtype=float))
        data = self.model(true_theta) + rng.normal(0.0, self.noise,
                                                   size=self.n_data)
        clone = ConjugateGaussian(self.prior_mean, self.prior_std, self.noise,
                                  self.design, data)
        clone.true_theta = true_theta
        return clone

    # -- the closed forms ------------------------------------------------
    def _require_data(self):
        if self.data is None:
            raise ValueError("no data; call generate() or pass `data`")
        if self.data.size != self.n_data:
            raise ValueError(f"data has {self.data.size} entries but the "
                             f"design matrix has {self.n_data} rows")

    def posterior_cov(self) -> np.ndarray:
        self._require_data()
        precision = (np.linalg.inv(self.prior_cov)
                     + self.design.T @ self.design / self.noise ** 2)
        return np.linalg.inv(precision)

    def posterior_mean(self) -> np.ndarray:
        self._require_data()
        return self.posterior_cov() @ (
            np.linalg.solve(self.prior_cov, self.prior_mean)
            + self.design.T @ self.data / self.noise ** 2)

    def posterior_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.posterior_cov()))

    def log_evidence(self) -> float:
        """``log p(y)``, exactly.

        The marginal of a linear-Gaussian model is Gaussian with covariance
        ``sigma^2 I + G Sigma_0 G'``, so this needs no integration at all --
        which is what makes it a fair check on TMCMC's estimate.
        """
        self._require_data()
        cov = (self.noise ** 2 * np.eye(self.n_data)
               + self.design @ self.prior_cov @ self.design.T)
        residual = self.data - self.design @ self.prior_mean
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("the marginal covariance is singular")
        return float(-0.5 * (residual @ np.linalg.solve(cov, residual)
                             + logdet + self.n_data * np.log(2.0 * np.pi)))

    def posterior_pdf(self, grid) -> np.ndarray:
        """Exact density on an ``(n, dim)`` set of points."""
        grid = np.atleast_2d(np.asarray(grid, dtype=float))
        cov = self.posterior_cov()
        centred = grid - self.posterior_mean()
        sign, logdet = np.linalg.slogdet(cov)
        quad = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(cov), centred)
        return np.exp(-0.5 * (quad + logdet + self.dim * np.log(2.0 * np.pi)))

    # -- the samplers' view ----------------------------------------------
    def as_problem(self) -> InferenceProblem:
        """The same thing as an :class:`~.problem.InferenceProblem`, with the
        analytic Jacobian attached so HMC is not crippled."""
        self._require_data()
        return InferenceProblem(
            prior=IndependentNormal(self.prior_mean, self.prior_std),
            likelihood=GaussianLikelihood(self.model, self.data, self.noise,
                                          jacobian=self.jacobian))

    def error(self, samples) -> dict:
        """Relative error of a sampler's moments against the closed form."""
        mean, std = self.posterior_mean(), self.posterior_std()
        got_mean = np.atleast_2d(samples.samples).mean(axis=0)
        got_std = np.atleast_2d(samples.samples).std(axis=0, ddof=1)
        out = {
            "mean_error": float(np.max(np.abs(got_mean - mean)
                                       / np.abs(std))),
            "std_error": float(np.max(np.abs(got_std - std) / std)),
        }
        if samples.log_evidence is not None:
            out["log_evidence_error"] = float(
                abs(samples.log_evidence - self.log_evidence()))
        return out

    def __repr__(self) -> str:
        return (f"ConjugateGaussian(dim={self.dim}, n_data={self.n_data}, "
                f"noise={self.noise:g})")
