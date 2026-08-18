r"""
Problems whose answers are known, for checking the ones that are not.

Same role as :mod:`simulacrum.uq.bayesian.reference`: every method in this
package is measured against a closed form here rather than against another
method, because two estimators can share a misconception and agree beautifully.

:class:`Ishigami`
    The standard test for variance-based sensitivity, and standard because its
    third input is a trap: ``S_3`` is **exactly zero** while ``S_T3`` is 0.244.
    An implementation that computes only first-order indices concludes that
    ``x_3`` does not matter, and it matters a great deal -- through its
    interaction with ``x_1``. Everything is analytic: total variance,
    first-order and total indices.

:class:`LinearLimitState`
    ``g(x) = a . x + b`` with Gaussian inputs, so the failure surface is a plane
    in standard-normal space and ``P_f = Phi(-beta)`` exactly. FORM is exact
    here by construction -- which is the point: it is the case that separates
    "FORM is implemented correctly" from "FORM happens to be accurate on this
    problem". The reliability index can be dialled to make the probability as
    rare as wanted, which is what turns it into a test of subset simulation.

:class:`QuadraticLimitState`
    A curved surface, where FORM is *not* exact and errs in a direction set by
    the curvature. There is no closed form, so the reference is a very large
    crude Monte Carlo run -- honest about being a reference with error bars
    rather than an exact answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from simulacrum.uq.forward.distributions import (
    JointIndependent,
    Normal,
    Uniform,
)

__all__ = ["Ishigami", "LinearLimitState", "QuadraticLimitState"]


class Ishigami:
    r"""``f = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)``, ``x_i ~ U(-pi, pi)``.

    With the conventional ``a = 7``, ``b = 0.1``:

        V     = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2   = 13.8446
        V_1   = 1/2 (1 + b pi^4/5)^2                   ->  S_1  = 0.3139
        V_2   = a^2/8                                  ->  S_2  = 0.4424
        V_3   = 0                                      ->  S_3  = 0
        V_13  = 8 b^2 pi^8 / 225                       ->  S_T3 = 0.2437

    Non-monotone, non-additive, and with one input that acts purely through an
    interaction. That last property is why it is the benchmark: a first-order
    index of zero and a total index of a quarter is a distinction no additive
    surrogate and no one-at-a-time sensitivity study can make.
    """

    def __init__(self, a: float = 7.0, b: float = 0.1):
        self.a, self.b = float(a), float(b)

    @property
    def distribution(self) -> JointIndependent:
        return JointIndependent([Uniform(-np.pi, np.pi) for _ in range(3)],
                                names=["x1", "x2", "x3"])

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = (np.sin(x[:, 0]) + self.a * np.sin(x[:, 1]) ** 2
               + self.b * x[:, 2] ** 4 * np.sin(x[:, 0]))
        return out if out.size > 1 else float(out[0])

    def evaluate(self, x) -> np.ndarray:
        """Vectorised form, always returning an array."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (np.sin(x[:, 0]) + self.a * np.sin(x[:, 1]) ** 2
                + self.b * x[:, 2] ** 4 * np.sin(x[:, 0]))

    # -- the closed forms ------------------------------------------------
    @property
    def mean(self) -> float:
        """``a/2``: the only term with a nonzero mean over the symmetric
        domain."""
        return self.a / 2.0

    @property
    def variance(self) -> float:
        a, b = self.a, self.b
        return (a ** 2 / 8.0 + b * np.pi ** 4 / 5.0
                + b ** 2 * np.pi ** 8 / 18.0 + 0.5)

    @property
    def first_order(self) -> np.ndarray:
        b = self.b
        v1 = 0.5 * (1.0 + b * np.pi ** 4 / 5.0) ** 2
        v2 = self.a ** 2 / 8.0
        return np.array([v1, v2, 0.0]) / self.variance

    @property
    def total(self) -> np.ndarray:
        v13 = 8.0 * self.b ** 2 * np.pi ** 8 / 225.0
        first = self.first_order * self.variance
        return np.array([first[0] + v13, first[1], v13]) / self.variance

    def __repr__(self) -> str:
        return f"Ishigami(a={self.a:g}, b={self.b:g})"


@dataclass
class LinearLimitState:
    r"""``g(x) = threshold - a . x`` with independent Gaussian inputs.

    Failure is ``g <= 0``. Because the inputs are Gaussian and ``g`` is linear,
    ``a . X`` is Gaussian too, so

        beta = (threshold - a . mu) / sqrt(sum (a_i sigma_i)^2),
        P_f  = Phi(-beta),

    exactly. Standard-normal space maps the failure surface to a plane at
    distance ``beta`` from the origin, which is precisely the situation FORM
    linearises -- so FORM is not approximate here, and any discrepancy is a bug
    rather than a modelling error.

    ``beta`` is the dial. Six inputs at ``beta = 4.75`` gives ``P_f = 1e-6``,
    which crude Monte Carlo cannot reach and subset simulation can.
    """

    coefficients: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    threshold: float

    @classmethod
    def with_beta(cls, beta: float, dim: int = 2) -> "LinearLimitState":
        """Unit coefficients on standard normals, scaled to hit ``beta``."""
        coefficients = np.ones(dim) / np.sqrt(dim)
        return cls(coefficients=coefficients, means=np.zeros(dim),
                   stds=np.ones(dim), threshold=float(beta))

    def __post_init__(self):
        self.coefficients = np.atleast_1d(np.asarray(self.coefficients,
                                                     dtype=float))
        self.means = np.atleast_1d(np.asarray(self.means, dtype=float))
        self.stds = np.atleast_1d(np.asarray(self.stds, dtype=float))
        self.threshold = float(self.threshold)

    @property
    def dim(self) -> int:
        return int(self.coefficients.size)

    @property
    def distribution(self) -> JointIndependent:
        return JointIndependent([Normal(m, s)
                                 for m, s in zip(self.means, self.stds)])

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = self.threshold - x @ self.coefficients
        return out if out.size > 1 else float(out[0])

    def evaluate(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.threshold - x @ self.coefficients

    @property
    def beta(self) -> float:
        spread = np.sqrt(np.sum((self.coefficients * self.stds) ** 2))
        return float((self.threshold - self.coefficients @ self.means) / spread)

    @property
    def probability(self) -> float:
        return float(stats.norm.cdf(-self.beta))

    @property
    def design_point(self) -> np.ndarray:
        """The exact most probable failure point, in input space.

        In ``u``-space it is ``beta * alpha`` with ``alpha`` the unit normal to
        the plane; mapping back is just the affine Gaussian transform.
        """
        weights = self.coefficients * self.stds
        alpha = weights / np.linalg.norm(weights)
        return self.means + self.stds * (self.beta * alpha)

    def __repr__(self) -> str:
        return (f"LinearLimitState(dim={self.dim}, beta={self.beta:.4f}, "
                f"Pf={self.probability:.4g})")


@dataclass
class QuadraticLimitState:
    r"""``g(u) = threshold - u_1 + kappa * sum_{i>1} u_i^2`` on standard
    normals.

    Der Kiureghian's parabolic limit state. The curvature is in the directions
    *orthogonal* to ``u_1`` only, which is what keeps the failure surface real:
    putting ``u_1`` into the quadratic term as well closes the surface off
    entirely for any ``kappa`` above ``1/(4 * threshold)``, and then ``g > 0``
    everywhere and the failure probability is exactly zero rather than small.
    At ``threshold = 3`` and ``kappa = 0.15`` that mistake gives ``min g =
    1.33``, and any sampler run on it correctly reports no failures.

    Included so that FORM has something to be wrong about. Agreement with
    :class:`LinearLimitState` shows only that FORM is implemented correctly; the
    design point here still sits at ``u = (threshold, 0, ...)`` with
    ``beta = threshold``, but the true surface curves away from the origin, so
    the tangent plane overestimates the failure region and FORM overestimates
    ``P_f``. Positive ``kappa`` bends it away, negative bends it towards.

    No closed form. Compare against a large crude Monte Carlo run and treat that
    reference as having its own error bars.
    """

    threshold: float = 3.0
    kappa: float = 0.1
    dim: int = 2

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError("the curvature lives in the directions orthogonal "
                             "to u_1, so at least two dimensions are needed")

    @property
    def distribution(self) -> JointIndependent:
        return JointIndependent([Normal(0.0, 1.0) for _ in range(self.dim)])

    @property
    def form_beta(self) -> float:
        """The reliability index FORM will find: the surface's closest approach
        is still on the ``u_1`` axis."""
        return float(self.threshold)

    def __call__(self, x) -> np.ndarray:
        out = self.evaluate(x)
        return out if out.size > 1 else float(out[0])

    def evaluate(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (self.threshold - x[:, 0]
                + self.kappa * np.sum(x[:, 1:] ** 2, axis=1))

    def __repr__(self) -> str:
        return (f"QuadraticLimitState(threshold={self.threshold:g}, "
                f"kappa={self.kappa:g}, dim={self.dim})")
