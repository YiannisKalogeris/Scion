r"""
Input distributions, and the transform every sampling design goes through.

Everything in forward UQ starts by turning points in the unit hypercube into
points in the input space, because that is the one thing crude Monte Carlo,
Latin hypercube and the quasi-random sequences all have in common: they are
designs on ``[0,1]^d``, and the distribution is applied afterwards by the
inverse CDF,

    x_i = F_i^-1(u_i).

So :meth:`RandomVariable.ppf` is the primitive here, not ``sample``. A variable
that can only be sampled directly cannot be used with LHS or Sobol at all, which
is why the interface is built around the quantile function and sampling is the
derived operation.

The standard-normal space
-------------------------
FORM and subset simulation both work in the space where every input is an
independent standard normal, and map back through

    x = F^-1(Phi(z)).

That is the Rosenblatt transform for independent inputs, and
:meth:`JointIndependent.from_standard_normal` is it. The reason to bother:
distance from the origin in that space is a probability, so "the most likely
failure point" becomes "the closest point to the origin", which is a
constrained minimisation rather than a search over an unknown density.

The lognormal trap
------------------
:class:`LogNormal` is parameterised by the mean and standard deviation of the
**underlying normal**, which is the mathematically natural choice and not what
an engineer usually has. If you have the mean and standard deviation of the
quantity itself -- a Young's modulus of 210 GPa with a 10% coefficient of
variation -- use :meth:`LogNormal.from_moments`. Passing those numbers to the
constructor instead is silently accepted and gives a distribution with the wrong
mean and a wildly wrong variance; at a CoV of 0.3 the mean comes out 4.6% high
and the standard deviation 3.2 times too large.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

__all__ = [
    "RandomVariable",
    "Normal",
    "LogNormal",
    "Uniform",
    "Exponential",
    "Gumbel",
    "Beta",
    "JointIndependent",
    "DISTRIBUTION_REGISTRY",
    "get_distribution",
]


class RandomVariable(ABC):
    """A scalar input, defined by its quantile function.

    ``ppf`` is the primitive because every design in this package is generated
    on the unit hypercube and mapped through it. Subclasses that wrap a
    ``scipy.stats`` distribution get everything else for free.
    """

    name: str = "variable"

    @property
    @abstractmethod
    def _frozen(self):
        """The underlying ``scipy.stats`` frozen distribution."""

    def ppf(self, u) -> np.ndarray:
        """Inverse CDF. ``u`` in ``[0, 1]``."""
        return np.asarray(self._frozen.ppf(np.asarray(u, dtype=float)),
                          dtype=float)

    def cdf(self, x) -> np.ndarray:
        return np.asarray(self._frozen.cdf(np.asarray(x, dtype=float)),
                          dtype=float)

    def pdf(self, x) -> np.ndarray:
        return np.asarray(self._frozen.pdf(np.asarray(x, dtype=float)),
                          dtype=float)

    def log_pdf(self, x) -> np.ndarray:
        return np.asarray(self._frozen.logpdf(np.asarray(x, dtype=float)),
                          dtype=float)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Derived from :meth:`ppf`, so a design and a direct draw agree."""
        return self.ppf(rng.random(n))

    @property
    def mean(self) -> float:
        return float(self._frozen.mean())

    @property
    def std(self) -> float:
        return float(self._frozen.std())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mean={self.mean:g}, std={self.std:g})"


class Normal(RandomVariable):
    name = "normal"

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        if std <= 0.0:
            raise ValueError("std must be positive")
        self._dist = stats.norm(loc=float(mean), scale=float(std))

    @property
    def _frozen(self):
        return self._dist


class LogNormal(RandomVariable):
    r"""``log X ~ N(mu, sigma^2)``.

    ``mu`` and ``sigma`` describe the **underlying normal**, not ``X``. See
    :meth:`from_moments` if what you have is the mean and standard deviation of
    ``X`` itself, which is usually the case in practice.
    """

    name = "lognormal"

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.mu, self.sigma = float(mu), float(sigma)
        self._dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    @classmethod
    def from_moments(cls, mean: float, std: float) -> "LogNormal":
        """Build from the mean and standard deviation of ``X`` itself.

        ``sigma^2 = log(1 + (std/mean)^2)`` and ``mu = log(mean) - sigma^2/2``.
        The inversion is exact, so ``from_moments(m, s).mean == m``.
        """
        mean, std = float(mean), float(std)
        if mean <= 0.0:
            raise ValueError("a lognormal has positive support, so its mean "
                             "must be positive")
        if std <= 0.0:
            raise ValueError("std must be positive")
        sigma_sq = np.log1p((std / mean) ** 2)
        return cls(mu=np.log(mean) - 0.5 * sigma_sq, sigma=np.sqrt(sigma_sq))

    @property
    def _frozen(self):
        return self._dist


class Uniform(RandomVariable):
    name = "uniform"

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        if upper <= lower:
            raise ValueError("upper must exceed lower")
        self.lower, self.upper = float(lower), float(upper)
        self._dist = stats.uniform(loc=self.lower, scale=self.upper - self.lower)

    @property
    def _frozen(self):
        return self._dist


class Exponential(RandomVariable):
    name = "exponential"

    def __init__(self, rate: float = 1.0):
        if rate <= 0.0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self._dist = stats.expon(scale=1.0 / self.rate)

    @property
    def _frozen(self):
        return self._dist


class Gumbel(RandomVariable):
    """Type-I extreme value. The usual model for an annual maximum load."""

    name = "gumbel"

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self._dist = stats.gumbel_r(loc=float(loc), scale=float(scale))

    @property
    def _frozen(self):
        return self._dist


class Beta(RandomVariable):
    """Beta on ``[lower, upper]``. The natural choice for a bounded input."""

    name = "beta"

    def __init__(self, a: float = 2.0, b: float = 2.0, lower: float = 0.0,
                 upper: float = 1.0):
        if a <= 0.0 or b <= 0.0:
            raise ValueError("both shape parameters must be positive")
        if upper <= lower:
            raise ValueError("upper must exceed lower")
        self._dist = stats.beta(a=float(a), b=float(b), loc=float(lower),
                                scale=float(upper) - float(lower))

    @property
    def _frozen(self):
        return self._dist


# ---------------------------------------------------------------------------

DISTRIBUTION_REGISTRY: dict[str, type[RandomVariable]] = {
    "normal": Normal,
    "gaussian": Normal,
    "lognormal": LogNormal,
    "uniform": Uniform,
    "exponential": Exponential,
    "gumbel": Gumbel,
    "beta": Beta,
}


def get_distribution(name: str, **kwargs) -> RandomVariable:
    key = name.lower().replace("_", "-")
    if key not in DISTRIBUTION_REGISTRY:
        raise NotImplementedError(
            f"Unknown distribution '{name}'. Available: "
            f"{sorted(set(DISTRIBUTION_REGISTRY))}")
    return DISTRIBUTION_REGISTRY[key](**kwargs)


class JointIndependent:
    """Independent inputs, and the transforms every method here needs.

    Independence is a real restriction and it is stated rather than hidden.
    Correlated inputs need a Nataf or Rosenblatt transform, which changes the
    mapping from the unit hypercube and is not implemented -- see the refusals
    in the project notes. Passing correlated data to this class silently
    propagates the wrong distribution.
    """

    def __init__(self, variables, names=None):
        self.variables = list(variables)
        if not self.variables:
            raise ValueError("a joint distribution needs at least one variable")
        self.names = (list(names) if names is not None
                      else [f"x{i + 1}" for i in range(len(self.variables))])
        if len(self.names) != len(self.variables):
            raise ValueError("names and variables must have the same length")

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def mean(self) -> np.ndarray:
        return np.array([v.mean for v in self.variables])

    @property
    def std(self) -> np.ndarray:
        return np.array([v.std for v in self.variables])

    # -- the transform every design goes through -------------------------
    def ppf(self, unit: np.ndarray) -> np.ndarray:
        """Map an ``(n, dim)`` design on the unit hypercube into input space.

        This is where a Latin hypercube or a Sobol sequence becomes a set of
        physical inputs, and it is the only place the distribution enters.
        """
        unit = np.atleast_2d(np.asarray(unit, dtype=float))
        if unit.shape[1] != self.dim:
            raise ValueError(f"the design has {unit.shape[1]} columns but the "
                             f"distribution has {self.dim} variables")
        return np.column_stack([v.ppf(unit[:, i])
                                for i, v in enumerate(self.variables)])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([v.cdf(x[:, i])
                                for i, v in enumerate(self.variables)])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.ppf(rng.random((n, self.dim)))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """``(n, dim) -> (n,)``. A sum of logs, by independence."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.sum([v.log_pdf(x[:, i])
                       for i, v in enumerate(self.variables)], axis=0)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    # -- standard-normal (u-) space --------------------------------------
    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        """``x = F^-1(Phi(z))``: independent standard normals to input space.

        The space FORM and subset simulation live in. Distance from the origin
        there is a probability, which is what turns "find the most likely
        failure" into "find the nearest point".
        """
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.ppf(stats.norm.cdf(z))

    def to_standard_normal(self, x: np.ndarray) -> np.ndarray:
        """``z = Phi^-1(F(x))``, the inverse of
        :meth:`from_standard_normal`."""
        return stats.norm.ppf(np.clip(self.cdf(x), 1e-15, 1.0 - 1e-15))

    def __len__(self) -> int:
        return self.dim

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}~{v.name}"
                          for n, v in zip(self.names, self.variables))
        return f"JointIndependent({inner})"
