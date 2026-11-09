r"""
Truncated Karhunen-Loeve expansion of a stationary random field.

A random *field* reaches the rest of this package as a finite vector of
coefficients: the field is written

    g(x) = sum_k sqrt(lam_k) phi_k(x) xi_k,     xi_k ~ N(0, 1) independent,

so a realization is a draw of ``n_modes`` standard normal variables and every
design in :mod:`scion.uq.forward.estimators` applies unchanged. The modes come
from the covariance operator, not from the mesh, which is why the expansion is
built on quadrature points and their weights rather than on nodes.

The quadrature weights are load-bearing, not a refinement
---------------------------------------------------------
The continuous eigenproblem is ``INT C(x,y) phi(y) dy = lam phi(x)``. Discretised
on points carrying weights ``w`` that is ``C W phi = lam phi``, a *generalised*
problem, and the symmetric form actually solved here is

    W^(1/2) C W^(1/2) psi = lam psi,        phi = W^(-1/2) psi

with ``psi`` orthonormal, hence ``phi^T W phi = 1``, which is
``INT phi^2 dV = 1`` -- the normalisation the expansion above assumes. Running
``eigh`` on ``C`` alone instead is the obvious thing and gives modes orthonormal
in the plain vector sense, so on a graded mesh the modes are ranked by their
energy in regions where the points happen to be dense rather than by the
variance they carry.

**The identity that looks like the check is not one.** With every mode retained
both forms reconstruct ``C`` exactly, so the pointwise variance
``sum_k lam_k phi_k(x)^2`` returns ``1`` at every point either way -- measured
to 1e-12 on a mesh whose weights span a factor of 3000. What separates them is
the *truncation*, and in both directions at once. The weighted form is the
optimal ``k``-term approximation in ``L2(D)``, which is the property the
expansion exists for; the naive one is not, and it simultaneously divides its
eigenvalue sum by ``trace(C) = n_points`` instead of ``INT dV``, so it reports a
larger variance fraction while reproducing the covariance worse. On that same
graded mesh at four modes: weighted ``0.739`` captured at an ``L2(D)`` error of
``7.52e-2``, naive ``0.782`` captured at ``9.70e-2``. A number that is both more
flattering and less accurate is the hazard here, since nothing in the run
complains.

Truncation understates the variance, and pointwise
--------------------------------------------------
``variance_captured`` is an integral over the domain, so a field capturing 95%
of the variance is not a field whose variance is 95% of target everywhere:
``pointwise_variance`` is what says where the expansion is thin, and it is
always worst where the modes are smoothest. Truncating therefore makes
realizations *less* extreme, which for a strength field is the unsafe
direction -- quote the pointwise minimum alongside the integral fraction.

Correlation length against mesh size
------------------------------------
The number of modes needed for a given variance fraction grows as the
correlation length falls, and below roughly two elements per correlation length
the eigenvectors are resolving the mesh rather than the covariance. There is no
check for that here because the honest one is a refinement study; the symptom
is that ``variance_captured`` at fixed ``n_modes`` moves with the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KERNEL_REGISTRY", "KarhunenLoeve", "LogNormalField"]


def _exponential(d: np.ndarray, ell: float) -> np.ndarray:
    return np.exp(-d / ell)


def _squared_exponential(d: np.ndarray, ell: float) -> np.ndarray:
    return np.exp(-(d / ell) ** 2)


KERNEL_REGISTRY = {
    "exponential": _exponential,
    "squared_exponential": _squared_exponential,
}


@dataclass(frozen=True)
class KarhunenLoeve:
    """A truncated expansion of a zero-mean, unit-variance Gaussian field.

    Build it with :meth:`from_points`. ``eigenvalues`` are in descending order
    and ``modes`` is ``(n_modes, n_points)``, normalised so that
    ``INT phi_k phi_l dV = delta_kl`` against the weights supplied.
    """

    eigenvalues: np.ndarray
    modes: np.ndarray
    weights: np.ndarray
    total_variance: float
    correlation_length: float
    kernel: str

    # ------------------------------------------------------------------
    @classmethod
    def from_points(cls, points, weights, correlation_length: float,
                    n_modes: int | None = None,
                    variance_target: float | None = None,
                    kernel: str = "exponential") -> "KarhunenLoeve":
        """Eigen-decompose the covariance on scattered points.

        Parameters
        ----------
        points : ``(n_points, dim)`` coordinates -- quadrature points, not
            nodes, because the weights have to be the ones the field will be
            integrated against.
        weights : ``(n_points,)`` quadrature weights. See the module docstring
            on why these enter the eigenproblem rather than scaling it
            afterwards.
        correlation_length : of the kernel, in the units of ``points``.
        n_modes : how many to retain. Mutually exclusive with
            ``variance_target``.
        variance_target : retain the fewest modes whose eigenvalues sum to at
            least this fraction of the total variance.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError(f"points must be (n_points, dim); got {pts.shape}")
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != pts.shape[0]:
            raise ValueError(
                f"{w.size} weights for {pts.shape[0]} points")
        if np.any(w <= 0.0):
            raise ValueError(
                "quadrature weights must be positive -- the symmetric form of "
                "the eigenproblem takes their square root")
        if (n_modes is None) == (variance_target is None):
            raise ValueError("give exactly one of n_modes, variance_target")
        if correlation_length <= 0.0:
            raise ValueError("correlation_length must be positive")
        try:
            kern = KERNEL_REGISTRY[kernel]
        except KeyError:
            raise ValueError(
                f"unknown kernel {kernel!r}; have "
                f"{sorted(KERNEL_REGISTRY)}") from None

        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        cov = kern(d, float(correlation_length))

        # the symmetric Nystrom form: eigenvectors orthonormal in l2 become
        # modes orthonormal in L2(D) on division by sqrt(w)
        root = np.sqrt(w)
        sym = cov * root[:, None] * root[None, :]
        sym = 0.5 * (sym + sym.T)                      # kill the asymmetry eigh would
        lam, vec = np.linalg.eigh(sym)
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]

        # a covariance operator is positive semidefinite, so a negative
        # eigenvalue is round-off and a mode that cannot carry variance
        lam = np.maximum(lam, 0.0)
        total = float(w.sum())                         # INT C(x,x) dV, C(x,x) = 1

        if variance_target is not None:
            if not 0.0 < variance_target <= 1.0:
                raise ValueError("variance_target must be in (0, 1]")
            share = np.cumsum(lam) / total
            keep = int(np.searchsorted(share, variance_target) + 1)
            keep = min(keep, lam.size)
        else:
            keep = int(n_modes)
            if not 1 <= keep <= lam.size:
                raise ValueError(
                    f"n_modes must be in [1, {lam.size}]; got {keep}")

        phi = (vec[:, :keep] / root[:, None]).T        # (keep, n_points)
        return cls(eigenvalues=lam[:keep].copy(), modes=np.ascontiguousarray(phi),
                   weights=w.copy(), total_variance=total,
                   correlation_length=float(correlation_length), kernel=kernel)

    # ------------------------------------------------------------------
    @property
    def n_modes(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def n_points(self) -> int:
        return int(self.modes.shape[1])

    @property
    def variance_captured(self) -> float:
        """Fraction of the domain-integrated variance the retained modes carry."""
        return float(self.eigenvalues.sum() / self.total_variance)

    @property
    def pointwise_variance(self) -> np.ndarray:
        """``sum_k lam_k phi_k(x)^2`` -- tends to 1 as every mode is retained.

        Read its *minimum* before quoting :attr:`variance_captured`: the
        integral fraction says nothing about where the expansion is thin, and
        truncation always makes realizations less extreme.
        """
        return self.eigenvalues @ (self.modes ** 2)

    # ------------------------------------------------------------------
    def realize(self, xi) -> np.ndarray:
        """The Gaussian field at the points, for coefficients ``xi``.

        ``xi`` is ``(n_modes,)`` or ``(n_samples, n_modes)`` standard normal;
        the return has a matching leading axis.
        """
        x = np.asarray(xi, dtype=float)
        if x.shape[-1] != self.n_modes:
            raise ValueError(
                f"xi must have {self.n_modes} coefficients; got {x.shape[-1]}")
        return (x * np.sqrt(self.eigenvalues)) @ self.modes


@dataclass(frozen=True)
class LogNormalField:
    """A lognormal field with prescribed mean and coefficient of variation.

    ``exp(mu + s g(x))`` for the underlying Gaussian ``g`` of
    :class:`KarhunenLoeve`, with ``mu`` and ``s`` from the quantity's *own*
    moments -- the same distinction :class:`~scion.uq.forward.distributions.LogNormal`
    draws, and the same trap: passing the target mean as the normal's ``mu``
    inflates it by ``exp(s^2/2)``.

    The mean is reproduced exactly only where the expansion carries the full
    unit variance. Truncation leaves ``pointwise_variance`` below one, which
    biases the realized mean *low* by ``exp(-s^2(1 - v)/2)``; the bias is
    reported by :attr:`mean_bias` rather than corrected, because correcting it
    would put the variance further out instead.
    """

    kl: KarhunenLoeve
    mean: float
    cov: float

    def __post_init__(self):
        if self.mean <= 0.0:
            raise ValueError("a lognormal field has a positive mean")
        if self.cov <= 0.0:
            raise ValueError("cov must be positive")

    @property
    def sigma(self) -> float:
        """The underlying normal's standard deviation."""
        return float(np.sqrt(np.log1p(self.cov ** 2)))

    @property
    def mu(self) -> float:
        """The underlying normal's mean."""
        return float(np.log(self.mean) - 0.5 * self.sigma ** 2)

    @property
    def mean_bias(self) -> np.ndarray:
        """Realized mean over target, pointwise, from truncation alone."""
        v = self.kl.pointwise_variance
        return np.exp(-0.5 * self.sigma ** 2 * (1.0 - v))

    def realize(self, xi) -> np.ndarray:
        """The field at the quadrature points."""
        return np.exp(self.mu + self.sigma * self.kl.realize(xi))
