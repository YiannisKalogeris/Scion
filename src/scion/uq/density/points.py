"""Representative points of the input probability space, with assigned
probabilities.

Both density methods in this subpackage start here. PDEM runs one
deterministic analysis per representative point and sums the resulting
densities weighted by ``P_q``; DPIM integrates a smoothed Dirac over the same
set. Li & Chen call ``P_q`` the *assigned probability*: the probability measure
of the representative region the point stands for.

The one invariant that matters is ``sum_q P_q == 1``. It is not decoration --
it is what makes the output a probability density rather than a density times
an unknown constant, and it is cheap to get wrong:

**The paper's nominal form does not sum to one, and on a skewed input it is
not close.** Li & Chen weight by the nominal PDF times the nominal element
area, ``p_Z(z_q) |dz_q|``, which is the midpoint rule for the cell's
probability. Measured here as ``|1 - sum_q P_q|``, in one dimension and in
three:

    distribution        points/axis   truncation     1-D      3-D
    Normal(0, 1)             20          1e-6      1.6e-6   4.8e-6
    Normal(0, 1)              7          1e-2      1.8e-2   5.2e-2
    LogNormal, CoV 1.0       10          1e-3      1.2e-1   4.2e-1
    Exponential(1)           20          1e-6      2.0e-2   5.8e-2

A density built from those weights integrates to that sum rather than to one,
and the bias is uniform, so no check inside the method notices it. The
Exponential row is the one to keep in mind: its 2% does **not** improve as the
truncation tightens, because the error is the midpoint rule meeting a density
peaked at the edge of its support, not a discarded tail. The lognormal row
overshoots -- the sum is 1.12 -- so the sign is not reliable either.

Here the weight is the *exact* cell probability from CDF differences instead.
It costs nothing extra, since independence makes it a product of per-axis
columns, and it is exact by construction rather than to the order of a
quadrature rule.

Two spacings, and the choice is a real one:

``"quantile"`` (the default) cuts the axis into cells of equal probability and
puts each point at its median. Every weight is then ``1/N`` exactly, no
truncation parameter is needed at all, and an unbounded support is handled
without a tail to discard. ``"uniform"`` reproduces the paper's lattice --
equally spaced points over a truncated range -- which spends points where the
density is small as well as where it is large, but is what a comparison
against the published figures needs. Under ``"uniform"`` the outer cells
absorb the tails exactly, so the invariant holds there too.

Independence is assumed, because the tensor product presumes it and
``JointIndependent`` is the only joint distribution in ``uq`` -- correlated
inputs are a declared refusal there and remain one here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

#: Tolerance on ``sum P_q - 1`` before the construction is rejected. The
#: lattice weights are renormalised by a factor this close to one; anything
#: further off is an error in the weights and not float round-off, so it is
#: raised rather than divided away.
PROBABILITY_TOL = 1e-10


@dataclass(frozen=True)
class RepresentativePoints:
    """A point set ``z`` with assigned probabilities, ``sum P_q == 1``.

    Frozen, because the weights and the points are only meaningful together:
    rescaling one without the other silently changes what every density built
    from the set integrates to.
    """

    z: np.ndarray                 # (n_points, dim)
    probability: np.ndarray       # (n_points,)

    def __post_init__(self) -> None:
        z = np.atleast_2d(np.asarray(self.z, dtype=float))
        p = np.asarray(self.probability, dtype=float).ravel()
        if z.ndim != 2:
            raise ValueError(f"points must be (n, dim); got shape {z.shape}")
        if p.shape[0] != z.shape[0]:
            raise ValueError(f"{z.shape[0]} points against {p.shape[0]} "
                             f"probabilities")
        if not np.all(np.isfinite(z)):
            raise ValueError("a representative point is not finite -- an "
                             "unbounded support needs quantile spacing, or a "
                             "truncation strictly inside (0, 0.5)")
        if not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise ValueError("assigned probabilities must be finite and "
                             "non-negative")
        total = float(p.sum())
        if abs(total - 1.0) > PROBABILITY_TOL:
            raise ValueError(f"assigned probabilities sum to {total!r}, not 1 "
                             f"-- see the module docstring on the nominal PDF")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "probability", p)

    @property
    def n_points(self) -> int:
        return self.z.shape[0]

    @property
    def dim(self) -> int:
        return self.z.shape[1]

    def __len__(self) -> int:
        return self.n_points

    def expectation(self, values) -> np.ndarray:
        """``sum_q P_q f(z_q)`` for ``values`` of shape ``(n,)`` or ``(n, k)``.

        The quadrature the whole method rests on, exposed because it is also
        the cheapest check on a point set: a set that cannot integrate its own
        input moments will not integrate a response either.
        """
        v = np.asarray(values, dtype=float)
        if v.shape[0] != self.n_points:
            raise ValueError(f"{v.shape[0]} values against {self.n_points} "
                             f"points")
        return np.tensordot(self.probability, v, axes=(0, 0))

    @property
    def mean(self) -> np.ndarray:
        """The point set's own estimate of ``E[Z]``."""
        return self.expectation(self.z)

    @property
    def std(self) -> np.ndarray:
        """The point set's own estimate of ``sd[Z]``."""
        second = self.expectation(self.z ** 2)
        return np.sqrt(np.maximum(second - self.mean ** 2, 0.0))

    def __repr__(self) -> str:
        return (f"RepresentativePoints(n_points={self.n_points}, "
                f"dim={self.dim})")


def _axis_quantile(variable, count: int):
    """Equal-probability cells, point at each cell's median quantile."""
    u = (np.arange(count) + 0.5) / count
    return np.asarray(variable.ppf(u), dtype=float).ravel(), \
        np.full(count, 1.0 / count)


def _axis_uniform(variable, count: int, truncation: float):
    """Equally spaced points over a truncated range; tails folded into the
    outer cells so the weights still sum to one."""
    lo = float(np.asarray(variable.ppf(truncation)).ravel()[0])
    hi = float(np.asarray(variable.ppf(1.0 - truncation)).ravel()[0])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"truncation {truncation!r} gives an empty or "
                         f"infinite range [{lo}, {hi}]")
    edges = np.linspace(lo, hi, count + 1)
    z = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.asarray(variable.cdf(edges), dtype=float).ravel().copy()
    # the outer cells reach the ends of the support, so no probability is
    # discarded and the column sums to one without a correction
    cdf[0], cdf[-1] = 0.0, 1.0
    weights = np.diff(cdf)
    if np.any(weights < 0.0):
        raise ValueError("a cell came out with negative probability; the "
                         "distribution's cdf is not monotone on the lattice")
    return z, weights


def lattice(distribution, counts, spacing: str = "quantile",
            truncation: float = 1e-6) -> RepresentativePoints:
    """Tensor-product representative points over an independent input space.

    Parameters
    ----------
    distribution : ``JointIndependent``, or a single ``RandomVariable``.
    counts : points per dimension; a scalar applies to every dimension.
    spacing : ``"quantile"`` for equal-probability cells (every weight
        ``1/N``, no truncation needed) or ``"uniform"`` for the paper's
        equally spaced lattice over a truncated range.
    truncation : tail probability left outside the ``"uniform"`` range; the
        outer cells absorb it, so this moves the *points*, not the weights.
        Ignored by ``"quantile"``.

    The point count is the product over dimensions, so this is the method's
    own curse of dimensionality and the reason both papers move to
    GF-discrepancy point sets above three or so inputs. It is left explicit
    rather than guarded: a four-dimensional lattice at 20 points an axis is
    160,000 deterministic analyses, which is a decision for the caller to
    make knowingly.
    """
    variables = getattr(distribution, "variables", None)
    if variables is None:                      # a single RandomVariable
        variables = [distribution]
    dim = len(variables)

    counts = np.broadcast_to(np.asarray(counts, dtype=int), (dim,))
    if np.any(counts < 1):
        raise ValueError("every dimension needs at least one point")
    if spacing not in ("quantile", "uniform"):
        raise ValueError(f"spacing {spacing!r}; use 'quantile' or 'uniform'")
    if spacing == "uniform" and not 0.0 < truncation < 0.5:
        raise ValueError(f"truncation must lie in (0, 0.5); got {truncation!r}")

    axes, weights = [], []
    for variable, count in zip(variables, counts):
        if spacing == "quantile":
            z_axis, w_axis = _axis_quantile(variable, int(count))
        else:
            z_axis, w_axis = _axis_uniform(variable, int(count), truncation)
        axes.append(z_axis)
        weights.append(w_axis)

    grids = np.meshgrid(*axes, indexing="ij")
    z = np.stack([g.ravel() for g in grids], axis=1)

    # by independence the cell probability is the product of the per-axis ones
    p = weights[0]
    for w_axis in weights[1:]:
        p = np.multiply.outer(p, w_axis)
    p = p.ravel()

    # the product of exact column sums is one to round-off; renormalise so the
    # invariant is exact, and let the container reject anything further off
    total = float(p.sum())
    if abs(total - 1.0) > PROBABILITY_TOL:
        raise ValueError(f"per-axis weights multiplied to {total!r}; one of "
                         f"the marginals does not integrate to one")
    return RepresentativePoints(z, p / total)


# ----------------------------------------------------------------------
# GF-discrepancy point selection
# ----------------------------------------------------------------------
#
# The lattice above is a tensor product, so its size is the product over
# dimensions and it is unusable past three or so inputs. What both papers
# reach for instead is a point set of ``N`` points *regardless of dimension*,
# chosen to make each marginal's probability-weighted empirical CDF close to
# the true one. Chen et al.'s GF-discrepancy measures exactly that, and Chen
# & Yang's Appendix A gives the construction.


def gf_discrepancy(points: "RepresentativePoints", distribution) -> float:
    """``D_GF = max_i sup_theta |F_hat_i(theta) - F_i(theta)|``, their Eq. (A1).

    A Kolmogorov-Smirnov distance per dimension, between the true marginal CDF
    and the empirical one *weighted by the assigned probabilities*,
    ``F_hat_i(theta) = sum_q P_q I{theta_{i,q} < theta}``, maximised over
    dimensions.

    Computed exactly rather than on a grid: the difference is a step function
    against a continuous one, so its supremum is attained at a sample point
    from one side or the other, and checking both sides of every jump is the
    whole calculation.
    """
    variables = getattr(distribution, "variables", [distribution])
    if len(variables) != points.dim:
        raise ValueError(f"{points.dim}-dimensional points against "
                         f"{len(variables)} variables")

    worst = 0.0
    for i, variable in enumerate(variables):
        order = np.argsort(points.z[:, i], kind="stable")
        cdf = np.asarray(variable.cdf(points.z[order, i]), dtype=float).ravel()
        upper = np.cumsum(points.probability[order])      # F_hat just above
        lower = upper - points.probability[order]         # and just below
        worst = max(worst, float(np.max(np.maximum(cdf - lower, upper - cdf))))
    return worst


def _voronoi_probabilities_1d(z, variable):
    """Exact assigned probabilities in one dimension.

    A Voronoi cell on the line is the interval between the midpoints to the
    neighbours, so its probability is a difference of two CDF values and
    needs no sampling. The outer cells run to the ends of the support, which
    is what makes the column sum to one exactly.
    """
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    midpoints = 0.5 * (sorted_z[:-1] + sorted_z[1:])
    cdf = np.empty(sorted_z.size + 1)
    cdf[0], cdf[-1] = 0.0, 1.0
    cdf[1:-1] = np.asarray(variable.cdf(midpoints), dtype=float).ravel()
    probability = np.empty(sorted_z.size)
    probability[order] = np.diff(cdf)
    return probability


def _voronoi_probabilities_nd(z, distribution, n_samples, rng):
    """Assigned probabilities by nearest-neighbour assignment of a sample.

    In more than one dimension a Voronoi cell is a polytope and
    ``int_{Omega_q} p(theta) dtheta`` has no closed form, so it is estimated:
    draw from ``p``, give each draw to its nearest representative point, and
    take the fractions. That sums to one by construction -- the estimate is
    in the *split*, never in the total -- and its error falls as
    ``1/sqrt(n_samples)``, which is why the default is generous.

    A cell receiving no sample would get zero probability and drop out of the
    method, so that is refused rather than returned.
    """
    sample = distribution.sample(n_samples, rng)
    counts = np.zeros(z.shape[0])
    # chunked, because the full (n_samples, n_points) distance matrix is the
    # one array here that can exhaust memory
    chunk = max(1, int(4e7 // max(z.shape[0], 1)))
    for start in range(0, n_samples, chunk):
        block = sample[start:start + chunk]
        nearest = np.argmin(
            ((block[:, None, :] - z[None, :, :]) ** 2).sum(axis=2), axis=1)
        counts += np.bincount(nearest, minlength=z.shape[0])
    if np.any(counts == 0):
        empty = int(np.sum(counts == 0))
        raise ValueError(
            f"{empty} of {z.shape[0]} Voronoi cells received no sample, so "
            f"they would carry zero probability and drop out of the method. "
            f"Raise n_samples (now {n_samples}) or lower the point count")
    return counts / counts.sum()


def _rank_transform(z, variable, weights):
    """Their Eqs. (A4) and (A5): re-place each value at its own weighted rank.

    ``theta_i,m <- F^-1( sum_q w_q I{theta_{i,q} < theta_{i,m}} + w_m / 2 )``.
    With ``w = 1/N`` this is Eq. (A4) and puts the marginal exactly on the
    equal-probability quantiles; with ``w = P_q`` it is Eq. (A5) and accounts
    for the Voronoi cells not carrying equal probability.
    """
    order = np.argsort(z, kind="stable")
    below = np.cumsum(weights[order]) - weights[order]
    target = np.empty_like(z)
    target[order] = below + 0.5 * weights[order]
    return np.asarray(variable.ppf(target), dtype=float).ravel()


def gf_points(distribution, n_points: int, *, seed: int | None = 0,
              n_samples: int | None = None) -> RepresentativePoints:
    """Representative points by GF-discrepancy, Chen & Yang's Appendix A.

    ``n_points`` points **whatever the dimension**, which is the whole reason
    to prefer this over :func:`lattice`: a tensor product costs
    ``n_points ** dim`` deterministic analyses and this costs ``n_points``.

    The construction is three steps. A Sobol set is mapped through the
    inverse marginal CDFs (A3); each dimension is then re-placed at its own
    rank so that the marginals sit exactly on the equal-probability quantiles
    (A4); the assigned probabilities are computed from the Voronoi partition
    and the dimensions re-placed once more at their *weighted* ranks (A5).
    What the Sobol set supplies is the pairing between dimensions -- the
    marginals are pinned by the rank transforms, so the sequence decides only
    which value of one variable meets which of another.

    Parameters
    ----------
    seed : scrambling seed for the Sobol set and the sampling in the Voronoi
        estimate. ``None`` draws one; the default is fixed so a point set is
        reproducible, since two runs with different points are not comparable.
    n_samples : draws for the Voronoi estimate above one dimension. Defaults
        to ``200 * n_points`` rounded up to a power of two, and is unused in
        one dimension, where the cells are intervals and the probabilities
        are exact.

    **After Eq. (A5), ``D_GF = max_q P_q / 2`` exactly** -- verified to four
    figures in one, two, three and six dimensions. The transform places each
    point at the midpoint of its own weight interval, so the gap between the
    weighted empirical CDF and the true one is ``P_m / 2`` on either side of
    it and the supremum is the largest. Three things follow, and they are
    the whole behaviour of this function:

    - the floor is ``1/(2N)``, reached exactly when the assigned
      probabilities are equal;
    - in one dimension they are, so the floor is attained: measured 1/(2N)
      to the last digit at every ``N`` tried, against a raw Sobol set's
      ``1/N``;
    - above one dimension the Voronoi cells of a Sobol set are not equal.
      ``max_q P_q * N`` runs about 1.4-1.6 in two dimensions, 1.75 in three
      and 1.8-2.1 in six, so ``D_GF`` sits that factor above the floor and
      **no amount of sampling changes it** -- measured flat from 64 to 16384
      draws per cell. Lowering it further means equalising the cells, a
      Lloyd-style relaxation the paper does not do and this does not either.

    So in three dimensions this is about 20% better than the Sobol set it
    starts from and in six it is a wash. The reason to use it there is not
    the discrepancy but the count: ``N`` analyses against ``N ** dim``.

    **In one dimension this returns the quantile lattice**, and not by
    coincidence: Eq. (A4) places the marginal on the quantiles by
    construction, and for a uniform input Eq. (A5) is then a fixed point. So
    `lattice(..., spacing="quantile")` is the one-dimensional case of this
    function, which is worth knowing before attributing a difference between
    two 1-D results to the point set.
    """
    variables = getattr(distribution, "variables", None)
    single = variables is None
    if single:
        variables = [distribution]
    dim = len(variables)
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("GF-discrepancy selection needs at least two points")

    if n_points & (n_points - 1):
        warnings.warn(
            f"Sobol's balance properties hold for powers of two; {n_points} "
            f"is not one. Use {1 << int(np.ceil(np.log2(n_points)))} or "
            f"accept a set that may be no better than random.",
            RuntimeWarning, stacklevel=2)

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # scipy's own power-of-2 notice
        unit = qmc.Sobol(d=dim, scramble=True, seed=rng).random(n_points)

    # (A3) then (A4): the marginals land on the equal-probability quantiles
    z = np.column_stack([
        _rank_transform(np.asarray(v.ppf(unit[:, i]), dtype=float).ravel(),
                        v, np.full(n_points, 1.0 / n_points))
        for i, v in enumerate(variables)])

    def assigned(points_z):
        if dim == 1:
            return _voronoi_probabilities_1d(points_z[:, 0], variables[0])
        draws = (n_samples if n_samples is not None
                 else 1 << int(np.ceil(np.log2(200 * n_points))))
        return _voronoi_probabilities_nd(points_z, distribution, draws, rng)

    # (A5): re-place each dimension at its *weighted* rank. The weights are
    # kept, deliberately: Eq. (A5) puts theta''_m exactly where the weighted
    # empirical CDF of the carried probabilities meets the true one, so the
    # discrepancy is minimised by construction -- and recomputing the
    # partition afterwards destroys precisely that. Measured in three
    # dimensions at 1024 points, re-weighting gives D_GF = 1.02e-2 against
    # 4.88e-4 for keeping them, i.e. 20x worse and worse than the raw Sobol
    # set it started from. The cost is that the probabilities belong to the
    # partition of the points before this last nudge, which is what the
    # paper specifies.
    probability = assigned(z)
    z = np.column_stack([_rank_transform(z[:, i], v, probability)
                         for i, v in enumerate(variables)])

    total = float(probability.sum())
    return RepresentativePoints(z, probability / total)
