r"""
Densities on a bounded support, and the atom a first-passage indicator carries.

Two things that the plain kernel sum of :mod:`~scion.uq.density.dpim` cannot
express, and that any indicator of inelastic activity needs.

**A kernel on a bounded support must be reflected.** A Gaussian centred near
the edge of the support puts mass outside it, where no realization can lie, so
the estimate reads *low* exactly at the bound -- which for a quantity that
measures damage is the end that matters. The correction is the mirror image of
each kernel about the bound, which returns the leaked mass to the inside
rather than discarding it.

**An indicator that is identically zero below a transition carries an atom,
and smoothing through it is wrong twice over.** Below the shakedown limit the
cyclic amplitude is not small, it is *zero*: with yield enforced as a
constraint it is zero to round-off, and the probability of that event is
finite. So the distribution is mixed -- a point mass at the bound plus a
density above it -- and running a kernel over the whole sample smears the
point mass across the continuous part. Measured on the probabilistic Bree
problem at a load factor where a quarter of realizations still shake down
(``pi_0 = 0.237``): smoothing everything **more than doubles** the apparent
density as the amplitude approaches zero. The atom has to be counted and
reported, not drawn as density.

The convention here is that :attr:`MixedDensity.atom` and the integral of
:attr:`MixedDensity.density` sum to one, so the pair is a distribution rather
than two separately normalised objects. The continuous part is therefore a
*sub*-density; divide by ``1 - atom`` for the density conditional on being off
the bound.

**What this does not do.** The reflection assumes the true density has zero
*derivative* at the bound, which is what makes it exact for a density meeting
the edge flat and merely better-than-nothing for one meeting it at a slope. A
boundary kernel or a log transform is the next step up and neither is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scion.uq.density.dpim import silverman_bandwidth
from scion.uq.density.points import RepresentativePoints

__all__ = ["MixedDensity", "mixed_density", "reflected_density"]


def reflected_density(grid, values, weights, bandwidth: float,
                      bound: float = 0.0) -> np.ndarray:
    """Weighted Gaussian kernel density, reflected about ``bound``.

    ``weights`` are used as given and are *not* normalised, so passing the
    assigned probabilities of a representative set returns a density that
    integrates to their sum. That is what lets the caller scale a continuous
    part by ``1 - atom`` without a second normalisation.

    The mirror of a sample about the bound is ``2 * bound - value``, which is
    the same expression whether the bound is below the data or above it, so
    one formula serves both half lines.
    """
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError(f"{values.size} values against {weights.size} "
                         f"weights")
    if not bandwidth > 0.0 or not np.isfinite(bandwidth):
        raise ValueError(f"bandwidth must be positive and finite; got "
                         f"{bandwidth!r}. The Dirac limit is not computable, "
                         f"which is what the smoothing exists to avoid")
    if values.size == 0:
        return np.zeros_like(grid)

    z = (grid[:, None] - values[None, :]) / bandwidth
    mirrored = (grid[:, None] - (2.0 * bound - values[None, :])) / bandwidth
    kernel = np.exp(-0.5 * z * z) + np.exp(-0.5 * mirrored * mirrored)
    return (kernel @ weights) / (bandwidth * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class MixedDensity:
    """A point mass at a bound plus a sub-density above (or below) it.

    ``atom`` plus the integral of ``density`` is one up to the grid's own quadrature
    error, which :attr:`mass` reports rather than asserting.
    """

    grid: np.ndarray = field(repr=False)
    density: np.ndarray = field(repr=False)
    atom: float
    bound: float
    bandwidth: float
    n_atom: int
    n_continuous: int

    @property
    def mass(self) -> float:
        """``atom`` plus the integral of the continuous part.

        The trapezoid rule written out rather than called: it is two lines,
        it works on a non-uniform grid, and ``numpy`` renamed ``trapz`` to
        ``trapezoid`` at 2.0, so naming either one picks a version.
        """
        return float(self.atom + self._integral(self.density))

    def _integral(self, y) -> float:
        return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(self.grid)))

    @property
    def conditional(self) -> np.ndarray:
        """The density given that the realization is off the bound."""
        if self.atom >= 1.0:
            raise ValueError(
                "every realization sits on the atom, so there is no "
                "conditional density to report")
        return self.density / (1.0 - self.atom)


def mixed_density(points: RepresentativePoints, values, *,
                  bound: float = 0.0, tol: float = 1.0e-6,
                  n_cells: int = 600, span: float = 1.25,
                  bandwidth: float | None = None) -> MixedDensity:
    """Split an indicator into its atom at ``bound`` and a reflected density.

    Parameters
    ----------
    points : the representative set; its assigned probabilities are the
        weights, so the atom is a probability and not a count.
    values : one response per representative point.
    bound : where the indicator is identically zero below (or above).
    tol : how far from ``bound`` still counts as *on* the atom. This is a
        tolerance on an exact zero rather than a threshold on a small number,
        which is what makes it safe to set loosely: with yield enforced as a
        constraint the amplitude below the transition is zero to round-off,
        measured at ``1e-16`` against a default of ``1e-6``. Against an
        indicator that merely becomes *small* -- a penalty formulation's, for
        instance -- there is no such gap and this parameter would be choosing
        the answer.
    bandwidth : defaults to :func:`~scion.uq.density.dpim.silverman_bandwidth`
        on the **continuous part alone**, which is the right sample for it;
        taking it over the whole set would let the atom's zeros widen the
        kernel that draws the part above the bound.

    Raises
    ------
    ValueError
        if the responses straddle the bound, or if fewer than two lie off it
        and no bandwidth is given -- a single point has no spread to scale by.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size != points.n_points:
        raise ValueError(f"{values.size} values against {points.n_points} "
                         f"representative points")
    if not np.all(np.isfinite(values)):
        raise ValueError("every response must be finite")

    above = values > bound + tol
    below = values < bound - tol
    if above.any() and below.any():
        raise ValueError(
            "the responses fall on both sides of the bound, so it is not a "
            "bound. A mixed distribution needs a half line: check the sign "
            "convention, or pass the bound the indicator actually respects")
    active = above | below
    atom = float(points.probability[~active].sum())
    weights = points.probability[active]
    present = values[active]

    if bandwidth is None:
        if present.size < 2:
            raise ValueError(
                f"only {present.size} response(s) lie off the bound, which is "
                f"not enough to choose a bandwidth from. Pass `bandwidth` "
                f"explicitly, or report the atom alone -- at this load level "
                f"the distribution is essentially the point mass")
        total = weights.sum()
        subset = RepresentativePoints(points.z[active], weights / total)
        bandwidth = silverman_bandwidth(subset, present)

    far = float(np.abs(present - bound).max()) if present.size else 1.0
    reach = span * (far if far > 0.0 else 1.0)
    grid = (np.linspace(bound, bound + reach, n_cells) if not below.any()
            else np.linspace(bound - reach, bound, n_cells))
    density = reflected_density(grid, present, weights, bandwidth, bound)
    return MixedDensity(grid=grid, density=density, atom=atom, bound=bound,
                        bandwidth=float(bandwidth),
                        n_atom=int((~active).sum()),
                        n_continuous=int(active.sum()))
