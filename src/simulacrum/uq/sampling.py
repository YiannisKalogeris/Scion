"""
Sampling methods for uncertainty propagation.

Provides Monte Carlo, Latin Hypercube, and Sobol quasi-random sampling
on hypercubes, with optional transformation to arbitrary distributions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import qmc


def monte_carlo(
    n_samples: int,
    dim: int,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Standard Monte Carlo sampling on [0,1]^d or within bounds.

    Parameters
    ----------
    n_samples : int
    dim : int
    bounds : (dim, 2) array of [low, high] per dimension, or None for [0,1]^d.
    seed : random seed.

    Returns
    -------
    samples : (n_samples, dim) array
    """
    rng = np.random.default_rng(seed)
    samples = rng.random((n_samples, dim))
    if bounds is not None:
        bounds = np.asarray(bounds)
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    return samples


def latin_hypercube(
    n_samples: int,
    dim: int,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube Sampling for better space-filling.

    Parameters
    ----------
    n_samples : int
    dim : int
    bounds : (dim, 2) array or None for [0,1]^d.
    seed : random seed.
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    samples = sampler.random(n=n_samples)
    if bounds is not None:
        bounds = np.asarray(bounds)
        samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return samples


def sobol_sequence(
    n_samples: int,
    dim: int,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Sobol quasi-random sequence for low-discrepancy sampling.

    Parameters
    ----------
    n_samples : int
        Should ideally be a power of 2.
    dim : int
    bounds : (dim, 2) array or None for [0,1]^d.
    seed : random seed.
    """
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    samples = sampler.random(n=n_samples)
    if bounds is not None:
        bounds = np.asarray(bounds)
        samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return samples


def importance_sampling(
    n_samples: int,
    target_log_pdf: Callable[[np.ndarray], float],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_log_pdf: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Basic importance sampling.

    Parameters
    ----------
    n_samples : int
    target_log_pdf : log of the target density
    proposal_sampler : draws samples from the proposal distribution
    proposal_log_pdf : log of the proposal density

    Returns
    -------
    samples : (n_samples, dim) array
    weights : (n_samples,) normalised importance weights
    """
    samples = proposal_sampler(n_samples)
    log_w = np.array([
        target_log_pdf(s) - proposal_log_pdf(s) for s in samples
    ])
    # Normalise in log-space for stability
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= weights.sum()
    return samples, weights
