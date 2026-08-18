r"""
Convergence diagnostics for MCMC output.

A sampler always returns samples. Nothing about the returned array says whether
they came from the posterior, and a chain stuck in one mode for its whole run
looks exactly like a converged one if you only plot a histogram. These are the
checks that distinguish the two, and they are the MCMC equivalent of the
tangent check elsewhere in this library: cheap, mechanical, and the only thing
that catches a failure whose symptom is a plausible number.

**Effective sample size.** Consecutive states of a Markov chain are correlated,
so ``n`` samples carry less information than ``n`` independent draws. The ESS
says how many independent draws they are worth,

    ESS = n / (1 + 2 sum_k rho_k),

with ``rho_k`` the autocorrelation at lag ``k``. The sum is truncated by
Geyer's initial positive sequence rule -- pairs of consecutive lags are summed
and the sum is stopped at the first negative pair, which is what keeps the
estimator from being swamped by the noise in the tail of ``rho``. Truncating at
a fixed lag instead reliably overestimates the ESS, which is the wrong direction
to be wrong in.

**R-hat.** Gelman-Rubin: run several chains from dispersed starts and compare
the variance between them to the variance within them. A single chain cannot
detect that it never visited a second mode; several chains started apart can.
Needs at least two chains -- that is a property of the diagnostic, not a
limitation here, so :func:`r_hat` refuses one rather than returning something
meaningless.

**What none of these apply to.** TMCMC returns an unordered particle set, not a
chain, so lag-based autocorrelation is undefined for it. The corresponding
health check there is the fraction of *unique* particles, since resampling
duplicates them and a collapsed population can have thousands of entries and a
handful of distinct values. :class:`~.chains.PosteriorSamples` knows which kind
it is holding and refuses the inapplicable one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "r_hat",
    "geweke",
]


def _as_chain(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def autocorrelation(chain: np.ndarray, max_lag: int | None = None
                    ) -> np.ndarray:
    """Normalised autocorrelation by FFT, ``(max_lag + 1, dim)``.

    Computed through the power spectrum rather than by an explicit double loop:
    the direct form is ``O(n^2)`` and the chains here are routinely 1e5 long.
    """
    x = _as_chain(chain)
    n = x.shape[0]
    if n < 2:
        raise ValueError("an autocorrelation needs at least two samples")
    max_lag = min(n - 1, int(max_lag) if max_lag is not None else n - 1)

    centred = x - x.mean(axis=0)
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(centred, n=size, axis=0)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=0)[:n]

    zero = acf[0].copy()
    zero[zero == 0.0] = 1.0            # a constant component has no structure
    return (acf / zero)[:max_lag + 1]


def integrated_autocorrelation_time(chain: np.ndarray) -> np.ndarray:
    """``tau = 1 + 2 sum rho_k`` per component, by Geyer's initial positive
    sequence. At least one, since a perfectly independent chain gives one."""
    rho = autocorrelation(chain)
    n_lags, dim = rho.shape
    tau = np.ones(dim)

    for d in range(dim):
        total = 0.0
        # sum consecutive pairs and stop at the first negative one; the pairs
        # of a reversible chain's autocorrelation are positive, so a negative
        # pair is noise rather than signal
        for k in range(1, n_lags - 1, 2):
            pair = rho[k, d] + rho[k + 1, d]
            if pair < 0.0:
                break
            total += pair
        tau[d] = max(1.0, 1.0 + 2.0 * total)
    return tau


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """``n / tau`` per component. The number of independent draws the chain is
    worth, and the number that should be quoted in a result."""
    x = _as_chain(chain)
    return x.shape[0] / integrated_autocorrelation_time(x)


def r_hat(chains) -> np.ndarray:
    """Gelman-Rubin potential scale reduction, one value per component.

    ``chains`` is a sequence of ``(n, dim)`` arrays, or one ``(m, n, dim)``
    array. Values below about 1.01 are conventionally taken as converged;
    anything above 1.1 means the chains have not mixed and the pooled samples
    are not a posterior.

    Refuses a single chain. Between-chain variance is the entire content of the
    statistic, and with one chain there is none.
    """
    array = np.asarray(chains, dtype=float)
    if array.ndim == 2:
        raise ValueError(
            "r_hat compares chains against each other and needs at least two; "
            "pass a sequence of (n, dim) arrays from dispersed starts")
    if array.ndim != 3:
        raise ValueError("expected (n_chains, n_samples, dim)")

    m, n, _ = array.shape
    if m < 2:
        raise ValueError("r_hat needs at least two chains")
    if n < 2:
        raise ValueError("r_hat needs at least two samples per chain")

    chain_means = array.mean(axis=1)                      # (m, dim)
    chain_vars = array.var(axis=1, ddof=1)                # (m, dim)

    between = n * chain_means.var(axis=0, ddof=1)
    within = chain_vars.mean(axis=0)
    pooled = (n - 1) / n * within + between / n

    within = np.where(within == 0.0, np.finfo(float).tiny, within)
    return np.sqrt(pooled / within)


def geweke(chain: np.ndarray, first: float = 0.1, last: float = 0.5
           ) -> np.ndarray:
    """Z-score comparing the mean of the first and last parts of one chain.

    A single-chain stationarity check: if the chain has converged, an early
    window and a late window estimate the same mean, and the difference scaled
    by its standard error is a standard normal. Magnitudes above about 2 say
    the chain was still moving.

    Weaker than :func:`r_hat` and worth exactly what it costs -- it cannot see
    a mode the chain never visited, because both windows are inside the same
    chain.
    """
    x = _as_chain(chain)
    n = x.shape[0]
    n_first, n_last = int(first * n), int(last * n)
    if min(n_first, n_last) < 2:
        raise ValueError("the chain is too short for these window fractions")

    head, tail = x[:n_first], x[-n_last:]
    var_head = head.var(axis=0, ddof=1) * integrated_autocorrelation_time(head)
    var_tail = tail.var(axis=0, ddof=1) * integrated_autocorrelation_time(tail)

    spread = np.sqrt(var_head / n_first + var_tail / n_last)
    spread = np.where(spread == 0.0, np.finfo(float).tiny, spread)
    return (head.mean(axis=0) - tail.mean(axis=0)) / spread
