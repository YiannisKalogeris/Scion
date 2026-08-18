r"""
Damage accumulation rules.

Rainflow gives ``n_i`` cycles at each amplitude and a life curve gives the
``N_i`` that amplitude would survive on its own. An accumulation rule combines
them into one number and says when it reaches failure.

Nothing here is a constitutive model. There is no stress, no state at a
quadrature point, no tangent: by the time these functions run the analysis is
over and the load order has already been discarded by the counting. Continuum
damage mechanics -- ``D`` as an internal variable, effective stress
``sigma/(1 - D)``, stiffness degradation inside the equilibrium solve -- is a
different subject that belongs with the other constitutive models, and it is the
sequence-*dependent* counterpart to what is here.

Miner's rule
------------
``D = sum n_i / N_i``, failure at ``D = 1``. Linear, order-independent, and
almost certainly wrong in detail -- which is why it is worth being precise about
*how*:

**It has no memory of sequence.** High-then-low loading and low-then-high give
the same ``D``, and measured lives differ by factors of two to five. A large
cycle early leaves a crack that the small cycles afterwards extend; the reverse
order does not. Nothing in a rainflow-plus-Miner calculation can see this,
because rainflow has already thrown the order away.

**It ignores the interaction with the endurance limit.** Under Miner with a
curve that has a flat cut-off, a cycle below the limit is exactly harmless
however many of them there are. Once larger cycles have started a crack it is
not. This one *is* fixable, and the fix belongs in the curve rather than here:
see :meth:`~simulacrum.fatigue.curves.TwoSlopeSN.haibach`.

**``D = 1`` is a nominal value.** Measured failure sums scatter roughly between
0.1 and 10, biased low for spectra containing a few large cycles. Design codes
therefore use ``critical`` well below one -- 0.5 is common, 0.2 for welded
details under a wide spectrum. It is a calibration constant, not a law, so it is
exposed rather than hard-coded.

Miner stays the default because the alternatives need parameters nobody has
measured for the material in hand, and because its failure mode -- indifference
to sequence -- is at least understood and documented rather than buried in a
fitted exponent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "cycle_damage",
    "AccumulationRule",
    "LinearAccumulation",
    "NonlinearAccumulation",
    "ACCUMULATION_REGISTRY",
    "get_accumulation_rule",
]


def cycle_damage(counts, cycles_to_failure) -> np.ndarray:
    """``n_i / N_i`` for each counted cycle.

    ``N = inf`` (runout) gives zero, and ``N = 0`` -- the amplitude is off the
    end of the curve, or a mean stress reached the static strength -- gives
    ``inf``. Both are propagated rather than clipped. An infinite contribution
    is the honest report that this one cycle fails the part; replacing it with a
    large finite number lets it be averaged away against the rest of the
    spectrum, which is how a static overload disappears from a fatigue report.
    """
    n = np.asarray(counts, dtype=float)
    N = np.asarray(cycles_to_failure, dtype=float)
    alive = N > 0.0
    return np.where(alive, n / np.where(alive, N, 1.0),
                    np.where(n > 0.0, np.inf, 0.0))


class AccumulationRule(ABC):
    """Combines per-cycle damage into a total, and says what failure means."""

    name: str = "accumulation"

    def __init__(self, critical: float = 1.0):
        if critical <= 0.0:
            raise ValueError("the critical damage must be positive")
        self.critical = float(critical)

    @abstractmethod
    def accumulate(self, damage, axis: int = -1) -> np.ndarray:
        """Total damage from the per-cycle contributions."""

    # ------------------------------------------------------------------
    def blocks_to_failure(self, total) -> np.ndarray:
        """Repetitions of the counted block the part survives, ``critical / D``.

        Meaningful only if the counted history is what actually repeats. A count
        of one transient scaled by ten is not a count of ten chained transients:
        the residual cycles at the joins differ, and those are the largest ones
        in the record. Count the block you mean, then scale here.
        """
        d = np.asarray(total, dtype=float)
        alive = d > 0.0
        return np.where(alive, self.critical / np.where(alive, d, 1.0), np.inf)

    def failed(self, total) -> np.ndarray:
        return np.asarray(total, dtype=float) >= self.critical

    def __repr__(self) -> str:
        return f"{type(self).__name__}(critical={self.critical:g})"


class LinearAccumulation(AccumulationRule):
    """Palmgren-Miner: ``D = sum n_i / N_i``."""

    name = "miner"

    def accumulate(self, damage, axis: int = -1):
        return np.sum(np.asarray(damage, dtype=float), axis=axis)


class NonlinearAccumulation(AccumulationRule):
    r"""Marco-Starkey: ``D = sum (n_i / N_i)^x``, ``x > 1``.

    A damage curve convex in the life fraction, so early cycles count for less
    than their share and late ones for more. At ``x = 1`` it is Miner.

    It is usually presented as the fix for Miner's sequence-independence, and as
    written here it is not: applied to a rainflow count it remains a sum over an
    unordered set, so every permutation of the load still gives the same answer.
    What it does change is the weighting between amplitudes, penalising a few
    large cycles relative to many small ones. A genuine sequence effect requires
    carrying a damage state through the history in order, which means giving up
    rainflow -- not changing an exponent.

    Kept because it is a published rule and costs nothing to evaluate. The
    exponent is material- and spectrum-specific and there is no default worth
    having, so there is none.
    """

    name = "marco-starkey"

    def __init__(self, exponent: float, critical: float = 1.0):
        super().__init__(critical)
        if exponent <= 0.0:
            raise ValueError("the Marco-Starkey exponent must be positive")
        self.exponent = float(exponent)

    def accumulate(self, damage, axis: int = -1):
        d = np.asarray(damage, dtype=float)
        return np.sum(d ** self.exponent, axis=axis)

    def __repr__(self) -> str:
        return (f"NonlinearAccumulation(exponent={self.exponent:g}, "
                f"critical={self.critical:g})")


# ---------------------------------------------------------------------------

ACCUMULATION_REGISTRY: dict[str, type[AccumulationRule]] = {
    "miner": LinearAccumulation,
    "linear": LinearAccumulation,
    "palmgren-miner": LinearAccumulation,
    "marco-starkey": NonlinearAccumulation,
    "nonlinear": NonlinearAccumulation,
}


def get_accumulation_rule(name: str, **kwargs) -> AccumulationRule:
    key = name.lower().replace("_", "-")
    if key not in ACCUMULATION_REGISTRY:
        raise NotImplementedError(
            f"Unknown accumulation rule '{name}'. Available: "
            f"{sorted(set(ACCUMULATION_REGISTRY))}")
    return ACCUMULATION_REGISTRY[key](**kwargs)
