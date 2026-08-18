r"""
Flow rules.

The yield criterion says *when* a material flows plastically; the flow
rule says *in which direction*:

    d eps^p = d lambda  dg/dsigma,

where ``g`` is the plastic potential. Taking ``g = f`` gives **associative**
flow, in which the plastic strain increment is normal to the yield surface.
Anything else is **non-associative**.

Why the distinction matters
---------------------------
For von Mises the question barely arises: the surface is pressure-insensitive,
its normal is deviatoric, and associative flow is both thermodynamically clean
and experimentally right for metals.

For a pressure-sensitive criterion it is the central modelling decision.
Associative flow on the Drucker-Prager cone has ``tr(dg/dsigma) = eta``, so the
plastic volume increases at a rate tied to the *friction* angle -- typically
30-45 degrees for a soil, which predicts dilatancy several times larger than
anything measured. The standard remedy is a potential with the same shape but a
smaller dilatancy angle ``psi``, often ``psi = 0`` for an entirely
volume-preserving flow.

The price
---------
Non-associative flow makes the consistent tangent **unsymmetric**: the
elastoplastic operator contains ``dg/dsigma (x) df/dsigma``, which is symmetric
only when the two coincide. Three consequences follow:

* the linear solver must not assume symmetry (no Cholesky, no ``eigh``, and
  symmetric storage formats will silently discard half the matrix);
* the problem is no longer a minimisation, so an energy-based line search or
  convergence test loses its footing;
* uniqueness and stability are not guaranteed -- strongly non-associative
  models can bifurcate before the yield surface is reached.

:attr:`FlowRule.symmetric_tangent` records which case a model is in, so the
assembler can pick a solver rather than discovering the problem numerically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .yield_criteria import DruckerPrager, VonMises, YieldCriterion

__all__ = [
    "FlowRule",
    "Associative",
    "NonAssociative",
    "FLOW_REGISTRY",
    "get_flow_rule",
]


class FlowRule(ABC):
    """Direction of plastic flow, ``dg/dsigma``."""

    name: str = "flow"

    #: True when ``g = f``, hence a symmetric consistent tangent.
    associative: bool = True

    @property
    def symmetric_tangent(self) -> bool:
        return self.associative

    # ------------------------------------------------------------------
    @abstractmethod
    def g(self, criterion: YieldCriterion, sigma: np.ndarray,
          kappa: np.ndarray) -> np.ndarray:
        """The potential itself, ``(n,)``.

        Only its gradient enters the constitutive equations -- the value is
        provided so the gradient can be verified against finite differences,
        which is the one check that catches a potential and its derivative
        drifting apart.
        """

    @abstractmethod
    def dg_dsigma(self, criterion: YieldCriterion,
                  sigma: np.ndarray) -> np.ndarray:
        """Flow direction, ``(n, 3, 3)``."""

    @abstractmethod
    def d2g_dsigma2(self, criterion: YieldCriterion,
                    sigma: np.ndarray) -> np.ndarray:
        """``d2g/dsigma2``, ``(n, 3, 3, 3, 3)``, for the closest-point solve."""

    # ------------------------------------------------------------------
    def dilatancy(self, criterion: YieldCriterion,
                  sigma: np.ndarray) -> np.ndarray:
        """``tr(dg/dsigma)``: the plastic volume change per unit multiplier.

        Zero for volume-preserving flow. Comparing this against the criterion's
        own ``tr(df/dsigma)`` is the quickest way to see how far a model departs
        from associativity.
        """
        return np.einsum("nii->n", self.dg_dsigma(criterion, sigma))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Associative(FlowRule):
    """``g = f``. Flow normal to the yield surface.

    The right default: it follows from the principle of maximum plastic
    dissipation, keeps the tangent symmetric, and is what the classical
    existence and uniqueness results assume. Depart from it only when the
    predicted dilatancy is the thing you are trying to fix.
    """

    name = "associative"
    associative = True

    def g(self, criterion, sigma, kappa):
        return criterion.f(sigma, kappa)

    def dg_dsigma(self, criterion, sigma):
        return criterion.df_dsigma(sigma)

    def d2g_dsigma2(self, criterion, sigma):
        return criterion.d2f_dsigma2(sigma)


class NonAssociative(FlowRule):
    r"""``g`` supplied as a second surface, independent of ``f``.

    The potential is itself a :class:`YieldCriterion` instance, so it inherits
    the same verified derivatives rather than duplicating them. Only its
    *shape* is used; its hardening force is irrelevant, since a constant offset
    does not change the gradient.

    Use :meth:`drucker_prager` for the usual case.
    """

    name = "non-associative"
    associative = False

    def __init__(self, potential: YieldCriterion):
        self.potential = potential

    # ------------------------------------------------------------------
    @classmethod
    def drucker_prager(cls, psi: float, match: str = "outer",
                       degrees: bool = True) -> "NonAssociative":
        """Cone potential with dilatancy angle ``psi``.

        ``psi = 0`` gives entirely volume-preserving plastic flow, the usual
        first choice for a soil or concrete when no dilatancy data exists.
        ``psi = phi`` recovers associative flow, and any value between
        interpolates. ``match`` should be the same calibration used for the
        yield surface, so that the two cones share a shape convention.
        """
        return cls(DruckerPrager.from_mohr_coulomb(psi, match=match,
                                                   degrees=degrees))

    @classmethod
    def volume_preserving(cls) -> "NonAssociative":
        """Deviatoric flow, ``dg/dsigma = s / (2 sqrt(J2))``.

        Equivalent to :meth:`drucker_prager` with ``psi = 0``, stated directly
        for the case where no friction angle is in play.
        """
        return cls(DruckerPrager(eta=0.0, xi=1.0))

    # ------------------------------------------------------------------
    def g(self, criterion, sigma, kappa):
        return self.potential.f(sigma, kappa)

    def dg_dsigma(self, criterion, sigma):
        return self.potential.df_dsigma(sigma)

    def d2g_dsigma2(self, criterion, sigma):
        return self.potential.d2f_dsigma2(sigma)

    def singular_mask(self, sigma: np.ndarray) -> np.ndarray:
        """Where the *potential* has no gradient.

        Distinct from the criterion's singular set: the flow direction can be
        undefined at a state that is perfectly regular on the yield surface,
        and the return map has to test both.
        """
        return self.potential.singular_mask(sigma)

    def __repr__(self) -> str:
        return f"NonAssociative({self.potential!r})"


# ---------------------------------------------------------------------------

FLOW_REGISTRY: dict[str, type[FlowRule]] = {
    "associative": Associative,
    "normal": Associative,
    "non-associative": NonAssociative,
    "nonassociative": NonAssociative,
}


def get_flow_rule(name: str, **kwargs) -> FlowRule:
    """Resolve a flow rule by name.

    ``NonAssociative`` needs a potential, so pass one::

        get_flow_rule("non-associative",
                      potential=DruckerPrager.from_mohr_coulomb(10.0))

    or use the classmethods, which are less error-prone::

        NonAssociative.drucker_prager(psi=10.0)
    """
    key = name.lower().replace("_", "-")
    if key not in FLOW_REGISTRY:
        raise NotImplementedError(
            f"Unknown flow rule '{name}'. Available: "
            f"{sorted(set(FLOW_REGISTRY))}")
    return FLOW_REGISTRY[key](**kwargs)
