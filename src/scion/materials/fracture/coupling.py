r"""
Coupling variants.

Axis 4, and the one most often confused with the split. The *split* decides how
``Psi_0`` decomposes; the *coupling* decides what is done with the two parts.
Only two questions, and the whole taxonomy of Ambati, Gerasimov & De Lorenzis
(2015) falls out of the answers:

    1. which stress enters the momentum equation?
    2. which energy drives the phase field?

                    momentum stress                     drives H
    isotropic       g(d) sigma_0                        Psi_0
    anisotropic     g(d) sigma_plus + sigma_minus       Psi_plus
    hybrid          g(d) sigma_0                        Psi_plus

The hybrid takes the linear momentum equation from the isotropic model and the
tension-driven damage from the anisotropic one. That is the whole idea: within
a staggered scheme it is incrementally linear, so it costs what the isotropic
model costs -- the paper measures roughly 85% less than anisotropic -- while
producing crack paths that match the anisotropic result.

What it gives up, and how that is repaired
------------------------------------------
Because the momentum equation degrades the *whole* energy, a closed crack under
compression transmits only ``g(d) sigma_0``: the faces interpenetrate and the
response is secant rather than elastic. The repair is a pointwise constraint,

    Psi_plus < Psi_minus  =>  d := 0

applied after the phase-field solve. It is not variationally derived -- the
hybrid model has no underlying functional, which the authors acknowledge and
justify by analogy with gradient-enhanced damage -- but it restores the
compressive branch. Their Fig. 2 is the constrained-versus-unconstrained
comparison; skip the constraint and the unloading branch is visibly wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Coupling",
    "Isotropic",
    "Anisotropic",
    "Hybrid",
    "COUPLING_REGISTRY",
    "get_coupling",
]


class Coupling(ABC):
    """How the split's two parts enter the two equations."""

    name: str = "coupling"

    #: True when the momentum equation is linear in u at fixed d, so the
    #: staggered scheme needs no inner Newton loop. This is the property the
    #: hybrid formulation exists to recover.
    linear_momentum: bool = False

    #: True when d must be zeroed where the state is compression-dominated
    enforce_compression_constraint: bool = False

    @abstractmethod
    def stress_tangent(self, split, eps: np.ndarray, lam: float, mu: float,
                       gd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Degraded stress and tangent for the momentum equation.

        ``gd`` is ``g(d)`` evaluated at the quadrature points.
        """

    @abstractmethod
    def driving_energy(self, split, eps: np.ndarray, lam: float,
                       mu: float) -> np.ndarray:
        """The energy whose running maximum becomes the history field H."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Isotropic(Coupling):
    """Everything degrades, everything drives. Bourdin et al.

    Linear momentum equation, and physically wrong under compression: it cracks
    in compressed regions and lets crack faces pass through each other. Useful
    as the cheap reference and as a deliberate demonstration of the failure --
    the two symmetric branches in the shear benchmark are this model behaving
    exactly as specified.
    """

    name = "isotropic"
    linear_momentum = True
    enforce_compression_constraint = False

    def stress_tangent(self, split, eps, lam, mu, gd):
        sp, sm, Cp, Cm = split.stress_tangent(eps, lam, mu)
        sigma0, C0 = sp + sm, Cp + Cm
        return (gd[:, None, None] * sigma0,
                gd[:, None, None, None, None] * C0)

    def driving_energy(self, split, eps, lam, mu):
        pp, pm = split.energies(eps, lam, mu)
        return pp + pm


class Anisotropic(Coupling):
    """Only the positive part degrades, only the positive part drives.

    Amor's or Miehe's model, depending on the split it is composed with. The
    physically consistent choice, and the expensive one: the stress now depends
    on the split's sign structure, so the momentum equation is nonlinear in u
    and needs Newton iterations at every staggered iteration.
    """

    name = "anisotropic"
    linear_momentum = False
    enforce_compression_constraint = False

    def stress_tangent(self, split, eps, lam, mu, gd):
        sp, sm, Cp, Cm = split.stress_tangent(eps, lam, mu)
        return (gd[:, None, None] * sp + sm,
                gd[:, None, None, None, None] * Cp + Cm)

    def driving_energy(self, split, eps, lam, mu):
        return split.energies(eps, lam, mu)[0]


class Hybrid(Coupling):
    """Isotropic stress, anisotropic driving energy, plus a constraint.

    Ambati, Gerasimov & De Lorenzis (2015). Incrementally linear within a
    staggered scheme, so it costs what the isotropic model costs while
    reproducing anisotropic crack paths.

    Variationally inconsistent by construction -- there is no functional whose
    Euler-Lagrange equations these are. The authors derive it instead by analogy
    with gradient-enhanced damage, where the same "postulate the stress relation,
    treat the gradient equation as auxiliary" structure is standard.
    """

    name = "hybrid"
    linear_momentum = True
    enforce_compression_constraint = True

    def stress_tangent(self, split, eps, lam, mu, gd):
        sp, sm, Cp, Cm = split.stress_tangent(eps, lam, mu)
        sigma0, C0 = sp + sm, Cp + Cm
        return (gd[:, None, None] * sigma0,
                gd[:, None, None, None, None] * C0)

    def driving_energy(self, split, eps, lam, mu):
        return split.energies(eps, lam, mu)[0]

    # ------------------------------------------------------------------
    @staticmethod
    def compression_mask(split, eps: np.ndarray, lam: float,
                         mu: float) -> np.ndarray:
        """Quadrature points where ``Psi_plus < Psi_minus``.

        Damage is forced to zero there, which is what keeps the crack faces
        from interpenetrating in the absence of an undegraded compressive
        stress term.
        """
        pp, pm = split.energies(eps, lam, mu)
        return pp < pm


COUPLING_REGISTRY: dict[str, type[Coupling]] = {
    "isotropic": Isotropic,
    "anisotropic": Anisotropic,
    "hybrid": Hybrid,
}


def get_coupling(name: str) -> Coupling:
    key = name.lower()
    if key not in COUPLING_REGISTRY:
        raise NotImplementedError(
            f"Unknown coupling '{name}'. Available: "
            f"{sorted(COUPLING_REGISTRY)}"
        )
    return COUPLING_REGISTRY[key]()
