"""
Constitutive models in principal-stretch space.

Every model here diagonalises ``F = U diag(s) V^T`` once and then works on the
three principal stretches, rotating back at the end. That is not an
implementation detail, it is the reason the module exists: in this space an
isotropic yield surface is a surface in ``R^3``, so a return map is a
projection of a 3-vector instead of an iterative solve on a tensor. Sand and
von Mises below are both closed-form for that reason.

The elastic laws use the *Hencky* (logarithmic) strain ``eps = log(s)``, which
is the natural strain measure here: it is additive under sequential stretching,
so a multiplicative split ``F = F_e F_p`` becomes an additive split of the
principal strains, and the plastic projection is then a subtraction.

Relation to the rest of ``materials/``
--------------------------------------
``hyperelastic_batched.py`` holds models written directly in terms of ``F``
(Neo-Hookean and friends) and shares none of this machinery -- no SVD, no
principal decomposition. ``StVenantKirchhoff`` in ``geometric_nonlinear.py`` is
Green strain, not Hencky; the two agree only to first order and diverge badly
in compression, which is exactly the regime this module is for.

Provenance
----------
Ported from CK-MPM (Liu, Wang & Li, *CK-MPM: A Compact-Kernel Material Point
Method*, ACM TOG 44(4), 2025; Apache-2.0), reimplemented in batched NumPy from
its CUDA sources. The underlying models are older and are cited on each class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from scion.core import small_linalg as sla
from scion.materials.hyperelastic_batched import BatchedMaterial

__all__ = [
    "PrincipalMaterial",
    "HenckyElastic",
    "FixedCorotated",
    "ReturnMap",
    "DruckerPragerSand",
    "VonMisesHencky",
    "PrincipalPlasticity",
    "PrincipalPlasticState",
    "ELASTIC_LAWS",
    "RETURN_MAPS",
]

#: Singular values below this are clamped before ``log``. An MPM particle can
#: be squashed arbitrarily flat by a bad step; ``log(0)`` would poison the whole
#: stress field rather than just that particle.
_MIN_STRETCH = 1.0e-6


def _reassemble(U: np.ndarray, diag: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``U diag(d) W^T`` for stacks."""
    return np.einsum("pij,pj,pkj->pik", U, diag, W)


# ---------------------------------------------------------------------------
# Elastic laws
# ---------------------------------------------------------------------------

class PrincipalMaterial(BatchedMaterial):
    """An isotropic law defined by its principal Kirchhoff stresses.

    Subclasses implement :meth:`principal_kirchhoff`, which receives the
    singular values and returns the three principal values of ``tau``. The base
    class owns the SVD and the rotation back, so a new isotropic model is one
    short method.

    Kirchhoff rather than Cauchy throughout, because ``tau = P F^T`` is what an
    MPM force loop wants: the ``1/J`` that would turn it into Cauchy cancels
    against the ``J`` in the volume update.
    """

    #: set by subclasses; used for the CFL wave speed
    mu: float
    lam: float

    @abstractmethod
    def principal_kirchhoff(self, s: np.ndarray) -> np.ndarray:
        """``(n, d) -> (n, d)`` principal Kirchhoff stresses from stretches."""

    @property
    def wave_speed_factor(self) -> float:
        """``sqrt(lambda + 2 mu)``; divide by ``sqrt(rho)`` for the P-wave speed."""
        return float(np.sqrt(self.lam + 2.0 * self.mu))

    def kirchhoff(self, F: np.ndarray) -> np.ndarray:
        """Kirchhoff stress ``tau = P F^T``. ``(n, d, d) -> (n, d, d)``."""
        U, s, Vt = sla.signed_svd(F)
        return _reassemble(U, self.principal_kirchhoff(s), U)

    def pk1(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First Piola-Kirchhoff stress and ``J``, for the hyperelastic path."""
        J = sla.det(F)
        tau = self.kirchhoff(F)
        # P = tau F^-T
        return np.einsum("pij,pkj->pik", tau, sla.inv(F, J)), J


class HenckyElastic(PrincipalMaterial):
    r"""St Venant-Kirchhoff elasticity in logarithmic strain.

    ``W = mu ||eps||^2 + (lambda/2) tr(eps)^2``  with ``eps = log(s)``, giving
    principal Kirchhoff stress ``tau_i = 2 mu eps_i + lambda tr(eps)``.

    This is the elastic law the sand and von Mises models below are built on,
    and the reason is that ``tau`` is *linear* in ``eps``: the yield surface is
    then a fixed cone or cylinder in principal-strain space rather than a
    surface that moves as the material deforms, so the return map is closed
    form.

    Not to be confused with :class:`~scion.materials.geometric_nonlinear.StVenantKirchhoff`,
    which uses Green strain. Under compression the two disagree strongly --
    Green-strain StVK famously loses stability past ``J ~ 0.6`` and offers no
    resistance to full collapse, whereas this one stiffens without bound as
    ``s -> 0`` because ``log(s) -> -inf``.
    """

    name = "hencky"

    def __init__(self, mu: float, lam: float):
        self.mu = float(mu)
        self.lam = float(lam)

    @classmethod
    def from_E_nu(cls, E: float, nu: float) -> "HenckyElastic":
        return cls(E / (2.0 * (1.0 + nu)),
                   E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    def principal_kirchhoff(self, s: np.ndarray) -> np.ndarray:
        eps = np.log(np.maximum(np.abs(s), _MIN_STRETCH))
        return 2.0 * self.mu * eps + self.lam * eps.sum(axis=1, keepdims=True)

    def energy(self, F: np.ndarray) -> np.ndarray:
        _, s, _ = sla.signed_svd(F)
        eps = np.log(np.maximum(np.abs(s), _MIN_STRETCH))
        tr = eps.sum(axis=1)
        return self.mu * (eps ** 2).sum(axis=1) + 0.5 * self.lam * tr ** 2


class FixedCorotated(PrincipalMaterial):
    r"""Fixed corotated elasticity (Stomakhin et al. 2012).

    ``W = mu sum_i (s_i - 1)^2 + (lambda/2)(J - 1)^2``, so

    ``tau_i = 2 mu s_i (s_i - 1) + lambda J (J - 1)``.

    The standard workhorse of graphics-style MPM. Its value over Neo-Hookean is
    behaviour under *inversion*: ``W`` stays finite and differentiable at
    ``J <= 0`` because it is built from the signed singular values rather than
    ``log J``, so an element that gets turned inside out by one bad step is
    pushed back instead of producing ``nan``. That robustness is why it
    tolerates the large time steps explicit MPM wants.

    The price is that it is not a physically-derived rubber model -- the
    quadratic in ``s_i`` is chosen for numerical behaviour. For a material whose
    finite-strain response actually matters, prefer Neo-Hookean or Hencky.
    """

    name = "fixed-corotated"

    def __init__(self, mu: float, lam: float):
        self.mu = float(mu)
        self.lam = float(lam)

    @classmethod
    def from_E_nu(cls, E: float, nu: float) -> "FixedCorotated":
        return cls(E / (2.0 * (1.0 + nu)),
                   E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    def principal_kirchhoff(self, s: np.ndarray) -> np.ndarray:
        J = np.prod(s, axis=1, keepdims=True)
        return 2.0 * self.mu * s * (s - 1.0) + self.lam * J * (J - 1.0)

    def energy(self, F: np.ndarray) -> np.ndarray:
        _, s, _ = sla.signed_svd(F)
        J = np.prod(s, axis=1)
        return self.mu * ((s - 1.0) ** 2).sum(axis=1) + 0.5 * self.lam * (J - 1.0) ** 2


# ---------------------------------------------------------------------------
# Return maps
# ---------------------------------------------------------------------------

class ReturnMap(ABC):
    """A projection of the principal log-strain vector onto an elastic domain.

    ``project(eps, log_jp) -> (eps, log_jp)``, both ``(n, d)`` and ``(n,)``.

    Only models whose yield surface is isotropic belong here: the projection
    acts on principal values alone and knows nothing about orientation, which
    is what makes it a 3-vector operation.
    """

    name: str

    @abstractmethod
    def project(self, eps: np.ndarray,
                log_jp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the admissible strain and the updated volume-correction term."""


class DruckerPragerSand(ReturnMap):
    r"""Drucker-Prager plasticity for granular media (Klar et al. 2016).

    Three branches, and the interesting one is the first:

    **Expansion** (``tr(eps) >= 0``). The material is being pulled apart. Sand
    carries no tension, so the whole strain is inadmissible and the state
    projects to the **cone tip** -- every stretch goes to ``exp(cohesion)`` and
    the stress is zero. The discarded volumetric strain is banked in ``log_jp``
    rather than thrown away, so a grain cloud that has been dilated and then
    recompressed has to close that volume again before it carries load. That
    bookkeeping is what makes a sand pile settle instead of springing back.

    **Inside the cone** (``delta_gamma <= 0``). Elastic; nothing happens.

    **Outside** . Return along the deviatoric direction to the cone surface.

    On the apex
    -----------
    This is a genuine admissible return at the tip, which is what the
    small-strain :class:`~scion.materials.plasticity.yield_criteria.DruckerPrager`
    return map does *not* have -- see the refusal in ``CLAUDE.md``. It works
    here only because the tip of the cone in log-strain space is a single point
    that the projection can name outright, and because ``log_jp`` gives the
    discarded volume somewhere to live. It does not lift that refusal: this is
    a different, finite-strain model, not a fix to the small-strain one.

    Parameters
    ----------
    friction_angle : in degrees. The cone half-angle; 30 degrees is the usual
        value for dry sand and is what CK-MPM hard-codes.
    cohesion : shifts the tip off the origin, allowing a little tension.
        Zero for dry sand.
    beta : fraction of the discarded expansion volume that is remembered in
        ``log_jp``. 1 keeps all of it, 0 discards it (no volume correction).
    """

    name = "drucker-prager-sand"

    def __init__(self, friction_angle: float = 30.0, cohesion: float = 0.0,
                 beta: float = 1.0, mu: float = 1.0, lam: float = 1.0):
        self.friction_angle = float(friction_angle)
        self.cohesion = float(cohesion)
        self.beta = float(beta)
        self.mu = float(mu)
        self.lam = float(lam)

        # sqrt(2/3) * 2 sin(phi) / (3 - sin(phi)): the inner-cone (compression
        # meridian) match to Mohr-Coulomb, in the sqrt(2/3)-scaled measure this
        # return map uses. At phi = 30 deg this reproduces CK-MPM's hard-coded
        # 0.816496580927726 * 2 * 0.5 / (3 - 0.5) exactly.
        sin_phi = np.sin(np.radians(self.friction_angle))
        self.alpha = float(np.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi))

    def project(self, eps, log_jp):
        eps = eps - self.cohesion
        eps_sum = eps.sum(axis=1)
        trace = eps_sum + log_jp

        dim = eps.shape[1]
        dev = eps - (trace / dim)[:, None]
        dev_norm = np.linalg.norm(dev, axis=1)

        out = eps.copy()
        new_log_jp = log_jp.copy()

        # branch 1: expansion -> project to the tip, bank the volume
        expanding = trace >= 0.0
        if np.any(expanding):
            out[expanding] = 0.0
            new_log_jp[expanding] = (self.beta * eps_sum[expanding]
                                     + log_jp[expanding])

        # branch 3: outside the cone -> return to the surface
        scaled_mu = 2.0 * self.mu
        delta = dev_norm + (dim * self.lam + scaled_mu) / scaled_mu * trace * self.alpha
        yielding = (~expanding) & (delta > 0.0) & (dev_norm > 0.0)
        if np.any(yielding):
            out[yielding] = (eps[yielding]
                             - (delta[yielding] / dev_norm[yielding])[:, None]
                             * dev[yielding])
            new_log_jp[yielding] = 0.0

        # branch 2 (inside the cone) leaves eps untouched but still clears the
        # banked volume: the material is carrying load again
        elastic = (~expanding) & ~yielding
        new_log_jp[elastic] = 0.0

        return out + self.cohesion, new_log_jp


class VonMisesHencky(ReturnMap):
    r"""J2 plasticity, perfectly plastic, in logarithmic strain.

    Radial return on the deviatoric log strain: the deviatoric Kirchhoff stress
    is ``2 mu dev(eps)``, so the admissible set is simply
    ``||dev(eps)|| <= tau_Y / (2 mu)`` and the projection scales the deviator
    back to that radius. No iteration, because the elastic law is linear in
    ``eps``.

    Yield convention
    ----------------
    ``yield_stress`` is compared against the **Frobenius norm** of the
    deviatoric Kirchhoff stress, which is CK-MPM's convention and is *not* the
    von Mises equivalent stress -- those differ by ``sqrt(3/2)``. Use
    :meth:`from_von_mises` to give a yield stress in the usual uniaxial sense
    and have the conversion done for you; passing a handbook ``sigma_y``
    straight into the constructor overpredicts strength by 22%.
    """

    name = "von-mises-hencky"

    def __init__(self, yield_stress: float, mu: float = 1.0):
        self.yield_stress = float(yield_stress)
        self.mu = float(mu)

    @classmethod
    def from_von_mises(cls, sigma_y: float, mu: float = 1.0) -> "VonMisesHencky":
        """Build from a uniaxial yield stress in the usual von Mises sense."""
        return cls(np.sqrt(2.0 / 3.0) * float(sigma_y), mu)

    def project(self, eps, log_jp):
        dim = eps.shape[1]
        dev = eps - eps.sum(axis=1, keepdims=True) / dim
        dev_norm = np.linalg.norm(dev, axis=1)

        excess = dev_norm - self.yield_stress / (2.0 * self.mu)
        yielding = (excess > 0.0) & (dev_norm > 0.0)

        out = eps.copy()
        if np.any(yielding):
            out[yielding] = (eps[yielding]
                             - (excess[yielding] / dev_norm[yielding])[:, None]
                             * dev[yielding])
        return out, log_jp


# ---------------------------------------------------------------------------
# The stateful model
# ---------------------------------------------------------------------------

@dataclass
class PrincipalPlasticState:
    """Per-particle history for :class:`PrincipalPlasticity`.

    ``F_e`` is the *elastic* deformation gradient, which is the state the return
    map acts on -- not the total ``F`` the stepper tracks. The two differ by the
    plastic part, and for sand they differ in volume as well, which is the whole
    point of ``log_jp``.
    """

    #: ``(n, 3, 3)`` elastic deformation gradient
    F_e: np.ndarray
    #: ``(n,)`` banked volumetric plastic strain (sand only; 0 otherwise)
    log_jp: np.ndarray

    @classmethod
    def identity(cls, n: int) -> "PrincipalPlasticState":
        return cls(F_e=sla.identity(n, 3), log_jp=np.zeros(n))

    def copy(self) -> "PrincipalPlasticState":
        return PrincipalPlasticState(self.F_e.copy(), self.log_jp.copy())

    @property
    def n(self) -> int:
        return int(self.log_jp.size)

    def __repr__(self) -> str:
        J = sla.det(self.F_e)
        return (f"PrincipalPlasticState({self.n} points, "
                f"J_e in [{J.min():.4g}, {J.max():.4g}], "
                f"max |log_jp| {np.abs(self.log_jp).max():.4g})")


class PrincipalPlasticity:
    """An elastic law plus a return map, on the stateful MPM path.

    ``update(F, L, dt, state) -> (sigma, state)``, matching
    :class:`~scion.materials.high_rate.JohnsonCookModel`. ``F`` is accepted and
    ignored: the elastic deformation gradient is carried in ``state`` and
    advanced by ``L`` here, because the return map *changes* it and the stepper
    has no way to write a correction back into ``particles.F``.

    The model never mutates the state it is handed.
    """

    def __init__(self, elastic: PrincipalMaterial, return_map: ReturnMap):
        self.elastic = elastic
        self.return_map = return_map
        # the return maps need the elastic constants to size their yield
        # surfaces; keep the two in step rather than making the caller do it
        for attr in ("mu", "lam"):
            if hasattr(return_map, attr):
                setattr(return_map, attr, getattr(elastic, attr))

    @property
    def name(self) -> str:
        return f"{self.elastic.name}+{self.return_map.name}"

    @property
    def wave_speed_factor(self) -> float:
        return self.elastic.wave_speed_factor

    def update(self, F: np.ndarray, L: np.ndarray, dt: float,
               state: PrincipalPlasticState):
        """Advance one step and return ``(cauchy_stress, new_state)``."""
        n = L.shape[0]
        if state is None:
            state = PrincipalPlasticState.identity(n)

        # elastic trial: push the stored F_e forward with the velocity gradient
        F_trial = np.einsum("pij,pjk->pik",
                            sla.identity(n, 3) + dt * L, state.F_e)

        U, s, Vt = sla.signed_svd(F_trial)
        eps = np.log(np.maximum(np.abs(s), _MIN_STRETCH))

        eps_new, log_jp = self.return_map.project(eps, state.log_jp)

        s_new = np.exp(eps_new)
        F_e = _reassemble(U, s_new, np.swapaxes(Vt, 1, 2))

        tau = _reassemble(U, self.elastic.principal_kirchhoff(s_new), U)
        J = np.prod(s_new, axis=1)
        sigma = tau / J[:, None, None]

        return sigma, PrincipalPlasticState(F_e=F_e, log_jp=log_jp)


ELASTIC_LAWS: dict[str, type[PrincipalMaterial]] = {
    "hencky": HenckyElastic,
    "fixed-corotated": FixedCorotated,
}

RETURN_MAPS: dict[str, type[ReturnMap]] = {
    "drucker-prager-sand": DruckerPragerSand,
    "von-mises-hencky": VonMisesHencky,
}
