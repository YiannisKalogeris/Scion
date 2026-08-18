r"""
Equivalent strain: the scalar that drives damage.

Axis 1. A damage model needs one number per point to compare against a
threshold, and the choice of that number *is* the failure theory. It decides
what counts as damaging: only tension, tension more than compression, or both
equally. Everything downstream -- the evolution law, the regularization, the
solver -- is indifferent to which one is in use.

The loading surface is

    f(eps, kappa) = eps_eq(eps) - kappa <= 0,

with ``kappa`` the largest ``eps_eq`` ever reached. Kuhn-Tucker, exactly as in
plasticity, and the reason damage is path-dependent even though the stress is an
explicit function of the current strain and ``kappa``.

The derivative is not optional
------------------------------
``deps_eq/deps`` enters the consistent tangent as a rank-one term. It is also
the term that makes that tangent *unsymmetric*, and the term everyone drops --
after which Newton still converges, linearly, and the only symptom is the
iteration count. Every measure here supplies it analytically and every one has a
finite-difference test.

Always full 3-D
---------------
Measures take ``(n, 3, 3)`` strain tensors whatever the element dimension, for
the same reason the plasticity criteria do. In plane strain ``eps_33 = 0`` but
the deviator's ``e_33 = -tr(eps)/3`` is not, and ``J2`` is built from the
deviator -- so a measure evaluated on the 2-D tensor gives a different, wrong
number. Note this differs from :mod:`simulacrum.materials.splits`, which is
written dimension-generically and is called by the damage model on the 3-D
tensor for the same reason.

Elastic constants
-----------------
Two of the measures need ``nu`` (and one needs ``E``). They are supplied by the
model through :meth:`EquivalentStrain.bind`, which fills them in and refuses a
measure that was constructed with constants contradicting the model's -- a
mismatch that otherwise silently shifts the compression response.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "EquivalentStrain",
    "Mazars",
    "ModifiedVonMises",
    "Rankine",
    "EnergyRelease",
    "EQUIVALENT_REGISTRY",
    "get_equivalent_strain",
]

_I = np.eye(3)


def _principal(eps: np.ndarray):
    """Eigenvalues and eigenvectors of a batch of symmetric tensors.

    ``eigh`` returns them ascending, with the eigenvectors as columns, so
    ``v[..., :, i]`` belongs to ``w[..., i]``.
    """
    return np.linalg.eigh(np.asarray(eps, dtype=float))


def _outer_eigenbasis(v: np.ndarray) -> np.ndarray:
    """``(n, 3, 3, 3)``: the projector ``n_i (x) n_i`` for each eigenvalue.

    ``dw_i/deps = n_i (x) n_i`` for a simple eigenvalue, which is what every
    principal-value derivative in this module is built from. At a repeated
    eigenvalue the individual projectors are not unique, but the sums that use
    them here are -- see the note in :meth:`Mazars.dvalue_deps`.
    """
    return np.einsum("nia,nja->naij", v, v)


class EquivalentStrain(ABC):
    """``eps_eq(eps)`` and its derivative."""

    name: str = "equivalent-strain"

    #: True when ``deps_eq/deps`` is parallel to the effective stress, which
    #: makes the damage model's rank-one tangent term symmetric. Only the
    #: energy measure has this property; for the others the tangent is
    #: unsymmetric and must be stored and factorised as such.
    symmetric_rank_one: bool = False

    #: elastic constants the measure's own formula reads. ``bind`` stores both
    #: regardless -- :meth:`uniaxial` needs ``nu`` to build a uniaxial *stress*
    #: state even for a measure whose formula does not use it -- but only these
    #: are required to be present before :meth:`value` is called.
    needs: tuple[str, ...] = ()

    def __init__(self, E: float | None = None, nu: float | None = None):
        self.E = None if E is None else float(E)
        self.nu = None if nu is None else float(nu)

    # ------------------------------------------------------------------
    def bind(self, E: float, nu: float) -> None:
        """Supply the model's elastic constants.

        Fills whatever was left unset and rejects a conflict. A measure carrying
        ``nu = 0.2`` inside a model built with ``nu = 0.3`` is not a harmless
        inconsistency: ``nu`` sets how much of a compressive state the modified
        von Mises measure counts, so the two would disagree about when the
        material damages while reporting the same stiffness.
        """
        for name, value in (("E", E), ("nu", nu)):
            held = getattr(self, name)
            if held is None:
                setattr(self, name, float(value))
            elif not np.isclose(held, value, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"{type(self).__name__} was built with {name}={held:g} but "
                    f"the model has {name}={value:g}; they must agree")

    def _require(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"{type(self).__name__} needs {name}; pass it to the "
                f"constructor or let DamageModel bind it")
        return float(value)

    # ------------------------------------------------------------------
    @abstractmethod
    def value(self, eps: np.ndarray) -> np.ndarray:
        """``eps_eq``, ``(n,)``. Non-negative."""

    @abstractmethod
    def dvalue_deps(self, eps: np.ndarray) -> np.ndarray:
        """``deps_eq/deps``, ``(n, 3, 3)``, symmetric."""

    # ------------------------------------------------------------------
    def uniaxial(self, strain) -> np.ndarray:
        """``eps_eq`` for a uniaxial *stress* state of the given axial strain.

        The calibration state: every measure here is normalised so that this
        returns the axial strain itself in tension. Comparing measures on
        anything else means comparing two different normalisations as well as
        two different theories.
        """
        e = np.atleast_1d(np.asarray(strain, dtype=float))
        nu = self._require("nu")
        eps = np.zeros((e.size, 3, 3))
        eps[:, 0, 0] = e
        eps[:, 1, 1] = eps[:, 2, 2] = -nu * e
        return self.value(eps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------

class Mazars(EquivalentStrain):
    r"""``eps_eq = sqrt( sum_i <eps_i>_+^2 )``, over the principal strains.

    Only tensile principal strains count, so a state in pure hydrostatic
    compression is undamaging however large. That is the right first statement
    for concrete and for any material that fails by opening microcracks
    perpendicular to the tensile direction.

    Its weakness is the same property: it makes shear look purely tensile. Pure
    shear has principal strains ``(g/2, 0, -g/2)`` and returns ``g/2``, the same
    as uniaxial tension of ``g/2`` -- no distinction between opening and
    sliding.

        ``deps_eq/deps = (1/eps_eq) sum_i <eps_i>_+ n_i (x) n_i``

    Non-differentiable at ``eps_eq = 0``, where the surface has a vertex; the
    derivative is reported as zero there, which is the correct subgradient
    choice for a point that is strictly inside the elastic domain and can
    therefore never be the linearisation point of a loading step.
    """

    name = "mazars"

    def value(self, eps):
        w, _ = _principal(eps)
        return np.sqrt(np.sum(np.maximum(w, 0.0) ** 2, axis=-1))

    def dvalue_deps(self, eps):
        w, v = _principal(eps)
        pos = np.maximum(w, 0.0)
        eq = np.sqrt(np.sum(pos ** 2, axis=-1))
        # A repeated eigenvalue makes the individual n_i (x) n_i ambiguous, but
        # equal eigenvalues carry equal weights <eps_i>_+, so the weighted sum
        # collapses onto the projector of the whole eigenspace and is unique.
        proj = _outer_eigenbasis(v)
        num = np.einsum("na,naij->nij", pos, proj)
        safe = eq > 0.0
        return np.where(safe[:, None, None],
                        num / np.where(safe, eq, 1.0)[:, None, None], 0.0)


class ModifiedVonMises(EquivalentStrain):
    r"""De Vree et al. (1995). One parameter, ``k = f_c / f_t``.

        A = (k-1) / (2k(1-2nu)),   B = ((k-1)/(1-2nu))^2,   C = 12k/(1+nu)^2

        eps_eq = A I1 + (1/2k) sqrt( B I1^2 + C J2 )

    with ``I1 = tr(eps)`` and ``J2 = (1/2) e:e`` on the strain deviator.

    Calibrated so that uniaxial tension of ``eps`` returns ``eps`` and uniaxial
    compression of ``eps`` returns ``eps/k``: the material takes ``k`` times more
    compressive strain to reach the same damage. For concrete ``k`` is 10 to 15.

    This is the measure to use when compression matters at all. Setting
    ``k = 1`` recovers a symmetric von Mises measure that damages equally either
    way -- occasionally what you want for a metal, never for concrete.

        ``deps_eq/deps = A I + (2 B I1 I + C e) / (4k sqrt(B I1^2 + C J2))``
    """

    name = "modified-von-mises"
    needs = ("nu",)

    def __init__(self, k: float = 10.0, nu: float | None = None):
        super().__init__(nu=nu)
        if k <= 0.0:
            raise ValueError("k = f_c/f_t must be positive")
        self.k = float(k)

    def _coefficients(self):
        nu, k = self._require("nu"), self.k
        if np.isclose(nu, 0.5):
            raise ValueError(
                "the modified von Mises measure divides by (1 - 2 nu) and is "
                "undefined for an incompressible material")
        A = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
        B = ((k - 1.0) / (1.0 - 2.0 * nu)) ** 2
        C = 12.0 * k / (1.0 + nu) ** 2
        return A, B, C, k

    @staticmethod
    def _invariants(eps):
        eps = np.asarray(eps, dtype=float)
        I1 = np.einsum("nii->n", eps)
        e = eps - (I1 / 3.0)[:, None, None] * _I
        J2 = 0.5 * np.einsum("nij,nij->n", e, e)
        return I1, e, J2

    def value(self, eps):
        A, B, C, k = self._coefficients()
        I1, _, J2 = self._invariants(eps)
        return A * I1 + np.sqrt(B * I1 ** 2 + C * J2) / (2.0 * k)

    def dvalue_deps(self, eps):
        A, B, C, k = self._coefficients()
        I1, e, J2 = self._invariants(eps)
        root = np.sqrt(B * I1 ** 2 + C * J2)
        safe = root > 0.0
        denom = 4.0 * k * np.where(safe, root, 1.0)
        term = (2.0 * B * I1[:, None, None] * _I + C * e) / denom[:, None, None]
        return A * _I + np.where(safe[:, None, None], term, 0.0)


class Rankine(EquivalentStrain):
    r"""``eps_eq = <sigma_1>_+ / E``, the largest principal *effective* stress.

    Effective meaning the undamaged stress ``sigma_0 = C_0 : eps``, divided by
    ``E`` so the measure is a strain and shares the calibration of the others.

    The most brittle member: it looks at one principal direction and ignores the
    other two entirely, so it cracks as soon as any direction reaches the
    tensile limit. That is the right statement for a maximum-stress fracture
    criterion and the wrong one for a material with any biaxial strength
    interaction.

    Note it is *not* the same as taking the largest principal strain. In plane
    strain the Poisson coupling makes the largest principal stress and the
    largest principal strain occur in the same direction but with different
    biaxial dependence, and the two measures disagree by ``1 + nu`` factors in
    biaxial states.
    """

    name = "rankine"
    needs = ("E", "nu")

    def _lame(self):
        E, nu = self._require("E"), self._require("nu")
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return E, lam, mu

    def value(self, eps):
        E, lam, mu = self._lame()
        eps = np.asarray(eps, dtype=float)
        tr = np.einsum("nii->n", eps)
        sigma = lam * tr[:, None, None] * _I + 2.0 * mu * eps
        w, _ = np.linalg.eigh(sigma)
        return np.maximum(w[:, -1], 0.0) / E

    def dvalue_deps(self, eps):
        E, lam, mu = self._lame()
        eps = np.asarray(eps, dtype=float)
        tr = np.einsum("nii->n", eps)
        sigma = lam * tr[:, None, None] * _I + 2.0 * mu * eps
        w, v = np.linalg.eigh(sigma)

        # d sigma_1 / d eps = (n_1 (x) n_1) : C_0, and C_0 is isotropic, so
        # this is lam I + 2 mu (n_1 (x) n_1).
        n1 = v[:, :, -1]
        proj = np.einsum("ni,nj->nij", n1, n1)
        d = (lam * _I + 2.0 * mu * proj) / E
        active = w[:, -1] > 0.0
        return np.where(active[:, None, None], d, 0.0)


class EnergyRelease(EquivalentStrain):
    r"""``eps_eq = sqrt(2 Psi_0 / E)``, Simo and Ju (1987).

    The energy-based measure: damage is driven by the elastic energy the point
    is storing, without asking how it is distributed among the principal
    directions.

        ``Psi_0 = (1/2) eps : C_0 : eps``
        ``deps_eq/deps = sigma_0 / (E eps_eq)``

    Symmetric in tension and compression, since the energy is quadratic. That
    makes it the natural choice for a metal and a poor one for concrete, and it
    is also why it is the only measure here whose derivative is *parallel* to
    the effective stress -- which makes the consistent tangent symmetric, unlike
    every other measure in this module.
    """

    name = "energy-release"
    needs = ("E", "nu")
    symmetric_rank_one = True

    def _lame(self):
        E, nu = self._require("E"), self._require("nu")
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return E, lam, mu

    def _stress(self, eps):
        _, lam, mu = self._lame()
        eps = np.asarray(eps, dtype=float)
        tr = np.einsum("nii->n", eps)
        return lam * tr[:, None, None] * _I + 2.0 * mu * eps

    def value(self, eps):
        E = self._require("E")
        sigma = self._stress(eps)
        psi = 0.5 * np.einsum("nij,nij->n", sigma, np.asarray(eps, dtype=float))
        return np.sqrt(2.0 * np.maximum(psi, 0.0) / E)

    def dvalue_deps(self, eps):
        E = self._require("E")
        sigma = self._stress(eps)
        eq = self.value(eps)
        safe = eq > 0.0
        return np.where(safe[:, None, None],
                        sigma / (E * np.where(safe, eq, 1.0))[:, None, None],
                        0.0)


# ---------------------------------------------------------------------------

EQUIVALENT_REGISTRY: dict[str, type[EquivalentStrain]] = {
    "mazars": Mazars,
    "positive-principal": Mazars,
    "modified-von-mises": ModifiedVonMises,
    "de-vree": ModifiedVonMises,
    "rankine": Rankine,
    "max-principal-stress": Rankine,
    "energy-release": EnergyRelease,
    "simo-ju": EnergyRelease,
}


def get_equivalent_strain(name: str, **kwargs) -> EquivalentStrain:
    key = name.lower().replace("_", "-")
    if key not in EQUIVALENT_REGISTRY:
        raise NotImplementedError(
            f"Unknown equivalent strain measure '{name}'. Available: "
            f"{sorted(set(EQUIVALENT_REGISTRY))}")
    return EQUIVALENT_REGISTRY[key](**kwargs)
