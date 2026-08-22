r"""
Linear isotropic micropolar (Cosserat) elasticity, in the material frame.

A micropolar continuum attaches a triad of *directors* to every material point
and lets it rotate independently of the deformation. That buys two things a
classical continuum cannot have: an intrinsic length scale, so the response
depends on the size of the body relative to its microstructure, and a stress
tensor that is **not required to be symmetric**, because couples can balance the
antisymmetric part.

Two deformation measures, and both are non-symmetric
---------------------------------------------------
``E = Q^T F - I``     first Cosserat deformation measure ("stretch")
``Gamma``             wryness, the material gradient of the microrotation

``E`` reduces to the Biot strain ``U - I`` when the microrotation follows the
macrorotation (``Q = R``), *not* to the Green-Lagrange strain, so the classical
limit of this model is Biot elasticity and not St Venant-Kirchhoff. Neither
measure is symmetric: ``skew(E)`` is exactly the lag between the micro- and
macro-rotation, and discarding it removes the entire micropolar effect.

Constitutive law (Neff et al.; O'Hare et al. Eqs. 20-21)
--------------------------------------------------------
    B = lam tr(E) I + mu(1 + a) E + mu(1 - a) E^T          = D : E
    S = alpha tr(Gam) I + (beta + gam) Gam + (beta - gam) Gam^T = Dt : Gam

``B`` and ``S`` are *Biot-like*: they are conjugate to ``E`` and ``Gamma`` and
push forward to the Cauchy measures as

    sigma = J^-1 Q B F^T ,      m = J^-1 Q S F^T .

Defining the law in the reference frame is what makes objectivity automatic --
``E`` and ``Gamma`` are both invariant under a superposed spatial rotation
(``F -> RF``, ``Q -> RQ``), so ``B`` and ``S`` are too, and the push-forward
supplies the rotation exactly once.

Technical constants
-------------------
The raw constants ``a, alpha, beta, gam`` have no direct physical reading;
:meth:`from_technical` takes the engineering set instead (Lakes 2016):

    N        coupling number, 0 to 1. Degree of microrotation-displacement
             coupling. ``a = N^2/(1 - N^2)`` **diverges as N -> 1**, which is
             the constrained-rotation (couple-stress) limit where ``Q`` is
             slaved to ``R``; approach it by reducing the tolerance on the
             solve, not by setting N = 1.
    l_t, l_b characteristic lengths in torsion and bending, in metres. These
             are what make the model size-dependent, and a body many
             characteristic lengths across behaves classically -- so a test at
             ``l/L << 1`` cannot tell this model from the classical one.
    psi      polar ratio, 0 to 1.5.

Setting ``a = alpha = beta = gam = 0`` recovers the Cauchy-Boltzmann continuum:
with no couple stress the angular momentum balance forces ``sigma`` symmetric,
which forces ``Q = R``. That reduction is asserted in the tests rather than
assumed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MicropolarHooke"]


def _isotropic_fourth(c_trace: float, c_direct: float,
                      c_transpose: float) -> np.ndarray:
    """``C_abgp = c_tr d_ab d_gp + c_di d_ag d_bp + c_tp d_ap d_bg``.

    The three isotropic invariants of a fourth-order tensor acting on a
    *non-symmetric* second-order one. A symmetric-strain theory cannot
    distinguish the last two -- which is exactly why a micropolar law needs
    both, and why reusing an ordinary elasticity tensor here silently
    symmetrises ``E`` and deletes the micropolar coupling.
    """
    d = np.eye(3)
    return (c_trace * np.einsum("ab,gp->abgp", d, d)
            + c_direct * np.einsum("ag,bp->abgp", d, d)
            + c_transpose * np.einsum("ap,bg->abgp", d, d))


class MicropolarHooke:
    """Linear isotropic centrosymmetric micropolar solid.

    Parameters
    ----------
    lam, mu : Lame parameters
    a : microrotation coupling, ``N^2/(1 - N^2)``. ``a = 0`` decouples.
    alpha, beta, gam : curvature moduli. All three zero removes the couple
        stress entirely, and with it the internal length scale.
    """

    def __init__(self, lam: float, mu: float, a: float = 0.0,
                 alpha: float = 0.0, beta: float = 0.0, gam: float = 0.0):
        self.lam, self.mu = float(lam), float(mu)
        self.a = float(a)
        self.alpha, self.beta, self.gam = float(alpha), float(beta), float(gam)
        if self.mu <= 0.0:
            raise ValueError(f"mu must be positive, got {mu}")
        if self.a < 0.0:
            raise ValueError(
                f"a = N^2/(1 - N^2) is non-negative for N in [0, 1); got {a}")

    # ------------------------------------------------------------------
    @classmethod
    def from_technical(cls, E: float, nu: float, N: float = 0.0,
                       l_t: float = 0.0, l_b: float = 0.0,
                       psi: float = 1.0) -> "MicropolarHooke":
        """Build from Young's modulus, Poisson's ratio and the engineering
        micropolar constants (O'Hare et al. Table 1 and Eqs. 22-25)."""
        if not 0.0 <= N < 1.0:
            raise ValueError(
                f"the coupling number N must lie in [0, 1); got {N}. N = 1 is "
                f"the constrained-rotation limit, where a = N^2/(1 - N^2) is "
                f"unbounded and the microrotation is no longer independent -- "
                f"that is couple-stress theory and a different formulation")
        if not 0.0 < psi <= 1.5:
            raise ValueError(f"the polar ratio psi must lie in (0, 1.5]; "
                             f"got {psi}")
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return cls(lam=lam, mu=mu,
                   a=N ** 2 / (1.0 - N ** 2),
                   alpha=2.0 * mu * l_t ** 2 * (1.0 - psi) / psi,
                   beta=mu * l_t ** 2,
                   gam=mu * (4.0 * l_b ** 2 - l_t ** 2))

    # ------------------------------------------------------------------
    @property
    def D(self) -> np.ndarray:
        """``(3, 3, 3, 3)`` stretch moduli, ``B = D : E``. Constant."""
        return _isotropic_fourth(self.lam, self.mu * (1.0 + self.a),
                                 self.mu * (1.0 - self.a))

    @property
    def D_curvature(self) -> np.ndarray:
        """``(3, 3, 3, 3)`` curvature moduli, ``S = D_curvature : Gamma``."""
        return _isotropic_fourth(self.alpha, self.beta + self.gam,
                                 self.beta - self.gam)

    @property
    def length_bending(self) -> float:
        """``sqrt((beta + gam)/(4 mu))``, the bending characteristic length."""
        return float(np.sqrt(max(self.beta + self.gam, 0.0) / (4.0 * self.mu)))

    @property
    def length_torsion(self) -> float:
        """``sqrt(beta/mu)``, the torsional characteristic length."""
        return float(np.sqrt(max(self.beta, 0.0) / self.mu))

    @property
    def coupling_number(self) -> float:
        """``N = sqrt(a/(1 + a))``. Zero means the microrotation does no work
        against the stretch and the model degenerates to two uncoupled fields."""
        return float(np.sqrt(self.a / (1.0 + self.a)))

    @property
    def wave_speed_factor(self) -> float:
        """``sqrt(lam + 2 mu)``; divide by ``sqrt(rho)`` for the P-wave speed.
        Present so a micropolar model can be handed to the explicit stepper's
        CFL estimate; the curvature moduli add a second, faster micro-inertial
        wave that this does not see."""
        return float(np.sqrt(self.lam + 2.0 * self.mu))

    # ------------------------------------------------------------------
    def stresses(self, E: np.ndarray,
                 Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(E, Gamma) -> (B, S)``, both ``(n, 3, 3)`` and non-symmetric.

        Written out rather than contracted with :attr:`D` because the isotropic
        form is three terms and the contraction is 81 multiplies per point.
        The two must agree, and the tests check that they do.
        """
        E = np.asarray(E, dtype=float)
        Gamma = np.asarray(Gamma, dtype=float)
        eye = np.eye(3)
        trE = np.einsum("nii->n", E)
        trG = np.einsum("nii->n", Gamma)
        B = (self.lam * trE[:, None, None] * eye
             + self.mu * (1.0 + self.a) * E
             + self.mu * (1.0 - self.a) * E.transpose(0, 2, 1))
        S = (self.alpha * trG[:, None, None] * eye
             + (self.beta + self.gam) * Gamma
             + (self.beta - self.gam) * Gamma.transpose(0, 2, 1))
        return B, S

    def energy(self, E: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
        """``W = (E : D : E + Gamma : Dt : Gamma) / 2``, per reference volume."""
        B, S = self.stresses(E, Gamma)
        return 0.5 * (np.einsum("nij,nij->n", B, E)
                      + np.einsum("nij,nij->n", S, Gamma))

    def __repr__(self) -> str:
        return (f"MicropolarHooke(lam={self.lam:.4g}, mu={self.mu:.4g}, "
                f"N={self.coupling_number:.3g}, l_b={self.length_bending:.3g}, "
                f"l_t={self.length_torsion:.3g})")


class MicropolarTransverselyIsotropic(MicropolarHooke):
    r"""Cosserat elasticity with one preferred direction.

    Why it exists
    -------------
    ``skew(Sigma)`` -- the configurational torque, and the density whose contour
    integral is the **L-integral** of Knowles & Sternberg -- vanishes
    *identically* for an isotropic solid. So the quantity that distinguishes a
    microstructured medium from a classical one is exactly zero until the
    material has an orientation to rotate. This is the cheapest law that gives
    it one while remaining a Cosserat continuum.

    Note the distinction from
    :class:`~scion.materials.hyperelastic_batched.TransverselyIsotropic`, which
    is *classical*: it also has a nonzero torque, so anisotropy alone is what
    makes ``skew(Sigma)`` nonzero and micropolarity is **not** what produces it.
    What micropolarity adds here is the second constant, ``k_fibre_curvature``,
    which makes the *couple*-stress response anisotropic too -- bending about
    the fibre differs from bending across it -- and that has no classical
    counterpart at all.

    The construction
    ----------------
    Quadratic, like the isotropic law, so ``D`` stays constant and the fast
    tangent path is untouched. With the structural tensor ``M = a (x) a``:

        W = W_iso(E, Gamma) + k_f (a.E.a)^2 / 2 + k_c (a.Gamma.a)^2 / 2

    giving ``D += k_f M (x) M`` and ``D_curvature += k_c M (x) M``. Both added
    terms are invariant under rotations about ``a``, which is transverse
    isotropy, and both vanish with their constant so the isotropic law is
    recovered exactly rather than approximately.

    One constant each rather than a complete transversely isotropic basis. For a
    *non-symmetric* argument -- and the Cosserat strain is not symmetric -- that
    basis has considerably more members than the classical five, and every extra
    one is a modulus nobody has measured for a foam. These two are the
    interpretable ones: stiffness along the fibre, and stiffness against
    twisting about it.

    Parameters
    ----------
    fibre : ``(3,)`` preferred direction in the **reference** configuration.
        Normalised on construction. It is a *material* direction, so it convects
        with the deformation rather than staying fixed in space -- which is what
        makes rotating it a meaningful configurational operation.
    k_fibre : extra stiffness for stretching along ``a``.
    k_fibre_curvature : extra stiffness for wryness about ``a``.
    """

    def __init__(self, lam: float, mu: float, fibre, k_fibre: float = 0.0,
                 k_fibre_curvature: float = 0.0, a: float = 0.0,
                 alpha: float = 0.0, beta: float = 0.0, gam: float = 0.0):
        super().__init__(lam=lam, mu=mu, a=a, alpha=alpha, beta=beta, gam=gam)
        f = np.asarray(fibre, dtype=float)
        if f.shape != (3,):
            raise ValueError(f"fibre must be a 3-vector, got {f.shape}")
        n = np.linalg.norm(f)
        if n == 0.0:
            raise ValueError("the fibre direction cannot be the zero vector")
        self.fibre = f / n
        self.k_fibre = float(k_fibre)
        self.k_fibre_curvature = float(k_fibre_curvature)
        if self.k_fibre < 0.0 or self.k_fibre_curvature < 0.0:
            raise ValueError(
                f"fibre stiffnesses must be non-negative, got "
                f"{k_fibre} and {k_fibre_curvature}: a negative one subtracts "
                f"energy along the preferred direction and can make the law "
                f"non-elliptic without any obvious symptom")

    @classmethod
    def from_technical(cls, E: float, nu: float, fibre, k_fibre: float = 0.0,
                       k_fibre_curvature: float = 0.0, N: float = 0.0,
                       l_t: float = 0.0, l_b: float = 0.0, psi: float = 1.0):
        """As :meth:`MicropolarHooke.from_technical`, plus the fibre."""
        iso = MicropolarHooke.from_technical(E=E, nu=nu, N=N, l_t=l_t,
                                             l_b=l_b, psi=psi)
        return cls(lam=iso.lam, mu=iso.mu, fibre=fibre, k_fibre=k_fibre,
                   k_fibre_curvature=k_fibre_curvature, a=iso.a,
                   alpha=iso.alpha, beta=iso.beta, gam=iso.gam)

    @property
    def _MM(self) -> np.ndarray:
        a = self.fibre
        return np.einsum("i,j,k,l->ijkl", a, a, a, a)

    @property
    def D(self) -> np.ndarray:
        return super().D + self.k_fibre * self._MM

    @property
    def D_curvature(self) -> np.ndarray:
        return super().D_curvature + self.k_fibre_curvature * self._MM

    def stresses(self, E: np.ndarray,
                 Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(E, Gamma) -> (B, S)``. The isotropic part written out, the fibre
        part added on; the tests check both against the ``D`` contraction."""
        B, S = super().stresses(E, Gamma)
        a = self.fibre
        M = np.einsum("i,j->ij", a, a)
        if self.k_fibre:
            aEa = np.einsum("i,nij,j->n", a, np.asarray(E, dtype=float), a)
            B = B + self.k_fibre * aEa[:, None, None] * M
        if self.k_fibre_curvature:
            aGa = np.einsum("i,nij,j->n", a, np.asarray(Gamma, dtype=float), a)
            S = S + self.k_fibre_curvature * aGa[:, None, None] * M
        return B, S

    def rotated(self, rotation) -> "MicropolarTransverselyIsotropic":
        """The same material with its fibre turned by ``rotation`` ``(3, 3)``.

        The operation the configurational torque is conjugate to: it is what
        ``-1/2 dW/d(material orientation)`` differentiates with respect to.
        """
        return MicropolarTransverselyIsotropic(
            lam=self.lam, mu=self.mu,
            fibre=np.asarray(rotation, dtype=float) @ self.fibre,
            k_fibre=self.k_fibre,
            k_fibre_curvature=self.k_fibre_curvature,
            a=self.a, alpha=self.alpha, beta=self.beta, gam=self.gam)

    def __repr__(self) -> str:
        return (f"MicropolarTransverselyIsotropic(mu={self.mu:.4g}, "
                f"N={self.coupling_number:.3g}, "
                f"l_b={self.length_bending:.3g}, "
                f"fibre={np.array2string(self.fibre, precision=3)}, "
                f"k_fibre={self.k_fibre:.4g}, "
                f"k_fibre_curvature={self.k_fibre_curvature:.4g})")
