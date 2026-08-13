r"""
Strain-energy splits for phase-field fracture.

Axis 1 of the four orthogonal choices that define a phase-field model. A split
decomposes the elastic energy density

    Psi_0(eps) = 1/2 lam tr(eps)^2 + mu tr(eps^2)

into a part that may drive damage and a part that may not,

    Psi_0 = Psi_plus + Psi_minus,

so that cracks form in tension but not in compression, and crack faces do not
interpenetrate on closure. Without a split (``NoSplit``) the model cracks under
compression -- the two symmetric branches of the classic shear benchmark.

Everything here is batched over quadrature points: strains arrive as
``(n, d, d)`` and stresses/tangents leave as ``(n, d, d)`` / ``(n, d, d, d, d)``.

Conventions
-----------
* 2-D means **plane strain**: eps_33 = 0, so tr(eps) = eps_xx + eps_yy and the
  in-plane response is complete. The n-dimensional bulk modulus
  ``K_n = lam + 2 mu / n`` and the n-dimensional deviator are used, with n the
  spatial dimension -- not a hardcoded 3. Mixing those up rescales the
  volumetric response and is invisible until you compare load levels.
* Tangents are returned with both minor symmetries, so the Voigt form built
  from them is symmetric.

Adding a split means implementing :meth:`Split.energies` and
:meth:`Split.stress_tangent` and registering the class. Directional,
star-convex and no-tension splits all fit this interface unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Split",
    "NoSplit",
    "VolumetricDeviatoric",
    "Spectral",
    "SPLIT_REGISTRY",
    "get_split",
    "voigt_tangent",
    "voigt_stress",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(n: int, dim: int, dtype=np.float64) -> np.ndarray:
    return np.tile(np.eye(dim, dtype=dtype), (n, 1, 1))


def _sym_identity(dim: int) -> np.ndarray:
    """Fourth-order symmetric identity, I_ijkl = (d_ik d_jl + d_il d_jk)/2."""
    I = np.eye(dim)
    return 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))


def _pos(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.abs(a))


def _neg(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a - np.abs(a))


def _heaviside(a: np.ndarray) -> np.ndarray:
    """H(a), with H(0) = 0. Only the measure-zero case differs between
    conventions; 0 keeps the undamaged state stress-free at eps = 0."""
    return (a > 0.0).astype(float)


_VOIGT = {2: [(0, 0), (1, 1), (0, 1)], 3: [(0, 0), (1, 1), (2, 2),
                                           (1, 2), (0, 2), (0, 1)]}


def voigt_stress(sigma: np.ndarray) -> np.ndarray:
    """``(n, d, d)`` tensor -> ``(n, n_voigt)`` vector."""
    idx = _VOIGT[sigma.shape[1]]
    return np.stack([sigma[:, i, j] for i, j in idx], axis=1)


def voigt_tangent(C4: np.ndarray) -> np.ndarray:
    """``(n, d, d, d, d)`` -> ``(n, n_voigt, n_voigt)``.

    Engineering shear convention: the strain vector carries ``gamma = 2 eps_xy``.
    No factor of 2 appears here -- the two off-diagonal slots ``(k,l)`` and
    ``(l,k)`` of the tensor already sum to twice the component, and ``gamma``
    absorbs exactly that. Doubling as well is the classic Voigt bug: it gives a
    shear stiffness twice too large, which still looks plausible.
    """
    dim = C4.shape[1]
    idx = _VOIGT[dim]
    nv = len(idx)
    out = np.zeros((C4.shape[0], nv, nv))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            out[:, a, b] = C4[:, i, j, k, l]
    return out


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Split(ABC):
    """Additive decomposition of the elastic energy density."""

    name: str = "split"

    @abstractmethod
    def energies(self, eps: np.ndarray, lam: float,
                 mu: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(psi_plus, psi_minus)``, each ``(n,)``."""

    @abstractmethod
    def stress_tangent(self, eps: np.ndarray, lam: float, mu: float
                       ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
        """Return ``(sigma_plus, sigma_minus, C_plus, C_minus)``.

        ``sigma_pm = d Psi_pm / d eps`` and ``C_pm = d sigma_pm / d eps``, so
        that a caller can assemble ``g(d) sigma_plus + sigma_minus`` for the
        anisotropic coupling or ``sigma_plus + sigma_minus`` for the isotropic
        one without knowing which split it holds.
        """

    # ------------------------------------------------------------------
    def total_energy(self, eps: np.ndarray, lam: float,
                     mu: float) -> np.ndarray:
        """Psi_0. Independent of the split -- used to check consistency."""
        tr = np.einsum("nii->n", eps)
        return 0.5 * lam * tr ** 2 + mu * np.einsum("nij,nij->n", eps, eps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class NoSplit(Split):
    """No decomposition: Psi_plus = Psi_0, Psi_minus = 0.

    The isotropic model of Bourdin et al. Cheapest, and the momentum equation
    stays linear -- but it cracks in compression and lets crack faces
    interpenetrate. Keep it as the reference and the cheap debugging path, not
    as a production model.
    """

    name = "none"

    def energies(self, eps, lam, mu):
        psi = self.total_energy(eps, lam, mu)
        return psi, np.zeros_like(psi)

    def stress_tangent(self, eps, lam, mu):
        n, dim = eps.shape[0], eps.shape[1]
        I = _identity(n, dim)
        tr = np.einsum("nii->n", eps)
        sigma = lam * tr[:, None, None] * I + 2.0 * mu * eps

        Id = np.eye(dim)
        C = (lam * np.einsum("ij,kl->ijkl", Id, Id)
             + 2.0 * mu * _sym_identity(dim))
        C = np.broadcast_to(C, (n,) + C.shape).copy()
        return sigma, np.zeros_like(sigma), C, np.zeros_like(C)


class VolumetricDeviatoric(Split):
    r"""Amor, Marigo & Maurini (2009): volumetric / deviatoric decomposition.

        Psi_plus  = 1/2 K_n <tr eps>_+^2 + mu (eps_dev : eps_dev)
        Psi_minus = 1/2 K_n <tr eps>_-^2

    with ``K_n = lam + 2 mu / n`` and ``eps_dev = eps - tr(eps) I / n``.

    Cheaper than the spectral split and its tangent is piecewise constant --
    only the sign of ``tr eps`` switches. The known weakness: the deviatoric
    energy is degraded even when all principal strains are negative, so cracking
    can appear in strongly compressed regions.
    """

    name = "amor"

    def _parts(self, eps, lam, mu):
        dim = eps.shape[1]
        K = lam + 2.0 * mu / dim
        tr = np.einsum("nii->n", eps)
        I = _identity(eps.shape[0], dim)
        dev = eps - (tr / dim)[:, None, None] * I
        return K, tr, I, dev

    def energies(self, eps, lam, mu):
        K, tr, _, dev = self._parts(eps, lam, mu)
        psi_p = 0.5 * K * _pos(tr) ** 2 + mu * np.einsum("nij,nij->n", dev, dev)
        psi_m = 0.5 * K * _neg(tr) ** 2
        return psi_p, psi_m

    def stress_tangent(self, eps, lam, mu):
        n, dim = eps.shape[0], eps.shape[1]
        K, tr, I, dev = self._parts(eps, lam, mu)

        sig_p = K * _pos(tr)[:, None, None] * I + 2.0 * mu * dev
        sig_m = K * _neg(tr)[:, None, None] * I

        Id = np.eye(dim)
        IxI = np.einsum("ij,kl->ijkl", Id, Id)
        P_dev = _sym_identity(dim) - IxI / dim

        h = _heaviside(tr)
        C_p = (K * h[:, None, None, None, None] * IxI
               + 2.0 * mu * np.broadcast_to(P_dev, (n,) + P_dev.shape))
        C_m = K * (1.0 - h)[:, None, None, None, None] * IxI
        return sig_p, sig_m, C_p, np.ascontiguousarray(C_m)


class Spectral(Split):
    r"""Miehe, Welschinger & Hofacker (2010): spectral decomposition.

        eps_pm = sum_I <eps_I>_pm  n_I (x) n_I
        Psi_pm = 1/2 lam <tr eps>_pm^2 + mu tr(eps_pm^2)

    The physically most convincing split, and the most expensive: the tangent
    needs the derivative of the eigenprojection,

        d eps_pm / d eps

    which in the eigenbasis is diagonal in the ``(II)(II)`` blocks and carries
    ``theta_IJ = (<e_I>_pm - <e_J>_pm) / (e_I - e_J)`` in the ``(IJ)(IJ)``
    blocks. That quotient is 0/0 when two principal strains coincide -- which is
    not exotic, it happens under any equibiaxial state and along symmetry lines
    -- so the degenerate branch is taken by tolerance, using the limit
    ``theta -> H(e_I)``. Getting that wrong gives NaNs or, worse, a tangent that
    is quietly wrong only near symmetry and makes Newton stall.
    """

    name = "spectral"

    #: relative tolerance for treating two principal strains as coincident
    DEGENERATE_TOL = 1.0e-9

    def _eig(self, eps):
        # eigh is fine: eps is symmetric by construction, and it returns
        # ascending eigenvalues with orthonormal eigenvectors as columns
        w, Q = np.linalg.eigh(eps)
        return w, Q

    def energies(self, eps, lam, mu):
        w, _ = self._eig(eps)
        tr = w.sum(axis=1)
        psi_p = 0.5 * lam * _pos(tr) ** 2 + mu * (_pos(w) ** 2).sum(axis=1)
        psi_m = 0.5 * lam * _neg(tr) ** 2 + mu * (_neg(w) ** 2).sum(axis=1)
        return psi_p, psi_m

    # ------------------------------------------------------------------
    def _projection(self, w, Q, positive: bool) -> np.ndarray:
        """d eps_pm / d eps, assembled in the eigenbasis then rotated back."""
        n, dim = w.shape
        f = _pos(w) if positive else _neg(w)
        df = _heaviside(w) if positive else 1.0 - _heaviside(w)

        T = np.zeros((n, dim, dim, dim, dim))
        for i in range(dim):
            T[:, i, i, i, i] = df[:, i]

        scale = np.maximum(np.abs(w).max(axis=1), 1.0)
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                gap = w[:, i] - w[:, j]
                degenerate = np.abs(gap) <= self.DEGENERATE_TOL * scale
                safe = np.where(degenerate, 1.0, gap)
                theta = np.where(degenerate,
                                 0.5 * (df[:, i] + df[:, j]),
                                 (f[:, i] - f[:, j]) / safe)
                # minor symmetries: the value is shared over the four slots
                T[:, i, j, i, j] += 0.5 * theta
                T[:, i, j, j, i] += 0.5 * theta

        # rotate: P_ijkl = Q_iI Q_jJ Q_kK Q_lL T_IJKL
        return np.einsum("nia,njb,nkc,nld,nabcd->nijkl", Q, Q, Q, Q, T)

    def stress_tangent(self, eps, lam, mu):
        n, dim = eps.shape[0], eps.shape[1]
        w, Q = self._eig(eps)
        tr = w.sum(axis=1)
        I = _identity(n, dim)

        eps_p = np.einsum("nia,na,nja->nij", Q, _pos(w), Q)
        eps_m = np.einsum("nia,na,nja->nij", Q, _neg(w), Q)

        sig_p = lam * _pos(tr)[:, None, None] * I + 2.0 * mu * eps_p
        sig_m = lam * _neg(tr)[:, None, None] * I + 2.0 * mu * eps_m

        Id = np.eye(dim)
        IxI = np.einsum("ij,kl->ijkl", Id, Id)
        h = _heaviside(tr)[:, None, None, None, None]

        C_p = lam * h * IxI + 2.0 * mu * self._projection(w, Q, True)
        C_m = lam * (1.0 - h) * IxI + 2.0 * mu * self._projection(w, Q, False)
        return sig_p, sig_m, C_p, C_m


# ---------------------------------------------------------------------------

SPLIT_REGISTRY: dict[str, type[Split]] = {
    "none": NoSplit,
    "isotropic": NoSplit,
    "amor": VolumetricDeviatoric,
    "volumetric-deviatoric": VolumetricDeviatoric,
    "miehe": Spectral,
    "spectral": Spectral,
}


def get_split(name: str) -> Split:
    key = name.lower()
    if key not in SPLIT_REGISTRY:
        raise NotImplementedError(
            f"Unknown energy split '{name}'. Available: "
            f"{sorted(set(SPLIT_REGISTRY))}"
        )
    return SPLIT_REGISTRY[key]()
