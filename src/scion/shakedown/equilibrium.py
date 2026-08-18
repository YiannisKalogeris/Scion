r"""
Statically admissible residual stress fields.

Melan's theorem asks for a time-independent residual stress field ``rho`` that
is *self-equilibrated*: in equilibrium with zero body force and zero surface
traction on the free boundary. In the discrete setting that is exactly

    A rho = 0,    A rho = sum over points of  w B^T rho   restricted to the
                  free degrees of freedom,

because ``A rho`` is the internal force vector of ``rho`` and a field with no
external load must produce none. The constrained degrees of freedom are exempt:
there the internal force is the reaction, and it is free to be anything.

The residual fields therefore form a *linear subspace*, and the whole of
:class:`SelfEquilibrium` is one thing -- the orthogonal projection onto it.
Everything Melan needs beyond that is convex optimisation.

The projection
--------------
``A rho = 0`` says exactly that ``rho`` is orthogonal to every compatible strain
field, since ``<rho, eps(u)>_w = u . A rho``. So the subspace is the orthogonal
complement of the compatible fields, and

    P rho = rho - eps(u),    M u = A rho,    M = sum w B^T B,

with the inner product ``<a, b>_w = sum w a:b``. ``M`` is the stiffness matrix
of a fictitious material with unit moduli; it is factorised once and every
projection after that is a back-substitution.

Mandel throughout
-----------------
Stress is stored in Mandel form, ``(n_points, 6)``. That makes ``<a, b>_w`` the
plain weighted dot product and ``a:b`` the tensor contraction it should be, with
no factor of two anywhere -- which matters here because the projection, the
objective and the yield function all contract stresses against different things
and a Voigt convention would need three different rules.

The strain-displacement matrix in Mandel form is the Voigt one with its shear
rows divided by ``sqrt(2)``, and ``M`` is then assembled by the ordinary
assembler with the constant "modulus" ``diag(1, 1, 1, 1/2, 1/2, 1/2)``. No new
assembly code.

Plane strain leaves rho_33 free, and that is not an oversight
-------------------------------------------------------------
In two dimensions the equilibrium equations involve only the in-plane
components. Nothing varies along ``x_3``, so ``d(sigma_33)/dx_3`` vanishes
identically and **any** ``rho_33`` field is self-equilibrated. Leaving it free
is correct, and it matters: at a point carrying in-plane residual stress, a free
``rho_33`` can be chosen to reduce ``J2``, which raises the lower bound. An
implementation that carried only the three in-plane components would be
conservative without saying so.

The out-of-plane shears ``rho_13`` and ``rho_23`` are the opposite case. They do
appear in an equilibrium equation of their own -- the anti-plane one -- which
this module does not assemble, so they are pinned to zero. That restricts the
feasible set, and restricting the feasible set of a *lower* bound can only make
it conservative, never wrong. It is the safe direction to be incomplete in, and
the reason the distinction is worth stating rather than assuming.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import linalg as spla

from ..materials.notation import (
    mandel_to_voigt_stress,
    voigt_strain_to_mandel,
)

__all__ = ["SelfEquilibrium"]

#: Mandel slots, in the order of ``notation.MANDEL_INDEX``:
#: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12.
_OUT_OF_PLANE_NORMAL = 2
_OUT_OF_PLANE_SHEAR = [3, 4]


class SelfEquilibrium:
    """Orthogonal projection onto self-equilibrated residual stress fields.

    Parameters
    ----------
    tables : :class:`~scion.discretization.fem.tables.QuadratureTables`.
    free_dofs : displacement degrees of freedom that are *not* prescribed. The
        equilibrium condition is imposed only there; at a support the internal
        force is a reaction and is unconstrained.
    """

    def __init__(self, tables, free_dofs):
        self.tables = tables
        self.dim = tables.dim
        self.n_points = tables.n_points
        self.free = np.asarray(free_dofs, dtype=np.int64)
        self.n_dof = tables.vector_dofs.n_total

        # The Mandel Gram matrix, assembled as an ordinary stiffness with the
        # constant modulus that turns Voigt B into Mandel B on both sides.
        # B_m = S B with S halving the shear rows, so B_m^T B_m = B^T S^T S B
        # and S^T S is 1 on the normal components, 1/2 on the shear ones. The
        # shear rows start at 2 in 2-D and at 3 in 3-D -- eps_33 is a normal
        # component and must keep its 1.
        nv = tables.n_voigt
        scale = np.ones(nv)
        scale[(2 if self.dim == 2 else 3):] = 0.5
        C = np.broadcast_to(np.diag(scale), (self.n_points, nv, nv))
        M, _ = tables.stiffness_and_force(
            np.zeros((self.n_points, nv)), C, self.n_dof)

        M_free = M.tocsr()[self.free][:, self.free].tocsc()
        if M_free.shape[0] == 0:
            raise ValueError("no free degrees of freedom; every displacement "
                             "is prescribed and there is nothing to project")
        try:
            self._solve = spla.factorized(M_free)
        except RuntimeError as exc:                       # singular
            raise ValueError(
                "the compatible-strain Gram matrix is singular, which means "
                "the free degrees of freedom admit a rigid-body motion. "
                "Restrain the structure before asking for residual stresses."
            ) from exc

    # ------------------------------------------------------------------
    def internal_force(self, rho: np.ndarray) -> np.ndarray:
        """``A rho``, the nodal internal force of a residual field, ``(n_dof,)``.

        Zero on the free degrees of freedom is precisely self-equilibrium; the
        entries at constrained degrees of freedom are reactions and carry no
        information about admissibility.
        """
        sigma_v = mandel_to_voigt_stress(np.asarray(rho, dtype=float), self.dim)
        _, f = self.tables.stiffness_and_force(
            sigma_v, np.zeros((self.n_points, self.tables.n_voigt,
                               self.tables.n_voigt)), self.n_dof)
        return f

    def project(self, rho: np.ndarray) -> np.ndarray:
        """Nearest self-equilibrated field, ``(n_points, 6)`` Mandel.

        Nearest in the quadrature-weighted L2 sense, which is the physically
        natural one for a stress field. Note this is *not* orthogonal in the
        plain Euclidean sense unless the weights are uniform -- see
        :meth:`scaled_projector` for the version an optimiser can use.
        """
        rho = np.array(rho, dtype=float, copy=True).reshape(self.n_points, 6)
        if self.dim == 2:
            rho[:, _OUT_OF_PLANE_SHEAR] = 0.0

        u = np.zeros(self.n_dof)
        u[self.free] = self._solve(self.internal_force(rho)[self.free])
        eps = voigt_strain_to_mandel(self.tables.strains_voigt(u), self.dim)
        # in 2-D eps has zeros in slots 2, 3, 4, so rho_33 passes through
        # untouched and the pinned shears stay pinned
        return rho - eps

    def residual(self, rho: np.ndarray) -> float:
        """Relative equilibrium violation of a field: ``0`` when admissible.

        Normalised by the norm of the force the field would produce if it were
        not projected, so it is a dimensionless number a test can assert on.
        """
        f = self.internal_force(rho)[self.free]
        scale = np.linalg.norm(
            self.internal_force(np.abs(rho))[self.free])
        return float(np.linalg.norm(f) / max(scale, 1e-300))

    # ------------------------------------------------------------------
    def scaled_projector(self):
        """``(project_scaled, to_scaled, from_scaled)`` for an optimiser.

        The L2 projection is orthogonal in the weighted inner product
        ``<a,b>_w``, not the Euclidean one, so handing it straight to a
        quasi-Newton method gives a wrong gradient: the chain rule
        ``grad(f . P) = P^T grad f`` needs ``P^T = P``, and it is not.

        Rescaling the variable fixes it exactly. With ``z = s * rho`` and
        ``s = sqrt(w / V)`` per point, the weighted inner product on ``rho``
        becomes the Euclidean one on ``z``, and ``P~ z = s * P(z / s)`` is a
        genuine Euclidean-orthogonal projector. Everything the optimiser does
        is then correct with no further care, and the scaling has the side
        benefit of weighting every point by its share of the volume rather than
        by an accident of the mesh.
        """
        w = self.tables.w.ravel()
        volume = float(w.sum())
        s = np.sqrt(w / volume)[:, None]

        def to_scaled(rho):
            return (s * np.asarray(rho).reshape(self.n_points, 6)).ravel()

        def from_scaled(z):
            return np.asarray(z).reshape(self.n_points, 6) / s

        def project_scaled(z):
            return (s * self.project(from_scaled(z))).ravel()

        return project_scaled, to_scaled, from_scaled

    def __repr__(self) -> str:
        pinned = "rho_13 = rho_23 = 0, rho_33 free" if self.dim == 2 else "full"
        return (f"SelfEquilibrium({self.n_points} points, "
                f"{self.free.size}/{self.n_dof} free dof, {pinned})")
