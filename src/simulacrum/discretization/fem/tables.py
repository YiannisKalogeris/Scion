r"""
Precomputed quadrature tables, and the element operators built from them.

A two-field problem cannot use :class:`~.assembler.FEMAssembler`: that builds a
fixed stiffness from a tangent that depends on nothing, and here every tangent
depends on the other field evaluated at the quadrature points. So the drivers
interleave their own element loops -- and they all need the same tables first.

Shape functions, their spatial gradients, the strain-displacement matrices and
the integration weights are properties of the mesh alone. The mesh does not move
in a small-strain problem, so all of them are computed once here and reused for
however many thousands of assemblies the run takes. That precomputation is most
of the cost of an element loop.

What this is not
----------------
Not an assembler. It holds tables and provides the four scalar-field operators
(:meth:`laplacian`, :meth:`mass`, :meth:`load`, :meth:`scatter`) that every
gradient-regularised second field needs, but it knows no physics and decides
nothing about equations, coupling or solution order. Those live in
``multiphysics``.

Quadrature ordering
-------------------
Points are numbered element-major over the concatenation of all element blocks,
then quadrature point within the element. This is the same order
:class:`~.quadrature_state.QuadratureState` uses and the same order the batched
materials expect, so a history field lines up with the strains with no
bookkeeping at all. Blocks with different element types are allowed provided
they share a quadrature count; mixed counts would break the flat indexing and
are rejected up front rather than producing silently misaligned arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from simulacrum.core.dof import DOFMap
from simulacrum.discretization.fem.assembler import FEMAssembler
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature

__all__ = ["QuadratureTables"]


class QuadratureTables:
    """Shape functions, gradients, weights and B matrices for a whole mesh.

    Parameters
    ----------
    domain    : the usual ``Domain``.
    thickness : out-of-plane thickness folded into the weights in 2-D.
    quad_order : points per direction; the element's default rule when omitted.

    Attributes
    ----------
    N   : ``(n_elements, n_qp, npe)``
    dN  : ``(n_elements, n_qp, npe, dim)`` -- spatial gradients
    w   : ``(n_elements, n_qp)`` -- ``|det J| * weight * thickness``
    B   : ``(n_elements, n_qp, n_voigt, npe * dim)``
    """

    def __init__(self, domain, thickness: float = 1.0,
                 quad_order: int | None = None, bbar: bool = False):
        self.domain = domain
        self.thickness = float(thickness)
        self.dim = domain.coordinates.shape[1]
        self.n_voigt = 3 if self.dim == 2 else 6
        self.bbar = bool(bbar)

        coords = domain.coordinates
        blocks = domain.mesh.element_blocks

        counts = {len(self._rule(b.etype, quad_order)[1]) for b in blocks}
        if len(counts) != 1:
            raise NotImplementedError(
                f"all element blocks must share a quadrature count; got "
                f"{sorted(counts)}. The flat point numbering that indexes every "
                f"state array assumes one count for the whole mesh.")
        self.n_qp = counts.pop()

        conn, N_all, dN_all, w_all = [], [], [], []
        for block in blocks:
            ref = get_element(block.etype)
            qpts, qwts = self._rule(block.etype, quad_order)
            for e in range(block.connectivity.shape[0]):
                enodes = block.connectivity[e]
                xe = coords[enodes]
                conn.append(enodes)
                for qp, qw in zip(qpts, qwts):
                    N, dN_dxi = ref.evaluate(qp)
                    J = dN_dxi.T @ xe
                    # dN_dx = dN_dxi @ inv(J).T -- the transpose is silently
                    # irrelevant for axis-aligned rectangles and wrong for
                    # everything else, so it is written the same way here as in
                    # FEMAssembler rather than rederived
                    dN_dx = dN_dxi @ np.linalg.inv(J).T
                    N_all.append(N)
                    dN_all.append(dN_dx)
                    w_all.append(abs(np.linalg.det(J)) * qw * self.thickness)

        self.conn = np.array(conn, dtype=np.int64)
        self.n_elements, self.npe = self.conn.shape
        self.n_points = self.n_elements * self.n_qp

        self.N = np.array(N_all).reshape(self.n_elements, self.n_qp, self.npe)
        self.dN = np.array(dN_all).reshape(self.n_elements, self.n_qp,
                                           self.npe, self.dim)
        self.w = np.array(w_all).reshape(self.n_elements, self.n_qp)

        self.B = np.array([[FEMAssembler._strain_displacement(
            self.dN[e, q], self.dim) for q in range(self.n_qp)]
            for e in range(self.n_elements)])

        if self.bbar:
            self._apply_bbar()

        self.vector_dofs = DOFMap(domain.n_nodes, self.dim)
        self.u_edofs = np.array([self.vector_dofs.element_dofs(c)
                                 for c in self.conn])
        #: a scalar field has one DOF per node, so its element DOFs are the
        #: connectivity itself
        self.scalar_edofs = self.conn

        self.element_size = float(
            np.sqrt(self.w.sum(axis=1) / self.thickness).min())

    @staticmethod
    def _rule(etype: str, order: int | None):
        return (get_quadrature(etype, order=order) if order
                else get_quadrature(etype))

    # ------------------------------------------------------------------
    def _apply_bbar(self) -> None:
        r"""Replace the dilatational part of B by its element average.

        The B-bar, or mean-dilatation, treatment of volumetric locking. An
        element that must deform at constant volume has one incompressibility
        constraint per quadrature point, and a bilinear quadrilateral has
        nowhere near enough displacement modes to satisfy four of them and still
        deform: it goes rigid, and the computed stiffness can be an order of
        magnitude too high.

        The fix is to stop asking. Split the strain into deviatoric and
        dilatational parts, ``eps = eps_dev + (theta/n) m`` with
        ``theta = m^T eps`` and ``n = m^T m``, and replace ``theta`` by its
        element mean:

            B_bar = B + (1/n) m (b_mean - m^T B),    b_mean = <m^T B>_element

        so that ``m^T B_bar`` is constant over the element -- one volumetric
        constraint instead of one per point -- while the deviatoric part keeps
        the full rule. This is equivalent to selectively under-integrating the
        volumetric term, and it is done here by modifying ``B`` instead of
        carrying two quadrature rules, so every operator built from the tables
        inherits it with no further care.

        ``n`` is 3 in three dimensions and **2 in plane strain**, because there
        the dilatation is ``eps_11 + eps_22``: ``eps_33`` is identically zero, so
        it contributes nothing to the volume change and ``m = [1, 1, 0]``. Using
        3 in 2-D leaves a third of the volumetric constraint in place and only
        partly relieves the locking.

        Not free. The mean-dilatation element is softer than it should be for
        genuinely compressible material, so it is worth switching on for
        near-incompressible problems -- plastic flow, the linear matching method
        -- and leaving off otherwise.
        """
        m = np.zeros(self.n_voigt)
        m[:self.dim] = 1.0                       # [1,1,0] in 2-D, [1,1,1,0,0,0]
        n = float(self.dim)

        dilatation = np.einsum("a,eqad->eqd", m, self.B)
        mean = (np.einsum("eqd,eq->ed", dilatation, self.w)
                / self.w.sum(axis=1)[:, None])
        self.B = self.B + np.einsum(
            "a,eqd->eqad", m, mean[:, None, :] - dilatation) / n

    # ------------------------------------------------------------------
    # Field evaluation
    # ------------------------------------------------------------------
    def strains_voigt(self, u: np.ndarray) -> np.ndarray:
        """Voigt strains at every point, ``(n_points, n_voigt)``.

        Engineering shears, matching what the constitutive models expect at the
        assembler boundary.
        """
        ue = np.asarray(u, dtype=float)[self.u_edofs]
        ev = np.einsum("eqab,eb->eqa", self.B, ue)
        return ev.reshape(self.n_points, self.n_voigt)

    def strain_tensors(self, u: np.ndarray) -> np.ndarray:
        """Strain tensors at every point, ``(n_points, dim, dim)``.

        The Voigt form with the engineering shears halved back into tensor
        components -- note this returns ``dim x dim``, not the full 3-D tensor;
        callers that need ``eps_33`` should use
        :func:`~simulacrum.materials.notation.voigt_strain_to_tensor`.
        """
        ev = self.strains_voigt(u)
        eps = np.zeros((self.n_points, self.dim, self.dim))
        if self.dim == 2:
            eps[:, 0, 0], eps[:, 1, 1] = ev[:, 0], ev[:, 1]
            eps[:, 0, 1] = eps[:, 1, 0] = 0.5 * ev[:, 2]
        else:
            eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2] = ev[:, 0], ev[:, 1], ev[:, 2]
            eps[:, 1, 2] = eps[:, 2, 1] = 0.5 * ev[:, 3]
            eps[:, 0, 2] = eps[:, 2, 0] = 0.5 * ev[:, 4]
            eps[:, 0, 1] = eps[:, 1, 0] = 0.5 * ev[:, 5]
        return eps

    def interpolate(self, nodal: np.ndarray) -> np.ndarray:
        """A nodal scalar field at the quadrature points, ``(n_points,)``."""
        return np.einsum("eqa,ea->eq", self.N,
                         np.asarray(nodal, dtype=float)[self.scalar_edofs]
                         ).ravel()

    def gradient(self, nodal: np.ndarray) -> np.ndarray:
        """Its spatial gradient, ``(n_points, dim)``."""
        return np.einsum("eqai,ea->eqi", self.dN,
                         np.asarray(nodal, dtype=float)[self.scalar_edofs]
                         ).reshape(-1, self.dim)

    def integrate(self, density: np.ndarray) -> float:
        """Integrate a quadrature-point density over the domain."""
        return float((np.asarray(density, dtype=float).ravel()
                      * self.w.ravel()).sum())

    # ------------------------------------------------------------------
    # Scalar-field operators
    # ------------------------------------------------------------------
    def scatter(self, ke: np.ndarray, n_dof: int) -> sparse.csr_matrix:
        """Assemble ``(n_elements, npe, npe)`` element matrices for a scalar."""
        rows = np.repeat(self.scalar_edofs, self.npe, axis=1).ravel()
        cols = np.tile(self.scalar_edofs, (1, self.npe)).ravel()
        return sparse.csr_matrix((ke.ravel(), (rows, cols)),
                                 shape=(n_dof, n_dof))

    def laplacian(self, coeff, n_dof: int) -> sparse.csr_matrix:
        """``INT c grad(u) . grad(v)``. ``coeff`` is scalar or per point."""
        c = np.asarray(coeff, dtype=float)
        c = (np.full(self.w.shape, float(c)) if c.ndim == 0
             else c.reshape(self.n_elements, self.n_qp))
        ke = np.einsum("eqai,eqbi,eq,eq->eab", self.dN, self.dN, c, self.w)
        return self.scatter(ke, n_dof)

    def mass(self, coeff, n_dof: int) -> sparse.csr_matrix:
        """``INT c u v``. ``coeff`` is scalar or per point."""
        c = np.asarray(coeff, dtype=float)
        c = (np.full(self.w.shape, float(c)) if c.ndim == 0
             else c.reshape(self.n_elements, self.n_qp))
        ke = np.einsum("eqa,eqb,eq,eq->eab", self.N, self.N, c, self.w)
        return self.scatter(ke, n_dof)

    def load(self, c_qp: np.ndarray, n_dof: int) -> np.ndarray:
        """``INT c v``, the scalar load vector."""
        c = np.asarray(c_qp, dtype=float).reshape(self.n_elements, self.n_qp)
        fe = np.einsum("eqa,eq,eq->ea", self.N, c, self.w)
        out = np.zeros(n_dof)
        np.add.at(out, self.scalar_edofs.ravel(), fe.ravel())
        return out

    # ------------------------------------------------------------------
    # Vector-field operators
    # ------------------------------------------------------------------
    def stiffness_and_force(self, sigma_voigt: np.ndarray,
                            C_voigt: np.ndarray, n_dof: int):
        """``(K, f_int)`` from quadrature-point stresses and tangents.

        ``sigma_voigt`` is ``(n_points, n_voigt)`` and ``C_voigt``
        ``(n_points, n_voigt, n_voigt)``. No symmetry is assumed of ``C``, so an
        unsymmetric consistent tangent -- non-associative plastic flow, or any
        damage model whose equivalent strain is not the energy measure --
        assembles correctly.
        """
        nv = self.n_voigt
        C = np.asarray(C_voigt, dtype=float).reshape(
            self.n_elements, self.n_qp, nv, nv)
        s = np.asarray(sigma_voigt, dtype=float).reshape(
            self.n_elements, self.n_qp, nv)

        ke = np.einsum("eqab,eqbc,eqcd,eq->ead",
                       np.swapaxes(self.B, 2, 3), C, self.B, self.w)
        fe = np.einsum("eqab,eqa,eq->eb", self.B, s, self.w)

        nd = self.u_edofs.shape[1]
        rows = np.repeat(self.u_edofs, nd, axis=1).ravel()
        cols = np.tile(self.u_edofs, (1, nd)).ravel()
        K = sparse.csr_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
        f_int = np.zeros(n_dof)
        np.add.at(f_int, self.u_edofs.ravel(), fe.ravel())
        return K, f_int

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"QuadratureTables({self.n_elements} elements x {self.n_qp} qp "
                f"= {self.n_points} points, dim={self.dim}, "
                f"h={self.element_size:.4g})")
