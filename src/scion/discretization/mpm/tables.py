"""An MPM backend for the quadrature tables the two-field problems loop over.

Why this is the whole port
--------------------------
`PhaseFieldFractureProblem` and `GradientDamageProblem` never touch a mesh.
They go entirely through
:class:`~scion.discretization.fem.tables.QuadratureTables` -- seven methods and
a handful of shape attributes. Supplying an object with the same interface,
backed by an MPM stencil instead of element tables, therefore runs the *same*
phase-field physics on particles with no change to the model, the staggered
solver, the energy split, the degradation function or the dissipation. The
coupling was already at the right seam; this is the adapter that uses it.

The mapping, and the one thing that is not a rename
---------------------------------------------------
Each **particle is an element carrying exactly one quadrature point** -- itself.
So ``n_elements = n_particles``, ``n_qp = 1``, ``npe`` is the stencil size (4
for a bilinear grid in 2-D), and ``w`` is the particle volume rather than a
Gauss weight times a Jacobian.

The consequence that is *not* cosmetic: FEM elements are disjoint, so element
matrices scatter into a matrix whose sparsity is fixed by the mesh. MPM stencils
**overlap** -- neighbouring particles share nodes -- so the same scatter is
correct but the assembled bandwidth follows the particle distribution. Nothing
in the assembly changes, because both go through ``np.add.at`` on a coordinate
list; it is the cost model that differs.

Known limitation: the tables are rebuilt whenever the particles move, so a
caller stepping a deforming body must rebuild rather than mutate. That is the
MPM equivalent of the reference's remark that its stiffness matrix has to be
re-factorised when material points cross cells.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["MPMQuadratureTables"]


class MPMQuadratureTables:
    """Quadrature tables built from an MPM stencil.

    Parameters
    ----------
    grid : a Cartesian background grid providing ``build_stencil`` and
        ``n_nodes``.
    particles : a particle set with ``x`` and ``vol0``.
    thickness : out-of-plane thickness in 2-D, multiplying every weight.
    volumes : optional ``(n,)`` override for the integration weights. Defaults
        to ``particles.vol0``, i.e. the **reference** volume, which is what a
        small-strain phase-field formulation integrates against.
    """

    def __init__(self, grid, particles, thickness: float = 1.0,
                 volumes=None):
        self.grid = grid
        self.dim = int(grid.dim)
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        self.thickness = float(thickness)

        st = grid.build_stencil(particles.x,
                                lp=getattr(particles, "lp", None))
        self.stencil = st
        n, npe = st.node_id.shape

        self.n_elements, self.n_qp, self.npe = n, 1, npe
        self.n_voigt = 3 if self.dim == 2 else 6
        self.n_points = n

        vol = (np.asarray(particles.vol0, dtype=float) if volumes is None
               else np.asarray(volumes, dtype=float))
        vol = np.broadcast_to(vol, (n,)).astype(float)
        self.w = (vol * (self.thickness if self.dim == 2 else 1.0)
                  ).reshape(n, 1)

        self.N = st.w.reshape(n, 1, npe)
        self.dN = st.grad_w.reshape(n, 1, npe, self.dim)
        self.conn = st.node_id
        self.scalar_edofs = st.node_id
        self.u_edofs = (st.node_id[:, :, None] * self.dim
                        + np.arange(self.dim)[None, None, :]
                        ).reshape(n, npe * self.dim)
        self.B = self._strain_operator()
        self.element_size = float(np.max(grid.dx) if np.ndim(grid.dx)
                                  else grid.dx)

    # ------------------------------------------------------------------
    def _strain_operator(self) -> np.ndarray:
        """``(n, 1, n_voigt, npe*dim)`` with **engineering** shears.

        The factor of one on the shear rows -- not one half -- is what makes
        this the operator the constitutive models expect at the assembler
        boundary, and it pairs with ``strain_tensors`` halving them back.
        """
        n, npe = self.conn.shape
        d, nv = self.dim, self.n_voigt
        B = np.zeros((n, 1, nv, npe * d))
        g = self.dN[:, 0]                       # (n, npe, dim)
        if d == 2:
            B[:, 0, 0, 0::2] = g[..., 0]
            B[:, 0, 1, 1::2] = g[..., 1]
            B[:, 0, 2, 0::2] = g[..., 1]
            B[:, 0, 2, 1::2] = g[..., 0]
        else:
            B[:, 0, 0, 0::3] = g[..., 0]
            B[:, 0, 1, 1::3] = g[..., 1]
            B[:, 0, 2, 2::3] = g[..., 2]
            B[:, 0, 3, 1::3] = g[..., 2]
            B[:, 0, 3, 2::3] = g[..., 1]
            B[:, 0, 4, 0::3] = g[..., 2]
            B[:, 0, 4, 2::3] = g[..., 0]
            B[:, 0, 5, 0::3] = g[..., 1]
            B[:, 0, 5, 1::3] = g[..., 0]
        return B

    # ------------------------------------------------------------------
    # point-wise quantities
    # ------------------------------------------------------------------
    def strains_voigt(self, u: np.ndarray) -> np.ndarray:
        ue = np.asarray(u, dtype=float)[self.u_edofs]
        return np.einsum("eqab,eb->eqa", self.B, ue).reshape(
            self.n_points, self.n_voigt)

    def strain_tensors(self, u: np.ndarray) -> np.ndarray:
        """``(n_points, dim, dim)``; engineering shears halved back."""
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
        return np.einsum("eqa,ea->eq", self.N,
                         np.asarray(nodal, dtype=float)[self.scalar_edofs]
                         ).ravel()

    def gradient(self, nodal: np.ndarray) -> np.ndarray:
        return np.einsum("eqai,ea->eqi", self.dN,
                         np.asarray(nodal, dtype=float)[self.scalar_edofs]
                         ).reshape(-1, self.dim)

    def integrate(self, density: np.ndarray) -> float:
        return float((np.asarray(density, dtype=float).ravel()
                      * self.w.ravel()).sum())

    # ------------------------------------------------------------------
    # scalar-field operators
    # ------------------------------------------------------------------
    def scatter(self, ke: np.ndarray, n_dof: int) -> sparse.csr_matrix:
        rows = np.repeat(self.scalar_edofs, self.npe, axis=1).ravel()
        cols = np.tile(self.scalar_edofs, (1, self.npe)).ravel()
        return sparse.csr_matrix((ke.ravel(), (rows, cols)),
                                 shape=(n_dof, n_dof))

    def laplacian(self, coeff, n_dof: int) -> sparse.csr_matrix:
        c = np.asarray(coeff, dtype=float)
        c = (np.full(self.w.shape, float(c)) if c.ndim == 0
             else c.reshape(self.n_elements, self.n_qp))
        ke = np.einsum("eqai,eqbi,eq,eq->eab", self.dN, self.dN, c, self.w)
        return self.scatter(ke, n_dof)

    def mass(self, coeff, n_dof: int) -> sparse.csr_matrix:
        c = np.asarray(coeff, dtype=float)
        c = (np.full(self.w.shape, float(c)) if c.ndim == 0
             else c.reshape(self.n_elements, self.n_qp))
        ke = np.einsum("eqa,eqb,eq,eq->eab", self.N, self.N, c, self.w)
        return self.scatter(ke, n_dof)

    def load(self, c_qp: np.ndarray, n_dof: int) -> np.ndarray:
        c = np.asarray(c_qp, dtype=float).reshape(self.n_elements, self.n_qp)
        fe = np.einsum("eqa,eq,eq->ea", self.N, c, self.w)
        out = np.zeros(n_dof)
        np.add.at(out, self.scalar_edofs.ravel(), fe.ravel())
        return out

    # ------------------------------------------------------------------
    # vector-field operators
    # ------------------------------------------------------------------
    def stiffness_and_force(self, sigma_voigt: np.ndarray,
                            C_voigt: np.ndarray, n_dof: int):
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
    def supported_nodes(self, tol: float = 1e-12) -> np.ndarray:
        """Nodes carrying real particle support.

        Needed because an MPM grid is larger than the body: unsupported nodes
        give identically zero rows, and both field solves must exclude them or
        the system is singular. A FEM mesh has no such nodes, which is why the
        FEM tables carry no equivalent.
        """
        support = np.zeros(self.grid.n_nodes)
        np.add.at(support, self.conn, self.N[:, 0] * self.w)
        return support > tol * support.max()

    def __repr__(self) -> str:
        return (f"MPMQuadratureTables({self.n_elements} particles, "
                f"npe={self.npe}, dim={self.dim}, dx={self.element_size:.4g})")
