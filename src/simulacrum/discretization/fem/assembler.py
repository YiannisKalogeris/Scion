"""
FEM global matrix and vector assembly.

Supports scalar (Poisson-type) and vector (elasticity-type) problems.
The assembler delegates constitutive-law evaluation to the material model.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from simulacrum.core.domain import Domain
from simulacrum.core.dof import DOFMap
from simulacrum.discretization.base import DiscretizationBase, LinearSystem
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature


class FEMAssembler(DiscretizationBase):
    """Standard Galerkin FEM assembly.

    Parameters
    ----------
    domain : Domain
        Computational domain with mesh, materials, and BCs.
    n_components : int
        DOFs per node (1 = scalar, dim = vector field).
    """

    def __init__(self, domain: Domain, n_components: int = 1):
        super().__init__(domain)
        self.n_components = n_components
        self.dof_map = DOFMap(domain.n_nodes, n_components)

    # ------------------------------------------------------------------
    def assemble(self) -> LinearSystem:
        n_dof = self.dof_map.n_total
        rows, cols, vals = [], [], []
        rhs = np.zeros(n_dof)

        for block in self.domain.mesh.element_blocks:
            ref_elem = get_element(block.etype)
            qpts, qwts = get_quadrature(block.etype)

            for e in range(block.connectivity.shape[0]):
                enodes = block.connectivity[e]
                coords = self.domain.coordinates[enodes]  # (n_el_nodes, dim)
                edofs = self.dof_map.element_dofs(enodes)

                ke = self._element_stiffness(ref_elem, coords, qpts, qwts, block.tag)

                for i_local, i_global in enumerate(edofs):
                    for j_local, j_global in enumerate(edofs):
                        rows.append(i_global)
                        cols.append(j_global)
                        vals.append(ke[i_local, j_local])

        K = sparse.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(n_dof, n_dof),
        )
        return LinearSystem(matrix=K, rhs=rhs)

    # ------------------------------------------------------------------
    def _element_stiffness(
        self, ref_elem, coords, qpts, qwts, material_tag: str
    ) -> np.ndarray:
        """Compute element stiffness matrix via numerical quadrature."""
        n_el_dof = ref_elem.n_nodes * self.n_components
        ke = np.zeros((n_el_dof, n_el_dof))

        material = self.domain.materials.get(material_tag)

        for qp, qw in zip(qpts, qwts):
            N, dN_dxi = ref_elem.evaluate(qp)
            J = dN_dxi.T @ coords  # (dim, dim)
            detJ = np.linalg.det(J)
            dN_dx = dN_dxi @ np.linalg.inv(J).T  # (n_nodes, dim)

            if self.n_components == 1:
                # Scalar (Poisson-type): ke += B^T B * detJ * w
                B = dN_dx  # (n_nodes, dim)
                if material is not None:
                    kappa = material.conductivity
                    ke += (B @ (kappa * np.eye(coords.shape[1])) @ B.T) * detJ * qw
                else:
                    ke += (B @ B.T) * detJ * qw
            else:
                # Vector (elasticity): delegate to B-matrix + C-matrix
                B = self._strain_displacement(dN_dx, coords.shape[1])
                if material is not None:
                    C = material.tangent(dim=coords.shape[1])
                else:
                    C = np.eye(B.shape[0])
                ke += (B.T @ C @ B) * detJ * qw

        return ke

    # ------------------------------------------------------------------
    @staticmethod
    def _strain_displacement(dN_dx: np.ndarray, dim: int) -> np.ndarray:
        """Build the Voigt strain-displacement matrix B.

        Parameters
        ----------
        dN_dx : (n_nodes, dim)
        dim   : spatial dimension

        Returns
        -------
        B : (n_voigt, n_nodes * dim)
        """
        n_nodes = dN_dx.shape[0]
        if dim == 2:
            B = np.zeros((3, n_nodes * 2))
            for i in range(n_nodes):
                B[0, 2 * i] = dN_dx[i, 0]
                B[1, 2 * i + 1] = dN_dx[i, 1]
                B[2, 2 * i] = dN_dx[i, 1]
                B[2, 2 * i + 1] = dN_dx[i, 0]
        elif dim == 3:
            B = np.zeros((6, n_nodes * 3))
            for i in range(n_nodes):
                B[0, 3 * i] = dN_dx[i, 0]
                B[1, 3 * i + 1] = dN_dx[i, 1]
                B[2, 3 * i + 2] = dN_dx[i, 2]
                B[3, 3 * i + 1] = dN_dx[i, 2]
                B[3, 3 * i + 2] = dN_dx[i, 1]
                B[4, 3 * i] = dN_dx[i, 2]
                B[4, 3 * i + 2] = dN_dx[i, 0]
                B[5, 3 * i] = dN_dx[i, 1]
                B[5, 3 * i + 1] = dN_dx[i, 0]
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        return B

    # ------------------------------------------------------------------
    def recover_fields(self, solution: np.ndarray) -> dict[str, np.ndarray]:
        n = self.domain.n_nodes
        c = self.n_components
        return {"primary": solution.reshape(n, c)}
