"""
Mass matrix assembly for finite element analysis.

Supports:
    - Consistent mass matrix: M_ij = ∫_Ω ρ N_i N_j dΩ
    - Lumped mass matrix: row-sum lumping of the consistent matrix

Usage:
    from scion.discretization.fem.mass import assemble_mass_matrix

    M_consistent = assemble_mass_matrix(mesh, dof_map, rho, etype, method="consistent")
    M_lumped = assemble_mass_matrix(mesh, dof_map, rho, etype, method="lumped")
"""

import numpy as np
from scipy import sparse
from typing import Optional

from scion.mesh.mesh import Mesh
from scion.core.dof import DOFMap
from scion.discretization.fem.elements import get_element
from scion.discretization.fem.quadrature import get_quadrature


def assemble_mass_matrix(
    mesh: Mesh,
    dof_map: DOFMap,
    rho: float,
    n_components: int = 2,
    method: str = "consistent",
    thickness: float = 1.0,
) -> sparse.csr_matrix:
    """
    Assemble the global mass matrix.

    Args:
        mesh: Finite element mesh
        dof_map: Degree of freedom mapping
        rho: Mass density
        n_components: Number of DOF per node (1 for scalar, 2 for 2D elasticity)
        method: "consistent" or "lumped"
        thickness: Thickness for plane stress/strain problems

    Returns:
        M: Sparse mass matrix (CSR format)
    """
    n_dof = dof_map.n_total
    rows, cols, vals = [], [], []

    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)

        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_el_dof = len(edofs)

            me = _element_mass_matrix(
                ref, coords, qpts, qwts, rho, n_components, thickness
            )

            for il, ig in enumerate(edofs):
                for jl, jg in enumerate(edofs):
                    rows.append(ig)
                    cols.append(jg)
                    vals.append(me[il, jl])

    M = sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_dof, n_dof),
    )

    if method == "lumped":
        M = _lump_mass_matrix(M)

    return M


def _element_mass_matrix(
    ref, coords, qpts, qwts, rho, n_components, thickness
):
    """
    Compute element consistent mass matrix.

    For n_components = 1 (scalar):
        m_ij = ∫ ρ N_i N_j |det J| dξ

    For n_components = 2 (2D vector):
        m_{2i,2j}     = ∫ ρ N_i N_j |det J| dξ   (x-x coupling)
        m_{2i+1,2j+1} = ∫ ρ N_i N_j |det J| dξ   (y-y coupling)
        m_{2i,2j+1}   = 0                          (no x-y coupling)
    """
    n_nodes = coords.shape[0]
    n_el_dof = n_nodes * n_components
    me = np.zeros((n_el_dof, n_el_dof))

    for qp, qw in zip(qpts, qwts):
        N, dN_dxi = ref.evaluate(qp)
        J = dN_dxi.T @ coords
        detJ = abs(np.linalg.det(J))

        # Scalar mass: N^T N
        nn = np.outer(N, N) * rho * detJ * qw * thickness

        if n_components == 1:
            me += nn
        else:
            # Expand to vector DOFs (interleaved ordering)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    for c in range(n_components):
                        me[n_components * i + c, n_components * j + c] += nn[i, j]

    return me


def _lump_mass_matrix(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Create lumped (diagonal) mass matrix by row-sum lumping.

    Each diagonal entry is the sum of its row in the consistent matrix.
    """
    diag = np.array(M.sum(axis=1)).ravel()
    return sparse.diags(diag, format="csr")
