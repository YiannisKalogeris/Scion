#!/usr/bin/env python3
"""
Geometric nonlinearity: small vs large displacement theory.

Problem:
    - Slender cantilever: 6m x 0.5m, thickness 0.2m
    - Material: E = 50 GPa, nu = 0.3 (linear elastic)
    - Left edge clamped
    - Point force at bottom-right corner: F = 10000 kN (downward)
    - Plane stress

Analysis:
    1. Linear (small displacement): standard FEM
    2. Geometrically nonlinear (large displacement): SVK + Total Lagrangian + NR
    3. Compare force-displacement curves and deformed shapes

The linear solution predicts deflection >> beam length (unphysical).
The nonlinear solution accounts for membrane stiffening under large rotations.

Run:       python examples/fem/cantilever_geometric_nonlinear_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from pathlib import Path
from scipy import sparse

from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.mesh.generators import rectangle_mesh
from simulacrum.core.dof import DOFMap
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature
from simulacrum.materials.linear_elastic import PlaneStress
from simulacrum.materials.geometric_nonlinear import StVenantKirchhoff
from simulacrum.solvers.nonlinear import LoadControl

# ====================================================================
# CONFIGURATION
# ====================================================================
_FIG_DIR = Path(__file__).resolve().parent / "Figures_cantilever_geometric_nl"
_FIG_DIR.mkdir(exist_ok=True)

# Geometry
L = 6.0          # length (m)
H = 0.5          # height (m)
THICKNESS = 0.20 # thickness (m)

# Material
E_YOUNG = 50e9   # Pa
NU_POISSON = 0.3

# Loading
F_TOTAL = 20000e3  # 20000 kN

# Mesh
NX = 60   # elements along length
NY = 6    # elements along height

# Nonlinear solver
N_LOAD_STEPS = 50
NR_TOL = 1e-8
NR_MAX_ITER = 30

# Geometrically-nonlinear internal force / tangent formulation:
#   "pk1"     -> f_int = P : grad(N),  K_T from full 4th-order tangent dP/dF
#   "pk2" -> f_int = B_L^T S,  K_T = K_L + K_NL
#                = B_L^T C B_L + B_grad^T S_bar B_grad
#                (2nd Piola-Kirchhoff + Green-Lagrange, Total Lagrangian
#                 B-matrix form as derived in the course slides)
# The two are mathematically equivalent and agree to machine precision.
FORMULATION = "pk1"   # "pk1" or "pk2"


# ====================================================================
# B-MATRIX
# ====================================================================
def build_B_matrix(dN_dx):
    n_nodes = dN_dx.shape[0]
    B = np.zeros((3, n_nodes * 2))
    for i in range(n_nodes):
        B[0, 2 * i] = dN_dx[i, 0]
        B[1, 2 * i + 1] = dN_dx[i, 1]
        B[2, 2 * i] = dN_dx[i, 1]
        B[2, 2 * i + 1] = dN_dx[i, 0]
    return B


# ====================================================================
# DEFORMATION GRADIENT
# ====================================================================
def compute_deformation_gradient(dN_dX, u_elem):
    grad_u = u_elem.T @ dN_dX
    return np.eye(2) + grad_u


# ====================================================================
# LINEAR SOLVE
# ====================================================================
def solve_linear(mesh, f_ext, constrained_dofs):
    """Small-displacement linear elastic solve."""
    print("\n" + "=" * 60)
    print("LINEAR ANALYSIS (small displacements)")
    print("=" * 60)

    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C_mat = material.tangent(dim=2)
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total

    # Assemble stiffness
    rows, cols, vals = [], [], []
    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_el_dof = len(edofs)
            ke = np.zeros((n_el_dof, n_el_dof))
            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords
                detJ = abs(np.linalg.det(J))
                dN_dx = dN_dxi @ np.linalg.inv(J)
                B = build_B_matrix(dN_dx)
                ke += (B.T @ C_mat @ B) * detJ * qw * THICKNESS
            for il, ig in enumerate(edofs):
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke[il, jl])

    K = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                          shape=(n_dof, n_dof))

    # Apply BCs and solve
    K_lil = K.tolil()
    rhs = f_ext.copy()
    for dof in constrained_dofs:
        K_lil[dof, :] = 0
        K_lil[:, dof] = 0
        K_lil[dof, dof] = 1.0
        rhs[dof] = 0.0
    K = K_lil.tocsr()

    from scipy.sparse.linalg import spsolve
    u = spsolve(K.tocsc(), rhs)

    disp = u.reshape(-1, 2)
    print(f"  max |u| = {np.sqrt(disp[:, 0]**2 + disp[:, 1]**2).max():.4f} m")
    print(f"  tip uy  = {disp[load_node, 1]:.4f} m")

    return u


# ====================================================================
# NONLINEAR ASSEMBLY (Total Lagrangian)
# ====================================================================
def assemble_internal_force_and_tangent(mesh, dof_map, u, material, plane_stress=True):
    """Assemble f_int and K_T using PK1 stress from the material."""
    n_dof = dof_map.n_total
    f_int = np.zeros(n_dof)
    rows, cols, vals = [], [], []
    disp = u.reshape(-1, 2)

    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            u_elem = disp[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_nodes_e = len(enodes)
            n_el_dof = len(edofs)
            fe_int = np.zeros(n_el_dof)
            ke_t = np.zeros((n_el_dof, n_el_dof))

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J_ref = dN_dxi.T @ coords
                detJ_ref = abs(np.linalg.det(J_ref))
                dN_dX = dN_dxi @ np.linalg.inv(J_ref)

                F = compute_deformation_gradient(dN_dX, u_elem)
                P, A, _ = material.pk1_stress_and_tangent_2d(F, plane_stress)

                w = detJ_ref * qw * THICKNESS

                for I in range(n_nodes_e):
                    for i in range(2):
                        val = 0.0
                        for J_idx in range(2):
                            val += P[i, J_idx] * dN_dX[I, J_idx]
                        fe_int[2 * I + i] += val * w

                for I in range(n_nodes_e):
                    for i in range(2):
                        for K_node in range(n_nodes_e):
                            for k in range(2):
                                val = 0.0
                                for J_idx in range(2):
                                    for L_idx in range(2):
                                        val += (A[i, J_idx, k, L_idx]
                                                * dN_dX[I, J_idx]
                                                * dN_dX[K_node, L_idx])
                                ke_t[2*I+i, 2*K_node+k] += val * w

            for il, ig in enumerate(edofs):
                f_int[ig] += fe_int[il]
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke_t[il, jl])

    K_T = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                            shape=(n_dof, n_dof))
    return f_int, K_T


# ====================================================================
# B-MATRIX FORMULATION (2nd Piola-Kirchhoff + Green-Lagrange)
# Total Lagrangian, as derived in the course slides.
# ====================================================================
def build_B_grad(dN_dX):
    """
    B_nabla: maps nodal increments to the displacement-gradient vector
        dg = [u,X ; v,Y ; u,Y ; v,X].
    Shape (4, 2*n_nodes). Element DOF ordering: [u1, v1, u2, v2, ...].
    """
    n = dN_dX.shape[0]
    Bg = np.zeros((4, 2 * n))
    for a in range(n):
        Nx, Ny = dN_dX[a, 0], dN_dX[a, 1]
        Bg[0, 2 * a]     = Nx   # u,X
        Bg[1, 2 * a + 1] = Ny   # v,Y
        Bg[2, 2 * a]     = Ny   # u,Y
        Bg[3, 2 * a + 1] = Nx   # v,X
    return Bg


def build_A_L(grad_u):
    """
    A_L(u^i): maps the displacement-gradient vector dg to the *linear part*
    of the Green-Lagrange strain increment, Voigt form [e_XX; e_YY; 2 e_XY].
    Shape (3, 4). `grad_u` is the 2x2 current displacement gradient with
    grad_u[i, J] = u_i,J  (grad_u[0,0]=u,X, grad_u[0,1]=u,Y,
    grad_u[1,0]=v,X, grad_u[1,1]=v,Y).
    """
    uX, uY = grad_u[0, 0], grad_u[0, 1]
    vX, vY = grad_u[1, 0], grad_u[1, 1]
    return np.array([
        [1.0 + uX, 0.0,      0.0,       vX      ],
        [0.0,      1.0 + vY, uY,        0.0     ],
        [uY,       vX,       1.0 + uX,  1.0 + vY],
    ])


def build_S_bar(S_voigt):
    """
    S_bar: 4x4 initial-stress matrix for the geometric stiffness, built from
    the 2nd PK stress S = [S_XX, S_YY, S_XY] (Voigt). Row/col ordering matches
    dg = [u,X ; v,Y ; u,Y ; v,X].
    """
    Sxx, Syy, Sxy = S_voigt
    return np.array([
        [Sxx, 0.0, Sxy, 0.0],
        [0.0, Syy, 0.0, Sxy],
        [Sxy, 0.0, Syy, 0.0],
        [0.0, Sxy, 0.0, Sxx],
    ])


def assemble_internal_force_and_tangent_bmatrix(mesh, dof_map, u, material,
                                                plane_stress=True):
    """
    Total Lagrangian assembly using the B-matrix formulation (slides):

        f_int = INT  B_L^T S          dV0
        K_T   = INT  B_L^T C B_L      dV0   (K_L,  material + initial-disp.)
              + INT  B_grad^T S_bar B_grad dV0   (K_NL, geometric/initial-stress)

    with B_L = A_L(u) B_grad, S = 2nd PK stress (Voigt), C = dS/dE (Voigt).
    Physically identical to assemble_internal_force_and_tangent (PK1 form).
    """
    n_dof = dof_map.n_total
    f_int = np.zeros(n_dof)
    rows, cols, vals = [], [], []
    disp = u.reshape(-1, 2)

    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            u_elem = disp[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_el_dof = len(edofs)
            fe_int = np.zeros(n_el_dof)
            ke_t = np.zeros((n_el_dof, n_el_dof))

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J_ref = dN_dxi.T @ coords
                detJ_ref = abs(np.linalg.det(J_ref))
                dN_dX = dN_dxi @ np.linalg.inv(J_ref)

                grad_u = u_elem.T @ dN_dX           # 2x2, grad_u[i,J] = u_i,J
                F = np.eye(2) + grad_u

                S_voigt, C_voigt = material.pk2_voigt_and_tangent_2d(
                    F, plane_stress)

                B_grad = build_B_grad(dN_dX)        # (4, n_el_dof)
                A_L = build_A_L(grad_u)             # (3, 4)
                B_L = A_L @ B_grad                  # (3, n_el_dof)
                S_bar = build_S_bar(S_voigt)        # (4, 4)

                w = detJ_ref * qw * THICKNESS

                fe_int += (B_L.T @ S_voigt) * w               # internal force
                ke_t   += (B_L.T @ C_voigt @ B_L) * w         # K_L  (material)
                ke_t   += (B_grad.T @ S_bar @ B_grad) * w     # K_NL (geometric)

            for il, ig in enumerate(edofs):
                f_int[ig] += fe_int[il]
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke_t[il, jl])

    K_T = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                            shape=(n_dof, n_dof))
    return f_int, K_T


def assemble_nonlinear(mesh, dof_map, u, material, plane_stress=True):
    """Dispatch to the active formulation (see the FORMULATION toggle)."""
    if FORMULATION == "pk1":
        return assemble_internal_force_and_tangent(
            mesh, dof_map, u, material, plane_stress)
    elif FORMULATION == "pk2":
        return assemble_internal_force_and_tangent_bmatrix(
            mesh, dof_map, u, material, plane_stress)
    else:
        raise ValueError(
            f"Unknown FORMULATION '{FORMULATION}' (use 'pk1' or 'pk2')")


def verify_formulations_agree(mesh, dof_map, material, scale=1e-3, seed=0):
    """
    Sanity check: the PK1 and B-matrix assemblies must produce the same
    f_int and K_T for an arbitrary displacement field, since they are the
    same physics in two algebraic forms. Returns relative errors
    (||df_int||/||f_int||, ||dK_T||/||K_T||); both should be ~1e-15.
    """
    rng = np.random.default_rng(seed)
    u = scale * rng.standard_normal(dof_map.n_total)
    f1, K1 = assemble_internal_force_and_tangent(
        mesh, dof_map, u, material, True)
    f2, K2 = assemble_internal_force_and_tangent_bmatrix(
        mesh, dof_map, u, material, True)
    rel_f = np.linalg.norm(f1 - f2) / max(np.linalg.norm(f1), 1e-30)
    rel_K = sparse.linalg.norm(K1 - K2) / max(sparse.linalg.norm(K1), 1e-30)
    return float(rel_f), float(rel_K)


def check_detF(mesh, dof_map, u):
    """Check if det(F) > 0 at all quadrature points."""
    disp = u.reshape(-1, 2)
    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            u_elem = disp[enodes]
            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J_ref = dN_dxi.T @ coords
                dN_dX = dN_dxi @ np.linalg.inv(J_ref)
                F = compute_deformation_gradient(dN_dX, u_elem)
                if np.linalg.det(F) <= 0:
                    return False
    return True


# ====================================================================
# NONLINEAR SOLVE (using LoadControl from library)
# ====================================================================
def solve_nonlinear(mesh, f_ext_total, constrained_dofs, material):
    """Geometrically nonlinear solve with incremental load control."""
    print("\n" + "=" * 60)
    print("NONLINEAR ANALYSIS (large displacements, SVK)")
    print("=" * 60)
    print(f"  Formulation: {FORMULATION}")

    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total

    def f_int_fn(u):
        f_int, _ = assemble_nonlinear(
            mesh, dof_map, u, material, plane_stress=True)
        return f_int

    def K_T_fn(u):
        _, K_T = assemble_nonlinear(
            mesh, dof_map, u, material, plane_stress=True)
        return K_T

    def detF_check(u):
        return check_detF(mesh, dof_map, u)

    lc = LoadControl(
        internal_force_fn=f_int_fn,
        tangent_fn=K_T_fn,
        f_ext=f_ext_total,
        n_dof=n_dof,
        constrained_dofs=constrained_dofs,
        n_steps=N_LOAD_STEPS,
        tol=NR_TOL,
        max_iter=NR_MAX_ITER,
        line_search=True,
        line_search_check=detF_check,
    )

    result = lc.solve()

    if not result.completed:
        print(f"  Analysis stopped at step {len(result.step_info)}")
    else:
        print(f"  All {N_LOAD_STEPS} steps converged")

    # Print convergence summary
    for si in result.step_info:
        status = "OK" if si.converged else "FAIL"
        print(f"    Step {si.step:3d}: {status} in {si.nr_iterations} iters, "
              f"|R|={si.residual_final:.2e}")

    return result


# ====================================================================
# LINEAR FORCE-DISPLACEMENT CURVE (incremental for plotting)
# ====================================================================
def linear_force_displacement(mesh, constrained_dofs, f_ext_total, n_points=20):
    """Compute linear force-displacement at the load point for multiple load levels."""
    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C_mat = material.tangent(dim=2)
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total

    # Assemble stiffness once
    rows, cols, vals = [], [], []
    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_el_dof = len(edofs)
            ke = np.zeros((n_el_dof, n_el_dof))
            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords
                detJ = abs(np.linalg.det(J))
                dN_dx = dN_dxi @ np.linalg.inv(J)
                B = build_B_matrix(dN_dx)
                ke += (B.T @ C_mat @ B) * detJ * qw * THICKNESS
            for il, ig in enumerate(edofs):
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke[il, jl])

    K = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                          shape=(n_dof, n_dof))

    K_lil = K.tolil()
    for dof in constrained_dofs:
        K_lil[dof, :] = 0
        K_lil[:, dof] = 0
        K_lil[dof, dof] = 1.0
    K_bc = K_lil.tocsr()

    from scipy.sparse.linalg import spsolve

    # Solve for unit load and scale
    f_unit = f_ext_total / np.linalg.norm(f_ext_total[f_ext_total != 0])
    f_unit[constrained_dofs] = 0.0
    u_unit = spsolve(K_bc.tocsc(), f_unit)

    fracs = np.linspace(0, 1, n_points + 1)
    disps = []
    forces = []
    for frac in fracs:
        u = frac * np.linalg.norm(f_ext_total[f_ext_total != 0]) * u_unit
        disps.append(u[load_dof_y])
        forces.append(frac * F_TOTAL)

    return np.array(forces), np.array(disps)


# ====================================================================
# VISUALISATION
# ====================================================================
def plot_mesh(mesh):
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    fig, ax = plt.subplots(figsize=(14, 3))
    polys = [nd[conn[e]] for e in range(conn.shape[0])]
    pc = PolyCollection(polys, edgecolors="steelblue", facecolors="none", linewidths=0.4)
    ax.add_collection(pc)
    left = mesh.boundary_tags.get("left", np.array([]))
    if len(left) > 0:
        ax.plot(nd[left, 0], nd[left, 1], "s", color="blue", ms=4, label="Clamped")
    ax.plot(nd[load_node, 0], nd[load_node, 1], "v", color="red", ms=10, label="Load point")
    ax.set_aspect("equal"); ax.legend(); ax.autoscale()
    ax.set_title("Mesh ({} Quad4, {} nodes)".format(mesh.n_elements, mesh.n_nodes))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mesh.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_force_displacement(forces_lin, disps_lin, result_nl):
    """Compare force-displacement curves."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Linear
    ax.plot(np.abs(disps_lin) * 1e3, forces_lin / 1e3, "b-o", lw=2, ms=4,
            label="Linear (small disp.)")

    # Nonlinear
    nl_forces = [frac * F_TOTAL / 1e3 for frac in result_nl.load_fractions]
    nl_disps = []
    for u_vec in result_nl.u_history:
        nl_disps.append(abs(u_vec[load_dof_y]) * 1e3)

    ax.plot(nl_disps, nl_forces, "r-s", lw=2, ms=4,
            label="Nonlinear (SVK, large disp.)")

    ax.set_xlabel("Tip displacement |$u_y$| (mm)")
    ax.set_ylabel("Applied force (kN)")
    ax.set_title("Force-displacement curve: small vs large displacement theory")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "force_displacement.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_deformed_shapes(mesh, u_lin, u_nl_final):
    """Compare deformed shapes."""
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    if mesh.element_blocks[0].etype == "quad4":
        tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    else:
        tri = conn

    disp_lin = u_lin.reshape(-1, 2)
    disp_nl = u_nl_final.reshape(-1, 2)
    umag_lin = np.sqrt(disp_lin[:, 0]**2 + disp_lin[:, 1]**2)
    umag_nl = np.sqrt(disp_nl[:, 0]**2 + disp_nl[:, 1]**2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Undeformed
    ax = axes[0]
    triang_ref = mtri.Triangulation(nd[:, 0], nd[:, 1], tri)
    ax.triplot(triang_ref, lw=0.4, color="steelblue")
    ax.plot(nd[load_node, 0], nd[load_node, 1], "v", color="red", ms=10)
    ax.set_aspect("equal"); ax.set_title("Undeformed")
    ax.set_xlim(-0.5, L + 1)

    # Linear deformed (scale down if too large)
    ax = axes[1]
    scale_lin = min(1.0, 0.5 * L / max(np.abs(disp_lin).max(), 1e-10))
    def_lin = nd + scale_lin * disp_lin
    triang_lin = mtri.Triangulation(def_lin[:, 0], def_lin[:, 1], tri)
    tcf = ax.tricontourf(triang_lin, umag_lin, levels=20, cmap="viridis")
    ax.triplot(triang_lin, lw=0.2, color="k", alpha=0.1)
    ax.triplot(triang_ref, lw=0.15, color="gray", alpha=0.3, ls="--")
    fig.colorbar(tcf, ax=ax, label="|u| (m)", shrink=0.6)
    ax.set_aspect("equal")
    scale_label = "" if scale_lin == 1.0 else " (x{:.1f})".format(scale_lin)
    ax.set_title("Linear — deformed shape{}".format(scale_label))

    # Nonlinear deformed (true scale)
    ax = axes[2]
    def_nl = nd + disp_nl  # true scale
    triang_nl = mtri.Triangulation(def_nl[:, 0], def_nl[:, 1], tri)
    tcf = ax.tricontourf(triang_nl, umag_nl, levels=20, cmap="viridis")
    ax.triplot(triang_nl, lw=0.2, color="k", alpha=0.1)
    ax.triplot(triang_ref, lw=0.15, color="gray", alpha=0.3, ls="--")
    fig.colorbar(tcf, ax=ax, label="|u| (m)", shrink=0.6)
    ax.set_aspect("equal")
    ax.set_title("Nonlinear (SVK) — deformed shape (true scale)")

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "deformed_shapes.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_load_steps(result_nl):
    """Plot intermediate deformed shapes at several load steps."""
    n_show = min(5, len(result_nl.u_history))
    indices = np.linspace(0, len(result_nl.u_history) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    conn = mesh.element_blocks[0].connectivity
    if mesh.element_blocks[0].etype == "quad4":
        tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    else:
        tri = conn
    nd = mesh.nodes

    for ax_idx, step_idx in enumerate(indices):
        ax = axes[ax_idx]
        u = result_nl.u_history[step_idx]
        disp = u.reshape(-1, 2)
        def_nodes = nd + disp
        umag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)

        triang = mtri.Triangulation(def_nodes[:, 0], def_nodes[:, 1], tri)
        tcf = ax.tricontourf(triang, umag, levels=20, cmap="viridis")
        ax.triplot(triang, lw=0.2, color="k", alpha=0.1)
        fig.colorbar(tcf, ax=ax, label="|u| (m)", shrink=0.6)
        ax.set_aspect("equal")

        frac = result_nl.load_fractions[step_idx]
        ax.set_title("Load = {:.0f}% ({:.0f} kN)".format(
            100 * frac, frac * F_TOTAL / 1e3))

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "load_steps.png", dpi=200, bbox_inches="tight")
    plt.show()


# ====================================================================
# MAIN
# ====================================================================
# Global references (set in main, used by plotting)
load_node = None
load_dof_y = None
mesh = None

def main():
    global load_node, load_dof_y, mesh

    print("=" * 60)
    print("GEOMETRIC NONLINEARITY: SMALL vs LARGE DISPLACEMENTS")
    print("=" * 60)
    print(f"  Beam: {L} x {H} m, thickness = {THICKNESS} m")
    print(f"  Material: E = {E_YOUNG/1e9:.0f} GPa, nu = {NU_POISSON}")
    print(f"  Load: F = {F_TOTAL/1e3:.0f} kN (downward at tip)")

    # ── Mesh ──
    mesh = rectangle_mesh((0, L), (0, H), NX, NY, etype="quad4")
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total
    print(f"\n  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} Quad4, {n_dof} DOFs")

    # Identify boundaries
    tol = 1e-10
    mesh.boundary_tags["left"] = np.where(mesh.nodes[:, 0] < tol)[0]
    mesh.boundary_tags["right"] = np.where(mesh.nodes[:, 0] > L - tol)[0]

    # Load node: bottom-right corner
    dists = np.sqrt((mesh.nodes[:, 0] - L)**2 + (mesh.nodes[:, 1] - 0)**2)
    load_node = np.argmin(dists)
    load_dof_y = 2 * load_node + 1
    print(f"  Load node: {load_node} at ({mesh.nodes[load_node, 0]:.2f}, "
          f"{mesh.nodes[load_node, 1]:.2f})")

    # Constrained DOFs (clamped left)
    left = mesh.boundary_tags["left"]
    constrained_dofs = np.array([d for nid in left for d in [2*nid, 2*nid+1]])
    print(f"  Clamped: {len(left)} nodes, {len(constrained_dofs)} DOFs")

    # External force vector
    f_ext = np.zeros(n_dof)
    f_ext[load_dof_y] = -F_TOTAL

    plot_mesh(mesh)

    # ── Tangent verification ──
    print("\n--- Tangent verification ---")
    material_svk = StVenantKirchhoff.from_E_nu(E=E_YOUNG, nu=NU_POISSON)
    F_test = np.array([[1.05, 0.02], [-0.01, 0.98]])
    P0, A_anal, _ = material_svk.pk1_stress_and_tangent_2d(F_test, plane_stress=True)

    A_num = np.zeros((2, 2, 2, 2))
    eps = 1e-7
    for k in range(2):
        for L_idx in range(2):
            F_p = F_test.copy(); F_p[k, L_idx] += eps
            F_m = F_test.copy(); F_m[k, L_idx] -= eps
            P_p, _ = material_svk.pk1_stress_2d(F_p, plane_stress=True)
            P_m, _ = material_svk.pk1_stress_2d(F_m, plane_stress=True)
            A_num[:, :, k, L_idx] = (P_p - P_m) / (2.0 * eps)

    max_diff = np.max(np.abs(A_anal - A_num))
    max_val = np.max(np.abs(A_anal))
    rel_err = max_diff / max_val if max_val > 0 else 0
    print(f"  SVK tangent: max |A_anal - A_num| = {max_diff:.4e}, "
          f"relative = {rel_err:.4e}")
    if rel_err > 1e-4:
        print("  WARNING: Tangent may be inconsistent!")
    else:
        print("  Tangent OK")

    # ── Cross-check: PK1 and B-matrix assemblies must agree ──
    rel_f, rel_K = verify_formulations_agree(mesh, dof_map, material_svk)
    print(f"  PK1 vs B-matrix: rel |df_int| = {rel_f:.2e}, "
          f"rel |dK_T| = {rel_K:.2e}")

    # ── Linear solve ──
    u_lin = solve_linear(mesh, f_ext, constrained_dofs)

    # ── Linear force-displacement curve ──
    forces_lin, disps_lin = linear_force_displacement(
        mesh, constrained_dofs, f_ext, n_points=N_LOAD_STEPS
    )

    # ── Nonlinear solve ──
    result_nl = solve_nonlinear(mesh, f_ext, constrained_dofs, material_svk)

    # ── Comparison ──
    u_nl_final = result_nl.u_history[-1]
    disp_lin = u_lin.reshape(-1, 2)
    disp_nl = u_nl_final.reshape(-1, 2)

    print("\n" + "=" * 60)
    print("COMPARISON AT FULL LOAD")
    print("=" * 60)
    print(f"  Linear  tip uy: {disp_lin[load_node, 1]*1e3:.2f} mm")
    print(f"  Nonlinear tip uy: {disp_nl[load_node, 1]*1e3:.2f} mm")
    print(f"  Ratio (lin/nl):  {abs(disp_lin[load_node, 1] / disp_nl[load_node, 1]):.2f}")

    # ── Plots ──
    plot_force_displacement(forces_lin, disps_lin, result_nl)
    plot_deformed_shapes(mesh, u_lin, u_nl_final)
    plot_load_steps(result_nl)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures saved to: {_FIG_DIR}")


if __name__ == "__main__":
    main()