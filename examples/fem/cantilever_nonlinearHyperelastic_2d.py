#!/usr/bin/env python3
"""
Εργασία 3 — Material nonlinearity: linear elastic vs decoupled Neo-Hookean.

Problem:
    - Rectangular body: 4m × 1m, thickness 0.2m
    - Material: E = 50 GPa, ν = 0.3
    - Left edge clamped (u_x = u_y = 0)
    - Right edge: prescribed horizontal displacement ū = 1.50 m
    - Plane strain (toggleable; see PLANE_STRESS below)

Analysis:
    1. Linear elastic: single solve with prescribed displacement
    2. Neo-Hookean: Newton-Raphson with incremental displacement control
    3. Compare force-displacement curves and stress fields

Neo-Hookean model (decoupled deviatoric-volumetric, as in the assignment):
    W(F) = (μ/2)(Ī₁ - 3) + (K/2)(J - 1)²,   Ī₁ = J^(-2/3) I₁
    μ = E/(2(1+ν)) = 19.23 GPa,  K = E/(3(1-2ν)) = 41.67 GPa
    Chosen so the initial tangent matches the linear elastic response.

Toggles:
    PLANE_STRESS : False -> plane strain (the assignment), True -> plane stress
    FORMULATION  : "pk1" (P:∇N) or "pk2" (B-matrix, 2nd PK + Green-Lagrange)
                   The two are equivalent and agree to machine precision.

Run:       python examples/fem/cantilever_nonlinearHyperelastic_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve

from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.core.dof import DOFMap
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature
from simulacrum.materials.linear_elastic import PlaneStress
from simulacrum.materials.hyperelastic import NeoHookean, NeoHookeanDecoupled
from simulacrum.solvers.nonlinear import DisplacementControl

# ====================================================================
# CONFIGURATION
# ====================================================================
_FIG_DIR = Path(__file__).resolve().parent / "Figures_cantilever_nonlinear"
_FIG_DIR.mkdir(exist_ok=True)

# Geometry
L = 4.0         # length (m)
H = 1.0         # height (m)
THICKNESS = 0.20  # thickness (m)

# Material
E_YOUNG = 50e9     # Pa
NU_POISSON = 0.3

# Prescribed displacement
U_PRESCRIBED = 1.50  # m (horizontal, on right edge)

# Mesh
MESH_LC = 0.1
ETYPE = "quad4"

# Nonlinear solver
N_LOAD_STEPS = 50        # number of increments
NR_TOL = 1e-6            # Newton-Raphson tolerance (relative residual)
NR_MAX_ITER = 50         # max Newton iterations per load step

# Idealisation: False -> plane strain (the assignment), True -> plane stress
PLANE_STRESS = False

# Internal force / tangent formulation:
#   "pk1" -> f_int = P : grad(N),  K_T from dP/dF
#   "pk2" -> f_int = B_L^T S,  K_T = B_L^T C B_L + B_grad^T S_bar B_grad
#            (2nd Piola-Kirchhoff + Green-Lagrange, B-matrix form)
# Equivalent; they agree to machine precision.
FORMULATION = "pk2"      # "pk1" or "pk2"


# ====================================================================
# MESH GENERATION
# ====================================================================
def mesh_cantilever_gmsh(L, H, lc, etype="quad4"):
    """Generate mesh for a rectangular cantilever."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cantilever")
    fac = gmsh.model.occ

    fac.addRectangle(0, 0, 0, L, H)
    fac.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    if etype == "quad4":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes_all = coords.reshape(-1, 3)[:, :2]

    etypes_gmsh, _, enodes = gmsh.model.mesh.getElements(dim=2)
    conn_raw, sim_etype = None, None
    for eid, en in zip(etypes_gmsh, enodes):
        _, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(eid)
        conn_raw = np.array([node_map[int(n)] for n in en]).reshape(-1, npe)
        sim_etype = "quad4" if npe == 4 else "tri3"

    # Filter orphan nodes
    used_nodes = np.unique(conn_raw.ravel())
    old_to_new = np.full(len(nodes_all), -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(len(used_nodes))
    nodes = nodes_all[used_nodes]
    conn = old_to_new[conn_raw]

    _, _, bnd_en = gmsh.model.mesh.getElements(dim=1)
    bnd_all = set()
    for en in bnd_en:
        for n in en:
            old_idx = node_map[int(n)]
            if old_to_new[old_idx] >= 0:
                bnd_all.add(old_to_new[old_idx])
    bnd_all = np.array(sorted(bnd_all), dtype=np.int64)

    gmsh.finalize()

    tol = lc * 0.5
    left = np.where(nodes[:, 0] < tol)[0]
    right = np.where(nodes[:, 0] > L - tol)[0]

    mesh = Mesh(nodes=nodes, element_blocks=[ElementBlock(sim_etype, conn)],
                boundary_tags={"boundary": bnd_all, "left": left, "right": right})
    return mesh


# ====================================================================
# B-MATRIX FOR ELASTICITY
# ====================================================================
def build_B_matrix(dN_dx):
    """Build Voigt strain-displacement matrix (small strain)."""
    n_nodes = dN_dx.shape[0]
    B = np.zeros((3, n_nodes * 2))
    for i in range(n_nodes):
        B[0, 2 * i] = dN_dx[i, 0]
        B[1, 2 * i + 1] = dN_dx[i, 1]
        B[2, 2 * i] = dN_dx[i, 1]
        B[2, 2 * i + 1] = dN_dx[i, 0]
    return B


def linear_elastic_C(plane_stress):
    """Small-strain isotropic Voigt elasticity matrix [xx, yy, xy]."""
    E, nu = E_YOUNG, NU_POISSON
    if plane_stress:
        c = E / (1.0 - nu**2)
        return np.array([
            [c,      c * nu, 0.0            ],
            [c * nu, c,      0.0            ],
            [0.0,    0.0,    c * (1 - nu) / 2.0],
        ])
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array([
        [lam + 2 * mu, lam,          0.0],
        [lam,          lam + 2 * mu, 0.0],
        [0.0,          0.0,          mu ],
    ])


# ====================================================================
# LINEAR ELASTIC SOLVE (with prescribed displacement)
# ====================================================================
def solve_linear(mesh, u_prescribed):
    """Solve linear elasticity with prescribed displacement on right edge."""
    print("\n" + "=" * 60)
    print("LINEAR ELASTIC ANALYSIS")
    print("=" * 60)

    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C_mat = linear_elastic_C(PLANE_STRESS)
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

    # Apply BCs
    left = mesh.boundary_tags["left"]
    right = mesh.boundary_tags["right"]

    # Build prescribed displacement vector
    u = np.zeros(n_dof)
    rhs = np.zeros(n_dof)

    # Prescribed DOFs
    fixed_dofs = {}
    for nid in left:
        fixed_dofs[2 * nid] = 0.0       # ux = 0
        fixed_dofs[2 * nid + 1] = 0.0   # uy = 0
    for nid in right:
        fixed_dofs[2 * nid] = u_prescribed  # ux = ū

    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
    fixed_dof_list = sorted(fixed_dofs.keys())
    fixed_vals = np.array([fixed_dofs[d] for d in fixed_dof_list])

    # Partition and solve: K_ff u_f = -K_fp u_p
    K_dense = K.toarray()
    K_ff = K_dense[np.ix_(free_dofs, free_dofs)]
    K_fp = K_dense[np.ix_(free_dofs, fixed_dof_list)]

    rhs_f = -K_fp @ fixed_vals
    u_f = np.linalg.solve(K_ff, rhs_f)

    u[free_dofs] = u_f
    for d, v in fixed_dofs.items():
        u[d] = v

    # Reaction forces
    reactions = K @ u
    rx_total = sum(reactions[2 * nid] for nid in right)

    disp = u.reshape(-1, 2)
    u_mag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)

    print(f"  max |u| = {u_mag.max():.6e} m")
    print(f"  Reaction on right edge (x): {rx_total:.2f} N = {rx_total/1e3:.2f} kN")

    return u, disp, u_mag, rx_total


# ====================================================================
# NONLINEAR ASSEMBLY (Total Lagrangian)
# ====================================================================
def compute_deformation_gradient(dN_dX, u_elem):
    """
    Compute deformation gradient F = I + ∂u/∂X at a quadrature point.

    Args:
        dN_dX: (n_nodes, 2) shape function gradients w.r.t. reference coords
        u_elem: (n_nodes, 2) nodal displacements

    Returns:
        F: (2, 2) deformation gradient
    """
    grad_u = u_elem.T @ dN_dX  # (2, 2): [∂u₁/∂X₁  ∂u₁/∂X₂]
                                #          [∂u₂/∂X₁  ∂u₂/∂X₂]
    F = np.eye(2) + grad_u
    return F


def assemble_internal_force_and_tangent(mesh, dof_map, u, material, plane_stress=True):
    """
    Assemble internal force vector and tangent stiffness for Total Lagrangian.

    Uses PK1 stress P and tangent A = ∂P/∂F from the material.

    Internal force:
        f_int[2*I + i] = ∫ P_iJ * ∂N_I/∂X_J dΩ₀

    Tangent stiffness:
        K_T[2*I+i, 2*K+k] = ∫ A_iJkL * ∂N_I/∂X_J * ∂N_K/∂X_L dΩ₀
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
            coords = mesh.nodes[enodes]  # reference coordinates
            u_elem = disp[enodes]         # nodal displacements
            edofs = dof_map.element_dofs(enodes)
            n_nodes_e = len(enodes)
            n_el_dof = len(edofs)

            fe_int = np.zeros(n_el_dof)
            ke_t = np.zeros((n_el_dof, n_el_dof))

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J_ref = dN_dxi.T @ coords
                detJ_ref = abs(np.linalg.det(J_ref))
                dN_dX = dN_dxi @ np.linalg.inv(J_ref)  # (n_nodes, 2)

                # Deformation gradient
                F = compute_deformation_gradient(dN_dX, u_elem)

                # PK1 stress and tangent from material
                P, A, lam3 = material.pk1_stress_and_tangent_2d(
                    F, plane_stress=plane_stress
                )

                w = detJ_ref * qw * THICKNESS

                # Internal force: f_int[2*I+i] += P_iJ * dN_I/dX_J * w
                for I in range(n_nodes_e):
                    for i in range(2):
                        val = 0.0
                        for J_idx in range(2):
                            val += P[i, J_idx] * dN_dX[I, J_idx]
                        fe_int[2 * I + i] += val * w

                # Tangent: K_T[2*I+i, 2*K+k] += A_iJkL * dN_I/dX_J * dN_K/dX_L * w
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

            # Scatter
            for il, ig in enumerate(edofs):
                f_int[ig] += fe_int[il]
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke_t[il, jl])

    K_T = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                            shape=(n_dof, n_dof))
    return f_int, K_T


# ====================================================================
# PK2 / B-MATRIX FORMULATION (2nd Piola-Kirchhoff + Green-Lagrange)
# ====================================================================
def build_B_grad(dN_dX):
    """B_nabla (4, 2n): nodal increments -> [u,X ; v,Y ; u,Y ; v,X]."""
    n = dN_dX.shape[0]
    Bg = np.zeros((4, 2 * n))
    for a in range(n):
        Nx, Ny = dN_dX[a, 0], dN_dX[a, 1]
        Bg[0, 2 * a]     = Nx
        Bg[1, 2 * a + 1] = Ny
        Bg[2, 2 * a]     = Ny
        Bg[3, 2 * a + 1] = Nx
    return Bg


def build_A_L(grad_u):
    """A_L (3, 4): displacement-gradient vector -> Green-Lagrange Voigt."""
    uX, uY = grad_u[0, 0], grad_u[0, 1]
    vX, vY = grad_u[1, 0], grad_u[1, 1]
    return np.array([
        [1.0 + uX, 0.0,      0.0,       vX      ],
        [0.0,      1.0 + vY, uY,        0.0     ],
        [uY,       vX,       1.0 + uX,  1.0 + vY],
    ])


def build_S_bar(S_voigt):
    """S_bar (4, 4) initial-stress matrix from 2nd PK [S_XX, S_YY, S_XY]."""
    Sxx, Syy, Sxy = S_voigt
    return np.array([
        [Sxx, 0.0, Sxy, 0.0],
        [0.0, Syy, 0.0, Sxy],
        [Sxy, 0.0, Syy, 0.0],
        [0.0, Sxy, 0.0, Sxx],
    ])


def assemble_internal_force_and_tangent_pk2(mesh, dof_map, u, material,
                                            plane_stress=True):
    """
    Total Lagrangian assembly in the B-matrix (PK2) form:
        f_int = ∫ B_L^T S dV0
        K_T   = ∫ B_L^T C B_L dV0 + ∫ B_grad^T S_bar B_grad dV0
    with (S, C) the 2nd PK stress and Lagrangian tangent from the material.
    Equivalent to assemble_internal_force_and_tangent (PK1).
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

                grad_u = u_elem.T @ dN_dX
                F = np.eye(2) + grad_u

                S_voigt, C_voigt, _ = material.pk2_voigt_and_tangent_2d(
                    F, plane_stress)

                B_grad = build_B_grad(dN_dX)
                B_L = build_A_L(grad_u) @ B_grad
                S_bar = build_S_bar(S_voigt)

                w = detJ_ref * qw * THICKNESS
                fe_int += (B_L.T @ S_voigt) * w
                ke_t   += (B_L.T @ C_voigt @ B_L) * w
                ke_t   += (B_grad.T @ S_bar @ B_grad) * w

            for il, ig in enumerate(edofs):
                f_int[ig] += fe_int[il]
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke_t[il, jl])

    K_T = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                            shape=(n_dof, n_dof))
    return f_int, K_T


def assemble_nonlinear(mesh, dof_map, u, material, plane_stress=True):
    """Dispatch to the active formulation (see FORMULATION)."""
    if FORMULATION == "pk1":
        return assemble_internal_force_and_tangent(
            mesh, dof_map, u, material, plane_stress)
    elif FORMULATION == "pk2":
        return assemble_internal_force_and_tangent_pk2(
            mesh, dof_map, u, material, plane_stress)
    else:
        raise ValueError(f"Unknown FORMULATION '{FORMULATION}' (use 'pk1'/'pk2')")


# ====================================================================
# NEWTON-RAPHSON WITH DISPLACEMENT CONTROL
# ====================================================================
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

def solve_nonlinear(mesh, material, u_total, n_steps, plane_stress=True):
    print("\n" + "=" * 60)
    print("NONLINEAR (NEO-HOOKEAN) ANALYSIS")
    print("=" * 60)
    print(f"  Idealisation: {'plane stress' if plane_stress else 'plane strain'}")
    print(f"  Formulation:  {FORMULATION}")

    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total

    left = mesh.boundary_tags["left"]
    right = mesh.boundary_tags["right"]

    # DOF sets
    fixed_dofs = []
    for nid in left:
        fixed_dofs.extend([2 * nid, 2 * nid + 1])

    # # enforce midline symmetry: uy = 0 on y = H/2 (kills the bow, keeps necking)
    # mid = np.where(np.abs(mesh.nodes[:, 1] - H / 2.0) < 1e-6)[0]
    fixed_dofs.extend(int(2 * nid + 1) for nid in mid)

    fixed_dofs = np.array(fixed_dofs)

    prescribed_dofs = {2 * nid: u_total for nid in right}

    all_constrained = np.union1d(fixed_dofs, np.array(list(prescribed_dofs.keys())))
    free_dofs = np.array([d for d in range(n_dof) if d not in all_constrained])

    # Wrap assembly into callables
    def f_int_fn(u):
        f_int, _ = assemble_nonlinear(
            mesh, dof_map, u, material, plane_stress)
        return f_int

    def K_T_fn(u):
        _, K_T = assemble_nonlinear(
            mesh, dof_map, u, material, plane_stress)
        return K_T

    def detF_check(u):
        return check_detF(mesh, dof_map, u)

    # Create solver
    dc = DisplacementControl(
        internal_force_fn=f_int_fn,
        tangent_fn=K_T_fn,
        n_dof=n_dof,
        free_dofs=free_dofs,
        prescribed_dofs=prescribed_dofs,
        fixed_dofs=fixed_dofs,
        n_steps=n_steps,
        tol=NR_TOL,
        max_iter=NR_MAX_ITER,
        line_search=True,
        line_search_check=detF_check,
    )

    result = dc.solve()

    if not result.completed:
        print("  Analysis stopped early!")

    return result.u_history, result.reaction_history, result.load_fractions



# ====================================================================
# VISUALISATION
# ====================================================================
def plot_mesh(mesh):
    """Plot the mesh."""
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity

    fig, ax = plt.subplots(figsize=(12, 4))
    polys = [nd[conn[e]] for e in range(conn.shape[0])]
    pc = PolyCollection(polys, edgecolors="steelblue", facecolors="none", linewidths=0.4)
    ax.add_collection(pc)

    left = mesh.boundary_tags["left"]
    right = mesh.boundary_tags["right"]
    ax.plot(nd[left, 0], nd[left, 1], "s", color="blue", ms=4, label="Clamped")
    ax.plot(nd[right, 0], nd[right, 1], ">", color="red", ms=4, label="Prescribed ux")
    ax.set_aspect("equal"); ax.legend(); ax.autoscale()
    ax.set_title("{} mesh ({} elements, {} nodes)".format(
        ETYPE.upper(), mesh.n_elements, mesh.n_nodes))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mesh.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_force_displacement(u_lin, rx_lin, u_hist_nl, rx_hist_nl, load_fracs_nl):
    """Compare force-displacement curves."""
    right_node = None  # will be filled

    fig, ax = plt.subplots(figsize=(10, 6))

    # Linear: single point + line from origin
    ax.plot([0, U_PRESCRIBED * 1e3], [0, rx_lin / 1e3], "b-o",
            lw=2, ms=8, label="Linear elastic", zorder=5)

    # Nonlinear: incremental curve
    u_right = [frac * U_PRESCRIBED * 1e3 for frac in load_fracs_nl]
    rx_nl = [r / 1e3 for r in rx_hist_nl]
    ax.plot(u_right, rx_nl, "r-s", lw=2, ms=5, label="Neo-Hookean")

    ax.set_xlabel("Prescribed displacement (mm)")
    ax.set_ylabel("Reaction force (kN)")
    ax.set_title("Force-displacement curve: Linear vs Neo-Hookean")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "force_displacement.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_deformed_shapes(mesh, u_lin, u_nl_final):
    """Compare deformed shapes at final load."""
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
    vmax = max(umag_lin.max(), umag_nl.max())

    scale = 1.0  # true scale (prescribed 0.2m is already visible)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, disp, umag, title in [
        (axes[0], disp_lin, umag_lin, "Linear elastic"),
        (axes[1], disp_nl, umag_nl, "Neo-Hookean"),
    ]:
        def_nodes = nd + scale * disp
        triang = mtri.Triangulation(def_nodes[:, 0], def_nodes[:, 1], tri)
        triang_ref = mtri.Triangulation(nd[:, 0], nd[:, 1], tri)

        tcf = ax.tricontourf(triang, umag, levels=20, cmap="viridis", vmin=0, vmax=vmax)
        ax.triplot(triang, lw=0.2, color="k", alpha=0.1)
        ax.triplot(triang_ref, lw=0.15, color="gray", alpha=0.3, ls="--")
        fig.colorbar(tcf, ax=ax, label="|u| (m)", shrink=0.7)
        ax.set_aspect("equal")
        ax.set_title("{} — deformed shape (true scale)".format(title))

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "deformed_shapes.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_stress_comparison(mesh, u_lin, u_nl, material_lin, material_nl):
    """Compare stress fields (von Mises) between linear and nonlinear."""
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity

    if mesh.element_blocks[0].etype == "quad4":
        tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    else:
        tri = conn

    # Compute stresses at nodes (averaged from elements)
    vm_lin = compute_von_mises(mesh, u_lin, material_lin, is_linear=True)
    vm_nl = compute_von_mises(mesh, u_nl, material_nl, is_linear=False)

    vmax = max(vm_lin.max(), vm_nl.max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, vm, disp_vec, title in [
        (axes[0], vm_lin, u_lin.reshape(-1, 2), "Linear elastic"),
        (axes[1], vm_nl, u_nl.reshape(-1, 2), "Neo-Hookean"),
    ]:
        def_nodes = nd + disp_vec
        triang = mtri.Triangulation(def_nodes[:, 0], def_nodes[:, 1], tri)
        tcf = ax.tricontourf(triang, vm / 1e6, levels=20, cmap="hot_r",
                             vmin=0, vmax=vmax / 1e6)
        ax.triplot(triang, lw=0.15, color="k", alpha=0.1)
        fig.colorbar(tcf, ax=ax, label="von Mises (MPa)", shrink=0.7)
        ax.set_aspect("equal")
        ax.set_title("{} — von Mises stress on deformed shape".format(title))

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "stress_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()


def compute_von_mises(mesh, u, material, is_linear=True):
    """Compute nodal von Mises stress (element-averaged at nodes).

    Uses the full 3D stress: σ₃₃ = 0 for plane stress, and
    σ₃₃ = ν(σ₁₁+σ₂₂) (linear) / the material's out-of-plane stress
    (nonlinear) for plane strain.
    """
    disp = u.reshape(-1, 2)
    vm_sum = np.zeros(mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    C_mat = linear_elastic_C(PLANE_STRESS) if is_linear else None

    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)

        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            u_elem = disp[enodes]

            sigma_avg = np.zeros((3, 3))   # full 3D stress (with σ₃₃)
            total_w = 0.0

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords
                detJ = abs(np.linalg.det(J))
                dN_dX = dN_dxi @ np.linalg.inv(J)

                sigma = np.zeros((3, 3))
                if is_linear:
                    B = build_B_matrix(dN_dX)
                    sv = C_mat @ (B @ u_elem.ravel())
                    sigma[0, 0], sigma[1, 1], sigma[0, 1] = sv[0], sv[1], sv[2]
                    sigma[1, 0] = sv[2]
                    if not PLANE_STRESS:
                        sigma[2, 2] = NU_POISSON * (sv[0] + sv[1])
                else:
                    F = compute_deformation_gradient(dN_dX, u_elem)
                    # NeoHookeanDecoupled.cauchy_stress_2d returns full 3x3
                    sigma = material.cauchy_stress_2d(F, plane_stress=PLANE_STRESS)

                sigma_avg += sigma * detJ * qw
                total_w += detJ * qw

            sigma_avg /= total_w

            s11, s22, s33 = sigma_avg[0, 0], sigma_avg[1, 1], sigma_avg[2, 2]
            s12 = sigma_avg[0, 1]
            vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2)
                         + 3.0 * s12**2)

            for nid in enodes:
                vm_sum[nid] += vm
                count[nid] += 1

    count[count == 0] = 1
    return vm_sum / count


def plot_convergence(u_hist, reaction_hist, mesh):
    """Plot NR convergence per load step (number of iterations)."""
    # This is a simple summary — the detailed convergence is printed during solve
    fig, ax = plt.subplots(figsize=(10, 4))

    disps = [frac * U_PRESCRIBED * 1e3 for frac in
             np.linspace(0, 1, len(reaction_hist))]
    reactions = [r / 1e3 for r in reaction_hist]

    ax.plot(disps, reactions, "r-o", ms=5)
    ax.set_xlabel("Prescribed displacement (mm)")
    ax.set_ylabel("Reaction force (kN)")
    ax.set_title("Load-displacement path (Neo-Hookean)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "load_path_nl.png", dpi=200, bbox_inches="tight")
    plt.show()


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("NONLINEAR ANALYSIS: LINEAR vs NEO-HOOKEAN")
    print("=" * 60)
    print(f"  Geometry: {L} x {H} m, thickness = {THICKNESS} m")
    print(f"  Material: E = {E_YOUNG/1e9:.0f} GPa, ν = {NU_POISSON}")
    print(f"  Prescribed displacement: ū = {U_PRESCRIBED} m")
    print(f"  Idealisation: {'plane stress' if PLANE_STRESS else 'plane strain'}")

    # Material parameters (decoupled Neo-Hookean: μ shear, K bulk)
    mu = E_YOUNG / (2.0 * (1.0 + NU_POISSON))
    K_bulk = E_YOUNG / (3.0 * (1.0 - 2.0 * NU_POISSON))
    print(f"  Neo-Hookean: μ = {mu/1e9:.4f} GPa, K = {K_bulk/1e9:.4f} GPa")

    # Mesh
    print("\n--- Mesh ---")
    mesh = mesh_cantilever_gmsh(L, H, MESH_LC, etype=ETYPE)
    print(f"  {mesh.n_nodes} nodes, {mesh.n_elements} {ETYPE.upper()} elements")
    print(f"  Left (clamped): {len(mesh.boundary_tags['left'])} nodes")
    print(f"  Right (prescribed): {len(mesh.boundary_tags['right'])} nodes")
    plot_mesh(mesh)

    # Linear elastic solve
    u_lin, disp_lin, umag_lin, rx_lin = solve_linear(mesh, U_PRESCRIBED)


    # Neo-Hookean nonlinear solve (decoupled deviatoric-volumetric model)
    material_nl = NeoHookeanDecoupled.from_E_nu(E=E_YOUNG, nu=NU_POISSON)


    # ── Tangent verification (run once, then remove) ──
    print("\n  Verifying material tangent with finite differences...")
    F_test = np.array([[1.02, 0.01], [0.005, 0.99]])
    P0, A_analytical, _ = material_nl.pk1_stress_and_tangent_2d(F_test, plane_stress=PLANE_STRESS)

    A_numerical = np.zeros((2, 2, 2, 2))
    eps = 1e-7
    for k in range(2):
        for L_idx in range(2):
            F_plus = F_test.copy()
            F_plus[k, L_idx] += eps
            P_plus, _ = material_nl.pk1_stress_2d(F_plus, plane_stress=PLANE_STRESS)

            F_minus = F_test.copy()
            F_minus[k, L_idx] -= eps
            P_minus, _ = material_nl.pk1_stress_2d(F_minus, plane_stress=PLANE_STRESS)

            A_numerical[:, :, k, L_idx] = (P_plus - P_minus) / (2.0 * eps)

    max_diff = np.max(np.abs(A_analytical - A_numerical))
    max_val = np.max(np.abs(A_analytical))
    print(f"    Tangent check: max |A_anal - A_num| = {max_diff:.4e}")
    print(f"    Relative error: {max_diff/max_val:.4e}")
    if max_diff / max_val > 1e-4:
        print("    WARNING: Tangent may be inconsistent!")
    else:
        print("    Tangent OK")

    # ── PK1 vs PK2 element-assembly cross-check ──
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    rng = np.random.default_rng(0)
    u_rand = 1e-3 * rng.standard_normal(dof_map.n_total)
    f1, K1 = assemble_internal_force_and_tangent(mesh, dof_map, u_rand, material_nl, PLANE_STRESS)
    f2, K2 = assemble_internal_force_and_tangent_pk2(mesh, dof_map, u_rand, material_nl, PLANE_STRESS)
    rel_f = np.linalg.norm(f1 - f2) / max(np.linalg.norm(f1), 1e-30)
    rel_K = sparse.linalg.norm(K1 - K2) / max(sparse.linalg.norm(K1), 1e-30)
    print(f"  PK1 vs PK2: rel |df_int| = {rel_f:.2e}, rel |dK_T| = {rel_K:.2e}")


    u_hist_nl, rx_hist_nl, load_fracs_nl = solve_nonlinear(
        mesh, material_nl, U_PRESCRIBED, N_LOAD_STEPS, plane_stress=PLANE_STRESS
    )
    u_nl_final = u_hist_nl[-1]

    # Compare final results
    print("\n" + "=" * 60)
    print("COMPARISON AT FULL LOAD")
    print("=" * 60)
    disp_nl = u_nl_final.reshape(-1, 2)
    umag_nl = np.sqrt(disp_nl[:, 0]**2 + disp_nl[:, 1]**2)
    print(f"  Linear:     max |u| = {umag_lin.max():.6e} m, "
          f"Reaction = {rx_lin/1e3:.2f} kN")
    print(f"  Neo-Hookean: max |u| = {umag_nl.max():.6e} m, "
          f"Reaction = {rx_hist_nl[-1]/1e3:.2f} kN")
    print(f"  Reaction ratio (NL/Lin): {rx_hist_nl[-1]/rx_lin:.4f}")

    # Linear stress is computed from linear_elastic_C(PLANE_STRESS) inside
    # compute_von_mises, so no separate material object is needed here.
    material_lin = None

    # Plots
    plot_force_displacement(u_lin, rx_lin, u_hist_nl, rx_hist_nl, load_fracs_nl)
    plot_deformed_shapes(mesh, u_lin, u_nl_final)
    plot_stress_comparison(mesh, u_lin, u_nl_final, material_lin, material_nl)
    plot_convergence(u_hist_nl, rx_hist_nl, mesh)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures saved to: {_FIG_DIR}")


if __name__ == "__main__":
    main()