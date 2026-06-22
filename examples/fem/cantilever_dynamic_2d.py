#!/usr/bin/env python3
"""
Dynamic analysis of a 2D cantilever beam under time-varying load.

Problem:
    - Rectangular beam: 4m × 1m, thickness 0.2m
    - Material: Steel (E=210 GPa, ν=0.3, ρ=7850 kg/m³)
    - Left edge clamped (u_x = u_y = 0)
    - Point force at bottom-right corner: F(t) = 100 sin(20t) kN (downward)
    - Plane stress

Analysis steps:
    1. Mesh with gmsh (Quad4 or Tri3)
    2. Assemble K (stiffness) and M (consistent + lumped mass)
    3. Eigenvalue problem → natural frequencies and mode shapes
    4. Rayleigh damping: C = a*M + b*K  (using first two modes)
    5. Time integration: Newmark-β and Central Difference
    6. Compare displacement history at the loaded node

Run:       python examples/fem/cantilever_dynamic_2d.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.core.domain import Domain
from simulacrum.core.dof import DOFMap
from simulacrum.boundary import DirichletBC, apply_dirichlet_elimination
from simulacrum.solvers.linear import DirectSolver
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature
from simulacrum.discretization.fem.mass import assemble_mass_matrix
from simulacrum.solvers.eigen import solve_eigenvalue_problem, compute_rayleigh_damping
from simulacrum.solvers.dynamic import NewmarkBeta, CentralDifference
from simulacrum.materials.linear_elastic import PlaneStress

from scipy import sparse

# ====================================================================
# CONFIGURATION
# ====================================================================
_FIG_DIR = Path(__file__).resolve().parent / "Figures_cantilever_dynamic"
_FIG_DIR.mkdir(exist_ok=True)

# Geometry
L = 4.0       # length (m)
H = 1.0       # height (m)
THICKNESS = 0.20  # thickness (m)

# Material
E_YOUNG = 210e9     # Pa
NU_POISSON = 0.3
RHO = 7850.0        # kg/m³

# Loading
F_AMPLITUDE = 100e3  # 100 kN
OMEGA_LOAD = 20.0    # rad/s
T_BURST = 0.5  # load for 0.5 seconds, then free vibration

# Mesh
MESH_LC = 0.15       # element size
ETYPE = "quad4"      # "tri3" or "quad4"

# Damping (target damping ratios for first two modes)
ZETA_1 = 0.02    # 2% damping for mode 1
ZETA_2 = 0.02    # 2% damping for mode 2

# Time integration
T_END = 5.0      # seconds
DT_NEWMARK = 1e-3
DT_CENTRAL = None  # auto-computed from critical timestep

# Number of modes to compute
N_MODES = 10


# ====================================================================
# MESH GENERATION
# ====================================================================
def mesh_cantilever_gmsh(L, H, lc, etype="quad4"):
    """Generate mesh for a rectangular cantilever using gmsh."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cantilever")
    fac = gmsh.model.occ

    # Rectangle from (0,0) to (L,H)
    fac.addRectangle(0, 0, 0, L, H)
    fac.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    if etype == "quad4":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)

    # Extract nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes_all = coords.reshape(-1, 3)[:, :2]

    # Extract 2D elements
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

    # Boundary nodes (remapped)
    _, _, bnd_en = gmsh.model.mesh.getElements(dim=1)
    bnd_all = set()
    for en in bnd_en:
        for n in en:
            old_idx = node_map[int(n)]
            if old_to_new[old_idx] >= 0:
                bnd_all.add(old_to_new[old_idx])
    bnd_all = np.array(sorted(bnd_all), dtype=np.int64)

    gmsh.finalize()

    # Identify boundary subsets
    tol = lc * 0.5
    left = np.where(nodes[:, 0] < tol)[0]
    right = np.where(nodes[:, 0] > L - tol)[0]
    bottom = np.where(nodes[:, 1] < tol)[0]
    top = np.where(nodes[:, 1] > H - tol)[0]

    mesh = Mesh(
        nodes=nodes,
        element_blocks=[ElementBlock(sim_etype, conn)],
        boundary_tags={
            "boundary": bnd_all,
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
        },
    )

    return mesh


# ====================================================================
# B-MATRIX FOR ELASTICITY
# ====================================================================
def build_B_matrix(dN_dx):
    """Build Voigt strain-displacement matrix."""
    n_nodes = dN_dx.shape[0]
    B = np.zeros((3, n_nodes * 2))
    for i in range(n_nodes):
        B[0, 2 * i] = dN_dx[i, 0]
        B[1, 2 * i + 1] = dN_dx[i, 1]
        B[2, 2 * i] = dN_dx[i, 1]
        B[2, 2 * i + 1] = dN_dx[i, 0]
    return B


# ====================================================================
# STIFFNESS MATRIX ASSEMBLY
# ====================================================================
def assemble_stiffness(mesh, dof_map, C_mat, thickness):
    """Assemble global stiffness matrix for plane stress."""
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
            ke = np.zeros((n_el_dof, n_el_dof))

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords
                detJ = abs(np.linalg.det(J))
                dN_dx = dN_dxi @ np.linalg.inv(J)

                B = build_B_matrix(dN_dx)
                ke += (B.T @ C_mat @ B) * detJ * qw * thickness

            for il, ig in enumerate(edofs):
                for jl, jg in enumerate(edofs):
                    rows.append(ig)
                    cols.append(jg)
                    vals.append(ke[il, jl])

    K = sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_dof, n_dof),
    )
    return K


# ====================================================================
# APPLY BCs TO K, M, C (elimination)
# ====================================================================
def apply_bc_to_system(K, M, C, constrained_dofs):
    """
    Apply Dirichlet BCs by zeroing rows/cols and setting diagonal to 1.

    Modifies K, M, C in-place for the constrained DOFs.
    Returns modified matrices.
    """
    K_lil = K.tolil()
    M_lil = M.tolil()
    C_lil = C.tolil()

    for dof in constrained_dofs:
        # K
        K_lil[dof, :] = 0
        K_lil[:, dof] = 0
        K_lil[dof, dof] = 1.0

        # M: set to small value (not 1, to avoid artificial eigenvalues)
        M_lil[dof, :] = 0
        M_lil[:, dof] = 0
        M_lil[dof, dof] = 1e-10

        # C
        C_lil[dof, :] = 0
        C_lil[:, dof] = 0
        C_lil[dof, dof] = 1e-10

    return K_lil.tocsr(), M_lil.tocsr(), C_lil.tocsr()


# ====================================================================
# FIND LOADED NODE
# ====================================================================
def find_node(mesh, x_target, y_target, tol=None):
    """Find node closest to (x_target, y_target)."""
    dists = np.sqrt((mesh.nodes[:, 0] - x_target)**2 +
                    (mesh.nodes[:, 1] - y_target)**2)
    idx = np.argmin(dists)
    if tol is not None and dists[idx] > tol:
        raise ValueError(f"No node within {tol} of ({x_target}, {y_target})")
    return idx


# ====================================================================
# VISUALISATION
# ====================================================================
def plot_mesh(mesh):
    """Plot the mesh."""
    from matplotlib.collections import PolyCollection

    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    etype = mesh.element_blocks[0].etype

    fig, ax = plt.subplots(figsize=(12, 4))
    polys = [nd[conn[e]] for e in range(conn.shape[0])]
    pc = PolyCollection(polys, edgecolors="steelblue", facecolors="none",
                        linewidths=0.4)
    ax.add_collection(pc)

    # Mark boundaries
    left = mesh.boundary_tags["left"]
    ax.plot(nd[left, 0], nd[left, 1], "s", color="blue", ms=4, label="Clamped")

    ax.set_aspect("equal")
    ax.set_title("{} mesh ({} elements, {} nodes)".format(
        etype.upper(), mesh.n_elements, mesh.n_nodes))
    ax.legend()
    ax.autoscale()
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mesh.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_mode_shapes(mesh, dof_map, frequencies, modes, n_plot=4):
    """Plot the first few mode shapes."""
    import matplotlib.tri as mtri

    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    if mesh.element_blocks[0].etype == "quad4":
        tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    else:
        tri = conn
    triang = mtri.Triangulation(nd[:, 0], nd[:, 1], tri)

    n_plot = min(n_plot, len(frequencies))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]

    for k in range(n_plot):
        ax = axes[k]
        mode = modes[:, k].reshape(-1, 2)
        u_mag = np.sqrt(mode[:, 0]**2 + mode[:, 1]**2)

        # Normalise mode for display
        scale = 0.3 * L / max(np.abs(mode).max(), 1e-15)
        def_nodes = nd + scale * mode

        if mesh.element_blocks[0].etype == "quad4":
            tri_def = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
        else:
            tri_def = conn
        triang_def = mtri.Triangulation(def_nodes[:, 0], def_nodes[:, 1], tri_def)

        tcf = ax.tricontourf(triang_def, u_mag, levels=20, cmap="viridis")
        ax.triplot(triang_def, lw=0.2, color="k", alpha=0.1)
        ax.triplot(triang, lw=0.15, color="gray", alpha=0.3, ls="--")
        fig.colorbar(tcf, ax=ax, shrink=0.6)
        ax.set_aspect("equal")
        ax.set_title("Mode {} — f = {:.2f} Hz".format(k + 1, frequencies[k]))

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mode_shapes.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_time_history(results, node_label="node 1"):
    """Plot displacement time histories comparing methods."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, data in results.items():
        t = data["t"]
        ux = data["u"][:, 0]  # x-displacement
        uy = data["u"][:, 1]  # y-displacement
        axes[0].plot(t, ux * 1e3, label=label, lw=0.8)
        axes[1].plot(t, uy * 1e3, label=label, lw=0.8)

    axes[0].set_ylabel("$u_x$ (mm)")
    axes[0].set_title("Displacement history at {} — horizontal".format(node_label))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("$u_y$ (mm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Displacement history at {} — vertical".format(node_label))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "time_history.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_mass_comparison(M_consistent, M_lumped):
    """Compare consistent and lumped mass diagonals."""
    diag_c = M_consistent.diagonal()
    diag_l = M_lumped.diagonal()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(diag_c, label="Consistent (diagonal)", alpha=0.7)
    ax.plot(diag_l, label="Lumped", alpha=0.7)
    ax.set_xlabel("DOF index")
    ax.set_ylabel("Mass diagonal entry")
    ax.set_title("Consistent vs Lumped Mass (diagonal entries)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mass_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("DYNAMIC ANALYSIS: 2D CANTILEVER BEAM")
    print("=" * 60)
    print(f"  Geometry: {L} x {H} m, thickness = {THICKNESS} m")
    print(f"  Material: E = {E_YOUNG/1e9:.0f} GPa, ν = {NU_POISSON}, ρ = {RHO} kg/m³")
    print(f"  Load:     F(t) = {F_AMPLITUDE/1e3:.0f} sin({OMEGA_LOAD}t) kN")

    # ── Step 1: Mesh ──
    print("\n--- Step 1: Mesh generation ---")
    mesh = mesh_cantilever_gmsh(L, H, MESH_LC, etype=ETYPE)
    print(f"  {mesh.n_nodes} nodes, {mesh.n_elements} {ETYPE.upper()} elements")
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total
    print(f"  Total DOFs: {n_dof}")
    plot_mesh(mesh)

    # Find loaded node (bottom-right corner)
    load_node = find_node(mesh, L, 0.0)
    load_dof_x = 2 * load_node
    load_dof_y = 2 * load_node + 1
    print(f"  Load node: {load_node} at ({mesh.nodes[load_node, 0]:.3f}, "
          f"{mesh.nodes[load_node, 1]:.3f})")
    print(f"  Load DOFs: ux={load_dof_x}, uy={load_dof_y}")

    # ── Step 2: Assemble K and M ──
    print("\n--- Step 2: Stiffness and mass matrices ---")
    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C_mat = material.tangent(dim=2)

    K = assemble_stiffness(mesh, dof_map, C_mat, THICKNESS)
    M_consistent = assemble_mass_matrix(
        mesh, dof_map, RHO, n_components=2, method="consistent", thickness=THICKNESS
    )
    M_lumped = assemble_mass_matrix(
        mesh, dof_map, RHO, n_components=2, method="lumped", thickness=THICKNESS
    )
    print(f"  K: {K.shape}, nnz={K.nnz}")
    print(f"  M_consistent: {M_consistent.shape}, nnz={M_consistent.nnz}")
    print(f"  M_lumped: diagonal, total mass = {M_lumped.diagonal().sum()/2:.2f} kg")

    # Expected total mass
    total_mass = RHO * L * H * THICKNESS
    print(f"  Expected total mass: {total_mass:.2f} kg")

    plot_mass_comparison(M_consistent, M_lumped)

    # ── Step 3: Apply BCs (clamped left edge) ──
    print("\n--- Step 3: Boundary conditions ---")
    left = mesh.boundary_tags["left"]
    constrained_dofs = []
    for nid in left:
        constrained_dofs.extend([2 * nid, 2 * nid + 1])
    constrained_dofs = np.array(constrained_dofs)
    print(f"  Clamped: {len(left)} nodes, {len(constrained_dofs)} DOFs")

    # Apply BCs to all matrices
    K_bc, M_con_bc, _ = apply_bc_to_system(
        K.copy(), M_consistent.copy(),
        sparse.csr_matrix((n_dof, n_dof)), constrained_dofs
    )
    _, M_lum_bc, _ = apply_bc_to_system(
        K.copy(), M_lumped.copy(),
        sparse.csr_matrix((n_dof, n_dof)), constrained_dofs
    )

    # ── Step 4: Eigenvalue problem ──
    print("\n--- Step 4: Eigenvalue analysis ---")

    # Consistent mass
    print("  With consistent mass:")
    freqs_con, modes_con = solve_eigenvalue_problem(K_bc, M_con_bc, n_modes=N_MODES)
    print(f"    First {min(5, N_MODES)} frequencies (Hz): "
          + ", ".join(f"{f:.2f}" for f in freqs_con[:5]))

    # Lumped mass
    print("  With lumped mass:")
    freqs_lum, modes_lum = solve_eigenvalue_problem(K_bc, M_lum_bc, n_modes=N_MODES)
    print(f"    First {min(5, N_MODES)} frequencies (Hz): "
          + ", ".join(f"{f:.2f}" for f in freqs_lum[:5]))

    # Compare
    print("\n  Frequency comparison (Hz):")
    print(f"    {'Mode':>5}  {'Consistent':>12}  {'Lumped':>12}  {'Diff (%)':>10}")
    print("    " + "-" * 42)
    for k in range(min(5, N_MODES)):
        diff = 100 * (freqs_lum[k] - freqs_con[k]) / freqs_con[k]
        print(f"    {k+1:>5d}  {freqs_con[k]:>12.2f}  {freqs_lum[k]:>12.2f}  {diff:>10.2f}")

    plot_mode_shapes(mesh, dof_map, freqs_con, modes_con, n_plot=4)

    # ── Step 5: Rayleigh damping ──
    print("\n--- Step 5: Rayleigh damping ---")
    omega1 = 2 * np.pi * freqs_con[0]
    omega2 = 2 * np.pi * freqs_con[1]

    # Consistent mass damping
    C_con, a_con, b_con = compute_rayleigh_damping(
        M_con_bc, K_bc, omega1, omega2, ZETA_1, ZETA_2
    )
    print(f"  Consistent: a={a_con:.6f}, b={b_con:.8f}")

    # Lumped mass damping
    C_lum, a_lum, b_lum = compute_rayleigh_damping(
        M_lum_bc, K_bc, omega1, omega2, ZETA_1, ZETA_2
    )
    print(f"  Lumped:     a={a_lum:.6f}, b={b_lum:.8f}")

    # ── Step 6: Force function ──
    def force_fn(t):
        F = np.zeros(n_dof)
        if t <= T_BURST:
            F[load_dof_y] = -F_AMPLITUDE * np.sin(OMEGA_LOAD * t)
        return F

    # ── Step 7: Time integration ──
    print("\n--- Step 6: Time integration ---")

    results = {}
    output_dofs = [load_dof_x, load_dof_y]

    # Newmark-β with consistent mass
    print("\n  Newmark-β (consistent mass) ...")
    nb_con = NewmarkBeta(M_con_bc, C_con, K_bc, dt=DT_NEWMARK)
    t_nb, u_nb, v_nb, a_nb = nb_con.solve(
        force_fn, T_END, dof_out=output_dofs
    )
    results["Newmark-β (consistent)"] = {"t": t_nb, "u": u_nb}

    # Newmark-β with lumped mass
    print("\n  Newmark-β (lumped mass) ...")
    nb_lum = NewmarkBeta(M_lum_bc, C_lum, K_bc, dt=DT_NEWMARK)
    t_nbl, u_nbl, v_nbl, a_nbl = nb_lum.solve(
        force_fn, T_END, dof_out=output_dofs
    )
    results["Newmark-β (lumped)"] = {"t": t_nbl, "u": u_nbl}

    # Central Difference with lumped mass
    print("\n  Central Difference (lumped mass) ...")
    # First compute critical timestep
    cd_temp = CentralDifference(M_lum_bc, C_lum, K_bc, dt=1e-6,
                                constrained_dofs=constrained_dofs)
    dt_cr = cd_temp.critical_timestep(K_bc, M_lum_bc)
    dt_cd = 0.5 * dt_cr   # use half the critical for safety
    print(f"    Critical timestep: {dt_cr:.6e} s")
    print(f"    Using dt = {dt_cd:.6e} s (ratio = {dt_cd/dt_cr:.4f})")

    cd = CentralDifference(M_lum_bc, C_lum, K_bc, dt=dt_cd,
                           constrained_dofs=constrained_dofs)
    t_cd, u_cd, v_cd, a_cd = cd.solve(
        force_fn, T_END, dof_out=output_dofs,
        store_every=max(1, int(DT_NEWMARK / dt_cd))
    )
    results["Central Diff (lumped)"] = {"t": t_cd, "u": u_cd}

    # Central Difference with consistent mass
    print("\n  Central Difference (consistent mass) ...")
    C_con_cd, _, _ = compute_rayleigh_damping(
        M_con_bc, K_bc, omega1, omega2, ZETA_1, ZETA_2
    )
    cd_con = CentralDifference(M_con_bc, C_con_cd, K_bc, dt=dt_cd,
                               constrained_dofs=constrained_dofs)
    t_cdc, u_cdc, v_cdc, a_cdc = cd_con.solve(
        force_fn, T_END, dof_out=output_dofs,
        store_every=max(1, int(DT_NEWMARK / dt_cd))
    )
    results["Central Diff (consistent)"] = {"t": t_cdc, "u": u_cdc}

    # ── Step 8: Plot results ──
    print("\n--- Step 7: Results ---")
    for label, data in results.items():
        uy_max = np.abs(data["u"][:, 1]).max()
        print(f"  {label}: max |u_y| = {uy_max*1e3:.4f} mm")

    plot_time_history(results, node_label="bottom-right corner")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Results saved to: {_FIG_DIR}")
    print(f"  Figures: mesh.png, mode_shapes.png, mass_comparison.png, time_history.png")
    print(f"\n  Next: compare with ANSYS modal + transient analysis")


if __name__ == "__main__":
    main()
