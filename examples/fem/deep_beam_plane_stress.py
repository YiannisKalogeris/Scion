"""
Plane-stress simply-supported deep beam (Final-Project SFEM course 2025-2026).

    L = 4 m, h = 2 m, thickness t = 0.20 m,  E = 1e5 MPa,  nu = 0.3
    uniform top pressure  q = 20 kN/m  (line load along the top edge, downward)
    pinned at the bottom-left corner, roller at the bottom-right corner.

The mesh is a structured 40 x 20 Quad4 grid generated with gmsh (transfinite +
recombine). The stiffness is assembled with the library's FEMAssembler; we add
the top-edge load and the supports, solve the direct problem, report the vertical
displacement at mid-span, and plot the deformed shape (shown only, never saved).

Run:
    python examples/fem/deep_beam_plane_stress.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ---- Simulacrum reusables -------------------------------------------------
from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.core.domain import Domain
from simulacrum.materials.linear_elastic import PlaneStress
from simulacrum.boundary import DirichletBC, apply_dirichlet_elimination
from simulacrum.solvers.linear import DirectSolver
from simulacrum.discretization.fem.assembler import FEMAssembler

# ===========================================================================
# Problem data  (given in the project sheet; converted to SI: N, m, Pa)
# ===========================================================================
L         = 4.0            # beam length            [m]
H         = 2.0            # beam height            [m]
THICKNESS = 0.20           # out-of-plane thickness [m]
E_YOUNG   = 1.0e5          # 1e5 kPa  
NU        = 0.3
Q_LOAD    = 20.0           # 20 kN/m  

NX, NY    = 40, 20         # elements along L and H  (40 x 20 Quad4)
TOL       = 1e-8


# ===========================================================================
# Mesh:  structured Quad4 rectangle via gmsh (transfinite + recombine)
# ===========================================================================
def mesh_beam_gmsh(length, height, nx, ny):
    """Structured nx*ny Quad4 mesh of the rectangle [0,L]x[0,H] using gmsh."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("deep_beam")
    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0,    0.0,    0.0)
    p2 = geo.addPoint(length, 0.0,    0.0)
    p3 = geo.addPoint(length, height, 0.0)
    p4 = geo.addPoint(0.0,    height, 0.0)
    l1 = geo.addLine(p1, p2)   # bottom
    l2 = geo.addLine(p2, p3)   # right
    l3 = geo.addLine(p3, p4)   # top
    l4 = geo.addLine(p4, p1)   # left
    cl = geo.addCurveLoop([l1, l2, l3, l4])
    surf = geo.addPlaneSurface([cl])
    geo.synchronize()

    # exactly nx x ny quads
    gmsh.model.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.mesh.setTransfiniteSurface(surf)
    gmsh.model.mesh.setRecombine(2, surf)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes = coords.reshape(-1, 3)[:, :2]

    etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
    conn = None
    for eid, en in zip(etypes, enodes):
        _, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(eid)
        if npe == 4:
            conn = np.array([node_map[int(n)] for n in en]).reshape(-1, 4)
    assert conn is not None, "gmsh did not return Quad4 elements"

    _, _, bnd_en = gmsh.model.mesh.getElements(dim=1)
    bnd = set()
    for en in bnd_en:
        for n in en:
            bnd.add(node_map[int(n)])
    bnd = np.array(sorted(bnd), dtype=np.int64)

    gmsh.finalize()
    return Mesh(nodes=nodes, element_blocks=[ElementBlock("quad4", conn)],
                boundary_tags={"boundary": bnd})


# ===========================================================================
# Assemble (via FEMAssembler) + load + supports + solve
# ===========================================================================
def solve_beam(mesh):
    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} Quad4 elements")

    material = PlaneStress(E=E_YOUNG, nu=NU)
    domain = Domain(mesh=mesh, dim=2)
    for blk in mesh.element_blocks:                 # register material per block tag
        domain.add_material(blk.tag, material)

    asm = FEMAssembler(domain, n_components=2)
    system = asm.assemble()                         # K = integral(B^T C B), rhs = 0
    dof_map = asm.dof_map

    K = THICKNESS * system.matrix                   # plane-stress thickness factor
    rhs = system.rhs                                # zeros; fill with the applied load

    # ---- consistent nodal load from the top-edge line load q (downward) ----
    top = np.where(np.abs(mesh.nodes[:, 1] - H) < TOL)[0]
    top = top[np.argsort(mesh.nodes[top, 0])]
    for i in range(len(top) - 1):
        n1, n2 = top[i], top[i + 1]
        seg = mesh.nodes[n2, 0] - mesh.nodes[n1, 0]  # linear edge length
        f = Q_LOAD * seg / 2.0                       # half to each end node
        for n in (n1, n2):
            uy = int(dof_map.element_dofs(np.array([n]))[1])
            rhs[uy] -= f                             # -y : downward

    # ---- supports : pin bottom-left, roller bottom-right ----
    bl = int(np.argmin(mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2))
    br = int(np.argmin((mesh.nodes[:, 0] - L) ** 2 + mesh.nodes[:, 1] ** 2))
    bcs = [
        DirichletBC(node_ids=np.array([bl]), component=0, value=0.0),  # pin  u_x
        DirichletBC(node_ids=np.array([bl]), component=1, value=0.0),  # pin  u_y
        DirichletBC(node_ids=np.array([br]), component=1, value=0.0),  # roller u_y
    ]
    K, rhs = apply_dirichlet_elimination(K, rhs, dof_map, bcs)

    u = DirectSolver().solve(K, rhs)
    return u.reshape(-1, 2)


# ===========================================================================
# Deformed-shape plot  
# ===========================================================================
def plot_deformed(mesh, disp, scale=None):
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    umag = np.sqrt((disp ** 2).sum(axis=1))

    # exaggeration factor (auto: largest displacement ~8% of the beam length)
    if scale is None:
        scale = 0.08 * max(L, H) / (umag.max() + 1e-30)
    defo = nd + scale * disp

    fig, ax = plt.subplots(figsize=(9, 5))

    # undeformed mesh (light grey)
    ax.add_collection(PolyCollection([nd[conn[e]] for e in range(conn.shape[0])],
                                     facecolors="none", edgecolors="0.8",
                                     linewidths=0.4))
    # deformed mesh, coloured by displacement magnitude
    pc = PolyCollection([defo[conn[e]] for e in range(conn.shape[0])],
                        array=umag[conn].mean(axis=1), cmap="viridis",
                        edgecolors="k", linewidths=0.25)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.85, label="|u|  [m]")

    # ---- support markers at the DEFORMED positions of the constrained nodes ----
    bl = int(np.argmin(nd[:, 0] ** 2 + nd[:, 1] ** 2))           # pin    near (0, 0)
    br = int(np.argmin((nd[:, 0] - L) ** 2 + nd[:, 1] ** 2))     # roller near (L, 0)
    off = 0.01 * max(L, H)                                       # tuck the triangle just below the node
    ax.plot(defo[[bl, br], 0], defo[[bl, br], 1] - off, "^",
            color="firebrick", ms=11, zorder=6, clip_on=False)

    pad = 0.15 * max(L, H)
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, H + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Deep beam - deformed shape (displacements x{scale:.0f})")
    plt.tight_layout()
    plt.show()


# ===========================================================================
def main():
    print("=" * 60)
    print("PLANE-STRESS SIMPLY-SUPPORTED DEEP BEAM")
    print("=" * 60)
    print(f"  L={L} m, h={H} m, t={THICKNESS} m, E={E_YOUNG:.3e} Pa, nu={NU}")
    print(f"  q = {Q_LOAD:.3e} N/m  (downward),  total = {Q_LOAD * L:.3e} N")

    mesh = mesh_beam_gmsh(L, H, NX, NY)
    disp = solve_beam(mesh)

    # vertical displacement at mid-span, bottom edge  (the u_mid of the figure)
    mid = int(np.argmin((mesh.nodes[:, 0] - L / 2.0) ** 2 +
                        (mesh.nodes[:, 1] - 0.0) ** 2))
    u_mid = disp[mid, 1]
    print(f"\n  mid-span node at ({mesh.nodes[mid,0]:.3f}, {mesh.nodes[mid,1]:.3f}) m")
    print(f"  vertical mid displacement  u_mid = {u_mid: .6e} m  "
          f"= {u_mid * 1e3: .6f} mm")
    print(f"  max |u| over the beam            = {np.sqrt((disp**2).sum(1)).max():.6e} m")

    plot_deformed(mesh, disp)
    print("\nDONE")


if __name__ == "__main__":
    main()
