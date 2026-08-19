#!/usr/bin/env python3
"""
2D plane-stress elasticity on an ellipse domain.

Demonstrates the full Scion FEM pipeline:
    1. Define an ellipse using scion.geometry.shapes
    2. Mesh it with gmsh (Tri3 elements)
    3. Identify boundary nodes and assign BCs
    4. Assemble the stiffness matrix
    5. Apply Dirichlet + Neumann BCs, solve, and visualise

BCs:
    - Left boundary (x < -a + tol): fixed in x and y (clamped)
    - Right boundary (x > a - tol): uniform traction in +x direction

Run from the project root:
    python examples/fem/ellipse_elasticity_2d.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ── Scion imports ──────────────────────────────────────────────
from scion.geometry.shapes import Ellipse
from scion.mesh.mesh import Mesh, ElementBlock
from scion.core.domain import Domain
from scion.core.dof import DOFMap
from scion.materials.linear_elastic import PlaneStress
from scion.discretization.fem.assembler import FEMAssembler
from scion.boundary import DirichletBC, apply_dirichlet_elimination
from scion.solvers.linear import DirectSolver


# ====================================================================
# 1. GEOMETRY
# ====================================================================
a, b = 2.0, 1.0                         # semi-axes
ellipse = Ellipse(center=(0.0, 0.0), a=a, b=b, theta=0.0)
print(f"Shape: {ellipse}")

# ====================================================================
# 2. MESH WITH GMSH
# ====================================================================
def mesh_ellipse_gmsh(a: float, b: float, lc: float = 0.15) -> Mesh:
    """
    Create a triangular mesh of an ellipse using the gmsh Python API.

    Parameters
    ----------
    a, b : semi-axes
    lc   : characteristic element length (smaller = finer mesh)

    Returns
    -------
    Mesh with boundary tags 'left', 'right', 'boundary'
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("ellipse")

    # Define ellipse boundary via OpenCASCADE
    factory = gmsh.model.occ
    # addEllipse(x, y, z, r1, r2) — returns a curve tag
    ellipse_tag = factory.addEllipse(0, 0, 0, a, b)
    loop_tag = factory.addCurveLoop([ellipse_tag])
    surf_tag = factory.addPlaneSurface([loop_tag])
    factory.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(2)

    # ── Extract nodes ──
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes = coords.reshape(-1, 3)[:, :2]  # keep x, y only

    # ── Extract triangle elements (dim=2) ──
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    connectivity = None
    for etype_id, enodes in zip(elem_types, elem_node_tags):
        name, _, _, n_per, _, _ = gmsh.model.mesh.getElementProperties(etype_id)
        conn = np.array([node_map[int(n)] for n in enodes]).reshape(-1, n_per)
        connectivity = conn

    # ── Extract boundary nodes (dim=1 curve elements) ──
    _, _, bnd_node_tags = gmsh.model.mesh.getElements(dim=1)
    boundary_node_set = set()
    for enodes in bnd_node_tags:
        for n in enodes:
            boundary_node_set.add(node_map[int(n)])
    boundary_all = np.array(sorted(boundary_node_set), dtype=np.int64)

    gmsh.finalize()

    # ── Identify left / right subsets geometrically ──
    tol = 0.10 * a  # ~10% of semi-axis
    left_nodes = boundary_all[nodes[boundary_all, 0] < -a + tol]
    right_nodes = boundary_all[nodes[boundary_all, 0] > a - tol]

    block = ElementBlock(etype="tri3", connectivity=connectivity)
    mesh = Mesh(
        nodes=nodes,
        element_blocks=[block],
        boundary_tags={
            "boundary": boundary_all,
            "left": left_nodes,
            "right": right_nodes,
        },
    )
    return mesh


mesh = mesh_ellipse_gmsh(a, b, lc=0.12)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} Tri3 elements")
print(f"  Left BC nodes:  {len(mesh.boundary_tags['left'])}")
print(f"  Right BC nodes: {len(mesh.boundary_tags['right'])}")

# ====================================================================
# 3. DOMAIN + MATERIAL
# ====================================================================
material = PlaneStress(E=210e3, nu=0.3)     # Steel-like (MPa)
domain = Domain(mesh=mesh, dim=2)
domain.add_material("default", material)

# ====================================================================
# 4. ASSEMBLE STIFFNESS MATRIX
# ====================================================================
assembler = FEMAssembler(domain, n_components=2)
system = assembler.assemble()
K, f = system.matrix, system.rhs
print(f"System: {K.shape[0]} DOFs, nnz = {K.nnz}")

# ====================================================================
# 5. BOUNDARY CONDITIONS
# ====================================================================
dof_map = assembler.dof_map

# 5a. Dirichlet: clamp the left boundary (ux = uy = 0)
left = mesh.boundary_tags["left"]
bcs = [
    DirichletBC(node_ids=left, component=0, value=0.0),   # ux = 0
    DirichletBC(node_ids=left, component=1, value=0.0),   # uy = 0
]

# 5b. Neumann: uniform traction on the right boundary
#     Distribute a total force among right-boundary nodes
right = mesh.boundary_tags["right"]
total_force_x = 1000.0   # total horizontal force (N)
force_per_node = total_force_x / len(right)
for node in right:
    dof_x = dof_map.dof(node, component=0)
    f[dof_x] += force_per_node

# Apply Dirichlet BCs by elimination
K, f = apply_dirichlet_elimination(K, f, dof_map, bcs)

# ====================================================================
# 6. SOLVE
# ====================================================================
solver = DirectSolver()
u = solver.solve(K, f)
print(f"Solved. max |u| = {np.max(np.abs(u)):.6e}")

# Reshape to (n_nodes, 2)
disp = u.reshape(-1, 2)
ux, uy = disp[:, 0], disp[:, 1]
u_mag = np.sqrt(ux**2 + uy**2)

# ====================================================================
# 7. VISUALISE
# ====================================================================
coords = mesh.nodes
triangles = mesh.element_blocks[0].connectivity
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Displacement magnitude ──
ax = axes[0]
tcf = ax.tricontourf(triang, u_mag, levels=32, cmap="viridis")
fig.colorbar(tcf, ax=ax, label="Displacement magnitude")
ax.triplot(triang, lw=0.3, color="k", alpha=0.2)
ax.set_aspect("equal")
ax.set_title("Displacement magnitude")

# ── Horizontal displacement ──
ax = axes[1]
tcf = ax.tricontourf(triang, ux, levels=32, cmap="RdBu_r")
fig.colorbar(tcf, ax=ax, label="$u_x$")
ax.triplot(triang, lw=0.3, color="k", alpha=0.2)
ax.set_aspect("equal")
ax.set_title("Horizontal displacement $u_x$")

# ── Deformed shape ──
ax = axes[2]
scale = 0.2 * a / max(np.max(np.abs(ux)), np.max(np.abs(uy)), 1e-15)
deformed_x = coords[:, 0] + scale * ux
deformed_y = coords[:, 1] + scale * uy
triang_def = mtri.Triangulation(deformed_x, deformed_y, triangles)
ax.triplot(triang, lw=0.3, color="gray", alpha=0.4, label="Undeformed")
tcf = ax.tricontourf(triang_def, u_mag, levels=32, cmap="viridis", alpha=0.85)
fig.colorbar(tcf, ax=ax, label="Displacement magnitude")
ax.triplot(triang_def, lw=0.3, color="k", alpha=0.3)
ax.set_aspect("equal")
ax.set_title(f"Deformed shape (×{scale:.0f})")

# Mark BCs
for ax_i in axes:
    ax_i.plot(
        coords[left, 0], coords[left, 1],
        "s", color="blue", ms=5, label="Fixed", zorder=5,
    )
    ax_i.plot(
        coords[right, 0], coords[right, 1],
        ">", color="red", ms=5, label="Load", zorder=5,
    )
    ax_i.legend(loc="upper left", fontsize=8)

plt.suptitle(
    f"2D Plane-Stress Elasticity on Ellipse (a={a}, b={b}) — "
    f"E={material.E:.0f} MPa, ν={material.nu}",
    fontsize=13,
)
plt.tight_layout()
# anchored to this file, not the working directory -- see poisson_2d.py
_out = Path(__file__).resolve().parent / "results" / "ellipse_elasticity_2d.png"
_out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_out, dpi=150)
plt.show()

print(f"\nDone. Saved figure to {_out}")
