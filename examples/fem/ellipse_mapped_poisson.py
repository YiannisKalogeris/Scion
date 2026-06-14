#!/usr/bin/env python3
"""
Mapped Poisson with REVERSED NSF:  ψ: □ → Ω  (square → ellipse)

The NSF forward pass maps square points to ellipse points.
No inverse is ever needed — only forward evaluation and autograd.

Mathematics:
    Original:  ∫_Ω ∇u · ∇v dx = ∫_Ω f v dx

    Change of variables  x = ψ(y),  dx = |det Dψ| dy,  ∇_x = (Dψ)⁻ᵀ ∇_y

    Stiffness:  ∫_□  ∇_y ũᵀ  G(y)  ∇_y ṽ  dy
    Load:       ∫_□  f(ψ(y))  ṽ(y)  |det Dψ|  dy

    where  F = Dψ(y),   G = F⁻¹ F⁻ᵀ |det F|

Prerequisites:
    cd projects/nsf_domain_mapping
    python train_single_ellipse.py

Run:
    python examples/fem/ellipse_mapped_poisson.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

import torch

from simulacrum.geometry.shapes import Ellipse
from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.mesh.generators import rectangle_mesh
from simulacrum.core.domain import Domain
from simulacrum.core.dof import DOFMap
from simulacrum.boundary import DirichletBC, apply_dirichlet_elimination
from simulacrum.solvers.linear import DirectSolver
from simulacrum.discretization.fem.assembler import FEMAssembler
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "nsf_domain_mapping"
sys.path.insert(0, str(PROJECT_DIR / "models"))
from neural_spline_flow import NeuralSplineFlow

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_single_ellipse"
DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps"  if torch.backends.mps.is_available() else "cpu"

DEVICE = "cpu"

_FIG_DIR = Path(__file__).resolve().parent / "Figures_ellipse_mapped_poisson"
_FIG_DIR.mkdir(exist_ok=True)

N_SQUARE = 30
DIRECT_ETYPE = "quad4"   # "tri3" or "quad4"
MESH_LC  = 0.08


class NSFMappingWrapper:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device

    def psi(self, y):
        y = y.to(self.device)
        with torch.no_grad():
            x, _ = self.model(y)
        return x

    def jacobian_matrix(self, y):
        y = y.to(self.device).detach().requires_grad_(True)
        x, _ = self.model(y)
        F = torch.zeros(y.shape[0], 2, 2, device=self.device)
        for j in range(2):
            g = torch.zeros_like(x)
            g[:, j] = 1.0
            dxdy = torch.autograd.grad(x, y, g, retain_graph=True)[0]
            F[:, j, :] = dxdy
        return F.detach()


def load_nsf_model():
    config_path = CHECKPOINT_DIR / "config.json"
    model_path = CHECKPOINT_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"\n  ERROR: No model at {model_path}\n  Run train_single_ellipse.py first.")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg.get("direction") == "square_to_ellipse", \
        f"Expected reversed model, got: {cfg.get('direction')}"
    a, b = cfg["ellipse_a"], cfg["ellipse_b"]
    model = NeuralSplineFlow(
        dim=2, n_layers=cfg["n_layers"], n_bins=cfg["n_bins"],
        hidden_dim=cfg["hidden_dim"], n_hidden=cfg["n_hidden"],
        domain_bounds=tuple(cfg["domain_bounds"]),
        codomain_bounds=tuple(cfg["codomain_bounds"]),
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  Loaded NSF (square to ellipse) | a={a}, b={b} | params={cfg['n_params']:,}")
    return NSFMappingWrapper(model, DEVICE), a, b


def u_exact(x, y, a, b):
    return 1.0 - x**2 / a**2 - y**2 / b**2

def f_source_value(a, b):
    return 2.0 / a**2 + 2.0 / b**2


def mesh_ellipse_gmsh(a, b, lc, etype="tri3"):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("ellipse")
    fac = gmsh.model.occ
    fac.addEllipse(0, 0, 0, a, b)
    fac.addCurveLoop([1]); fac.addPlaneSurface([1]); fac.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    if etype == "quad4":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads

    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes = coords.reshape(-1, 3)[:, :2]

    etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
    conn = None
    sim_etype = None
    for eid, en in zip(etypes, enodes):
        name, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(eid)
        conn = np.array([node_map[int(n)] for n in en]).reshape(-1, npe)
        sim_etype = "quad4" if npe == 4 else "tri3"

    _, _, bnd_en = gmsh.model.mesh.getElements(dim=1)
    bnd = set()
    for en in bnd_en:
        for n in en:
            bnd.add(node_map[int(n)])
    bnd = np.array(sorted(bnd), dtype=np.int64)

    gmsh.finalize()
    return Mesh(nodes=nodes, element_blocks=[ElementBlock(sim_etype, conn)],
                boundary_tags={"boundary": bnd})


def assemble_load_vector(mesh, dof_map, source_val, weight_fn=None):
    f = np.zeros(dof_map.n_total)
    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            edofs = dof_map.element_dofs(enodes)
            for qp, qw in zip(qpts, qwts):
                N, dN = ref.evaluate(qp)
                J = dN.T @ coords
                detJ = abs(np.linalg.det(J))
                x_qp = N @ coords
                w = weight_fn(x_qp) if weight_fn else 1.0
                fe = N * source_val * w * detJ * qw
                for il, ig in enumerate(edofs):
                    f[ig] += fe[il]
    return f


def solve_direct(a, b, f_val):
    print("\n" + "=" * 60)
    print("DIRECT SOLVE ON ELLIPSE")
    print("=" * 60)
    mesh = mesh_ellipse_gmsh(a, b, MESH_LC, etype=DIRECT_ETYPE)
    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    domain = Domain(mesh=mesh, dim=2)
    asm = FEMAssembler(domain, n_components=1)
    sys_ = asm.assemble()
    K, rhs = sys_.matrix, assemble_load_vector(mesh, asm.dof_map, f_val)
    bnd = mesh.boundary_tags["boundary"]
    bcs = [DirichletBC(node_ids=bnd, component=0, value=0.0)]
    K, rhs = apply_dirichlet_elimination(K, rhs, asm.dof_map, bcs)
    u = DirectSolver().solve(K, rhs)
    u_ex = u_exact(mesh.nodes[:, 0], mesh.nodes[:, 1], a, b)
    print(f"  L2  error: {np.sqrt(np.mean((u - u_ex)**2)):.6e}")
    print(f"  Linf error: {np.max(np.abs(u - u_ex)):.6e}")
    return mesh, u


def precompute_metric(mapping, mesh, a, b):
    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype)

    all_y = []
    for e in range(block.connectivity.shape[0]):
        coords = mesh.nodes[block.connectivity[e]]
        for qp, _ in zip(qpts, qwts):
            N, _ = ref.evaluate(qp)
            all_y.append(N @ coords)

    all_y = np.array(all_y, dtype=np.float32)
    print(f"  Computing metric at {len(all_y)} quadrature points ...")

    y_t = torch.tensor(all_y, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        x_phys = mapping.psi(y_t).cpu().numpy()

    F_t = mapping.jacobian_matrix(y_t)
    detF = torch.det(F_t)
    abs_detF = detF.abs()

    F_inv = torch.linalg.inv(F_t)
    G_t = torch.bmm(F_inv, F_inv.transpose(1, 2)) * abs_detF.unsqueeze(-1).unsqueeze(-1)

    G = G_t.detach().cpu().numpy()
    detF_np = detF.detach().cpu().numpy()
    abs_detF_np = abs_detF.detach().cpu().numpy()

    # Diagnostics
    print(f"  det(Dψ) range: [{detF_np.min():.4f}, {detF_np.max():.4f}]  "
          f"({100*(detF_np>0).mean():.0f}% positive)")

    bnd_ids = np.unique(np.concatenate([
        mesh.boundary_tags["left"], mesh.boundary_tags["right"],
        mesh.boundary_tags["top"], mesh.boundary_tags["bottom"],
    ]))
    bnd_y = torch.tensor(mesh.nodes[bnd_ids], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        bnd_x = mapping.psi(bnd_y).cpu().numpy()
    ell_check = (bnd_x[:, 0] / a)**2 + (bnd_x[:, 1] / b)**2
    print(f"  ψ(∂□) on ellipse: (x/a)²+(y/b)² = [{ell_check.min():.4f}, {ell_check.max():.4f}]"
          f"  (should ≈ 1.0)")

    eig_min = np.array([np.linalg.eigvalsh(G[i]).min() for i in range(len(G))])
    print(f"  G min eigenvalue: {eig_min.min():.6e}  (must be > 0)")

    return {"G": G, "detF": abs_detF_np, "x_phys": x_phys}


def assemble_mapped_system(mesh, metric, f_val):
    from scipy import sparse
    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype)
    dof_map = DOFMap(mesh.n_nodes, 1)
    n_dof = dof_map.n_total
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_dof)

    qp_idx = 0
    for e in range(block.connectivity.shape[0]):
        enodes = block.connectivity[e]
        coords = mesh.nodes[enodes]
        edofs = dof_map.element_dofs(enodes)
        n_el = len(enodes)
        ke = np.zeros((n_el, n_el))
        fe = np.zeros(n_el)
        for qp, qw in zip(qpts, qwts):
            N, dN = ref.evaluate(qp)
            J_iso = dN.T @ coords
            detJ_iso = abs(np.linalg.det(J_iso))
            dN_dy = dN @ np.linalg.inv(J_iso)

            G = metric["G"][qp_idx]
            absdetF = metric["detF"][qp_idx]

            ke += (dN_dy @ G @ dN_dy.T) * detJ_iso * qw
            fe += N * f_val * absdetF * detJ_iso * qw
            qp_idx += 1

        for il, ig in enumerate(edofs):
            rhs[ig] += fe[il]
            for jl, jg in enumerate(edofs):
                rows.append(ig); cols.append(jg); vals.append(ke[il, jl])

    K = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                          shape=(n_dof, n_dof))
    return K, rhs, dof_map


def solve_mapped(mapping, a, b, f_val):
    print("\n" + "=" * 60)
    print("MAPPED SOLVE  (ψ: □ → Ω)")
    print("=" * 60)
    sq = rectangle_mesh((-1.0, 1.0), (-1.0, 1.0), N_SQUARE, N_SQUARE, etype="quad4")
    print(f"  Square mesh: {sq.n_nodes} nodes, {sq.n_elements} Quad4")

    metric = precompute_metric(mapping, sq, a, b)
    K, rhs, dof_map = assemble_mapped_system(sq, metric, f_val)
    print(f"  System: {K.shape[0]} DOFs, nnz = {K.nnz}")

    all_bnd = np.unique(np.concatenate([
        sq.boundary_tags["left"], sq.boundary_tags["right"],
        sq.boundary_tags["top"], sq.boundary_tags["bottom"],
    ]))
    bcs = [DirichletBC(node_ids=all_bnd, component=0, value=0.0)]
    K, rhs = apply_dirichlet_elimination(K, rhs, dof_map, bcs)

    u_sq = DirectSolver().solve(K, rhs)

    nodes_t = torch.tensor(sq.nodes, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_phys = mapping.psi(nodes_t).cpu().numpy()

    u_ex = u_exact(x_phys[:, 0], x_phys[:, 1], a, b)
    interior = np.setdiff1d(np.arange(sq.n_nodes), all_bnd)

    print(f"  Solution range: [{u_sq.min():.6f}, {u_sq.max():.6f}]")
    print(f"  Expected range: [{u_ex[interior].min():.4f}, {u_ex[interior].max():.4f}]")

    l2 = np.sqrt(np.mean((u_sq[interior] - u_ex[interior]) ** 2))
    linf = np.max(np.abs(u_sq[interior] - u_ex[interior]))
    print(f"  L2  error: {l2:.6e}")
    print(f"  Linf error: {linf:.6e}")

    return sq, u_sq, x_phys, all_bnd


def visualise(mesh_d, u_d, mesh_sq, u_m, x_m, a, b, f_val, all_bnd):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    nd = mesh_d.nodes
    u_ex_d = u_exact(nd[:, 0], nd[:, 1], a, b)
    u_ex_m = u_exact(x_m[:, 0], x_m[:, 1], a, b)
    err_d = np.abs(u_d - u_ex_d)
    err_m = np.abs(u_m - u_ex_m)

    tri_d = mesh_d.element_blocks[0].connectivity
    if mesh_d.element_blocks[0].etype == "quad4":
        tri_d = np.vstack([tri_d[:, [0, 1, 2]], tri_d[:, [0, 2, 3]]])
    triang_d = mtri.Triangulation(nd[:, 0], nd[:, 1], tri_d)
    quad_conn = mesh_sq.element_blocks[0].connectivity
    tri_conn = np.vstack([quad_conn[:, [0, 1, 2]], quad_conn[:, [0, 2, 3]]])
    triang_sq = mtri.Triangulation(mesh_sq.nodes[:, 0], mesh_sq.nodes[:, 1], tri_conn)
    triang_m = mtri.Triangulation(x_m[:, 0], x_m[:, 1], tri_conn)

    ax = axes[0, 0]
    tcf = ax.tricontourf(triang_d, u_d, levels=32, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.8)
    ax.triplot(triang_d, lw=0.2, color="k", alpha=0.15)
    ax.set_aspect("equal"); ax.set_title("Direct FEM on ellipse")

    ax = axes[0, 1]
    tcf = ax.tricontourf(triang_sq, u_m, levels=32, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_aspect("equal"); ax.set_title("Solution on square")

    ax = axes[0, 2]
    tcf = ax.tricontourf(triang_m, u_m, levels=32, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.8)
    ax.triplot(triang_m, lw=0.2, color="k", alpha=0.15)
    ax.set_aspect("equal"); ax.set_title("Mapped back to ellipse")

    ax = axes[1, 0]
    tcf = ax.tricontourf(triang_d, err_d, levels=32, cmap="hot_r")
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|error|")
    ax.set_aspect("equal")
    ax.set_title(f"Direct error (L2={np.sqrt(np.mean(err_d**2)):.2e})")

    ax = axes[1, 1]
    tcf = ax.tricontourf(triang_sq, u_ex_m, levels=32, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_aspect("equal"); ax.set_title("Expected u(ψ(y)) on square")

    ax = axes[1, 2]
    tcf = ax.tricontourf(triang_m, err_m, levels=32, cmap="hot_r")
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|error|")
    ax.set_aspect("equal")
    ax.set_title(f"Mapped error (L2={np.sqrt(np.mean(err_m**2)):.2e})")

    plt.suptitle(
        f"Poisson on ellipse (a={a}, b={b}): direct vs mapped FEM\n"
        f"u = 1 - x²/a² - y²/b²,  f = {f_val:.4f}  |  ψ: □ → Ω",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "ellipse_mapped_poisson.png", dpi=150)
    plt.show()
    print("Saved ellipse_mapped_poisson.png")

def plot_individual_fields(mesh_d, u_d, mesh_sq, u_m, x_m, a, b, all_bnd):
    """Generate separate publication-quality plots for each solution field."""
    nd = mesh_d.nodes
    u_ex_d = u_exact(nd[:, 0], nd[:, 1], a, b)
    u_ex_m = u_exact(x_m[:, 0], x_m[:, 1], a, b)

    tri_d = mesh_d.element_blocks[0].connectivity
    if mesh_d.element_blocks[0].etype == "quad4":
        tri_d = np.vstack([tri_d[:, [0, 1, 2]], tri_d[:, [0, 2, 3]]])
    triang_d = mtri.Triangulation(nd[:, 0], nd[:, 1], tri_d)

    quad_conn = mesh_sq.element_blocks[0].connectivity
    tri_conn = np.vstack([quad_conn[:, [0, 1, 2]], quad_conn[:, [0, 2, 3]]])
    triang_m = mtri.Triangulation(x_m[:, 0], x_m[:, 1], tri_conn)

    vmin, vmax = 0.0, 1.0
    theta = np.linspace(0, 2 * np.pi, 200)
    ell_x = a * np.cos(theta)
    ell_y = b * np.sin(theta)

    # Plot 1: Exact solution
    fig, ax = plt.subplots(figsize=(8, 5))
    tcf = ax.tricontourf(triang_d, u_ex_d, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(ell_x, ell_y, "k-", lw=1.5)
    ax.set_xlim(-a * 1.15, a * 1.15); ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal"); ax.set_title("Exact solution")
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_exact.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Plot 2: Direct FEM
    fig, ax = plt.subplots(figsize=(8, 5))
    tcf = ax.tricontourf(triang_d, u_d, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.triplot(triang_d, lw=0.15, color="k", alpha=0.1)
    ax.plot(ell_x, ell_y, "k-", lw=1.5)
    ax.set_xlim(-a * 1.15, a * 1.15); ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal")
    err_l2_d = np.sqrt(np.mean((u_d - u_ex_d)**2))
    ax.set_title("Direct FEM (Tri3, L2 = {:.2e})".format(err_l2_d))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_direct_fem.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Plot 3: Pullback FEM
    fig, ax = plt.subplots(figsize=(8, 5))
    tcf = ax.tricontourf(triang_m, u_m, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.triplot(triang_m, lw=0.15, color="k", alpha=0.1)
    ax.plot(ell_x, ell_y, "k-", lw=1.5)
    ax.set_xlim(-a * 1.15, a * 1.15); ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal")
    interior = np.setdiff1d(np.arange(mesh_sq.n_nodes), all_bnd)
    err_l2_m = np.sqrt(np.mean((u_m[interior] - u_ex_m[interior])**2))
    ax.set_title("Pullback FEM (Quad4, L2 = {:.2e})".format(err_l2_m))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_pullback_fem.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Plot 4: Error fields side by side
    err_d = np.abs(u_d - u_ex_d)
    err_m = np.abs(u_m - u_ex_m)
    err_max = max(err_d.max(), err_m.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    tcf = ax.tricontourf(triang_d, err_d, levels=32, cmap="hot_r", vmin=0, vmax=err_max)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(ell_x, ell_y, "k-", lw=1.5)
    ax.set_xlim(-a * 1.15, a * 1.15); ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal")
    ax.set_title("Direct FEM error (L2 = {:.2e})".format(np.sqrt(np.mean(err_d**2))))

    ax = axes[1]
    tcf = ax.tricontourf(triang_m, err_m, levels=32, cmap="hot_r", vmin=0, vmax=err_max)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(ell_x, ell_y, "k-", lw=1.5)
    ax.set_xlim(-a * 1.15, a * 1.15); ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal")
    ax.set_title("Pullback FEM error (L2 = {:.2e})".format(np.sqrt(np.mean(err_m**2))))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_error_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Plot 5: Solution on the reference square
    triang_sq = mtri.Triangulation(mesh_sq.nodes[:, 0], mesh_sq.nodes[:, 1], tri_conn)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    tcf = ax.tricontourf(triang_sq, u_m, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.set_aspect("equal"); ax.set_title("Computed solution on square")
    ax = axes[1]
    tcf = ax.tricontourf(triang_sq, u_ex_m, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.set_aspect("equal"); ax.set_title("Expected solution on square")
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_solution_on_square.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_mesh(mesh, a, b, title=None):
    """Plot the mesh showing actual element shapes."""
    from matplotlib.collections import PolyCollection

    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    etype = mesh.element_blocks[0].etype

    theta = np.linspace(0, 2 * np.pi, 200)
    ell_x = a * np.cos(theta)
    ell_y = b * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Build polygons from connectivity (works for both tri3 and quad4)
    polys = [nd[conn[e]] for e in range(conn.shape[0])]
    pc = PolyCollection(polys, edgecolors="steelblue", facecolors="none",
                        linewidths=0.4, alpha=0.8)
    ax.add_collection(pc)

    ax.plot(ell_x, ell_y, "k-", lw=2)

    bnd = mesh.boundary_tags.get("boundary", np.array([]))
    if len(bnd) > 0:
        ax.plot(nd[bnd, 0], nd[bnd, 1], ".", color="red", ms=2, zorder=5)

    ax.set_xlim(-a * 1.15, a * 1.15)
    ax.set_ylim(-b * 1.5, b * 1.5)
    ax.set_aspect("equal")
    if title is None:
        title = "{} mesh ({} elements, {} nodes)".format(
            etype.upper(), mesh.n_elements, mesh.n_nodes)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_mesh_{}.png".format(etype), dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved poisson_mesh_{}.png".format(etype))

def main():
    print("=" * 60)
    print("MAPPED POISSON  (ψ: □ → Ω)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    mapping, a, b = load_nsf_model()
    f_val = f_source_value(a, b)
    print(f"  Ellipse: a={a}, b={b},  f = {f_val:.4f}")

    mesh_d, u_d = solve_direct(a, b, f_val)
    plot_mesh(mesh_d, a, b)
    mesh_sq, u_m, x_m, all_bnd = solve_mapped(mapping, a, b, f_val)
    visualise(mesh_d, u_d, mesh_sq, u_m, x_m, a, b, f_val, all_bnd)
    plot_individual_fields(mesh_d, u_d, mesh_sq, u_m, x_m, a, b, all_bnd)

    print("\nDONE")


if __name__ == "__main__":
    main()
















































