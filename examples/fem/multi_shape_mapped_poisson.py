#!/usr/bin/env python3
"""
Mapped Poisson on Multiple Superellipse Domains via Auto-Decoder NSF.

Direction:  ψ: □ → Ω   (forward pass = square → physical)

Mathematics:
    Original:  -Δu = f  in Ω,   u = 0 on ∂Ω

    Pullback:
        Stiffness:  ∫_□  ∇_y ũᵀ  G(y)  ∇_y ṽ  dy
        Load:       ∫_□  f(ψ(y))  ṽ(y)  |det Dψ|  dy
        where  F = Dψ(y),   G = F⁻¹ F⁻ᵀ |det F|

Prerequisites:
    Train the auto-decoder NSF on ellipses/superellipses:
        cd projects/nsf_domain_mapping
        python Main_nsf.py   (with shape_distribution: Ellipse + Superellipse)
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from pathlib import Path

import torch

# ── Scion library ──
from scion.geometry.shapes import ShapeGenerator
from scion.mesh.mesh import Mesh, ElementBlock
from scion.mesh.generators import rectangle_mesh
from scion.core.domain import Domain
from scion.core.dof import DOFMap
from scion.boundary import DirichletBC, apply_dirichlet_elimination
from scion.solvers.linear import DirectSolver
from scion.discretization.fem.assembler import FEMAssembler
from scion.discretization.fem.elements import get_element
from scion.discretization.fem.quadrature import get_quadrature

# ── NSF model ──
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "nsf_domain_mapping"
sys.path.insert(0, str(PROJECT_DIR / "models"))
sys.path.insert(0, str(PROJECT_DIR))
from neural_spline_flow import NeuralSplineFlow
from Main_nsf import (AutoDecoderNSF, AutoDecoderMapping, create_autodecoder,
                      load_inference_model, EncoderMapping)
from dataset import DiffeomorphismDataset
from scion.geometry.sdf import SDFRenderer

# ====================================================================
# CONFIGURATION
# ====================================================================
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_ellipse"
DATA_DIR = PROJECT_DIR / "data_ellipse"
DEVICE = "cpu"
DOUBLE_METRIC = False   # evaluate the flow in float64 when building the metric

_FIG_DIR = Path(__file__).resolve().parent / "Figures_multi_shape_poisson"
_FIG_DIR.mkdir(exist_ok=True)

N_SQUARE = 30
MESH_LC = 0.05
QUAD_ORDER = 2        # points per direction for the pullback assembly only
DIRECT_ETYPE = "quad4"       # "tri3" or "quad4"

# Constant source term for Poisson
F_SOURCE = 10.0

# Number of training shapes to test on
N_TEST_SHAPES = 5
SEED = 42

# ── Article comparison (ground-truth) settings ──
HIFI_TARGET_ELEMS = 40000     # high-fidelity reference solve ("ground truth")
HIFI_ETYPE        = "quad4"     # robust element type for the fine reference mesh
RELATIVE_L2       = True       # report ||e||_L2 / ||u_ref||_L2 (False => absolute, as in example 1)
#ERROR_CLIM = (0.0, 0.025) # only if i want my plots to be in a common range (otherwise auto-scaled)

# ── Mapping source ──
# Encoder is the primary path; if its checkpoint is absent we fall back to the stored
# Stage-1 codes (perfect-z testing). Rename encoder_best.pt (e.g. -> encoder_best_REMOVE.pt)
# to force the stored-code path, exactly as in Main_nsf.py.
ENCODER_CKPT = "encoder_best_REMOVE.pt"
SDF_GRID     = 64              # encoder SDF image resolution (must match training)
SDF_BOUNDS   = (-2.5, 2.5)     # encoder SDF bounding box   (must match training)

# ====================================================================
# LOAD AUTO-DECODER (Stage 1 only, stored codes)
# ====================================================================
def load_model():
    """Load trained auto-decoder with stored codes."""
    config_path = CHECKPOINT_DIR / "config.json"
    model_path = CHECKPOINT_DIR / "stage1_best.pt"

    if not model_path.exists():
        print(f"\n  ERROR: No Stage 1 model at {model_path}")
        print(f"  Train with: cd {PROJECT_DIR} && python Main_nsf.py")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")
    n_train = len(train_dataset)

    model_cfg = cfg["model"]

    model = AutoDecoderNSF(
        n_shapes=n_train,
        latent_dim=model_cfg["latent_dim"],
        n_layers=model_cfg["n_layers"],
        n_bins=model_cfg["n_bins"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden=model_cfg["n_hidden"],
        domain_bounds=tuple(model_cfg["domain_bounds"]),
        codomain_bounds=tuple(model_cfg["codomain_bounds"]),
    ).to("cpu")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    n_params = sum(p.numel() for p in model.flow.parameters())
    print(f"  Loaded auto-decoder | n_shapes={n_train} | "
          f"latent_dim={model_cfg['latent_dim']} | flow params={n_params:,}")
    return model, train_dataset


# ====================================================================
# MESH ANY SHAPE WITH GMSH
# ====================================================================
def mesh_shape_gmsh(shape, lc=0.12, etype="tri3", n_boundary=200):
    """Mesh any ClosedCurve shape using gmsh."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("shape")
    fac = gmsh.model.occ

    t = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    pts = shape(t)

    point_tags = []
    for i in range(len(pts)):
        tag = fac.addPoint(pts[i, 0], pts[i, 1], 0, lc)
        point_tags.append(tag)

    spline_tag = fac.addBSpline(point_tags + [point_tags[0]])
    loop_tag = fac.addCurveLoop([spline_tag])
    fac.addPlaneSurface([loop_tag])
    fac.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    if etype == "quad4":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map_raw = {int(t_): i for i, t_ in enumerate(node_tags)}
    nodes_all = coords.reshape(-1, 3)[:, :2]

    # Extract 2D elements
    etypes_gmsh, _, enodes = gmsh.model.mesh.getElements(dim=2)
    conn_raw, sim_etype = None, None
    for eid, en in zip(etypes_gmsh, enodes):
        _, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(eid)
        conn_raw = np.array([node_map_raw[int(n)] for n in en]).reshape(-1, npe)
        sim_etype = "quad4" if npe == 4 else "tri3"

    # Filter orphan nodes
    used_nodes = np.unique(conn_raw.ravel())
    old_to_new = np.full(len(nodes_all), -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(len(used_nodes))
    nodes = nodes_all[used_nodes]
    conn = old_to_new[conn_raw]

    # Boundary nodes (remapped)
    _, _, bnd_en = gmsh.model.mesh.getElements(dim=1)
    bnd = set()
    for en in bnd_en:
        for n in en:
            old_idx = node_map_raw[int(n)]
            if old_to_new[old_idx] >= 0:
                bnd.add(old_to_new[old_idx])
    bnd = np.array(sorted(bnd), dtype=np.int64)

    gmsh.finalize()

    mesh = Mesh(nodes=nodes, element_blocks=[ElementBlock(sim_etype, conn)],
                boundary_tags={"boundary": bnd})
    return mesh


# ====================================================================
# DIRECT POISSON SOLVE
# ====================================================================
def assemble_load_vector(mesh, dof_map, source_val):
    """Assemble load vector for constant source f."""
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
                fe = N * source_val * detJ * qw
                for il, ig in enumerate(edofs):
                    f[ig] += fe[il]
    return f


def solve_direct_poisson(shape, mesh):
    """Solve -Δu = f on the physical mesh with u=0 on ∂Ω."""
    print(f"    Direct solve: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    domain = Domain(mesh=mesh, dim=2)
    asm = FEMAssembler(domain, n_components=1)
    sys_ = asm.assemble()
    K = sys_.matrix
    rhs = assemble_load_vector(mesh, asm.dof_map, F_SOURCE)

    bnd = mesh.boundary_tags["boundary"]
    bcs = [DirichletBC(node_ids=bnd, component=0, value=0.0)]
    K, rhs = apply_dirichlet_elimination(K, rhs, asm.dof_map, bcs)

    u = DirectSolver().solve(K, rhs)
    print(f"    u range: [{u.min():.6e}, {u.max():.6e}]")
    return u


# ====================================================================
# PRECOMPUTE METRIC (same as ellipse_mapped_poisson.py)
# ====================================================================
def precompute_metric(mapping, mesh):
    """Precompute G = F⁻¹ F⁻ᵀ |det F| and |det F| at all quadrature points."""
    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype, order=QUAD_ORDER)

    all_y = []
    for e in range(block.connectivity.shape[0]):
        coords = mesh.nodes[block.connectivity[e]]
        for qp, _ in zip(qpts, qwts):
            N, _ = ref.evaluate(qp)
            all_y.append(N @ coords)

    all_y = np.array(all_y, dtype=np.float64)
    print(f"    Metric at {len(all_y)} quadrature points ...")

    # ── optionally run the flow in double for this pass ──────────────────
    # det = ad - bc in float32 can cancel to exactly 0.0 even though the NSF
    # guarantees det > 0, which makes torch.linalg.inv raise. Double precision
    # removes the cancellation without clamping anything, so the real value is
    # kept rather than replaced by a floor.
    restore = []
    dtype = torch.float32
    if DOUBLE_METRIC:
        dtype = torch.float64
        mapping.model.double()
        restore.append(lambda: mapping.model.float())
        if hasattr(mapping, "z") and torch.is_tensor(mapping.z):
            z_old = mapping.z
            mapping.z = mapping.z.double()
            restore.append(lambda mp=mapping, z=z_old: setattr(mp, "z", z))

    try:
        y_t = torch.tensor(all_y, dtype=dtype, device=DEVICE)

        with torch.no_grad():
            x_phys = mapping.psi(y_t).cpu().numpy()

        F_t = mapping.jacobian_matrix(y_t)
        detF = torch.det(F_t)
        abs_detF = detF.abs()

        det_np = detF.detach().cpu().numpy()
        tiny = np.abs(det_np) < 1e-10
        if tiny.any():
            pts = all_y[tiny]
            print(f"    {tiny.sum()} of {len(det_np)} quadrature points have "
                  f"|det| < 1e-10  (min {np.abs(det_np).min():.3e})")
            print(f"      max |xi|,|eta| there: {np.abs(pts).max(axis=0)}")

        F_inv = torch.linalg.inv(F_t)
        G_t = torch.bmm(F_inv, F_inv.transpose(1, 2)) \
            * abs_detF.unsqueeze(-1).unsqueeze(-1)

        G = G_t.detach().cpu().numpy().astype(np.float64)
        abs_detF_np = abs_detF.detach().cpu().numpy().astype(np.float64)
    finally:
        for fn in reversed(restore):
            fn()

    print(f"    det(Dψ) range: [{detF.min():.4f}, {detF.max():.4f}]  "
          f"({100*(detF>0).float().mean():.0f}% positive)")

    return {"G": G, "detF": abs_detF_np, "x_phys": x_phys,
            "quad_order": QUAD_ORDER}


# ====================================================================
# MAPPED POISSON ASSEMBLY
# ====================================================================
def assemble_mapped_poisson(mesh, metric, f_val):
    """Assemble pulled-back Poisson system on the reference square."""
    from scipy import sparse
    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype, order=metric["quad_order"])
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


# ====================================================================
# MAPPED POISSON SOLVE
# ====================================================================
def solve_mapped_poisson(mapping):
    """Solve Poisson on the reference square with pulled-back operators."""
    sq = rectangle_mesh((-1.0, 1.0), (-1.0, 1.0), N_SQUARE, N_SQUARE, etype="quad4")
    print(f"    Square mesh: {sq.n_nodes} nodes, {sq.n_elements} Quad4")

    metric = precompute_metric(mapping, sq)
    K, rhs, dof_map = assemble_mapped_poisson(sq, metric, F_SOURCE)
    print(f"    System: {K.shape[0]} DOFs, nnz = {K.nnz}")

    # BC: u = 0 on entire boundary of square
    all_bnd = np.unique(np.concatenate([
        sq.boundary_tags["left"], sq.boundary_tags["right"],
        sq.boundary_tags["top"], sq.boundary_tags["bottom"],
    ]))
    bcs = [DirichletBC(node_ids=all_bnd, component=0, value=0.0)]
    K, rhs = apply_dirichlet_elimination(K, rhs, dof_map, bcs)

    u_sq = DirectSolver().solve(K, rhs)

    # Map nodes to physical coords
    nodes_t = torch.tensor(sq.nodes, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_phys = mapping.psi(nodes_t).cpu().numpy()

    print(f"    u range: [{u_sq.min():.6e}, {u_sq.max():.6e}]")
    return sq, u_sq, x_phys, all_bnd


# ====================================================================
# VISUALISATION
# ====================================================================
def visualise_poisson(shape, mesh_d, u_d, mesh_sq, u_m, x_m, all_bnd, shape_idx):
    """Compare direct and mapped Poisson solutions."""
    nd = mesh_d.nodes

    # Triangulations
    tri_d = mesh_d.element_blocks[0].connectivity
    if mesh_d.element_blocks[0].etype == "quad4":
        tri_d = np.vstack([tri_d[:, [0, 1, 2]], tri_d[:, [0, 2, 3]]])
    triang_d = mtri.Triangulation(nd[:, 0], nd[:, 1], tri_d)

    quad_conn = mesh_sq.element_blocks[0].connectivity
    tri_conn = np.vstack([quad_conn[:, [0, 1, 2]], quad_conn[:, [0, 2, 3]]])
    triang_m = mtri.Triangulation(x_m[:, 0], x_m[:, 1], tri_conn)
    triang_sq = mtri.Triangulation(mesh_sq.nodes[:, 0], mesh_sq.nodes[:, 1], tri_conn)

    # Shape boundary
    t = np.linspace(0, 2 * np.pi, 200)
    bnd_pts = shape(t)

    vmin = 0.0
    vmax = max(u_d.max(), u_m.max())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: solution fields
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang_d, u_d, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="u")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Direct FEM")

    ax = axes[0, 1]
    tcf = ax.tricontourf(triang_sq, u_m, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="u")
    ax.set_aspect("equal"); ax.set_title("Mapped solution on square")

    ax = axes[0, 2]
    tcf = ax.tricontourf(triang_m, u_m, levels=32, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="u")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Mapped back to shape")

    # Row 2: mesh, comparison, boundary check
    ax = axes[1, 0]
    ax.triplot(triang_d, lw=0.3, color="steelblue", alpha=0.5)
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Direct mesh")

    ax = axes[1, 1]
    ax.triplot(triang_m, lw=0.3, color="steelblue", alpha=0.5)
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Mapped mesh")

    # Comparison: max values
    ax = axes[1, 2]
    ax.bar(["Direct", "Mapped"], [u_d.max(), u_m.max()], color=["steelblue", "coral"])
    ax.set_ylabel("max(u)")
    ax.set_title("max(u): {:.4e} vs {:.4e}".format(u_d.max(), u_m.max()))

    plt.suptitle("Poisson on shape {} | f = {}".format(shape_idx + 1, F_SOURCE), fontsize=14)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_shape_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()


def plot_comparison(u_d, u_m, mesh_d, x_m, mesh_sq, shape, shape_idx):
    """Side-by-side comparison of solution fields."""
    nd = mesh_d.nodes

    tri_d = mesh_d.element_blocks[0].connectivity
    if mesh_d.element_blocks[0].etype == "quad4":
        tri_d = np.vstack([tri_d[:, [0, 1, 2]], tri_d[:, [0, 2, 3]]])
    triang_d = mtri.Triangulation(nd[:, 0], nd[:, 1], tri_d)

    quad_conn = mesh_sq.element_blocks[0].connectivity
    tri_conn = np.vstack([quad_conn[:, [0, 1, 2]], quad_conn[:, [0, 2, 3]]])
    triang_m = mtri.Triangulation(x_m[:, 0], x_m[:, 1], tri_conn)

    t = np.linspace(0, 2 * np.pi, 200)
    bnd_pts = shape(t)
    vmax = max(u_d.max(), u_m.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tcf = ax.tricontourf(triang_d, u_d, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="u")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("Direct FEM (max u = {:.4e})".format(u_d.max()))

    ax = axes[1]
    tcf = ax.tricontourf(triang_m, u_m, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="u")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("Mapped FEM (max u = {:.4e})".format(u_m.max()))

    plt.suptitle("Shape {} — Poisson comparison".format(shape_idx + 1))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "poisson_comparison_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()


def plot_mesh_shape(mesh, shape, shape_idx):
    """Plot the mesh for a shape."""
    nd = mesh.nodes
    conn = mesh.element_blocks[0].connectivity
    etype = mesh.element_blocks[0].etype

    t = np.linspace(0, 2 * np.pi, 200)
    bnd_pts = shape(t)

    fig, ax = plt.subplots(figsize=(8, 6))
    polys = [nd[conn[e]] for e in range(conn.shape[0])]
    pc = PolyCollection(polys, edgecolors="steelblue", facecolors="none",
                        linewidths=0.3, alpha=0.8)
    ax.add_collection(pc)
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=2)

    bnd = mesh.boundary_tags.get("boundary", np.array([]))
    if len(bnd) > 0:
        ax.plot(nd[bnd, 0], nd[bnd, 1], ".", color="red", ms=2, zorder=5)

    ax.set_aspect("equal")
    ax.set_title("{} mesh ({} elements, {} nodes)".format(
        etype.upper(), mesh.n_elements, mesh.n_nodes))
    ax.autoscale()
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mesh_shape_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()


# ====================================================================
# GROUND-TRUTH COMPARISON  (high-fidelity reference + cross-mesh errors)
# ====================================================================
def estimate_area(shape, n=400):
    """Shoelace area of the shape boundary, used to target an element count."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = shape(t)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def mesh_shape_to_target(shape, target_elems, etype, n_iter=3):
    """Mesh a shape at approximately `target_elems` elements by adjusting the gmsh
    characteristic length, refining the estimate a couple of times."""
    area = estimate_area(shape)
    factor = 1.0 if etype == "quad4" else 2.0       # ~2 tris per quad cell at equal lc
    lc = float(np.sqrt(factor * area / max(target_elems, 1)))
    mesh = mesh_shape_gmsh(shape, lc=lc, etype=etype)
    for _ in range(n_iter - 1):
        n_el = mesh.n_elements
        if n_el <= 0:
            break
        lc *= float(np.sqrt(n_el / target_elems))   # shrink / grow toward the target
        mesh = mesh_shape_gmsh(shape, lc=lc, etype=etype)
    return mesh


def _tri_conn(mesh):
    """Triangulated connectivity (quad4 split into two triangles)."""
    conn = mesh.element_blocks[0].connectivity
    if mesh.element_blocks[0].etype == "quad4":
        return np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    return conn


def diagnose_mapped_mesh(x_m, tri_conn, label=""):
    """Why a mapped mesh fails to triangulate."""
    a, b, c = x_m[tri_conn[:, 0]], x_m[tri_conn[:, 1]], x_m[tri_conn[:, 2]]
    area = 0.5 * ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                  - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    scale = np.abs(area).max()
    n_dup = len(x_m) - len(np.unique(np.round(x_m.astype(np.float64), 12), axis=0))
    print(f"    [{label}] duplicate nodes : {n_dup}")
    print(f"    [{label}] inverted tris   : {(area < -1e-12*scale).sum()} / {len(area)}"
          f"  ({100*(area < -1e-12*scale).mean():.1f}%)")
    print(f"    [{label}] degenerate tris : {(np.abs(area) < 1e-9*scale).sum()}")
    print(f"    [{label}] area ratio max/min(|.|) : "
          f"{scale/max(np.abs(area).min(), 1e-30):.3e}")
    return area

def _interp_nodal(nodes, tri_conn, u, pts, fill=0.0, label=""):
    """Linearly interpolate a nodal field onto arbitrary points, using the
    ACTUAL mesh connectivity.
    """
    import matplotlib.tri as mtri
    a, b, c = nodes[tri_conn[:, 0]], nodes[tri_conn[:, 1]], nodes[tri_conn[:, 2]]
    area = 0.5 * ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                  - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    bad = area <= 1e-10 * np.abs(area).max()

    try:
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tri_conn)
        if bad.any():
            print(f"    warning{' '+label if label else ''}: masking "
                  f"{bad.sum()}/{len(area)} folded or degenerate triangles "
                  f"({100*bad.mean():.1f}%)")
            triang.set_mask(bad)
        vals = mtri.LinearTriInterpolator(triang, u)(pts[:, 0], pts[:, 1])
        mask = np.ma.getmaskarray(vals)
        return np.ma.filled(vals, fill), mask
    except RuntimeError:
        from scipy.interpolate import LinearNDInterpolator
        print(f"    warning{' '+label if label else ''}: mesh triangulation "
              f"invalid, falling back to Delaunay (accuracy degraded)")
        v = LinearNDInterpolator(nodes, u, fill_value=np.nan)(pts)
        mask = np.isnan(v)
        return np.where(mask, fill, v), mask


def l2_norm_on_mesh(mesh, nodal):
    """Proper L2 norm  (∫_Ω field² dx)^{1/2}  of a nodal field, via element quadrature."""
    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype)
    acc = 0.0
    for e in range(block.connectivity.shape[0]):
        enodes = block.connectivity[e]
        coords = mesh.nodes[enodes]
        ue = nodal[enodes]
        for qp, qw in zip(qpts, qwts):
            N, dN = ref.evaluate(qp)
            detJ = abs(np.linalg.det(dN.T @ coords))
            val = N @ ue
            acc += val * val * detJ * qw
    return float(np.sqrt(acc))


def evaluate_errors(mesh_hi, u_hi, mesh_d, u_d, x_m, tri_conn_m, u_m):
    """Interpolate the direct and pullback solutions onto the high-fidelity nodes and
    return their absolute error fields there together with the (relative) L2 errors,
    measured against the high-fidelity reference by integration over its mesh."""
    pts = mesh_hi.nodes
    ud_i, mask_d = _interp_nodal(mesh_d.nodes, _tri_conn(mesh_d), u_d, pts)
    um_i, mask_m = _interp_nodal(x_m, tri_conn_m, u_m, pts)

    frac = 100.0 * max(mask_d.mean(), mask_m.mean())
    if frac > 1.0:
        print(f"    note: {frac:.1f}% of reference nodes fell outside a coarse mesh "
              f"(filled with boundary value)")

    err_d = np.abs(ud_i - u_hi)
    err_m = np.abs(um_i - u_hi)

    l2_d = l2_norm_on_mesh(mesh_hi, ud_i - u_hi)
    l2_m = l2_norm_on_mesh(mesh_hi, um_i - u_hi)
    if RELATIVE_L2:
        ref = l2_norm_on_mesh(mesh_hi, u_hi)
        if ref > 0:
            l2_d /= ref
            l2_m /= ref
    return err_d, err_m, l2_d, l2_m


def _shape_limits(shape, pad=0.12):
    t = np.linspace(0, 2 * np.pi, 400)
    bp = shape(t)
    x0, x1 = bp[:, 0].min(), bp[:, 0].max()
    y0, y1 = bp[:, 1].min(), bp[:, 1].max()
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    return bp, (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def plot_groundtruth_errors(shape, shape_idx, mesh_hi, u_hi,
                            err_d, l2_d, etype_d, err_m, l2_m):
    """Example-1-style figures: the high-fidelity ground-truth solution, and the L2 error
    maps of the element-matched direct FEM and the pullback FEM over the same domain."""
    tg_hi = mtri.Triangulation(mesh_hi.nodes[:, 0], mesh_hi.nodes[:, 1], _tri_conn(mesh_hi))
    bp, xlim, ylim = _shape_limits(shape)
    lbl = {"tri3": "Tri3", "quad4": "Quad4"}.get(etype_d, etype_d)
    metric = "rel. L2" if RELATIVE_L2 else "L2"

    # (a) ground-truth solution
    fig, ax = plt.subplots(figsize=(7, 6))
    tcf = ax.tricontourf(tg_hi, u_hi, levels=32, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(bp[:, 0], bp[:, 1], "k-", lw=1.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title("Ground truth (high-fidelity FEM)")
    plt.tight_layout()
    plt.savefig(_FIG_DIR / f"groundtruth_shape_{shape_idx}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) error maps, shared colour scale
    #err_max = max(err_d.max(), err_m.max(), 1e-30) #use if I want to scale to actual max error
    '''    
    vmin, vmax = ERROR_CLIM
    levels = np.linspace(vmin, vmax, 33)        # 32 bands over [0, 0.25] uncomment this if I want to use a common error scale for all shapes
    '''


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    #tcf = ax.tricontourf(tg_hi, err_d, levels=32, cmap="hot_r", vmin=0, vmax=err_max) #use if I want to use automatic error scales
    tcf = ax.tricontourf(tg_hi, err_d, levels=32, cmap="hot_r", extend="max") #levels=levels for a common error scale for all shapes
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(bp[:, 0], bp[:, 1], "k-", lw=1.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title(f"Direct FEM ({lbl}) error ({metric} = {l2_d:.2e})")

    ax = axes[1]
    #tcf = ax.tricontourf(tg_hi, err_m, levels=32, cmap="hot_r", vmin=0, vmax=err_max)
    tcf = ax.tricontourf(tg_hi, err_m, levels=32, cmap="hot_r", extend="max") #levels=levels for a common error scale for all shapes
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.plot(bp[:, 0], bp[:, 1], "k-", lw=1.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title(f"Pullback FEM error ({metric} = {l2_m:.2e})")

    plt.tight_layout()
    plt.savefig(_FIG_DIR / f"errors_shape_{shape_idx}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved groundtruth_shape_{shape_idx}.png and errors_shape_{shape_idx}.png")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("MAPPED POISSON ON ELLIPSES / SUPERELLIPSES")
    print("=" * 60)
    print(f"  Device:  {DEVICE}")
    print(f"  Source:  f = {F_SOURCE}")

    # ── Mapping source: encoder if its checkpoint exists, else stored Stage-1 codes ──
    use_encoder = (CHECKPOINT_DIR / ENCODER_CKPT).exists()
    if use_encoder:
        model = load_inference_model(CHECKPOINT_DIR)
        renderer = SDFRenderer(grid_size=SDF_GRID, bounds=SDF_BOUNDS)
        print(f"  Mapping source: ENCODER  (found {ENCODER_CKPT})")
    else:
        model, train_dataset = load_model()
        print(f"  Mapping source: STORED Stage-1 codes  ({ENCODER_CKPT} not found)")

    # Regenerate the same shapes used in training (identical seed / distribution)
    with open(CHECKPOINT_DIR / "config.json") as f:
        cfg = json.load(f)
    data_cfg = cfg["data"]
    n_train = data_cfg["n_train"]
    gen = ShapeGenerator(
        seed=cfg.get("seed", SEED),
        shape_distribution=data_cfg["shape_distribution"],
    )
    train_shapes = [gen.generate() for _ in range(n_train)]

    n_shapes = min(N_TEST_SHAPES, n_train)
    print(f"\n  Testing on {n_shapes} training shapes")

    results = []

    for i in range(n_shapes):
        shape = train_shapes[i]

        print("\n" + "=" * 60)
        print(f"  SHAPE {i+1}/{n_shapes}")
        print("=" * 60)

        # Build the mapping for this shape
        if use_encoder:
            image_np = renderer.render(shape)
            image_t = torch.tensor(image_np, dtype=torch.float32, device=DEVICE)
            mapping = EncoderMapping(model, image_t)
            print(f"  Encoded shape -> z (norm={mapping.z.norm():.4f})")
        else:
            with torch.no_grad():
                z = model.latent_codes(torch.tensor(i, device=DEVICE))
            mapping = AutoDecoderMapping(model, z)
            print(f"  Stored code z (norm={z.norm():.4f})")

        # Direct solve
        print("\n  Direct solve:")
        mesh_d = mesh_shape_gmsh(shape, lc=MESH_LC, etype=DIRECT_ETYPE)
        plot_mesh_shape(mesh_d, shape, i)
        u_d = solve_direct_poisson(shape, mesh_d)

        # Mapped solve
        print("\n  Mapped solve:")
        mesh_sq, u_m, x_m, all_bnd = solve_mapped_poisson(mapping)

        # ── Article comparison vs a high-fidelity ground truth ──
        # Reference: a much finer direct FEM solve of the same shape.
        # Baseline:  a direct FEM solve element-matched to the mapped mesh.
        print("\n  Ground-truth comparison:")
        mesh_hi = mesh_shape_to_target(shape, HIFI_TARGET_ELEMS, HIFI_ETYPE)
        u_hi = solve_direct_poisson(shape, mesh_hi)
        print(f"    Reference : {mesh_hi.n_elements} {HIFI_ETYPE} elements")

        mesh_dc = mesh_shape_to_target(shape, mesh_sq.n_elements, DIRECT_ETYPE)
        u_dc = solve_direct_poisson(shape, mesh_dc)
        print(f"    Matched   : {mesh_dc.n_elements} {DIRECT_ETYPE} (direct) vs "
              f"{mesh_sq.n_elements} quad4 (mapped)")
        quad_conn = mesh_sq.element_blocks[0].connectivity
        tri_conn_m = np.vstack([quad_conn[:, [0, 1, 2]], quad_conn[:, [0, 2, 3]]])
        diagnose_mapped_mesh(x_m, tri_conn_m, f"shape {i+1}")      # <-- add
        err_d, err_m, l2_d, l2_m = evaluate_errors(
            mesh_hi, u_hi, mesh_dc, u_dc, x_m, tri_conn_m, u_m)
        tag = "rel L2" if RELATIVE_L2 else "L2"
        print(f"    {tag} — direct: {l2_d:.3e} | pullback: {l2_m:.3e}")

        plot_groundtruth_errors(shape, i, mesh_hi, u_hi,
                                err_d, l2_d, DIRECT_ETYPE, err_m, l2_m)

        # Compare
        print(f"\n  Comparison:")
        print(f"    Direct  max(u): {u_d.max():.6e}")
        print(f"    Mapped  max(u): {u_m.max():.6e}")
        ratio = u_m.max() / max(u_d.max(), 1e-15)
        print(f"    Ratio:          {ratio:.4f}")

        results.append({
            "shape": i + 1,
            "direct_max": u_d.max(),
            "mapped_max": u_m.max(),
            "ratio": ratio,
            "direct_nodes": mesh_d.n_nodes,
            "direct_elements": mesh_d.n_elements,
            "l2_direct": l2_d,
            "l2_mapped": l2_m,
        })

        # Visualise
        visualise_poisson(shape, mesh_d, u_d, mesh_sq, u_m, x_m, all_bnd, i)
        plot_comparison(u_d, u_m, mesh_d, x_m, mesh_sq, shape, i)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    tag = "rel L2" if RELATIVE_L2 else "L2"
    print(f"  {'Shape':>5}  {'Direct max(u)':>14}  {'Mapped max(u)':>14}  "
          f"{tag+' direct':>12}  {tag+' pullbk':>12}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['shape']:>5d}  {r['direct_max']:>14.6e}  {r['mapped_max']:>14.6e}  "
              f"{r['l2_direct']:>12.3e}  {r['l2_mapped']:>12.3e}")

    print("\n" + "=" * 60)
    print("ALL SHAPES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()