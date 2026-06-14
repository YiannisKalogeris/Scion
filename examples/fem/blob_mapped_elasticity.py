#!/usr/bin/env python3
"""
Mapped Plane-Stress Elasticity on Fourier Blob Domains via Auto-Decoder NSF.

Direction:  ψ: □ → Ω   (forward pass = square → physical)

Mathematics:
    Physical domain:
        ∫_Ω ε(u) : C : ε(v) dx = ∫_Ω f·v dx

    Change of variables  x = ψ(y),  F = Dψ,  dx = |det F| dy:
        ∇_x = ∇_y · F⁻¹

    Pulled-back stiffness:
        ke = Σ_q  Bᵀ(y_q) C B(y_q) |det F_q| |det J_iso,q| w_q

    where B is built from dN/dx = dN/dy · F⁻¹ (physical gradients).

    Pulled-back body force:
        fe = Σ_q  N(y_q) f(ψ(y_q)) |det F_q| |det J_iso,q| w_q

Prerequisites:
    Train the auto-decoder NSF on Fourier blobs:
        cd projects/nsf_domain_mapping
        python Main_nsf.py   (with shape_distribution = {"FourierBlob": 1.0})

Place at:  examples/fem/blob_mapped_elasticity.py
Run:       python examples/fem/blob_mapped_elasticity.py
"""

import sys
import json
from xml.parsers.expat import model
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from pathlib import Path
import pickle

import torch

# ── Simulacrum library ──
from simulacrum.geometry.shapes import FourierBlob, ShapeGenerator
from simulacrum.mesh.mesh import Mesh, ElementBlock
from simulacrum.mesh.generators import rectangle_mesh
from simulacrum.core.domain import Domain
from simulacrum.core.dof import DOFMap
from simulacrum.boundary import DirichletBC, apply_dirichlet_elimination
from simulacrum.solvers.linear import DirectSolver
from simulacrum.discretization.fem.assembler import FEMAssembler
from simulacrum.discretization.fem.elements import get_element
from simulacrum.discretization.fem.quadrature import get_quadrature
from simulacrum.materials.linear_elastic import PlaneStress
from simulacrum.geometry.sdf import sample_points_in_shape
from simulacrum.geometry.harmonic_maps import HarmonicMapSolver

# ── NSF model ──
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "nsf_domain_mapping"
sys.path.insert(0, str(PROJECT_DIR / "models"))
sys.path.insert(0, str(PROJECT_DIR))
from neural_spline_flow import NeuralSplineFlow
from Main_nsf import load_inference_model, EncoderMapping
from dataset import DataSample

# ====================================================================
# CONFIGURATION
# ====================================================================
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_nsf"
DEVICE = "cpu"   # use "mps" or "cuda" if available

_FIG_DIR = Path(__file__).resolve().parent / "Figures_blob_mapped_elasticity"
_FIG_DIR.mkdir(exist_ok=True)

N_SQUARE = 60               # elements per side on reference square
MESH_LC = 0.02              # gmsh element size for direct solve
DIRECT_ETYPE = "quad4"       # "tri3" or "quad4"

# Material: steel-like (plane stress, MPa)
E_YOUNG = 210e3
NU_POISSON = 0.3
THICKNESS = 1.0             # unit thickness for plane stress

# Body force (uniform gravity-like, in -y direction)
BODY_FORCE = np.array([0.0, -10.0])   # (fx, fy) per unit volume

# Number of test shapes to solve
N_TEST_SHAPES = 5
SEED = 42


# ====================================================================
# NSF MAPPING WRAPPER (auto-decoder version)
# ====================================================================
class AutoDecoderMappingWrapper:
    """Wraps AutoDecoderNSF + fitted z for FEM use. Direction: □ → Ω."""

    def __init__(self, model, z, device):
        self.model = model
        self.model.eval()
        self.z = z
        self.device = device

    def psi(self, y):
        y = y.to(self.device)
        with torch.no_grad():
            x, _ = self.model.forward_with_z(y, self.z)
        return x

    def jacobian_matrix(self, y):
        y = y.to(self.device).detach().requires_grad_(True)
        x, _ = self.model.forward_with_z(y, self.z)
        F = torch.zeros(y.shape[0], 2, 2, device=self.device)
        for j in range(2):
            g = torch.zeros_like(x)
            g[:, j] = 1.0
            dxdy = torch.autograd.grad(x, y, g, retain_graph=True)[0]
            F[:, j, :] = dxdy
        return F.detach()


# ====================================================================
# LOAD AUTO-DECODER MODEL
# ====================================================================
def load_autodecoder():
    """Load trained auto-decoder NSF."""
    config_path = CHECKPOINT_DIR / "config.json"
    model_path = CHECKPOINT_DIR / "best_model.pt"

    if not model_path.exists():
        print(f"\n  ERROR: No model at {model_path}")
        print(f"  Train with: cd {PROJECT_DIR} && python Main_nsf.py")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = cfg["model"]
    # Need n_shapes from training data to reconstruct embedding table
    data_cfg = cfg["data"]
    n_train = data_cfg["n_train"]

    model = AutoDecoderNSF(
        n_shapes=n_train,
        latent_dim=model_cfg["latent_dim"],
        n_layers=model_cfg["n_layers"],
        n_bins=model_cfg["n_bins"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden=model_cfg["n_hidden"],
        domain_bounds=tuple(model_cfg["domain_bounds"]),
        codomain_bounds=tuple(model_cfg["codomain_bounds"]),
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    n_params = sum(p.numel() for p in model.flow.parameters())
    print(f"  Loaded auto-decoder NSF | latent_dim={model_cfg['latent_dim']} | "
          f"flow params={n_params:,}")
    return model


# ====================================================================
# FIT NSF FOR A SHAPE (using harmonic map ground truth)
# ====================================================================
def fit_nsf_for_shape(model, shape, n_interior=2000, n_boundary=128):
    """Compute harmonic map and fit z for a given shape."""
    bounds = (-2.5, 2.5)
    solver = HarmonicMapSolver(grid_size=128, bounds=bounds)
    solver.solve(shape)

    # Sample interior points
    rng = np.random.default_rng(42)
    ellipse_pts = np.empty((0, 2), dtype=np.float32)
    square_pts = np.empty((0, 2), dtype=np.float32)

    while len(ellipse_pts) < n_interior:
        n_req = int((n_interior - len(ellipse_pts)) * 1.3) + 50
        cands = sample_points_in_shape(shape, n_req, bounds, rng)
        tgts = solver.evaluate(cands)
        valid = ~np.isnan(tgts).any(axis=1)
        n_take = min(valid.sum(), n_interior - len(ellipse_pts))
        ellipse_pts = np.concatenate([ellipse_pts, cands[valid][:n_take]])
        square_pts = np.concatenate([square_pts, tgts[valid][:n_take].astype(np.float32)])

    # Boundary
    t = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    bnd_phys = shape(t).astype(np.float32)

    # Boundary targets on square
    def param_to_square(t_val):
        s = ((t_val / (2 * np.pi)) % 1.0) * 8
        if s < 2: return np.array([1.0, -1.0 + s])
        elif s < 4: return np.array([1.0 - (s - 2), 1.0])
        elif s < 6: return np.array([-1.0, 1.0 - (s - 4)])
        else: return np.array([-1.0 + (s - 6), -1.0])

    bnd_square = np.array([param_to_square(ti) for ti in t], dtype=np.float32)

    # Create DataSample (reversed: interior_targets = square, interior_points = phys)
    sample = DataSample(
        image=np.zeros((64, 64, 1), dtype=np.float32),  # dummy
        interior_points=ellipse_pts,
        interior_targets=square_pts,
        boundary_points=bnd_phys,
        boundary_targets=bnd_square,
    )

    # Fit z
    z = fit_new_shape(model, sample, device=DEVICE)
    return z, sample


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

    # Keep only nodes that appear in 2D elements
    used_nodes = np.unique(conn_raw.ravel())
    old_to_new = np.full(len(nodes_all), -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(len(used_nodes))
    nodes = nodes_all[used_nodes]
    conn = old_to_new[conn_raw]

    # Boundary: 1D element nodes, remapped
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

    # Identify left and right boundary subsets geometrically
    x_min, x_max = nodes[bnd, 0].min(), nodes[bnd, 0].max()
    x_range = x_max - x_min
    tol = 0.15 * x_range
    left = bnd[nodes[bnd, 0] < x_min + tol]
    right = bnd[nodes[bnd, 0] > x_max - tol]

    mesh.boundary_tags["left"] = left
    mesh.boundary_tags["right"] = right

    return mesh


# ====================================================================
# STRAIN-DISPLACEMENT MATRIX
# ====================================================================
def build_B_matrix(dN_dx, dim=2):
    """Build Voigt strain-displacement matrix from shape function gradients.

    Args:
        dN_dx: (n_nodes, 2) gradients w.r.t. physical coordinates
    Returns:
        B: (3, n_nodes*2) for 2D plane stress/strain
    """
    n_nodes = dN_dx.shape[0]
    B = np.zeros((3, n_nodes * 2))
    for i in range(n_nodes):
        B[0, 2 * i] = dN_dx[i, 0]
        B[1, 2 * i + 1] = dN_dx[i, 1]
        B[2, 2 * i] = dN_dx[i, 1]
        B[2, 2 * i + 1] = dN_dx[i, 0]
    return B


# ====================================================================
# DIRECT ELASTICITY SOLVE
# ====================================================================
def solve_direct_elasticity(shape, mesh):
    """Solve plane-stress elasticity directly on the physical mesh."""
    print(f"    Direct solve: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C = material.tangent(dim=2)

    domain = Domain(mesh=mesh, dim=2)
    domain.add_material("default", material)
    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total

    # Assemble stiffness and body force
    from scipy import sparse
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_dof)

    for block in mesh.element_blocks:
        ref = get_element(block.etype)
        qpts, qwts = get_quadrature(block.etype)

        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            edofs = dof_map.element_dofs(enodes)
            n_el_dof = len(edofs)
            ke = np.zeros((n_el_dof, n_el_dof))
            fe = np.zeros(n_el_dof)

            for qp, qw in zip(qpts, qwts):
                N, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords
                detJ = abs(np.linalg.det(J))
                dN_dx = dN_dxi @ np.linalg.inv(J)

                B = build_B_matrix(dN_dx)
                ke += (B.T @ C @ B) * detJ * qw

                # Body force: N_i * f_j
                for i in range(len(N)):
                    fe[2 * i] += N[i] * BODY_FORCE[0] * detJ * qw
                    fe[2 * i + 1] += N[i] * BODY_FORCE[1] * detJ * qw

            for il, ig in enumerate(edofs):
                rhs[ig] += fe[il]
                for jl, jg in enumerate(edofs):
                    rows.append(ig); cols.append(jg); vals.append(ke[il, jl])

    K = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                          shape=(n_dof, n_dof))


    # BCs: clamp left boundary (ux = uy = 0)
    left = mesh.boundary_tags["left"]
    print(f"    Clamping {len(left)} nodes on left boundary")
    bcs = [
        DirichletBC(node_ids=left, component=0, value=0.0),
        DirichletBC(node_ids=left, component=1, value=0.0),
    ]

    K, rhs = apply_dirichlet_elimination(K, rhs, dof_map, bcs)


    u = DirectSolver().solve(K, rhs)
    disp = u.reshape(-1, 2)
    u_mag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)
    print(f"    max |u| = {u_mag.max():.6e}")

    return u, disp, u_mag


# ====================================================================
# PRECOMPUTE METRIC FOR ELASTICITY
# ====================================================================
def precompute_metric_elasticity(mapping, mesh):
    """Precompute F, F_inv, |det F|, and physical coords at all quad points."""
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
    print(f"    Metric at {len(all_y)} quadrature points ...")

    y_t = torch.tensor(all_y, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        x_phys = mapping.psi(y_t).cpu().numpy()

    F_t = mapping.jacobian_matrix(y_t)
    detF = torch.det(F_t)
    abs_detF = detF.abs()
    F_inv = torch.linalg.inv(F_t)

    print(f"    det(Dψ) range: [{detF.min():.4f}, {detF.max():.4f}]  "
          f"({100*(detF>0).float().mean():.0f}% positive)")

    return {
        "F_inv": F_inv.detach().cpu().numpy(),
        "detF": abs_detF.detach().cpu().numpy(),
        "x_phys": x_phys,
    }


# ====================================================================
# MAPPED ELASTICITY ASSEMBLY
# ====================================================================
def assemble_mapped_elasticity(mesh, metric):
    """Assemble pulled-back elasticity system on the reference square."""
    from scipy import sparse

    material = PlaneStress(E=E_YOUNG, nu=NU_POISSON)
    C = material.tangent(dim=2)

    block = mesh.element_blocks[0]
    ref = get_element(block.etype)
    qpts, qwts = get_quadrature(block.etype)

    dof_map = DOFMap(mesh.n_nodes, n_components=2)
    n_dof = dof_map.n_total
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_dof)

    qp_idx = 0
    for e in range(block.connectivity.shape[0]):
        enodes = block.connectivity[e]
        coords = mesh.nodes[enodes]
        edofs = dof_map.element_dofs(enodes)
        n_el_dof = len(edofs)
        ke = np.zeros((n_el_dof, n_el_dof))
        fe = np.zeros(n_el_dof)

        for qp, qw in zip(qpts, qwts):
            N, dN_dxi = ref.evaluate(qp)
            J_iso = dN_dxi.T @ coords
            detJ_iso = abs(np.linalg.det(J_iso))
            dN_dy = dN_dxi @ np.linalg.inv(J_iso)

            # Physical gradients: dN/dx = dN/dy · F⁻¹
            F_inv = metric["F_inv"][qp_idx]     # (2, 2)
            abs_detF = metric["detF"][qp_idx]   # scalar
            x_qp = metric["x_phys"][qp_idx]     # (2,) physical coord

            dN_dx = dN_dy @ F_inv

            # B-matrix using physical gradients
            B = build_B_matrix(dN_dx)

            # Stiffness: Bᵀ C B |det F| |det J_iso| w
            ke += (B.T @ C @ B) * abs_detF * detJ_iso * qw

            # Body force: N_i f_j |det F| |det J_iso| w
            for i in range(len(N)):
                fe[2 * i] += N[i] * BODY_FORCE[0] * abs_detF * detJ_iso * qw
                fe[2 * i + 1] += N[i] * BODY_FORCE[1] * abs_detF * detJ_iso * qw

            qp_idx += 1

        for il, ig in enumerate(edofs):
            rhs[ig] += fe[il]
            for jl, jg in enumerate(edofs):
                rows.append(ig); cols.append(jg); vals.append(ke[il, jl])

    K = sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                          shape=(n_dof, n_dof))
    return K, rhs, dof_map


# ====================================================================
# MAPPED ELASTICITY SOLVE
# ====================================================================
def solve_mapped_elasticity(mapping):
    sq = rectangle_mesh((-1.0, 1.0), (-1.0, 1.0), N_SQUARE, N_SQUARE, etype="quad4")
    print(f"    Square mesh: {sq.n_nodes} nodes, {sq.n_elements} Quad4")

    metric = precompute_metric_elasticity(mapping, sq)
    K, rhs, dof_map = assemble_mapped_elasticity(sq, metric)
    print(f"    System: {K.shape[0]} DOFs, nnz = {K.nnz}")

    # Map ALL boundary nodes to physical coordinates
    all_bnd = np.unique(np.concatenate([
        sq.boundary_tags["left"], sq.boundary_tags["right"],
        sq.boundary_tags["top"], sq.boundary_tags["bottom"],
    ]))
    bnd_y = torch.tensor(sq.nodes[all_bnd], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        bnd_x = mapping.psi(bnd_y).cpu().numpy()

    # Apply SAME geometric criterion as the direct solve
    x_min = bnd_x[:, 0].min()
    x_max = bnd_x[:, 0].max()
    x_range = x_max - x_min
    left_mask = bnd_x[:, 0] < x_min + 0.15 * x_range
    left_nodes = all_bnd[left_mask]

    print(f"    Clamping {len(left_nodes)} boundary nodes (physical left)")

    bcs = [
        DirichletBC(node_ids=left_nodes, component=0, value=0.0),
        DirichletBC(node_ids=left_nodes, component=1, value=0.0),
    ]
    K, rhs = apply_dirichlet_elimination(K, rhs, dof_map, bcs)

    u_sq = DirectSolver().solve(K, rhs)
    disp = u_sq.reshape(-1, 2)
    u_mag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)

    # Map nodes to physical coords
    nodes_t = torch.tensor(sq.nodes, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_phys = mapping.psi(nodes_t).cpu().numpy()

    print(f"    max |u| = {u_mag.max():.6e}")
    return sq, u_sq, disp, u_mag, x_phys


# ====================================================================
# VISUALISATION
# ====================================================================
def visualise_elasticity(shape, mesh_d, disp_d, umag_d,
                         mesh_sq, disp_m, umag_m, x_m,
                         shape_idx):
    """Compare direct and mapped elasticity solutions."""
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

    vmax = max(umag_d.max(), umag_m.max())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: displacement magnitude
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang_d, umag_d, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|u|")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Direct FEM |u|")

    ax = axes[0, 1]
    tcf = ax.tricontourf(triang_sq, umag_m, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|u|")
    ax.set_aspect("equal"); ax.set_title("Mapped |u| on square")

    ax = axes[0, 2]
    tcf = ax.tricontourf(triang_m, umag_m, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|u|")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Mapped |u| on shape")

    # Row 2: ux, uy, deformed
    ax = axes[1, 0]
    tcf = ax.tricontourf(triang_d, disp_d[:, 0], levels=32, cmap="RdBu_r")
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="ux")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Direct FEM ux")

    ax = axes[1, 1]
    tcf = ax.tricontourf(triang_d, disp_d[:, 1], levels=32, cmap="RdBu_r")
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="uy")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Direct FEM uy")

    # Deformed shape
    ax = axes[1, 2]
    scale = 0.2 * (nd[:, 0].max() - nd[:, 0].min()) / max(np.abs(disp_d).max(), 1e-15)
    def_d = nd + scale * disp_d
    tri_def = tri_d
    triang_def = mtri.Triangulation(def_d[:, 0], def_d[:, 1], tri_def)
    tcf = ax.tricontourf(triang_def, umag_d, levels=32, cmap="viridis",
                         vmin=0, vmax=vmax, alpha=0.8)
    fig.colorbar(tcf, ax=ax, shrink=0.8, label="|u|")
    ax.triplot(triang_def, lw=0.2, color="k", alpha=0.1)
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k--", lw=1, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title("Deformed shape (x{:.0f})".format(scale))

    plt.suptitle("Plane-stress elasticity on Fourier blob (shape {})".format(shape_idx + 1),
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "elasticity_shape_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()

    # ── Individual: mapped displacement components ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    tcf = ax.tricontourf(triang_m, disp_m[:, 0], levels=32, cmap="RdBu_r")
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="ux")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Mapped FEM ux")

    ax = axes[1]
    tcf = ax.tricontourf(triang_m, disp_m[:, 1], levels=32, cmap="RdBu_r")
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="uy")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal"); ax.set_title("Mapped FEM uy")

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "elasticity_mapped_disp_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()


def plot_comparison(umag_d, umag_m, mesh_d, x_m, mesh_sq, shape, shape_idx):
    """Direct comparison of displacement magnitude between methods."""
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
    vmax = max(umag_d.max(), umag_m.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tcf = ax.tricontourf(triang_d, umag_d, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="|u|")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("Direct FEM (max |u| = {:.4e})".format(umag_d.max()))

    ax = axes[1]
    tcf = ax.tricontourf(triang_m, umag_m, levels=32, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(tcf, ax=ax, shrink=0.85, label="|u|")
    ax.plot(bnd_pts[:, 0], bnd_pts[:, 1], "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("Mapped FEM (max |u| = {:.4e})".format(umag_m.max()))

    plt.suptitle("Shape {} — displacement magnitude comparison".format(shape_idx + 1))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "elasticity_comparison_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()


def plot_mesh_shape(mesh, shape, shape_idx):
    """Plot the mesh for a shape."""
    from matplotlib.collections import PolyCollection

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

    # Mark left (clamped) and right boundary
    left = mesh.boundary_tags.get("left", np.array([]))
    if len(left) > 0:
        ax.plot(nd[left, 0], nd[left, 1], "s", color="blue", ms=3, label="Clamped", zorder=5)

    ax.set_aspect("equal")
    ax.set_title("{} mesh ({} elements, {} nodes)".format(
        etype.upper(), mesh.n_elements, mesh.n_nodes))
    ax.legend(loc="upper right", fontsize=8)
    ax.autoscale()
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "mesh_shape_{}.png".format(shape_idx),
                dpi=200, bbox_inches="tight")
    plt.show()

def check_harmonic_map(shape, shape_idx, grid_size=128, bounds=(-2.5, 2.5)):
    """Visualise harmonic map quality for a shape."""
    from simulacrum.geometry.harmonic_maps import HarmonicMapSolver
    from simulacrum.geometry.sdf import sample_points_in_shape

    solver = HarmonicMapSolver(grid_size=grid_size, bounds=bounds)
    solver.solve(shape)

    # Sample interior points
    rng = np.random.default_rng(42)
    pts = sample_points_in_shape(shape, 2000, bounds, rng)
    mapped = solver.evaluate(pts)
    valid = ~np.isnan(mapped).any(axis=1)
    pts = pts[valid]
    mapped = mapped[valid]

    # Boundary
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    bnd = shape(t)

    # Check quality
    in_square = (np.abs(mapped) <= 1.0).all(axis=1).mean()
    print(f"  Harmonic map: {valid.sum()} valid pts, "
          f"{100*in_square:.1f}% land in [-1,1]^2")

    # Compute Jacobian determinant via finite differences
    eps = 1e-4
    det_samples = []
    test_pts = pts[:500]
    for p in test_pts:
        mx = solver.evaluate(np.array([p + [eps, 0]]))[0]
        my = solver.evaluate(np.array([p + [0, eps]]))[0]
        m0 = solver.evaluate(np.array([p]))[0]
        if np.any(np.isnan(mx)) or np.any(np.isnan(my)) or np.any(np.isnan(m0)):
            continue
        J = np.column_stack([(mx - m0) / eps, (my - m0) / eps])
        det_samples.append(np.linalg.det(J))
    det_samples = np.array(det_samples)
    n_neg = (det_samples < 0).sum()
    print(f"  Jacobian det: min={det_samples.min():.4f}, max={det_samples.max():.4f}, "
          f"negative={n_neg}/{len(det_samples)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Physical points coloured by mapped x
    ax = axes[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=mapped[:, 0], s=3, cmap='RdBu_r')
    ax.plot(bnd[:, 0], bnd[:, 1], 'k-', lw=2)
    fig.colorbar(sc, ax=ax, label='mapped x')
    ax.set_aspect('equal'); ax.set_title('Harmonic map: x-component')

    # Mapped points on the square
    ax = axes[1]
    ax.scatter(mapped[:, 0], mapped[:, 1], s=3, alpha=0.3, c='blue')
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='green', lw=2))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.set_title('Points mapped to square')

    # Jacobian determinant
    ax = axes[2]
    ax.hist(det_samples, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_title('Harmonic map det(J) ({} negative)'.format(n_neg))

    plt.suptitle('Harmonic map diagnostic — shape {}'.format(shape_idx + 1))
    plt.tight_layout()
    plt.savefig(_FIG_DIR / 'harmonic_check_{}.png'.format(shape_idx),
                dpi=200, bbox_inches='tight')
    plt.show()

# ====================================================================
# MAIN (the correct way)
# ====================================================================
# def main():
#     print("=" * 60)
#     print("MAPPED PLANE-STRESS ELASTICITY ON FOURIER BLOBS")
#     print("=" * 60)
#     print(f"  Device:    {DEVICE}")
#     print(f"  Material:  E={E_YOUNG}, nu={NU_POISSON}")
#     print(f"  Body force: {BODY_FORCE}")

#     from dataset import DiffeomorphismDataset
#     train_dataset = DiffeomorphismDataset.load(PROJECT_DIR / "data" / "train.pkl")

#     # Load encoder + flow model
#     model = load_inference_model(CHECKPOINT_DIR)

#     # Generate test shapes
#     gen = ShapeGenerator(seed=SEED, shape_distribution={"FourierBlob": 1.0})
#     test_shapes = [gen.generate() for _ in range(N_TEST_SHAPES)]
#     print(f"\n  Generated {N_TEST_SHAPES} test Fourier blobs")

#     for i, shape in enumerate(test_shapes):
#         print("\n" + "=" * 60)
#         print(f"  SHAPE {i+1}/{N_TEST_SHAPES}")
#         print("=" * 60)
#         check_harmonic_map(shape, i)

#         # Get SDF image for this shape
#         from simulacrum.geometry.sdf import SDFRenderer
#         renderer = SDFRenderer(grid_size=64, bounds=(-2.5, 2.5))
#         image_np = renderer.render(shape) 
#         image_t = torch.tensor(image_np, dtype=torch.float32, device=DEVICE)

#         # Encoder: image → z → mapping  (single forward pass, no fitting!)
#         mapping = EncoderMapping(model, image_t)
#         print(f"  Encoded shape to z (norm={mapping.z.norm():.4f})")

#         # Direct solve
#         print("\n  Direct solve:")
#         mesh_d = mesh_shape_gmsh(shape, lc=MESH_LC, etype=DIRECT_ETYPE)
#         plot_mesh_shape(mesh_d, shape, i)
#         u_d, disp_d, umag_d = solve_direct_elasticity(shape, mesh_d)

#         # Mapped solve
#         print("\n  Mapped solve:")
#         mesh_sq, u_m, disp_m, umag_m, x_m = solve_mapped_elasticity(mapping)

#         # Mapped solve
#         print("\n  Mapped solve:")
#         mesh_sq, u_m, disp_m, umag_m, x_m = solve_mapped_elasticity(mapping)

#         # Boundary check
#         t = np.linspace(0, 2 * np.pi, 200)
#         true_bnd = shape(t)
#         print(f"  True boundary x: [{true_bnd[:,0].min():.3f}, {true_bnd[:,0].max():.3f}]")
#         print(f"  True boundary y: [{true_bnd[:,1].min():.3f}, {true_bnd[:,1].max():.3f}]")
#         print(f"  Mapped nodes  x: [{x_m[:,0].min():.3f}, {x_m[:,0].max():.3f}]")
#         print(f"  Mapped nodes  y: [{x_m[:,1].min():.3f}, {x_m[:,1].max():.3f}]")

#         # Verify shape matches training data
#         sample = train_dataset[i]
#         train_bnd = sample.boundary_points
#         regen_bnd = shape(np.linspace(0, 2*np.pi, len(train_bnd), endpoint=False))
#         mismatch = np.max(np.abs(train_bnd - regen_bnd))
#         print(f"  Shape match check: max boundary difference = {mismatch:.6f}")

#         # Compare
#         print(f"\n  Comparison:")
#         print(f"    Direct  max |u|: {umag_d.max():.6e}")
#         print(f"    Mapped  max |u|: {umag_m.max():.6e}")
#         ratio = umag_m.max() / max(umag_d.max(), 1e-15)
#         print(f"    Ratio:           {ratio:.4f}")

#         # Visualise
#         visualise_elasticity(shape, mesh_d, disp_d, umag_d,
#                              mesh_sq, disp_m, umag_m, x_m, i)
#         plot_comparison(umag_d, umag_m, mesh_d, x_m, mesh_sq, shape, i)

#     print("\n" + "=" * 60)
#     print("ALL SHAPES COMPLETE")
#     print("=" * 60)

# ====================================================================
# MAIN hardcoded to use stored codes (bypass encoder)
# ====================================================================
def main():
    print("=" * 60)
    print("MAPPED PLANE-STRESS ELASTICITY ON FOURIER BLOBS")
    print("=" * 60)
    print(f"  Device:    {DEVICE}")
    print(f"  Material:  E={E_YOUNG}, nu={NU_POISSON}")
    print(f"  Body force: {BODY_FORCE}")

    from dataset import DiffeomorphismDataset
    train_dataset = DiffeomorphismDataset.load(PROJECT_DIR / "data" / "train.pkl")
    
    # Load auto-decoder with stored codes (bypass encoder)
    from Main_nsf import create_autodecoder, AutoDecoderMapping
    from dataset import DiffeomorphismDataset
    train_dataset = DiffeomorphismDataset.load(PROJECT_DIR / "data" / "train.pkl")
    model = create_autodecoder(len(train_dataset))
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "stage1_best.pt", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    # Generate test shapes
    gen = ShapeGenerator(seed=SEED, shape_distribution={"FourierBlob": 1.0})
    test_shapes = [gen.generate() for _ in range(N_TEST_SHAPES)]
    print(f"\n  Generated {N_TEST_SHAPES} test Fourier blobs")

    for i, shape in enumerate(test_shapes):
        print("\n" + "=" * 60)
        print(f"  SHAPE {i+1}/{N_TEST_SHAPES}")
        print("=" * 60)
        check_harmonic_map(shape, i)

        # Use stored code directly
        with torch.no_grad():
            z = model.latent_codes(torch.tensor(i, device=DEVICE))
        mapping = AutoDecoderMapping(model, z)
        print(f"  Stored code z (norm={z.norm():.4f})")


        # Direct solve
        print("\n  Direct solve:")
        mesh_d = mesh_shape_gmsh(shape, lc=MESH_LC, etype=DIRECT_ETYPE)
        plot_mesh_shape(mesh_d, shape, i)
        u_d, disp_d, umag_d = solve_direct_elasticity(shape, mesh_d)

        # Mapped solve
        print("\n  Mapped solve:")
        mesh_sq, u_m, disp_m, umag_m, x_m = solve_mapped_elasticity(mapping)

        # Mapped solve
        print("\n  Mapped solve:")
        mesh_sq, u_m, disp_m, umag_m, x_m = solve_mapped_elasticity(mapping)

        # Boundary check
        t = np.linspace(0, 2 * np.pi, 200)
        true_bnd = shape(t)
        print(f"  True boundary x: [{true_bnd[:,0].min():.3f}, {true_bnd[:,0].max():.3f}]")
        print(f"  True boundary y: [{true_bnd[:,1].min():.3f}, {true_bnd[:,1].max():.3f}]")
        print(f"  Mapped nodes  x: [{x_m[:,0].min():.3f}, {x_m[:,0].max():.3f}]")
        print(f"  Mapped nodes  y: [{x_m[:,1].min():.3f}, {x_m[:,1].max():.3f}]")

        # Verify shape matches training data
        sample = train_dataset[i]
        train_bnd = sample.boundary_points
        regen_bnd = shape(np.linspace(0, 2*np.pi, len(train_bnd), endpoint=False))
        mismatch = np.max(np.abs(train_bnd - regen_bnd))
        print(f"  Shape match check: max boundary difference = {mismatch:.6f}")

        # Compare
        print(f"\n  Comparison:")
        print(f"    Direct  max |u|: {umag_d.max():.6e}")
        print(f"    Mapped  max |u|: {umag_m.max():.6e}")
        ratio = umag_m.max() / max(umag_d.max(), 1e-15)
        print(f"    Ratio:           {ratio:.4f}")

        # Visualise
        visualise_elasticity(shape, mesh_d, disp_d, umag_d,
                             mesh_sq, disp_m, umag_m, x_m, i)
        plot_comparison(umag_d, umag_m, mesh_d, x_m, mesh_sq, shape, i)

    print("\n" + "=" * 60)
    print("ALL SHAPES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
