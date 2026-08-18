#!/usr/bin/env python3
"""
Train an (unconditional) NSF in the REVERSED direction:  ψ: □ → Ω

The forward pass of the flow maps reference square points to ellipse points.
This is the correct direction for the FEM pipeline because:
    - Given any y ∈ □, ψ(y) gives the physical coordinate  (forward, reliable)
    - Dψ(y) via autograd gives the Jacobian matrix
    - No inverse is needed in the FEM assembly

Training data:
    - Input:  harmonic map values (points in [-1,1]²)
    - Target: corresponding ellipse points

Place at:  projects/nsf_domain_mapping/train_single_ellipse.py
Run:       python train_single_ellipse.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
import json

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR / "models"))

from scion.geometry.shapes import Ellipse
from scion.geometry.harmonic_maps import HarmonicMapSolver
from scion.geometry.sdf import sample_points_in_shape
from neural_spline_flow import NeuralSplineFlow

# ====================================================================
# CONFIGURATION
# ====================================================================
ELLIPSE_A = 2.0
ELLIPSE_B = 1.0
BOUNDS = (-2.1, 2.1)

MODEL_CONFIG = {
    "n_layers": 8,
    "n_bins": 20,
    "hidden_dim": 128,
    "n_hidden": 3,
    # Domain is the SQUARE (input to forward pass)
    "domain_bounds": (-1.1, 1.1),
    # Codomain is the ELLIPSE region (output of forward pass)
    "codomain_bounds": (-2.1, 2.1),
}

TRAIN_CONFIG = {
    "n_interior": 8000,
    "n_boundary": 2048,
    "n_epochs": 20000,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "boundary_weight": 5.0,
    "logdet_weight": 0.001,
    "grad_clip": 1.0,
    "print_every": 500,
}

CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints_single_ellipse"
DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps"  if torch.backends.mps.is_available() else "cpu"


# ====================================================================
# DATA GENERATION  (reversed: input=square, target=ellipse)
# ====================================================================
def _param_to_square(t: float) -> np.ndarray:
    """Map t ∈ [0, 2π) → square boundary [-1,1]²."""
    s = ((t / (2 * np.pi)) % 1.0) * 8
    if s < 2:
        return np.array([1.0, -1.0 + s])
    elif s < 4:
        return np.array([1.0 - (s - 2), 1.0])
    elif s < 6:
        return np.array([-1.0, 1.0 - (s - 4)])
    else:
        return np.array([-1.0 + (s - 6), -1.0])


def generate_data():
    """
    Generate training data for ψ: □ → Ω.

    Returns:
        square_pts   : (N, 2) points in [-1,1]²  (NSF input)
        ellipse_pts  : (N, 2) corresponding points in Ω  (NSF target)
        bnd_square   : (M, 2) square boundary points
        bnd_ellipse  : (M, 2) ellipse boundary points
    """
    print("Generating ground truth data (reversed direction) ...")

    shape = Ellipse(center=(0.0, 0.0), a=ELLIPSE_A, b=ELLIPSE_B, theta=0.0)
    solver = HarmonicMapSolver(grid_size=768, bounds=BOUNDS)
    solver.solve(shape)

    # ── Interior: sample in ellipse, get harmonic map → square coords ──
    rng = np.random.default_rng(42)
    n_target = TRAIN_CONFIG["n_interior"]
    ellipse_pts = np.empty((0, 2), dtype=np.float32)
    square_pts = np.empty((0, 2), dtype=np.float32)

    while len(ellipse_pts) < n_target:
        n_request = int((n_target - len(ellipse_pts)) * 1.3) + 50
        candidates = sample_points_in_shape(shape, n_request, BOUNDS, rng)
        targets = solver.evaluate(candidates)  # ellipse → square
        valid = ~np.isnan(targets).any(axis=1)
        n_take = min(valid.sum(), n_target - len(ellipse_pts))
        ellipse_pts = np.concatenate([ellipse_pts, candidates[valid][:n_take]])
        square_pts = np.concatenate([square_pts, targets[valid][:n_take].astype(np.float32)])

    # ── Boundary ──
    n_bnd = TRAIN_CONFIG["n_boundary"]
    t = np.linspace(0, 2 * np.pi, n_bnd, endpoint=False)
    bnd_ellipse = shape(t).astype(np.float32)
    bnd_square = np.array([_param_to_square(ti) for ti in t], dtype=np.float32)

    print(f"  Interior: {len(square_pts)}  (exact)")
    print(f"  Boundary: {len(bnd_square)}")
    print(f"  Square input range:  [{square_pts.min():.3f}, {square_pts.max():.3f}]")
    print(f"  Ellipse target range: x=[{ellipse_pts[:,0].min():.3f}, {ellipse_pts[:,0].max():.3f}]"
          f"  y=[{ellipse_pts[:,1].min():.3f}, {ellipse_pts[:,1].max():.3f}]")

    return square_pts, ellipse_pts, bnd_square, bnd_ellipse


# ====================================================================
# TRAINING
# ====================================================================
def train():
    print(f"\nDevice: {DEVICE}")
    print(f"Direction: ψ: □ → Ω  (square → ellipse)")
    square_pts, ellipse_pts, bnd_square, bnd_ellipse = generate_data()

    # NSF input = square, NSF target = ellipse
    input_t = torch.tensor(square_pts, dtype=torch.float32, device=DEVICE)
    target_t = torch.tensor(ellipse_pts, dtype=torch.float32, device=DEVICE)
    bnd_in_t = torch.tensor(bnd_square, dtype=torch.float32, device=DEVICE)
    bnd_tgt_t = torch.tensor(bnd_ellipse, dtype=torch.float32, device=DEVICE)

    print("\nCreating NeuralSplineFlow (□ → Ω) ...")
    model = NeuralSplineFlow(dim=2, **MODEL_CONFIG).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Domain (input):    {MODEL_CONFIG['domain_bounds']}")
    print(f"  Codomain (output): {MODEL_CONFIG['codomain_bounds']}")

    with torch.no_grad():
        test_out, _ = model(input_t[:10])
        print(f"  Initial output range: [{test_out.min():.3f}, {test_out.max():.3f}]")
        det0 = model.jacobian_det(input_t[:100])
        print(f"  Initial det(Dψ): [{det0.min():.4f}, {det0.max():.4f}]")

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["n_epochs"], eta_min=TRAIN_CONFIG["lr_min"],
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    losses_int, losses_bnd = [], []

    print(f"\nTraining for {TRAIN_CONFIG['n_epochs']} epochs ...")
    for epoch in range(TRAIN_CONFIG["n_epochs"]):
        optimizer.zero_grad()

        pred, logdet = model(input_t)
        mse_int = nn.functional.mse_loss(pred, target_t)

        pred_bnd, _ = model(bnd_in_t)
        mse_bnd = nn.functional.mse_loss(pred_bnd, bnd_tgt_t)

        logdet_reg = TRAIN_CONFIG["logdet_weight"] * logdet.pow(2).mean()

        loss = mse_int + TRAIN_CONFIG["boundary_weight"] * mse_bnd + logdet_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
        optimizer.step()
        scheduler.step()

        losses_int.append(mse_int.item())
        losses_bnd.append(mse_bnd.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

        if (epoch + 1) % TRAIN_CONFIG["print_every"] == 0:
            with torch.no_grad():
                det = model.jacobian_det(input_t[:200])
            print(f"  Epoch {epoch+1:5d}: MSE_int={mse_int.item():.6f}  "
                  f"MSE_bnd={mse_bnd.item():.6f}  "
                  f"det=[{det.min().item():.4f}, {det.max().item():.4f}]  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pt")

    config_out = {
        **MODEL_CONFIG,
        "ellipse_a": ELLIPSE_A,
        "ellipse_b": ELLIPSE_B,
        "bounds": BOUNDS,
        "n_params": n_params,
        "direction": "square_to_ellipse",
    }
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nCheckpoints saved to {CHECKPOINT_DIR}/")
    print(f"  best_model.pt   (loss={best_loss:.6f})")

    return (model, square_pts, ellipse_pts, bnd_square, bnd_ellipse,
            losses_int, losses_bnd)


# ====================================================================
# EVALUATION
# ====================================================================
def evaluate(model, square_pts, ellipse_pts, bnd_square, bnd_ellipse,
             losses_int, losses_bnd):

    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt",
                                     map_location=DEVICE))
    model.eval()

    in_t = torch.tensor(square_pts, dtype=torch.float32, device=DEVICE)
    bnd_in_t = torch.tensor(bnd_square, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred = model(in_t)[0].cpu().numpy()
        pred_bnd = model(bnd_in_t)[0].cpu().numpy()
        det_all = model.jacobian_det(in_t).cpu().numpy()

    mse_int = np.mean((pred - ellipse_pts) ** 2)
    mse_bnd = np.mean((pred_bnd - bnd_ellipse) ** 2)

    bnd_check = (pred_bnd[:, 0] / ELLIPSE_A)**2 + (pred_bnd[:, 1] / ELLIPSE_B)**2

    with torch.no_grad():
        fwd = model(in_t[:200])[0]
        back = model.inverse(fwd)[0]
        rt_err = (in_t[:200] - back).abs().mean().item()

    print("\n" + "=" * 60)
    print("EVALUATION (best model)")
    print("=" * 60)
    print(f"  Direction:          □ → Ω  (square → ellipse)")
    print(f"  MSE interior:       {mse_int:.6f}")
    print(f"  MSE boundary:       {mse_bnd:.6f}")
    print(f"  Boundary on ellipse: (x/a)²+(y/b)² = [{bnd_check.min():.4f}, {bnd_check.max():.4f}]")
    print(f"  det(Dψ) range:      [{det_all.min():.4f}, {det_all.max():.4f}]")
    print(f"  Positive det(Dψ):   {100 * (det_all > 0).mean():.1f}%")
    print(f"  Round-trip error:    {rt_err:.6f}")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    theta = np.linspace(0, 2 * np.pi, 200)

    ax = axes[0, 0]
    ax.scatter(square_pts[:, 0], square_pts[:, 1], s=3, alpha=0.3, c="blue")
    ax.scatter(bnd_square[:, 0], bnd_square[:, 1], s=15, c="red")
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color="green", lw=2))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_title("Input (square)"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(ellipse_pts[:, 0], ellipse_pts[:, 1], s=3, alpha=0.3, c="blue")
    ax.scatter(bnd_ellipse[:, 0], bnd_ellipse[:, 1], s=15, c="red")
    ax.plot(ELLIPSE_A * np.cos(theta), ELLIPSE_B * np.sin(theta), "k-", lw=2)
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_title("Target (ellipse)"); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(pred[:, 0], pred[:, 1], s=3, alpha=0.3, c="blue")
    ax.scatter(pred_bnd[:, 0], pred_bnd[:, 1], s=15, c="red")
    ax.plot(ELLIPSE_A * np.cos(theta), ELLIPSE_B * np.sin(theta), "k-", lw=2)
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_title(f"ψ(□) predicted (MSE={mse_int:.5f})"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(det_all, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.set_title(f"det(Dψ)  ({100*(det_all>0).mean():.0f}% positive)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    n_grid = 20
    with torch.no_grad():
        for v in np.linspace(-0.95, 0.95, n_grid):
            xs = np.linspace(-0.95, 0.95, 60).astype(np.float32)
            line = np.stack([xs, np.full_like(xs, v)], axis=1)
            m = model(torch.tensor(line, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            ax.plot(m[:, 0], m[:, 1], "b-", lw=0.4, alpha=0.7)
        for v in np.linspace(-0.95, 0.95, n_grid):
            ys = np.linspace(-0.95, 0.95, 60).astype(np.float32)
            line = np.stack([np.full_like(ys, v), ys], axis=1)
            m = model(torch.tensor(line, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            ax.plot(m[:, 0], m[:, 1], "b-", lw=0.4, alpha=0.7)
    ax.plot(ELLIPSE_A * np.cos(theta), ELLIPSE_B * np.sin(theta), "r-", lw=2)
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_title("Deformed grid: ψ(□) → Ω"); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    sc = ax.scatter(square_pts[:, 0], square_pts[:, 1],
                    c=det_all, cmap="viridis", s=5,
                    vmin=0, vmax=np.percentile(det_all, 95))
    plt.colorbar(sc, ax=ax, label="det(Dψ)")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.set_title("Jacobian field on □"); ax.grid(True, alpha=0.3)

    plt.suptitle(f"NSF  ψ: □ → Ω  (a={ELLIPSE_A}, b={ELLIPSE_B})", fontsize=14)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "nsf_square_to_ellipse.png", dpi=150)
    print(f"Saved {CHECKPOINT_DIR / 'nsf_square_to_ellipse.png'}")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(losses_int, label="interior MSE", alpha=0.8)
    ax.semilogy(losses_bnd, label="boundary MSE", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training curve (ψ: □ → Ω)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "training_curve.png", dpi=150)
    print(f"Saved {CHECKPOINT_DIR / 'training_curve.png'}")
    plt.show()


if __name__ == "__main__":
    model, *data_and_losses = train()
    evaluate(model, *data_and_losses)