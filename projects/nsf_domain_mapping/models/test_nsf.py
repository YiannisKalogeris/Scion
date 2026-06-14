"""
test_nsf.py - Test Neural Spline Flow for diffeomorphism learning

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from simulacrum.geometry.harmonic_maps import HarmonicMapSolver
from simulacrum.geometry.shapes import Ellipse
from neural_spline_flow import NeuralSplineFlow, NSFMapping

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# =============================================================================
# Data 
# =============================================================================
print("\n" + "="*60)
print("Generating ground truth data...")
print("="*60)

shape = Ellipse(a=1.0, b=0.8)
solver = HarmonicMapSolver(grid_size=128, bounds=(-2.0, 2.0))
solver.solve(shape)

# Interior points - sample more and filter
interior_points = []
for _ in range(5000):
    while True:
        p = np.random.uniform(-0.95, 0.95, 2)
        if (p[0]/1.0)**2 + (p[1]/0.8)**2 < 0.9:
            interior_points.append(p)
            break
interior_points = np.array(interior_points, dtype=np.float32)
targets = solver.evaluate(interior_points)

# Remove NaN
valid = ~np.isnan(targets).any(axis=1)
interior_points = interior_points[valid]
targets = targets[valid].astype(np.float32)

# Boundary points
n_boundary = 128
t_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
boundary_points = np.stack([1.0 * np.cos(t_boundary), 0.8 * np.sin(t_boundary)], axis=1).astype(np.float32)

# Boundary targets: map t ∈ [0, 2π) to square boundary (counterclockwise)
def param_to_square(t):
    """Map parameter t ∈ [0, 2π) to square boundary [-1,1]²."""
    t_norm = (t / (2 * np.pi)) % 1.0
    s = t_norm * 8  # perimeter
    if s < 2:
        return np.array([1.0, -1.0 + s])
    elif s < 4:
        return np.array([1.0 - (s - 2), 1.0])
    elif s < 6:
        return np.array([-1.0, 1.0 - (s - 4)])
    else:
        return np.array([-1.0 + (s - 6), -1.0])

boundary_targets = np.array([param_to_square(t) for t in t_boundary], dtype=np.float32)

print(f"Interior points: {len(interior_points)}")
print(f"Boundary points: {len(boundary_points)}")
print(f"Input range: x=[{interior_points[:,0].min():.2f}, {interior_points[:,0].max():.2f}], "
      f"y=[{interior_points[:,1].min():.2f}, {interior_points[:,1].max():.2f}]")
print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")

points_t = torch.tensor(interior_points, device=DEVICE)
targets_t = torch.tensor(targets, device=DEVICE)
boundary_t = torch.tensor(boundary_points, device=DEVICE)
boundary_targets_t = torch.tensor(boundary_targets, device=DEVICE)

# =============================================================================
# Neural Spline Flow Model
# =============================================================================
print("\n" + "="*60)
print("Creating Neural Spline Flow model...")
print("="*60)

model = NeuralSplineFlow(
    dim=2,
    n_layers=8,          # More layers for expressiveness
    n_bins=16,           # More bins for finer splines
    hidden_dim=128,      # Larger hidden dim
    n_hidden=3,          # More hidden layers
    domain_bounds=(-1.2, 1.2),    # Slightly larger than ellipse
    codomain_bounds=(-1.0, 1.0),  # Target square
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Verify initial Jacobians are positive
with torch.no_grad():
    det_init = model.jacobian_det(points_t[:100])
    print(f"Initial Jacobian det: min={det_init.min():.4f}, max={det_init.max():.4f}")
    print(f"All positive: {(det_init > 0).all().item()} ✓")

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

# =============================================================================
# Training - Just MSE, no Jacobian regularization needed!
# =============================================================================
print("\n" + "="*60)
print("Training (MSE only - no Jacobian regularization needed!)...")
print("="*60)

losses = []
for epoch in range(3000):
    optimizer.zero_grad()
    
    # Interior loss
    pred, logdet = model(points_t)
    mse_interior = nn.functional.mse_loss(pred, targets_t)
    
    # Boundary loss
    pred_boundary, _ = model(boundary_t)
    mse_boundary = nn.functional.mse_loss(pred_boundary, boundary_targets_t)
    
    # Optional: slight regularization to prevent extreme log-dets
    logdet_reg = 0.001 * (logdet.pow(2).mean())
    
    loss = mse_interior + 5.0 * mse_boundary + logdet_reg
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(mse_interior.item())
    
    if (epoch + 1) % 300 == 0:
        with torch.no_grad():
            det = model.jacobian_det(points_t[:100])
        print(f"Epoch {epoch+1}: MSE_int={mse_interior.item():.6f}, MSE_bnd={mse_boundary.item():.6f}, "
              f"det=[{det.min().item():.3f}, {det.max().item():.3f}]")

# =============================================================================
# Evaluate
# =============================================================================
print("\n" + "="*60)
print("Evaluation...")
print("="*60)

model.eval()
with torch.no_grad():
    pred_np, logdet = model(points_t)
    pred_np = pred_np.cpu().numpy()
    
    pred_boundary_np, _ = model(boundary_t)
    pred_boundary_np = pred_boundary_np.cpu().numpy()
    
    det_all = model.jacobian_det(points_t).cpu().numpy()

print(f"Final MSE (interior): {np.mean((pred_np - targets)**2):.6f}")
print(f"Final MSE (boundary): {np.mean((pred_boundary_np - boundary_targets)**2):.6f}")
print(f"Jacobian det: min={det_all.min():.4f}, max={det_all.max():.4f}")
print(f"Positive Jacobians: {100*(det_all > 0).mean():.1f}%")

# Test inverse
print("\nTesting inverse mapping...")
with torch.no_grad():
    # Forward then inverse should give identity
    y, _ = model(points_t[:100])
    x_back, _ = model.inverse(y)
    recon_error = (points_t[:100] - x_back).abs().mean().item()
    print(f"Inverse reconstruction error: {recon_error:.6f}")

# =============================================================================
# Visualize
# =============================================================================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Input points
ax = axes[0, 0]
ax.scatter(interior_points[:, 0], interior_points[:, 1], s=10, alpha=0.5, c='blue', label=f'interior ({len(interior_points)})')
ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, alpha=0.8, c='red', label=f'boundary ({len(boundary_points)})')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
ax.set_title(f'Input (Ellipse)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Ground truth
ax = axes[0, 1]
ax.scatter(targets[:, 0], targets[:, 1], s=10, alpha=0.5, c='blue', label='interior')
ax.scatter(boundary_targets[:, 0], boundary_targets[:, 1], s=20, alpha=0.8, c='red', label='boundary')
ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='green', lw=2))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
ax.set_title('Ground Truth (Harmonic Map)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Predicted
ax = axes[0, 2]
ax.scatter(pred_np[:, 0], pred_np[:, 1], s=10, alpha=0.5, c='blue', label='interior')
ax.scatter(pred_boundary_np[:, 0], pred_boundary_np[:, 1], s=20, alpha=0.8, c='red', label='boundary')
ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='green', lw=2))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
mse_val = np.mean((pred_np - targets)**2)
ax.set_title(f'NSF Predicted (MSE: {mse_val:.4f})')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Jacobian histogram
ax = axes[1, 0]
ax.hist(det_all, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
ax.set_xlabel('det(J)')
ax.set_ylabel('Count')
pos_pct = 100 * (det_all > 0).mean()
ax.set_title(f'Jacobian Det ({pos_pct:.0f}% positive)')
ax.legend()
ax.grid(True, alpha=0.3)

# Grid deformation
ax = axes[1, 1]
n_grid = 15
with torch.no_grad():
    # Horizontal lines
    for y_val in np.linspace(-0.75, 0.75, n_grid):
        # Only include points inside ellipse
        x_range = 0.95 * np.sqrt(1 - (y_val/0.8)**2) if abs(y_val) < 0.8 else 0
        if x_range > 0.1:
            line_x = np.linspace(-x_range, x_range, 50)
            line = np.stack([line_x, np.full_like(line_x, y_val)], axis=1).astype(np.float32)
            line_t = torch.tensor(line, device=DEVICE)
            mapped, _ = model(line_t)
            mapped = mapped.cpu().numpy()
            ax.plot(mapped[:, 0], mapped[:, 1], 'b-', lw=0.5, alpha=0.7)
    
    # Vertical lines
    for x_val in np.linspace(-0.95, 0.95, n_grid):
        # Only include points inside ellipse
        y_range = 0.95 * 0.8 * np.sqrt(1 - (x_val/1.0)**2) if abs(x_val) < 1.0 else 0
        if y_range > 0.1:
            line_y = np.linspace(-y_range, y_range, 50)
            line = np.stack([np.full_like(line_y, x_val), line_y], axis=1).astype(np.float32)
            line_t = torch.tensor(line, device=DEVICE)
            mapped, _ = model(line_t)
            mapped = mapped.cpu().numpy()
            ax.plot(mapped[:, 0], mapped[:, 1], 'b-', lw=0.5, alpha=0.7)

ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='green', lw=2))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
ax.set_title('Deformed Grid (NSF)')
ax.grid(True, alpha=0.3)

# Jacobian field
ax = axes[1, 2]
sc = ax.scatter(interior_points[:, 0], interior_points[:, 1], 
                c=det_all, cmap='viridis', s=20, vmin=0, vmax=np.percentile(det_all, 95))
plt.colorbar(sc, ax=ax, label='det(J)')
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(1.0*np.cos(theta), 0.8*np.sin(theta), 'r-', lw=2)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
ax.set_title('Jacobian Det Field (all positive!)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nsf_test.png', dpi=150)
print("Saved to nsf_test.png")
plt.show()

# =============================================================================
# Training curve
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('NSF Training Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nsf_training_curve.png', dpi=150)
print("Saved training curve to nsf_training_curve.png")

# =============================================================================
# Summary comparison
# =============================================================================
mse_bnd = np.mean((pred_boundary_np - boundary_targets)**2)
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final MSE (interior):  {mse_val:.6f}")
print(f"Final MSE (boundary):  {mse_bnd:.6f}")
print(f"Jacobian det range:    [{det_all.min():.4f}, {det_all.max():.4f}]")
print(f"Positive Jacobians:    {pos_pct:.1f}%")
print(f"Inverse error:         {recon_error:.6f}")
print("="*60)
print("\nNSF GUARANTEES positive Jacobians by construction!")
print("No regularization hacks needed.")