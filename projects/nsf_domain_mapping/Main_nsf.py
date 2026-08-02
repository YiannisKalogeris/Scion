#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 22:09:45 2026

@author: ioannis
"""

"""
Main entry point for the Diffeomorphism Network project.

Direction:  ψ: □ → Ω   (reference square → physical domain)

Two-stage architecture:
    Stage 1: Auto-Decoder — jointly train flow + per-shape latent codes z_i
    Stage 2: Encoder      — freeze flow, train CNN to predict z from SDF image

At test time:  SDF image → Encoder → z → Flow → mapping  (single forward pass)

Pipeline:
    1. Data generation (harmonic maps as ground truth)
    2. Stage 1: Auto-decoder training
    3. Stage 2: Encoder training (supervised by stored codes)
    4. Evaluation on test shapes (encoder-based, no fitting)
    5. Visualization

Run:
    python Main_nsf.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from simulacrum.geometry.shapes import ShapeGenerator, SHAPE_REGISTRY

from dataset import DiffeomorphismDataset, generate_datasets
from models.neural_spline_flow import NeuralSplineFlow, NSFMapping


# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent

DATA_DIR = _SCRIPT_DIR / "data_fourierBlob"                 #"data_ellipse", "data_fourierBlob"
CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints_fourierBlob" #"checkpoints_ellipse", "checkpoints_fourierBlob"
OUTPUT_DIR = _SCRIPT_DIR / "outputs_fourierBlob"         #"outputs_ellipse", "outputs_fourierBlob"

DEVICE = 'cuda' if torch.cuda.is_available() else \
         'mps'  if torch.backends.mps.is_available() else 'cpu'

SEED = 42

# -----------------------------------------------------------------------------
# Data Generation Settings
# -----------------------------------------------------------------------------
DATA_CONFIG = {
    "n_train": 500,
    "n_val": 10,
    "n_test": 10,
    "image_size": 64,
    "n_interior_points": 1024,
    "n_boundary_points": 256,
    "harmonic_grid_size": 128,
    "shape_distribution": {"Ellipse": 0.0, "Superellipse": 0.0, "FourierBlob": 1.0, "SmoothPolygon": 0.0, "StarPolygon": 0.0},
}

# -----------------------------------------------------------------------------
# Model Architecture Settings
# Direction: □ → Ω  (domain = square, codomain = physical shape region)
# -----------------------------------------------------------------------------
MODEL_CONFIG = {
    "n_layers": 14,
    "n_bins": 20,
    "hidden_dim": 128,
    "n_hidden": 3,
    "domain_bounds": (-1.05, 1.05),
    "codomain_bounds": (-2.5, 2.5),
    "latent_dim": 6,
    "image_size": 64,
}

# -----------------------------------------------------------------------------
# Stage 1: Auto-Decoder Training
# -----------------------------------------------------------------------------
STAGE1_CONFIG = {
    "n_epochs": 8000,
    "batch_size": 50,
    "learning_rate_flow": 1e-3,
    "learning_rate_codes": 1e-2,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,

    "loss_weights": {
        "interior": 1.0,
        "boundary": 5.0,
        "logdet_reg": 0.001,
        "code_reg": 1e-3,
    },

    "checkpoint_every": 1000,
    "validate_every": 1000,
    "early_stopping_patience": 50,
}

# -----------------------------------------------------------------------------
# Stage 2: Encoder Training
# -----------------------------------------------------------------------------
STAGE2_CONFIG = {
    "n_epochs": 4000,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,

    # Loss: encoder prediction error + mapping quality
    "lambda_code": 0.1,       # ||Enc(I) - z_i||²
    "lambda_mapping": 0.5,    # mapping MSE using predicted z
    "lambda_boundary": 0.5,   # boundary mapping MSE

    "validate_every": 20,
    "early_stopping_patience": 200,
}

# Settings for test-time fitting (fallback, used in Stage 1 validation)
FIT_CONFIG = {
    "n_steps": 200,
    "lr": 1e-2,
    "boundary_weight": 5.0,
    "code_reg": 1e-3,
}

# -----------------------------------------------------------------------------
# Pipeline Control
# -----------------------------------------------------------------------------
PIPELINE = {
    "generate_data": False,
    "train_stage1": False,
    "train_stage2": False,
    "evaluate": False,
    "visualize": True,
    "n_visualizations": 5,
}


# =============================================================================
# AUTO-DECODER NSF MODEL  (Stage 1)
# =============================================================================

class AutoDecoderNSF(nn.Module):
    """
    NSF with per-shape learnable latent codes. Direction: □ → Ω.
    Used in Stage 1 to jointly learn the flow and latent codes.
    """

    def __init__(
        self,
        n_shapes: int,
        latent_dim: int = 32,
        n_layers: int = 8,
        n_bins: int = 16,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        domain_bounds: tuple = (-1.05, 1.05),
        codomain_bounds: tuple = (-2.5, 2.5),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_shapes = n_shapes

        self.latent_codes = nn.Embedding(n_shapes, latent_dim)
        nn.init.normal_(self.latent_codes.weight, 0.0, 0.01)

        self.flow = NeuralSplineFlow(
            dim=2,
            n_layers=n_layers,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            domain_bounds=domain_bounds,
            codomain_bounds=codomain_bounds,
            condition_dim=latent_dim,
        )

    def _get_z(self, shape_idx):
        return self.latent_codes(shape_idx)

    def forward(self, y, shape_idx):
        z = self._get_z(shape_idx)
        B, N, _ = y.shape
        y_flat = y.view(B * N, 2)
        z_exp = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        x, logdet = self.flow.forward(y_flat, z_exp)
        return x.view(B, N, 2), logdet.view(B, N)

    def forward_with_z(self, y, z):
        if y.dim() == 2:
            z_exp = z.unsqueeze(0).expand(y.shape[0], -1)
            return self.flow.forward(y, z_exp)
        else:
            B, N, _ = y.shape
            y_flat = y.view(B * N, 2)
            z_exp = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            x, logdet = self.flow.forward(y_flat, z_exp)
            return x.view(B, N, 2), logdet.view(B, N)

    def inverse_with_z(self, x, z):
        if x.dim() == 2:
            z_exp = z.unsqueeze(0).expand(x.shape[0], -1)
            return self.flow.inverse(x, z_exp)
        else:
            B, N, _ = x.shape
            x_flat = x.view(B * N, 2)
            z_exp = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            y, logdet = self.flow.inverse(x_flat, z_exp)
            return y.view(B, N, 2), logdet.view(B, N)


# =============================================================================
# CNN ENCODER  (Stage 2)
# =============================================================================

class SDFEncoder(nn.Module):
    """
    CNN encoder:  SDF image (H, W, 1) → latent code z ∈ R^d_z.

    Trained in Stage 2 to predict the auto-decoder's stored z_i from images.
    """

    def __init__(self, latent_dim: int = 32, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),    # 64→32
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32→16
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16→8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                      # 8→1×1
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, image):
        """
        Args:
            image: (B, H, W, 1) SDF image
        Returns:
            z: (B, latent_dim)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # (B, H, W, C) → (B, C, H, W)
        x = image.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.head(x)


# =============================================================================
# ENCODER + FLOW  (inference model)
# =============================================================================

class EncoderNSF(nn.Module):
    """
    Full inference model: SDF image → z → flow → physical coordinates.

    Combines a trained encoder with a trained (frozen) flow.
    """

    def __init__(self, encoder: SDFEncoder, flow: NeuralSplineFlow):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.latent_dim = encoder.latent_dim

    def encode(self, image):
        """Encode SDF image to latent code."""
        return self.encoder(image)

    def forward(self, y, image):
        """
        Forward map: □ → Ω conditioned on image.

        Args:
            y: (N, 2) or (B, N, 2) square points
            image: (H, W, 1) or (B, H, W, 1) SDF image
        Returns:
            x: physical coordinates
            logabsdet: log |det Dψ|
        """
        z = self.encode(image)  # (B, latent_dim) or (1, latent_dim)

        if y.dim() == 2:
            z = z.expand(y.shape[0], -1)
            return self.flow.forward(y, z)
        else:
            B, N, _ = y.shape
            y_flat = y.view(B * N, 2)
            z_exp = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            x, logdet = self.flow.forward(y_flat, z_exp)
            return x.view(B, N, 2), logdet.view(B, N)

    def forward_with_z(self, y, z):
        """Forward with explicit z (bypass encoder)."""
        if y.dim() == 2:
            z_exp = z.unsqueeze(0).expand(y.shape[0], -1)
            return self.flow.forward(y, z_exp)
        else:
            B, N, _ = y.shape
            y_flat = y.view(B * N, 2)
            z_exp = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            x, logdet = self.flow.forward(y_flat, z_exp)
            return x.view(B, N, 2), logdet.view(B, N)


# =============================================================================
# FEM MAPPING WRAPPER
# =============================================================================

class EncoderMapping:
    """
    Wraps EncoderNSF for use in FEM pipelines.

    Direction: ψ(y; image): □ → Ω
    At construction, the image is encoded once to get z.
    """

    def __init__(self, model: EncoderNSF, image: torch.Tensor):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        # Encode once and cache
        with torch.no_grad():
            self.z = model.encode(image.to(self.device)).squeeze(0)

    def psi(self, y):
        y = y.to(self.device)
        with torch.no_grad():
            x, _ = self.model.forward_with_z(y, self.z)
        return x

    def psi_inverse(self, x):
        x = x.to(self.device)
        z_exp = self.z.unsqueeze(0).expand(x.shape[0], -1)
        with torch.no_grad():
            y, _ = self.model.flow.inverse(x, z_exp)
        return y

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


# Also keep the auto-decoder mapping for Stage 1 validation
class AutoDecoderMapping:
    def __init__(self, model, z):
        self.model = model
        self.model.eval()
        self.z = z
        self.device = z.device

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


# =============================================================================
# TEST-TIME FITTING (fallback / Stage 1 validation)
# =============================================================================

def fit_new_shape(model, shape_data, config=None, device=None):
    if config is None:
        config = FIT_CONFIG
    if device is None:
        device = DEVICE

    model.eval()
    for p in model.flow.parameters():
        p.requires_grad_(False)

    z = torch.zeros(model.latent_dim, dtype=torch.float32,
                    device=device, requires_grad=True)
    opt = optim.Adam([z], lr=config["lr"])

    square_pts = torch.tensor(shape_data.interior_targets,
                              dtype=torch.float32, device=device)
    phys_pts = torch.tensor(shape_data.interior_points,
                            dtype=torch.float32, device=device)
    bnd_square = torch.tensor(shape_data.boundary_targets,
                              dtype=torch.float32, device=device)
    bnd_phys = torch.tensor(shape_data.boundary_points,
                            dtype=torch.float32, device=device)

    for step in range(config["n_steps"]):
        opt.zero_grad()
        pred, _ = model.forward_with_z(square_pts, z)
        pred_bnd, _ = model.forward_with_z(bnd_square, z)

        loss = (
            nn.functional.mse_loss(pred, phys_pts)
            + config["boundary_weight"] * nn.functional.mse_loss(pred_bnd, bnd_phys)
            + config["code_reg"] * z.pow(2).sum()
        )
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"    fit step {step+1}: loss={loss.item():.6f}")

    for p in model.flow.parameters():
        p.requires_grad_(True)

    return z.detach()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_directories():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CHECKPOINT_DIR)


def save_config(run_dir):
    config = {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG,
        "stage1": STAGE1_CONFIG,
        "stage2": STAGE2_CONFIG,
        "fit": FIT_CONFIG,
        "pipeline": PIPELINE,
        "seed": SEED,
        "device": DEVICE,
        "model_type": "TwoStage_AutoDecoder_Encoder",
        "direction": "square_to_shape",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def collate_samples(samples, device=None):
    batch = {
        'image': torch.stack([
            torch.tensor(s.image, dtype=torch.float32) for s in samples
        ], dim=0),
        'square_interior': torch.stack([
            torch.tensor(s.interior_targets, dtype=torch.float32) for s in samples
        ], dim=0),
        'square_boundary': torch.stack([
            torch.tensor(s.boundary_targets, dtype=torch.float32) for s in samples
        ], dim=0),
        'phys_interior': torch.stack([
            torch.tensor(s.interior_points, dtype=torch.float32) for s in samples
        ], dim=0),
        'phys_boundary': torch.stack([
            torch.tensor(s.boundary_points, dtype=torch.float32) for s in samples
        ], dim=0),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def plot_loss_history(history, stage_name, run_dir):
    """Plot the training loss curve for one stage and save to run_dir."""
    train = history.get("train_loss", [])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(train) + 1), train,
            "-", color="tab:blue", lw=1.5, label="train loss")

    ax.set_yscale("log")            # loss spans orders of magnitude
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{stage_name} training")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = run_dir / f"{stage_name}_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curve to {out}")

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_loss_stage1(model, batch, shape_indices, weights):
    """Stage 1 loss: auto-decoder."""
    sq_int = batch['square_interior']
    sq_bnd = batch['square_boundary']
    phys_int = batch['phys_interior']
    phys_bnd = batch['phys_boundary']

    pred_int, logdet = model.forward(sq_int, shape_indices)
    pred_bnd, _ = model.forward(sq_bnd, shape_indices)

    loss_int = nn.functional.mse_loss(pred_int, phys_int)
    loss_bnd = nn.functional.mse_loss(pred_bnd, phys_bnd)
    loss_logdet = logdet.pow(2).mean()
    z = model._get_z(shape_indices)
    loss_code = z.pow(2).mean()

    total = (
        weights["interior"] * loss_int
        + weights["boundary"] * loss_bnd
        + weights["logdet_reg"] * loss_logdet
        + weights["code_reg"] * loss_code
    )
    return total, {
        "interior": loss_int.item(),
        "boundary": loss_bnd.item(),
        "logdet_reg": loss_logdet.item(),
        "code_reg": loss_code.item(),
    }


# =============================================================================
# PIPELINE: DATA GENERATION
# =============================================================================

def run_data_generation():
    print("\n" + "=" * 60)
    print("STEP 1: DATA GENERATION")
    print("=" * 60)

    train_ds, val_ds, test_ds = generate_datasets(
        data_dir=DATA_DIR,
        n_train=DATA_CONFIG["n_train"],
        n_val=DATA_CONFIG["n_val"],
        n_test=DATA_CONFIG["n_test"],
        image_size=DATA_CONFIG["image_size"],
        n_interior_points=DATA_CONFIG["n_interior_points"],
        n_boundary_points=DATA_CONFIG["n_boundary_points"],
        harmonic_grid_size=DATA_CONFIG.get("harmonic_grid_size", 128),
        shape_distribution=DATA_CONFIG["shape_distribution"],
    )

    print(f"\nDatasets generated:")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# =============================================================================
# PIPELINE: STAGE 1 — AUTO-DECODER
# =============================================================================

def create_autodecoder(n_shapes):
    model = AutoDecoderNSF(
        n_shapes=n_shapes,
        latent_dim=MODEL_CONFIG["latent_dim"],
        n_layers=MODEL_CONFIG["n_layers"],
        n_bins=MODEL_CONFIG["n_bins"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        n_hidden=MODEL_CONFIG["n_hidden"],
        domain_bounds=MODEL_CONFIG["domain_bounds"],
        codomain_bounds=MODEL_CONFIG["codomain_bounds"],
    )
    return model.to(DEVICE)


def train_stage1(run_dir):
    print("\n" + "=" * 60)
    print("STAGE 1: AUTO-DECODER TRAINING  (ψ: □ → Ω)")
    print("=" * 60)

    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")
    val_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "val.pkl")
    n_train = len(train_dataset)
    cfg = STAGE1_CONFIG
    print(f"  Train: {n_train}, Val: {len(val_dataset)}")

    model = create_autodecoder(n_train)
    n_flow = sum(p.numel() for p in model.flow.parameters())
    n_codes = model.latent_codes.weight.numel()
    print(f"  Flow params:  {n_flow:,}")
    print(f"  Latent codes: {n_codes:,}  ({n_train} x {MODEL_CONFIG['latent_dim']})")

    optimizer = optim.AdamW([
        {"params": model.flow.parameters(),
         "lr": cfg["learning_rate_flow"],
         "weight_decay": cfg["weight_decay"]},
        {"params": model.latent_codes.parameters(),
         "lr": cfg["learning_rate_codes"],
         "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"],
        eta_min=cfg["learning_rate_flow"] * 0.01, #0.1 better for the ellipse
    )

    from torch.utils.data import Dataset, DataLoader

    class IndexedDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return idx, self.ds[idx]

    def indexed_collate(items):
        indices, samples = zip(*items)
        indices_t = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        batch = collate_samples(samples, DEVICE)
        return indices_t, batch

    train_loader = DataLoader(
        IndexedDataset(train_dataset),
        batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=indexed_collate,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n  Training for {cfg['n_epochs']} epochs ...")
    for epoch in range(cfg["n_epochs"]):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{cfg['n_epochs']}")
        for indices, batch in pbar:
            optimizer.zero_grad()
            loss, comp = compute_loss_stage1(model, batch, indices, cfg["loss_weights"])
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        if (epoch + 1) % cfg["validate_every"] == 0:
            model.eval()
            val_mses = []
            for i in range(min(len(val_dataset), 20)):
                z_fit = fit_new_shape(
                    model, val_dataset[i],
                    config={"n_steps": 100, "lr": 1e-2,
                            "boundary_weight": 5.0, "code_reg": 1e-3},
                    device=DEVICE,
                )
                sq = torch.tensor(val_dataset[i].interior_targets,
                                  dtype=torch.float32, device=DEVICE)
                phys = torch.tensor(val_dataset[i].interior_points,
                                    dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    pred, _ = model.forward_with_z(sq, z_fit)
                val_mses.append(nn.functional.mse_loss(pred, phys).item())

            avg_val = np.mean(val_mses)
            history["val_loss"].append(avg_val)
            print(f"\n  Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), run_dir / "stage1_best.pt")
                print(f"    -> Saved best Stage 1 model")
            else:
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % cfg["checkpoint_every"] == 0:
            torch.save(model.state_dict(), run_dir / f"stage1_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), run_dir / "stage1_final.pt")
    with open(run_dir / "stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    plot_loss_history(history, "stage1", run_dir)

    print(f"\n  Stage 1 complete. Best val loss: {best_val_loss:.6f}")
    return model, history


# =============================================================================
# PIPELINE: STAGE 2 — ENCODER TRAINING
# =============================================================================

def create_encoder():
    return SDFEncoder(
        latent_dim=MODEL_CONFIG["latent_dim"],
        image_size=MODEL_CONFIG["image_size"],
    ).to(DEVICE)


def train_stage2(run_dir):
    print("\n" + "=" * 60)
    print("STAGE 2: ENCODER TRAINING  (SDF image → z)")
    print("=" * 60)

    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")
    val_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "val.pkl")
    n_train = len(train_dataset)
    cfg = STAGE2_CONFIG

    # Load Stage 1 model (for flow weights and stored codes)
    ad_model = create_autodecoder(n_train)
    stage1_path = run_dir / "stage1_best.pt"
    if not stage1_path.exists():
        print(f"  ERROR: Stage 1 model not found at {stage1_path}")
        print(f"  Run Stage 1 first.")
        return None
    ad_model.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
    ad_model.eval()

    # Extract stored codes as targets
    with torch.no_grad():
        target_codes = ad_model.latent_codes.weight.clone()  # (N, latent_dim)
    print(f"  Target codes: {target_codes.shape}")
    print(f"  Code norm range: [{target_codes.norm(dim=1).min():.4f}, "
          f"{target_codes.norm(dim=1).max():.4f}]")

    # Freeze the flow — it's already trained
    flow = ad_model.flow
    for p in flow.parameters():
        p.requires_grad_(False)

    # Create encoder
    encoder = create_encoder()
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params: {n_enc_params:,}")

    # Combine into inference model
    enc_nsf = EncoderNSF(encoder, flow).to(DEVICE)

    optimizer = optim.AdamW(
        encoder.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"],
        eta_min=cfg["learning_rate"] * 0.01,  ##0.05 better for the ellipse
    )

    from torch.utils.data import Dataset, DataLoader

    class IndexedDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return idx, self.ds[idx]

    def indexed_collate(items):
        indices, samples = zip(*items)
        indices_t = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        batch = collate_samples(samples, DEVICE)
        return indices_t, batch

    train_loader = DataLoader(
        IndexedDataset(train_dataset),
        batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=indexed_collate,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n  Training encoder for {cfg['n_epochs']} epochs ...")
    for epoch in range(cfg["n_epochs"]):
        encoder.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"S2 Epoch {epoch+1}/{cfg['n_epochs']}")
        for indices, batch in pbar:
            optimizer.zero_grad()

            images = batch['image']          # (B, H, W, 1)
            sq_int = batch['square_interior']
            sq_bnd = batch['square_boundary']
            phys_int = batch['phys_interior']
            phys_bnd = batch['phys_boundary']

            # Predict z from image
            z_pred = encoder(images)         # (B, latent_dim)
            z_target = target_codes[indices] # (B, latent_dim)

            # Loss 1: code prediction error
            loss_code = nn.functional.mse_loss(z_pred, z_target)

            # Loss 2: mapping quality with predicted z
            B, N, _ = sq_int.shape
            y_flat = sq_int.view(B * N, 2)
            z_exp = z_pred.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            pred_int, _ = flow.forward(y_flat, z_exp)
            pred_int = pred_int.view(B, N, 2)
            loss_mapping = nn.functional.mse_loss(pred_int, phys_int)

            # Loss 3: boundary mapping with predicted z
            _, M, _ = sq_bnd.shape
            y_bnd_flat = sq_bnd.view(B * M, 2)
            z_bnd_exp = z_pred.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
            pred_bnd, _ = flow.forward(y_bnd_flat, z_bnd_exp)
            pred_bnd = pred_bnd.view(B, M, 2)
            loss_bnd = nn.functional.mse_loss(pred_bnd, phys_bnd)

            loss = (cfg["lambda_code"] * loss_code
                    + cfg["lambda_mapping"] * loss_mapping
                    + cfg["lambda_boundary"] * loss_bnd)

            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({
                "code": f"{loss_code.item():.4f}",
                "map": f"{loss_mapping.item():.4f}",
            })

        scheduler.step()
        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # Validation: encoder on val images (no fitting needed!)
        if (epoch + 1) % cfg["validate_every"] == 0:
            encoder.eval()
            val_mses = []

            with torch.no_grad():
                for i in range(len(val_dataset)):
                    sample = val_dataset[i]
                    img = torch.tensor(sample.image, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                    z_pred = encoder(img).squeeze(0)

                    sq = torch.tensor(sample.interior_targets,
                                      dtype=torch.float32, device=DEVICE)
                    phys = torch.tensor(sample.interior_points,
                                        dtype=torch.float32, device=DEVICE)

                    z_exp = z_pred.unsqueeze(0).expand(sq.shape[0], -1)
                    pred, _ = flow.forward(sq, z_exp)
                    val_mses.append(nn.functional.mse_loss(pred, phys).item())

            avg_val = np.mean(val_mses)
            history["val_loss"].append(avg_val)
            print(f"\n  Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(encoder.state_dict(), run_dir / "encoder_best.pt")
                print(f"    -> Saved best encoder")
            else:
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

    torch.save(encoder.state_dict(), run_dir / "encoder_final.pt")
    with open(run_dir / "stage2_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    plot_loss_history(history, "stage2", run_dir)

    print(f"\n  Stage 2 complete. Best val loss: {best_val_loss:.6f}")
    return enc_nsf, history


# =============================================================================
# PIPELINE: EVALUATION  (encoder-based, no fitting)
# =============================================================================

def load_inference_model(run_dir):
    """Load the full encoder + flow inference model."""
    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")
    n_train = len(train_dataset)

    # Reconstruct auto-decoder and load flow weights
    ad_model = create_autodecoder(n_train)
    ad_model.load_state_dict(torch.load(run_dir / "stage1_best.pt", map_location=DEVICE))
    flow = ad_model.flow

    # Load encoder
    encoder = create_encoder()
    encoder.load_state_dict(torch.load(run_dir / "encoder_best.pt", map_location=DEVICE))

    # Combine
    model = EncoderNSF(encoder, flow).to(DEVICE)
    model.eval()
    return model


def evaluate_model(run_dir):
    print("\n" + "=" * 60)
    print("EVALUATION  (encoder-based, no fitting)")
    print("=" * 60)

    model = load_inference_model(run_dir)
    test_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "test.pkl")
    print(f"  Test shapes: {len(test_dataset)}")

    metrics = {"mse_interior": [], "mse_boundary": [],
               "jacobian_min": [], "jacobian_max": []}

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[i]
            img = torch.tensor(sample.image, dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
            z = model.encode(img).squeeze(0)

            sq = torch.tensor(sample.interior_targets,
                              dtype=torch.float32, device=DEVICE)
            phys = torch.tensor(sample.interior_points,
                                dtype=torch.float32, device=DEVICE)
            bnd_sq = torch.tensor(sample.boundary_targets,
                                  dtype=torch.float32, device=DEVICE)
            bnd_phys = torch.tensor(sample.boundary_points,
                                    dtype=torch.float32, device=DEVICE)

            pred, _ = model.forward_with_z(sq, z)
            pred_bnd, _ = model.forward_with_z(bnd_sq, z)

            metrics["mse_interior"].append(nn.functional.mse_loss(pred, phys).item())
            metrics["mse_boundary"].append(nn.functional.mse_loss(pred_bnd, bnd_phys).item())

            _, logdet = model.forward_with_z(sq[:50], z)
            det = torch.exp(logdet).cpu().numpy()
            metrics["jacobian_min"].append(det.min())
            metrics["jacobian_max"].append(det.max())

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["mse_interior_std"] = float(np.std(metrics["mse_interior"]))

    print("\n" + "-" * 50)
    print("RESULTS (Encoder-based, single forward pass)")
    print("-" * 50)
    print(f"  MSE interior: {summary['mse_interior']:.6f} +/- {summary['mse_interior_std']:.6f}")
    print(f"  MSE boundary: {summary['mse_boundary']:.6f}")
    print(f"  det(Dψ) range: [{summary['jacobian_min']:.3f}, {summary['jacobian_max']:.3f}]")
    print("-" * 50)

    with open(run_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
# PIPELINE: VISUALIZATION  (encoder-based)
# =============================================================================

def generate_visualizations(run_dir):
    print("\n" + "=" * 60)
    print("VISUALIZATION  (encoder-based)")
    print("=" * 60)

    model = load_inference_model(run_dir)
    test_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "test.pkl")
    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")

    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    n_samples = min(PIPELINE["n_visualizations"], len(test_dataset))

    for i in range(n_samples):
        sample = test_dataset[i]
        print(f"\n  Shape {i+1}/{n_samples} ...")

        # Encode image → z  (single forward pass, no fitting!)
        img = torch.tensor(sample.image, dtype=torch.float32,
                           device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            z = model.encode(img).squeeze(0)

        sq_pts = sample.interior_targets
        phys_pts = sample.interior_points
        bnd_sq = sample.boundary_targets
        bnd_phys = sample.boundary_points

        sq_t = torch.tensor(sq_pts, dtype=torch.float32, device=DEVICE)
        bnd_sq_t = torch.tensor(bnd_sq, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred = model.forward_with_z(sq_t, z)[0].cpu().numpy()
            pred_bnd = model.forward_with_z(bnd_sq_t, z)[0].cpu().numpy()

        mse = np.mean((pred - phys_pts) ** 2)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        ax = axes[0, 0]
        ax.scatter(sq_pts[:, 0], sq_pts[:, 1], s=5, alpha=0.5, c='blue')
        ax.scatter(bnd_sq[:, 0], bnd_sq[:, 1], s=15, c='red')
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color='green', lw=2))
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.set_title('Input (square)')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.imshow(sample.image[:, :, 0], extent=[-2, 2, -2, 2],
                  origin='lower', cmap='RdBu', alpha=0.7)
        ax.scatter(phys_pts[:, 0], phys_pts[:, 1], s=5, alpha=0.5, c='blue')
        ax.scatter(bnd_phys[:, 0], bnd_phys[:, 1], s=15, c='red')
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal'); ax.set_title('Target (shape)')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.scatter(pred[:, 0], pred[:, 1], s=5, alpha=0.5, c='blue')
        ax.scatter(pred_bnd[:, 0], pred_bnd[:, 1], s=15, c='red')
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'Predicted (MSE={mse:.5f})')
        ax.grid(True, alpha=0.3)

        # Deformed grid
        ax = axes[1, 0]
        n_grid = 15
        with torch.no_grad():
            for v in np.linspace(-0.95, 0.95, n_grid):
                xs = np.linspace(-0.95, 0.95, 50).astype(np.float32)
                line = np.stack([xs, np.full_like(xs, v)], axis=1)
                lt = torch.tensor(line, dtype=torch.float32, device=DEVICE)
                m = model.forward_with_z(lt, z)[0].cpu().numpy()
                ax.plot(m[:, 0], m[:, 1], 'b-', lw=0.5, alpha=0.7)
            for v in np.linspace(-0.95, 0.95, n_grid):
                ys = np.linspace(-0.95, 0.95, 50).astype(np.float32)
                line = np.stack([np.full_like(ys, v), ys], axis=1)
                lt = torch.tensor(line, dtype=torch.float32, device=DEVICE)
                m = model.forward_with_z(lt, z)[0].cpu().numpy()
                ax.plot(m[:, 0], m[:, 1], 'b-', lw=0.5, alpha=0.7)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal'); ax.set_title('Deformed grid')
        ax.grid(True, alpha=0.3)

        # Jacobian field and histogram
        with torch.no_grad():
            _, logdet = model.forward_with_z(sq_t, z)
            det = torch.exp(logdet).cpu().numpy()

        ax = axes[1, 1]
        sc = ax.scatter(sq_pts[:, 0], sq_pts[:, 1], c=det, cmap='viridis', s=10,
                        vmin=0, vmax=np.percentile(det, 95))
        plt.colorbar(sc, ax=ax, label='det(Dpsi)')
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal'); ax.set_title('det(Dpsi) on square')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.hist(det, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', ls='--', lw=2)
        ax.set_title('det(Dpsi) ({:.0f}% positive)'.format(100 * (det > 0).mean()))
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Encoder NSF | test shape {i+1} | MSE={mse:.5f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(viz_dir / f"mapping_{i}.png", dpi=150)
        plt.show()
        plt.close()

    print(f"\n  Saved {n_samples} visualizations to {viz_dir}")

def visualize_mapping_quality(run_dir, n_shapes=5, source="train"):
    """
    Plot N shapes: ground truth boundary vs NSF-predicted boundary side by side.
    
    For each shape, draws:
      - Left: ground truth physical points and boundary
      - Right: NSF-mapped points from square coordinates
    """
    print("\n" + "=" * 60)
    print(f"MAPPING QUALITY CHECK ({source} set, {n_shapes} shapes)")
    print("=" * 60)

    train_dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / "train.pkl")
    dataset = DiffeomorphismDataset.load(Path(DATA_DIR) / f"{source}.pkl")

    # Load model depending on what's available
    if (run_dir / "encoder_best_REMOVE.pt").exists():   #if i dont use this, it visualizes using only the stored codes, use "encoder_best.pt"
        # Two-stage: use encoder
        model = load_inference_model(run_dir)
        use_encoder = True
        print("  Using encoder for z prediction")
    else:
        # Stage 1 only: use stored codes
        model_ad = create_autodecoder(len(train_dataset))
        model_ad.load_state_dict(torch.load(run_dir / "stage1_best.pt", map_location=DEVICE))
        model_ad.eval()
        use_encoder = False
        print("  Using stored auto-decoder codes")

    n_shapes = min(n_shapes, len(dataset))
    fig, axes = plt.subplots(2, n_shapes, figsize=(4 * n_shapes, 8))
    if n_shapes == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_shapes):
        sample = dataset[i]

        # Get z
        if use_encoder:
            img = torch.tensor(sample.image, dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                z = model.encode(img).squeeze(0)
            flow_forward = model.forward_with_z
        else:
            if source == "train":
                with torch.no_grad():
                    z = model_ad.latent_codes(torch.tensor(i, device=DEVICE))
            else:
                z = fit_new_shape(model_ad, sample, device=DEVICE)
            flow_forward = model_ad.forward_with_z

        # Ground truth
        phys_pts = sample.interior_points
        phys_bnd = sample.boundary_points
        sq_pts = sample.interior_targets
        sq_bnd = sample.boundary_targets

        # NSF prediction
        sq_t = torch.tensor(sq_pts, dtype=torch.float32, device=DEVICE)
        bnd_t = torch.tensor(sq_bnd, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_pts = flow_forward(sq_t, z)[0].cpu().numpy()
            pred_bnd = flow_forward(bnd_t, z)[0].cpu().numpy()

        mse = np.mean((pred_pts - phys_pts) ** 2)

        # Top row: ground truth shape
        ax = axes[0, i]
        ax.scatter(phys_pts[:, 0], phys_pts[:, 1], s=1, alpha=0.3, c='blue')
        ax.plot(np.append(phys_bnd[:, 0], phys_bnd[0, 0]),
                np.append(phys_bnd[:, 1], phys_bnd[0, 1]),
                'k-', lw=2)
        ax.set_aspect('equal')
        ax.set_title('Ground truth' if i == 0 else '')
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.set_ylabel('Target (physical)')

        # Bottom row: NSF prediction
        ax = axes[1, i]
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=1, alpha=0.3, c='red')
        ax.plot(np.append(pred_bnd[:, 0], pred_bnd[0, 0]),
                np.append(pred_bnd[:, 1], pred_bnd[0, 1]),
                'k-', lw=2)
        # Also overlay ground truth boundary as dashed
        ax.plot(np.append(phys_bnd[:, 0], phys_bnd[0, 0]),
                np.append(phys_bnd[:, 1], phys_bnd[0, 1]),
                'g--', lw=1.5, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_title('MSE = {:.2e}'.format(mse))
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.set_ylabel('NSF prediction')

    plt.suptitle('Mapping quality: ground truth (top) vs NSF (bottom)\n'
                 'Green dashed = true boundary overlaid on prediction',
                 fontsize=13)
    plt.tight_layout()

    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / "mapping_quality_{}.png".format(source), dpi=200, bbox_inches="tight")
    plt.show()
    print("  Saved mapping_quality_{}.png".format(source))

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("TWO-STAGE NSF  ψ: □ → Ω  (PyTorch)")
    print("=" * 60)
    print("\n  Stage 1: Auto-decoder (flow + per-shape codes)")
    print("  Stage 2: CNN encoder (image → z, supervised by stored codes)")
    print("  Inference: image → encoder → z → flow → mapping")
    print("  Jacobians: guaranteed positive by NSF construction")

    print(f"\n  Data dir:       {DATA_DIR}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Device:         {DEVICE}")
    print(f"  Latent dim:     {MODEL_CONFIG['latent_dim']}")
    print(f"  Domain:         {MODEL_CONFIG['domain_bounds']} -> {MODEL_CONFIG['codomain_bounds']}")

    print(f"\nPipeline:")
    for step, enabled in PIPELINE.items():
        print(f"  [{'Y' if enabled else 'N'}] {step}")

    set_seed(SEED)
    run_dir = setup_directories()
    save_config(run_dir)

    if PIPELINE["generate_data"]:
        run_data_generation()

    if PIPELINE["train_stage1"]:
        train_stage1(run_dir)

    if PIPELINE["train_stage2"]:
        train_stage2(run_dir)

    if PIPELINE["evaluate"]:
        evaluate_model(run_dir)

    if PIPELINE["visualize"]:
        generate_visualizations(run_dir)
        visualize_mapping_quality(run_dir, n_shapes=5, source="train")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
To use the trained model for FEM on a new shape:

    from Main_nsf import load_inference_model, EncoderMapping

    model = load_inference_model(Path("checkpoints_nsf"))
    image = ...  # SDF image (H, W, 1) as torch.Tensor

    mapping = EncoderMapping(model, image)
    x_phys = mapping.psi(y_square)           # □ → Ω
    F = mapping.jacobian_matrix(y_square)     # 2×2 Jacobian
    """)


if __name__ == "__main__":
    main()