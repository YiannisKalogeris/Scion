#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sdf_representation.py

Generates the four-panel figure for the "Input representation via Signed
Distance Functions" section:

    (a) physical domain Omega with sampled interior / boundary points
    (b) binary occupancy mask  (for the "richer than a mask" contrast)
    (c) raw signed distance field  phi
    (d) normalised network input   phi_norm = tanh(phi / delta)

It reuses the project's own geometry code, so the rendered field, sign
convention and tanh scale are identical to the training pipeline:
    simulacrum.geometry.shapes  -> shape classes
    simulacrum.geometry.sdf     -> SDF field, normalisation, interior sampling

Run:
    python plot_sdf_representation.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from simulacrum.geometry.shapes import FourierBlob, Ellipse, Superellipse
from simulacrum.geometry.sdf import (
    compute_sdf_from_boundary,
    normalize_sdf,
    render_shape_to_binary,
    sample_points_in_shape,
)

# ----------------------------------------------------------------------------- 
# Configuration
# ----------------------------------------------------------------------------- 
SEED          = 7
GRID_SIZE     = 64                  # resolution of the encoder input image
BOUNDS        = (-2.0, 2.0)         # bounding box for the regular grid
N_BOUNDARY    = 600                 # boundary samples used to build the SDF
N_INTERIOR_PT = 220                 # interior points shown in panel (a)
N_BOUNDARY_PT = 40                  # boundary points shown in panel (a)
OUT_STEM      = "fig_sdf_representation"

# Representative domain (convex Fourier blob). Swap for Ellipse(...) etc.
SHAPE = FourierBlob(center=(0.0, 0.0), r0=1.0,
                    coeffs_cos=[0.18, -0.08], coeffs_sin=[0.12, 0.05])

# ----------------------------------------------------------------------------- 
# Matplotlib style (paper-friendly)
# ----------------------------------------------------------------------------- 
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "mathtext.fontset": "cm",
    "axes.linewidth":   0.8,
    "savefig.dpi":      300,
})

C_INSIDE   = "#2c6fbb"   # interior point colour
C_BOUNDARY = "#c0392b"   # boundary point colour
C_FILL     = "#dce6f2"   # domain fill


def main():
    rng = np.random.default_rng(SEED)

    # --- geometry -----------------------------------------------------------
    boundary_dense = SHAPE.sample(N_BOUNDARY)                 # for SDF + curve
    sdf, x_grid, y_grid = compute_sdf_from_boundary(
        boundary_dense, grid_size=GRID_SIZE, bounds=BOUNDS)
    sdf_norm = normalize_sdf(sdf, method="tanh")             # tanh(phi/delta)
    mask = render_shape_to_binary(
        SHAPE, grid_size=GRID_SIZE, bounds=BOUNDS, n_boundary_samples=N_BOUNDARY)

    # report the delta actually used (95th percentile of |phi|)
    delta = float(np.percentile(np.abs(sdf), 95))

    # sampled points for panel (a)
    interior_pts = sample_points_in_shape(SHAPE, N_INTERIOR_PT, BOUNDS, rng)
    boundary_pts = SHAPE.sample(N_BOUNDARY_PT)
    curve = SHAPE.sample(400)

    X, Y = np.meshgrid(x_grid, y_grid)
    extent = [BOUNDS[0], BOUNDS[1], BOUNDS[0], BOUNDS[1]]

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), constrained_layout=True)

    # (a) domain + sampled points
    ax = axes[0]
    ax.fill(curve[:, 0], curve[:, 1], color=C_FILL, zorder=0)
    ax.plot(np.append(curve[:, 0], curve[0, 0]),
            np.append(curve[:, 1], curve[0, 1]),
            color="k", lw=1.2, zorder=3)
    ax.scatter(interior_pts[:, 0], interior_pts[:, 1], s=5,
               color=C_INSIDE, alpha=0.8, zorder=2,
               label=r"interior ($\phi<0$)")
    ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], s=16,
               color=C_BOUNDARY, zorder=4, marker="o",
               edgecolors="k", linewidths=0.3, label=r"boundary")
    ax.set_title(r"(a) Domain $\Omega$ and samples")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, handletextpad=0.2)

    # (b) binary mask
    ax = axes[1]
    ax.imshow(mask[:, :, 0] if mask.ndim == 3 else mask,
              extent=extent, origin="lower", cmap="gray_r",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(r"(b) Binary mask")

    # (c) signed distance field
    ax = axes[2]
    norm = TwoSlopeNorm(vmin=float(np.nanmin(sdf)), vcenter=0.0,
                        vmax=float(np.nanmax(sdf)))
    im = ax.imshow(sdf, extent=extent, origin="lower", cmap="RdBu_r",
                   norm=norm, interpolation="bilinear")
    ax.contour(X, Y, sdf, levels=8, colors="k", linewidths=0.3, alpha=0.4)
    ax.contour(X, Y, sdf, levels=[0.0], colors="k", linewidths=1.4)  # boundary
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\phi$", rotation=0, labelpad=8)
    ax.set_title(r"(c) Signed distance $\phi$")

    # (d) normalised input
    ax = axes[3]
    im = ax.imshow(sdf_norm, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.contour(X, Y, sdf_norm, levels=[0.0], colors="k", linewidths=1.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\phi_{\mathrm{norm}}$", rotation=90, labelpad=4)
    ax.set_title(r"(d) Input $\tanh(\phi/\delta)$, $%d\times%d$" % (GRID_SIZE, GRID_SIZE))

    for ax in axes:
        ax.set_xlim(BOUNDS); ax.set_ylim(BOUNDS)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight", dpi=200)
    print(f"delta (95th pct of |phi|) = {delta:.4f}")
    print(f"saved {OUT_STEM}.pdf and {OUT_STEM}.png")


if __name__ == "__main__":
    main()
