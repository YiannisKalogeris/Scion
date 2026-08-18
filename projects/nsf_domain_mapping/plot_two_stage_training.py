#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_two_stage_training.py

Schematic of the two-stage multi-shape training procedure.

    Stage 1 (auto-decoder): training shapes + a learnable per-shape code table
        {z_i} are fed to the shared NSF G_theta(.;z_i); flow and codes are
        optimised jointly against the harmonic-map ground truth -> {z_i^*}.
    Stage 2 (encoder): the SDF images are mapped by a CNN encoder to predicted
        codes, trained to match the stored {z_i^*} with the flow frozen.

The code table {z_i^*} is the hinge: produced in Stage 1, used as the
regression target in Stage 2.

Uses real shape / SDF thumbnails from scion.geometry.

Run:  python plot_two_stage_training.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from scion.geometry.shapes import Ellipse, Superellipse, FourierBlob
from scion.geometry.sdf import compute_sdf_from_boundary, normalize_sdf

# ----------------------------------------------------------------------------
# Style / palette
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
})

C_TRAIN = "#dce6f2"   # trainable component
C_TRAIN_E = "#2c6fbb"
C_FROZEN = "#e9e9e9"   # frozen component
C_FROZEN_E = "#7f7f7f"
C_LOSS = "#f6dcdc"    # loss
C_LOSS_E = "#b03030"
C_DATA = "#ffffff"    # data / io
C_DATA_E = "#444444"

FIGW, FIGH = 13.0, 7.6
SQ = 0.82             # thumbnail side, inches
WF, HF = SQ / FIGW, SQ / FIGH

# ----------------------------------------------------------------------------
# Geometry thumbnails (real shapes)
# ----------------------------------------------------------------------------
SHAPES = [
    Ellipse(center=(0, 0), a=1.5, b=0.9, theta=0.3),
    FourierBlob(center=(0, 0), r0=1.0, coeffs_cos=[0.18, -0.08], coeffs_sin=[0.12, 0.05]),
    Superellipse(center=(0, 0), a=1.1, b=1.1, n=3.5, theta=0.2),
]
BOUNDS = (-2.0, 2.0)
_curves = [s.sample(400) for s in SHAPES]
_sdfs = []
for s in SHAPES:
    _sdf, _, _ = compute_sdf_from_boundary(s.sample(500), grid_size=64, bounds=BOUNDS)
    _sdfs.append(normalize_sdf(_sdf, "tanh"))


def _new_ax(fig, cx, cy, wf=WF, hf=HF, z=5):
    a = fig.add_axes([cx - wf / 2, cy - hf / 2, wf, hf], zorder=z)
    a.set_xticks([]); a.set_yticks([])
    return a


def shape_thumb(fig, cx, cy, idx, z=5, ec=C_DATA_E):
    a = _new_ax(fig, cx, cy, z=z)
    c = _curves[idx]
    a.fill(c[:, 0], c[:, 1], color=C_TRAIN, zorder=1)
    a.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), color="k", lw=1.0)
    a.set_xlim(BOUNDS); a.set_ylim(BOUNDS); a.set_aspect("equal")
    for sp in a.spines.values():
        sp.set_edgecolor(ec); sp.set_linewidth(0.8)


def sdf_thumb(fig, cx, cy, idx, z=5, ec=C_DATA_E):
    a = _new_ax(fig, cx, cy, z=z)
    a.imshow(_sdfs[idx], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    for sp in a.spines.values():
        sp.set_edgecolor(ec); sp.set_linewidth(0.8)


def deck(fig, cx, cy, kind):
    """Overlapping 'deck' of thumbnails to suggest a dataset of shapes."""
    offs = [(0.020, 0.044), (0.010, 0.022), (0.0, 0.0)]
    for k, (dx, dy) in enumerate(offs):
        idx = k % len(SHAPES)
        if kind == "shape":
            shape_thumb(fig, cx + dx, cy + dy, idx, z=3 + k)
        else:
            sdf_thumb(fig, cx + dx, cy + dy, idx, z=3 + k)


def box(ax, cx, cy, w, h, text, fc, ec, fs=9, lw=1.3, tc="black"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.012",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5)


def arrow(ax, p0, p1, color="k", lw=1.4, ls="-", rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                 lw=lw, color=color, linestyle=ls, zorder=3,
                 connectionstyle=f"arc3,rad={rad}"))


def code_table(fig, cx, cy, label, star=False, z=5):
    a = fig.add_axes([cx - 0.026, cy - 0.085, 0.052, 0.17], zorder=z)
    rng = np.random.default_rng(1)
    vals = rng.uniform(-1, 1, size=(5, 3))
    a.imshow(vals, cmap="viridis", aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor(C_TRAIN_E); sp.set_linewidth(1.1)
    a.set_title(label, fontsize=10, pad=3)
    a.set_xlabel(r"$d_z$", fontsize=9, labelpad=1)
    a.text(-0.45, 0.3, r"$N_{shape}$", transform=a.transAxes, fontsize=9,
           rotation=90, va="center")


def main():
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # divider
    ax.plot([0.02, 0.98], [0.50, 0.50], color="0.7", lw=0.8, ls=(0, (4, 3)), zorder=1)

    # ---------------- STAGE 1 ----------------
    ax.text(0.015, 0.965,
            r"$\bf{Stage\ 1}$ — Auto-decoder: jointly train flow $\boldsymbol{\theta}$ and per-shape codes $\{\boldsymbol{z}_i\}$",
            fontsize=13, va="center")

    deck(fig, 0.065, 0.75, "shape")
    ax.text(0.075, 0.635, r"training shapes $\{\Omega_i\}_{i=1}^{N_{shape}}$", ha="center", fontsize=10)

    box(ax, 0.27, 0.87, 0.165, 0.085,
        "harmonic-map\nground truth\n" + r"$\{(\boldsymbol{\xi}_i,\boldsymbol{x}_i)\}_{i=1}^{N_{shape}}$",
        C_DATA, C_DATA_E, fs=10)
    code_table(fig, 0.27, 0.66, r"$\{\boldsymbol{z}_i\}$ (learnable)")

    box(ax, 0.50, 0.78, 0.155, 0.115, "shared NSF\n" + r"$G_{\theta}(\cdot\,;\boldsymbol{z}_i)$",
        C_TRAIN, C_TRAIN_E, fs=10)
    box(ax, 0.74, 0.78, 0.175, 0.135,
        r"$\mathcal{L}_{S1}$" + "\ninterior + boundary\n+ log-det + code",
        C_LOSS, C_LOSS_E, fs=10)

    # arrows stage 1
    arrow(ax, (0.115, 0.78), (0.185, 0.86))                       # shapes -> GT
    arrow(ax, (0.105, 0.71), (0.245, 0.66), ls=(0, (3, 2)),
          color=C_TRAIN_E)                                        # shapes -> table (index)
    ax.text(0.155, 0.725, "one code\nper shape", fontsize=9, color=C_TRAIN_E, ha="center")
    arrow(ax, (0.296, 0.66), (0.43, 0.74), color=C_TRAIN_E)       # table -> flow (z_i)
    ax.text(0.37, 0.685, r"$\boldsymbol{z}_i$", fontsize=11, color=C_TRAIN_E)
    arrow(ax, (0.353, 0.88), (0.652, 0.83))                       # GT -> loss (target)
    ax.text(0.50, 0.895, r"target $\boldsymbol{x}_i$", fontsize=10, ha="center")
    arrow(ax, (0.578, 0.78), (0.652, 0.78))                       # flow -> loss (pred)
    ax.text(0.615, 0.80, r"$\hat{\boldsymbol{x}}_i$", fontsize=10, ha="center")
    # update loop
    arrow(ax, (0.74, 0.712), (0.50, 0.712), color=C_LOSS_E, ls=(0, (4, 2)),
          rad=-0.32, lw=1.2)
    ax.text(0.60, 0.60, r"backprop: update $\boldsymbol{\theta}$ and $\{\boldsymbol{z}_i\}$ to reduce $\mathcal{L}_{S1}$",
            fontsize=10, color=C_LOSS_E, ha="center")

    # ---------------- STAGE 2 ----------------
    ax.text(0.015, 0.455,
            r"$\bf{Stage\ 2}$ — Encoder: freeze flow, regress SDF image $\to$ code",
            fontsize=13, va="center")

    deck(fig, 0.065, 0.27, "sdf")
    ax.text(0.075, 0.155, r"SDF images $\{I_i\}$", ha="center", fontsize=10)

    box(ax, 0.27, 0.27, 0.15, 0.12, "CNN encoder\n" + r"$\mathrm{Enc}_{\theta_E}$",
        C_TRAIN, C_TRAIN_E, fs=10)
    box(ax, 0.475, 0.32, 0.10, 0.06, r"$\hat{\boldsymbol{z}}_i$", C_DATA, C_DATA_E, fs=11)
    box(ax, 0.475, 0.18, 0.14, 0.075, "frozen NSF\n" + r"$G_{\theta}$  (locked)",
        C_FROZEN, C_FROZEN_E, fs=10, tc="0.25")
    box(ax, 0.74, 0.27, 0.175, 0.125,
        r"$\mathcal{L}_{S2}$" + "\ncode regression\n+ mapping",
        C_LOSS, C_LOSS_E, fs=10)

    arrow(ax, (0.115, 0.27), (0.195, 0.27))                       # sdf -> enc
    arrow(ax, (0.345, 0.29), (0.425, 0.32), color=C_TRAIN_E)      # enc -> zhat
    arrow(ax, (0.345, 0.25), (0.405, 0.19), color=C_TRAIN_E)      # enc -> frozen flow
    arrow(ax, (0.525, 0.32), (0.652, 0.30))                       # zhat -> loss (regression)
    arrow(ax, (0.545, 0.18), (0.652, 0.24))                       # frozen flow -> loss (mapping)

    # ---------------- HINGE: code table -> Stage 2 target ----------------
    arrow(ax, (0.27, 0.572), (0.70, 0.345), color=C_TRAIN_E, ls=(0, (4, 2)),
          rad=-0.12, lw=1.4)
    ax.text(0.50, 0.37, r"stored codes $\{z_i^*\}$  (regression targets)",
            fontsize=11, color=C_TRAIN_E, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    fig.savefig("fig_two_stage_training.pdf", bbox_inches="tight")
    fig.savefig("fig_two_stage_training.png", bbox_inches="tight", dpi=200)
    print("saved fig_two_stage_training.pdf / .png")


if __name__ == "__main__":
    main()
