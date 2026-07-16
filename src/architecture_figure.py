# -*- coding: utf-8 -*-
"""
Generate the AFDD-Net architecture figure for the thesis/paper.
Output: figures/paper/architecture_AFDDNet.png (+ .pdf)

Usage:
  python -m src.architecture_figure
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from model_identity import MODEL_NAME


# Palette (clean print-friendly)
C_ENC = "#2E86AB"
C_BN = "#1B4F72"
C_ANAT = "#27AE60"
C_PATH = "#E74C3C"
C_GATE = "#F39C12"
C_SKIP = "#7F8C8D"
C_IN = "#5D6D7E"
C_OUT = "#8E44AD"
C_BOX_EDGE = "#2C3E50"
C_BG = "white"


def _box(ax, x, y, w, h, text, facecolor, fontsize=8, textcolor="white",
         weight="bold", alpha=1.0, lw=1.2, radius=0.02):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor, edgecolor=C_BOX_EDGE, linewidth=lw, alpha=alpha,
        mutation_aspect=0.5,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=textcolor, fontweight=weight, linespacing=1.25,
        wrap=False,
    )
    return patch


def _arrow(ax, x1, y1, x2, y2, color=C_BOX_EDGE, lw=1.4,
           connectionstyle="arc3,rad=0", mutation_scale=12):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, lw=lw, connectionstyle=connectionstyle,
        shrinkA=0, shrinkB=0,
    ))


def _dashed_arrow(ax, x1, y1, x2, y2, color=C_SKIP, lw=1.1):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=(0, (4, 2.5)), mutation_scale=10,
            connectionstyle="arc3,rad=0",
        ),
    )


def draw_architecture(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=200)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Title
    ax.text(
        7.25, 8.85, f"{MODEL_NAME} Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold", color=C_BOX_EDGE,
    )
    ax.text(
        7.25, 8.48,
        "Anisotropic Factorized Dual-Decoder with MYO Soft-Gating",
        ha="center", va="center", fontsize=9.5, color="#566573", style="italic",
    )

    # ---------- Input ----------
    _box(ax, 0.25, 4.15, 1.35, 1.1,
         "LGE-MRI\nInput\n1 x D x H x W\n16 x 128 x 128",
         C_IN, fontsize=7.5)

    # ---------- Shared Encoder column ----------
    enc_x = 2.0
    enc_w = 1.7
    enc_h = 0.72
    enc_stages = [
        (6.55, "Enc-1\n32 ch\nAF-Conv", C_ENC),
        (5.55, "Enc-2\n64 ch\nAF-Conv", C_ENC),
        (4.55, "Enc-3\n128 ch\nAF-Conv", C_ENC),
        (3.55, "Enc-4\n256 ch\nAF-Conv", C_ENC),
        (2.40, "Bottleneck\n512 ch\nAF-Conv", C_BN),
    ]
    for y, label, color in enc_stages:
        _box(ax, enc_x, y, enc_w, enc_h, label, color, fontsize=7.5)

    for y in (6.40, 5.40, 4.40, 3.40):
        ax.text(enc_x + enc_w / 2, y, "v MaxPool 2x", ha="center", va="center",
                fontsize=6.5, color="#566573")

    _arrow(ax, 1.60, 4.70, 2.0, 6.91, color=C_IN, lw=1.5)

    for y1, y2 in ((6.55, 6.27), (5.55, 5.27), (4.55, 4.27), (3.55, 3.12)):
        _arrow(ax, enc_x + enc_w / 2, y1, enc_x + enc_w / 2, y2 + enc_h,
               color=C_ENC, lw=1.2, mutation_scale=10)

    ax.annotate(
        "Shared Encoder\n(factorized 3x3x1 + 1x1x3)",
        xy=(enc_x - 0.08, 4.6), xytext=(0.15, 7.55),
        fontsize=8, color=C_ENC, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_ENC, lw=1.0),
        ha="left", va="center",
    )

    # ---------- Anatomy decoder ----------
    anat_x = 5.6
    anat_w = 1.85
    anat_h = 0.68
    anat_stages = [
        (6.55, "Dec-A1\n32 + AG"),
        (5.55, "Dec-A2\n64 + AG"),
        (4.55, "Dec-A3\n128 + AG"),
        (3.55, "Dec-A4\n256 + AG"),
    ]
    for y, label in anat_stages:
        _box(ax, anat_x, y, anat_w, anat_h, label, C_ANAT, fontsize=7.5)

    _arrow(ax, enc_x + enc_w, 2.76, anat_x, 3.89, color=C_BN, lw=1.3)
    for y1, y2 in ((3.55 + anat_h, 4.55), (4.55 + anat_h, 5.55), (5.55 + anat_h, 6.55)):
        _arrow(ax, anat_x + anat_w / 2, y1, anat_x + anat_w / 2, y2,
               color=C_ANAT, lw=1.2, mutation_scale=10)

    _box(ax, 7.75, 6.50, 1.55, 0.82,
         "Anatomy Head\nSoftmax\nBG / LV / MYO",
         C_ANAT, fontsize=7.5)
    _arrow(ax, anat_x + anat_w, 6.89, 7.75, 6.91, color=C_ANAT, lw=1.3)

    ax.text(anat_x + anat_w / 2, 7.45, "Anatomy Decoder",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_ANAT)

    # ---------- Pathology decoder ----------
    path_x = 10.0
    path_w = 1.85
    path_stages = [
        (6.55, "Dec-P1\n32 + AG + G"),
        (5.55, "Dec-P2\n64 + AG + G"),
        (4.55, "Dec-P3\n128 + AG + G"),
        (3.55, "Dec-P4\n256 + AG + G"),
    ]
    for y, label in path_stages:
        _box(ax, path_x, y, path_w, anat_h, label, C_PATH, fontsize=7.2)

    _arrow(ax, enc_x + enc_w, 2.55, path_x, 3.89,
           color=C_BN, lw=1.3, connectionstyle="arc3,rad=-0.15")

    for y1, y2 in ((3.55 + anat_h, 4.55), (4.55 + anat_h, 5.55), (5.55 + anat_h, 6.55)):
        _arrow(ax, path_x + path_w / 2, y1, path_x + path_w / 2, y2,
               color=C_PATH, lw=1.2, mutation_scale=10)

    _box(ax, 12.15, 6.50, 1.55, 0.82,
         "Pathology Head\nSigmoid\nMI / MVO",
         C_PATH, fontsize=7.5)
    _arrow(ax, path_x + path_w, 6.89, 12.15, 6.91, color=C_PATH, lw=1.3)

    ax.text(path_x + path_w / 2, 7.45, "Pathology Decoder",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_PATH)

    # ---------- Skip connections ----------
    skip_ys = [6.91, 5.91, 4.91, 3.91]
    for y in skip_ys:
        _dashed_arrow(ax, enc_x + enc_w, y, anat_x, y, color=C_SKIP, lw=1.0)
        _dashed_arrow(ax, enc_x + enc_w, y - 0.08, path_x, y, color="#AAB7B8", lw=0.9)

    # ---------- MYO soft-gating ----------
    _box(ax, 7.75, 4.85, 1.55, 0.95,
         "Soft MYO\nP(MYO)\n(detached)",
         C_GATE, fontsize=7.5, textcolor="#1C2833")
    _arrow(ax, 8.525, 6.50, 8.525, 5.80, color=C_GATE, lw=1.5)

    for y in skip_ys:
        ax.annotate(
            "", xy=(path_x, y), xytext=(9.30, 5.32),
            arrowprops=dict(
                arrowstyle="-|>", color=C_GATE, lw=1.05,
                linestyle=(0, (3, 2)), mutation_scale=9,
                connectionstyle="arc3,rad=0.05",
            ),
        )

    ax.text(
        8.9, 4.55, "MYO soft-gate\nat every scale",
        ha="center", va="top", fontsize=7, color="#B9770E", fontweight="bold",
    )

    # ---------- Soft restrict + outputs ----------
    _box(ax, 12.0, 4.85, 1.85, 0.95,
         "Soft Restrict\npath x MYO.detach()",
         "#D35400", fontsize=7.2)
    _arrow(ax, 12.925, 6.50, 12.925, 5.80, color="#D35400", lw=1.4)
    _arrow(ax, 9.30, 5.32, 12.0, 5.32, color=C_GATE, lw=1.1)

    _box(ax, 11.85, 3.55, 2.15, 0.85,
         "Outputs\nLV / MYO / MI / MVO",
         C_OUT, fontsize=8)
    _arrow(ax, 12.925, 4.85, 12.925, 4.40, color=C_OUT, lw=1.4)

    # ---------- Factorized conv inset ----------
    inset = FancyBboxPatch(
        (0.25, 0.25), 4.6, 1.85,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#F8F9F9", edgecolor=C_ENC, linewidth=1.4,
    )
    ax.add_patch(inset)
    ax.text(2.55, 1.90, "Anisotropic Factorized Block (AF-Conv)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_ENC)

    _box(ax, 0.45, 0.95, 1.25, 0.65, "In-plane\nConv 1x3x3", "#5DADE2",
         fontsize=7, textcolor="#1B2631")
    _box(ax, 1.95, 0.95, 1.35, 0.65, "Through-plane\nConv 3x1x1", "#3498DB",
         fontsize=7, textcolor="white")
    _box(ax, 3.55, 0.95, 1.05, 0.65, "+ Residual", "#1A5276",
         fontsize=7)
    _arrow(ax, 1.70, 1.27, 1.95, 1.27, color=C_ENC, lw=1.2, mutation_scale=10)
    _arrow(ax, 3.30, 1.27, 3.55, 1.27, color=C_ENC, lw=1.2, mutation_scale=10)
    ax.text(2.55, 0.55,
            "Matches EMIDEC spacing ~ 1.5 x 1.5 x 10 mm",
            ha="center", va="center", fontsize=7, color="#566573", style="italic")

    # ---------- Training losses inset ----------
    loss_box = FancyBboxPatch(
        (5.15, 0.25), 4.55, 1.85,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#FDFEFE", edgecolor=C_PATH, linewidth=1.4,
    )
    ax.add_patch(loss_box)
    ax.text(7.42, 1.90, "Training Objectives (M5)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_PATH)
    ax.text(
        7.42, 1.15,
        "Anatomy: Soft Dice + weighted CE\n"
        "Pathology: Focal Tversky (a=0.65, b=0.35)\n"
        "Topology: L_topo = path * (1 - MYO)\n"
        "          curriculum: warmup -> lambda=0.05",
        ha="center", va="center", fontsize=7.2, color="#2C3E50",
    )

    # ---------- Legend ----------
    legend_box = FancyBboxPatch(
        (9.95, 0.25), 4.25, 1.85,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#F4F6F7", edgecolor=C_BOX_EDGE, linewidth=1.2,
    )
    ax.add_patch(legend_box)
    ax.text(12.07, 1.90, "Legend",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_BOX_EDGE)

    legend_items = [
        (C_ENC, "Shared encoder (AF-Conv)"),
        (C_ANAT, "Anatomy decoder + AG"),
        (C_PATH, "Pathology decoder + AG"),
        (C_GATE, "MYO soft-gating"),
        (C_SKIP, "Skip connection (dashed)"),
    ]
    for i, (color, label) in enumerate(legend_items):
        yy = 1.55 - i * 0.26
        ax.add_patch(Rectangle((10.15, yy - 0.08), 0.28, 0.16,
                               facecolor=color, edgecolor=C_BOX_EDGE, lw=0.6))
        ax.text(10.55, yy, label, ha="left", va="center", fontsize=7, color="#2C3E50")

    ax.text(
        7.25, 8.12,
        "AG = Attention Gate  |  G = MYO soft-gate channel  |  Filters: 32 -> 64 -> 128 -> 256 -> 512",
        ha="center", va="center", fontsize=7.5, color="#7F8C8D",
    )

    fig.tight_layout(pad=0.3)
    saved = []
    for ext in ("png", "pdf"):
        path = out_dir / f"architecture_AFDDNet.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=C_BG)
        saved.append(path)
    plt.close(fig)
    return saved


def main():
    out = cfg.PAPER_FIGURES_DIR
    paths = draw_architecture(out)
    print(f"{MODEL_NAME} architecture figure written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
