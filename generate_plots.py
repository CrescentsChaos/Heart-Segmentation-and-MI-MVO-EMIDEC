"""
Generate all training curves and confusion matrices from EMIDEC results.

Produces:
  1. Per-fold Dice bar charts (training curves proxy) for all models
  2. Cross-validation Dice comparison bar charts (all classes)
  3. Per-class confusion matrices from per-case predictions
  4. Disease classification confusion matrices (M5 only)
  5. Radar chart comparing all models across metrics
  6. Box plots showing per-fold variance
  7. Heatmap of all metrics across all models

All figures are saved to: <project>/figures/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = ["M1", "M2", "M3", "M4", "M5"]
BASELINE_VARIANTS = ["UNET", "SEGRESNET", "SWINUNETR", "DYNUNET"]
ALL_VARIANTS = ABLATION_VARIANTS + BASELINE_VARIANTS

DISPLAY_NAMES = {
    "M1": "M1 (Baseline 3D U-Net)",
    "M2": "M2 (AFDD-Net-F)",
    "M3": "M3 (AFDD-Net-D)",
    "M4": "M4 (AFDD-Net-T)",
    "M5": "M5 (AFDD-Net)",
    "UNET": "UNet",
    "SEGRESNET": "SegResNet",
    "SWINUNETR": "SwinUNETR",
    "DYNUNET": "DynUNet",
}

# Color palettes
ABLATION_COLORS = {
    "M1": "#6366f1",  # Indigo
    "M2": "#8b5cf6",  # Violet
    "M3": "#a78bfa",  # Light violet
    "M4": "#c084fc",  # Purple
    "M5": "#e11d48",  # Rose (highlight)
}
BASELINE_COLORS = {
    "UNET": "#64748b",    # Slate
    "SEGRESNET": "#94a3b8",   # Light slate
    "SWINUNETR": "#78716c",  # Stone
    "DYNUNET": "#a8a29e",    # Warm gray
}
ALL_COLORS = {**ABLATION_COLORS, **BASELINE_COLORS}

CLASSES = ["LV", "MYO", "MI", "MVO"]
CLASS_COLORS = {
    "LV": "#3b82f6",
    "MYO": "#10b981",
    "MI": "#f59e0b",
    "MVO": "#ef4444",
}

N_FOLDS = 5


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def setup_style():
    """Apply a professional dark theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "text.color": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#475569",
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0f172a",
        "savefig.edgecolor": "none",
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cv_metrics(variant: str) -> dict | None:
    """Load the cross-validation metrics JSON for a variant."""
    path = RESULTS_DIR / f"{variant}_cv_metrics.json"
    if not path.exists():
        print(f"  [warn] Missing: {path.name}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fold_test_metrics(variant: str, fold: int) -> dict | None:
    """Load per-fold test metrics JSON."""
    path = RESULTS_DIR / f"{variant}_fold{fold}_test_metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_cv_data() -> dict:
    """Load CV metrics for all variants."""
    data = {}
    for v in ALL_VARIANTS:
        cv = load_cv_metrics(v)
        if cv:
            data[v] = cv
    return data


def load_all_fold_test_data() -> dict:
    """Load per-fold test metrics (with per_case data) for all variants."""
    data = {}
    for v in ALL_VARIANTS:
        fold_data = {}
        for fold in range(N_FOLDS):
            fd = load_fold_test_metrics(v, fold)
            if fd:
                fold_data[fold] = fd
        if fold_data:
            data[v] = fold_data
    return data


# ---------------------------------------------------------------------------
# Plot 1: Per-Fold Dice Scores (Training Curve Proxy)
# ---------------------------------------------------------------------------
def plot_per_fold_dice(all_cv: dict):
    """Bar chart showing per-fold Dice for each class across folds (per model)."""
    for variant, cv_data in all_cv.items():
        folds = cv_data.get("folds", [])
        if not folds:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{DISPLAY_NAMES.get(variant, variant)} — Per-Fold Dice Scores (5-Fold CV)",
            fontsize=16, fontweight="bold", color="#f8fafc", y=0.98
        )

        for idx, cls in enumerate(CLASSES):
            ax = axes[idx // 2, idx % 2]
            fold_indices = []
            dice_means = []
            dice_stds = []

            for fold_info in folds:
                fold_num = fold_info["fold"]
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_indices.append(fold_num)
                    dice_means.append(summary[cls]["dice"]["mean"])
                    dice_stds.append(summary[cls]["dice"]["std"])

            if not fold_indices:
                ax.set_visible(False)
                continue

            x = np.arange(len(fold_indices))
            bars = ax.bar(
                x, dice_means, yerr=dice_stds,
                color=CLASS_COLORS[cls], alpha=0.85,
                edgecolor="white", linewidth=0.5,
                capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5, "color": "#94a3b8"}
            )

            # Add value labels
            for bar_obj, val in zip(bars, dice_means):
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#e2e8f0"
                )

            # Mean line
            mean_val = np.mean(dice_means)
            ax.axhline(mean_val, color="#f43f5e", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(
                len(fold_indices) - 0.5, mean_val + 0.005,
                f"Mean: {mean_val:.3f}", fontsize=9, color="#f43f5e", ha="right"
            )

            ax.set_xticks(x)
            ax.set_xticklabels([f"Fold {i}" for i in fold_indices])
            ax.set_ylabel("Dice Score")
            ax.set_title(f"{cls} Dice", fontsize=13, color=CLASS_COLORS[cls])
            ax.set_ylim(0, min(1.05, max(dice_means) + 0.15))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = FIGURES_DIR / f"per_fold_dice_{variant}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-Validation Comparison Bar Charts
# ---------------------------------------------------------------------------
def plot_cv_comparison_bars(all_cv: dict):
    """Side-by-side grouped bar chart comparing all models across classes."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        "Cross-Validation Dice Score Comparison — All Models",
        fontsize=18, fontweight="bold", color="#f8fafc", y=0.98
    )

    for idx, cls in enumerate(CLASSES):
        ax = axes[idx // 2, idx % 2]
        variants_present = []
        means = []
        stds = []
        colors = []

        for v in ALL_VARIANTS:
            if v not in all_cv:
                continue
            folds = all_cv[v].get("folds", [])
            fold_dices = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_dices.append(summary[cls]["dice"]["mean"])
            if fold_dices:
                variants_present.append(v)
                means.append(np.mean(fold_dices))
                stds.append(np.std(fold_dices))
                colors.append(ALL_COLORS.get(v, "#64748b"))

        if not variants_present:
            continue

        x = np.arange(len(variants_present))
        bars = ax.bar(
            x, means, yerr=stds, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"elinewidth": 1.2, "capthick": 1.2, "color": "#94a3b8"}
        )

        for bar_obj, val in zip(bars, means):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0",
                fontweight="bold"
            )

        # Highlight best
        best_idx = int(np.argmax(means))
        bars[best_idx].set_edgecolor("#22d3ee")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_NAMES.get(v, v) for v in variants_present],
            rotation=35, ha="right", fontsize=8
        )
        ax.set_ylabel("Dice Score")
        ax.set_title(f"{cls} Dice", fontsize=14, color=CLASS_COLORS[cls])
        max_y = max(means) + max(stds) + 0.05 if means else 1.0
        ax.set_ylim(0, min(1.1, max_y))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = FIGURES_DIR / "cv_comparison_all_models.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Per-Case Confusion Matrices
# ---------------------------------------------------------------------------
def compute_confusion_from_per_case(all_fold_data: dict):
    """
    For each model, compute per-class confusion using per-case dice scores.
    Binary classification: presence detected (Dice > threshold) vs absent (Dice == 0).
    Ground truth presence: pathological flag + non-zero GT labels.
    """
    results = {}
    for variant, fold_data in all_fold_data.items():
        class_cm = {}
        for cls in CLASSES:
            tp = fp = fn = tn = 0
            for fold_num, fold_metrics in fold_data.items():
                per_case = fold_metrics.get("per_case", [])
                for case in per_case:
                    case_cls = case.get(cls, {})
                    if not case_cls:
                        continue

                    dice = case_cls.get("dice", 0) if isinstance(case_cls, dict) else 0
                    recall = case_cls.get("recall", 0) if isinstance(case_cls, dict) else 0
                    precision = case_cls.get("precision", 0) if isinstance(case_cls, dict) else 0

                    # For anatomy (LV, MYO): always present in GT
                    if cls in ["LV", "MYO"]:
                        gt_present = True
                    else:
                        # For pathology (MI, MVO): check pathological flag
                        is_pathological = case.get("pathological", False)
                        # If recall > 0, GT had voxels; if recall == 0 and dice == 0,
                        # could be either no GT or complete miss
                        # Use pathological flag as primary indicator
                        if cls == "MI":
                            gt_present = is_pathological
                        else:  # MVO
                            # MVO can be absent even in pathological cases
                            # Dice == 1.0 with all zeros means both pred and GT are empty (TN)
                            gt_present = is_pathological and dice != 1.0

                    pred_present = dice > 0.01 and dice != 1.0  # Exclude perfect empty-vs-empty match

                    # For LV/MYO, always present, so pred_present is dice > threshold
                    if cls in ["LV", "MYO"]:
                        pred_present = dice > 0.3  # Low threshold; these should always be detected

                    if gt_present and pred_present:
                        tp += 1
                    elif gt_present and not pred_present:
                        fn += 1
                    elif not gt_present and pred_present:
                        fp += 1
                    else:
                        tn += 1

            class_cm[cls] = np.array([[tp, fn], [fp, tn]])
        results[variant] = class_cm
    return results


def plot_confusion_matrices(confusion_data: dict):
    """Plot confusion matrices for pathology classes (MI, MVO) for all models."""
    for variant, class_cm in confusion_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"{DISPLAY_NAMES.get(variant, variant)} — Pathology Detection Confusion Matrices",
            fontsize=14, fontweight="bold", color="#f8fafc", y=1.02
        )

        for idx, cls in enumerate(["MI", "MVO"]):
            ax = axes[idx]
            cm = class_cm[cls]
            total = cm.sum()

            # Normalize for display
            cm_pct = cm / total * 100 if total > 0 else cm * 0

            im = ax.imshow(cm_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

            # Labels
            labels_pred = ["Detected", "Missed"]
            labels_gt = ["Present\n(GT)", "Absent\n(GT)"]

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels_pred, fontsize=10)
            ax.set_yticklabels(labels_gt, fontsize=10)
            ax.set_xlabel("Prediction", fontsize=11)
            ax.set_ylabel("Ground Truth", fontsize=11)

            # Annotate cells
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = cm_pct[i, j]
                    text_color = "white" if pct > 50 else "#1e293b"
                    ax.text(
                        j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color=text_color
                    )

            ax.set_title(f"{cls}", fontsize=13, color=CLASS_COLORS[cls], pad=10)
            plt.colorbar(im, ax=ax, shrink=0.8, label="Percentage (%)")

        plt.tight_layout()
        save_path = FIGURES_DIR / f"confusion_matrix_{variant}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 4: Disease Classification Confusion Matrix (M5)
# ---------------------------------------------------------------------------
def plot_disease_classification_cm(all_fold_data: dict):
    """Plot confusion matrix for the disease classifier (normal vs pathological)."""
    for variant in ["M5"]:
        if variant not in all_fold_data:
            continue

        tp = fp = fn = tn = 0
        for fold_num, fold_metrics in all_fold_data[variant].items():
            per_case = fold_metrics.get("per_case", [])
            for case in per_case:
                gt_pathological = case.get("pathological", False)
                disease_pred = case.get("disease_pred", None)
                if disease_pred is None:
                    continue

                pred_pathological = disease_pred == 1

                if gt_pathological and pred_pathological:
                    tp += 1
                elif gt_pathological and not pred_pathological:
                    fn += 1
                elif not gt_pathological and pred_pathological:
                    fp += 1
                else:
                    tn += 1

        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()
        if total == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle(
            f"{DISPLAY_NAMES.get(variant, variant)} — Disease Classification\n"
            f"(Normal vs Pathological)",
            fontsize=14, fontweight="bold", color="#f8fafc", y=1.0
        )

        cm_pct = cm / total * 100
        im = ax.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)

        labels = ["Normal", "Pathological"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_pct[i, j]
                text_color = "white" if pct > 50 else "#1e293b"
                ax.text(
                    j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=text_color
                )

        # Overall accuracy
        acc = (tp + tn) / total * 100
        ax.set_title(
            f"Overall Accuracy: {acc:.1f}% ({tp + tn}/{total})",
            fontsize=12, color="#22d3ee", pad=15
        )

        plt.colorbar(im, ax=ax, shrink=0.8, label="Percentage (%)")
        plt.tight_layout()
        save_path = FIGURES_DIR / f"disease_classification_cm_{variant}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 5: Radar Chart Comparing All Models
# ---------------------------------------------------------------------------
def plot_radar_comparison(all_cv: dict):
    """Radar (spider) chart comparing models across all metrics."""
    metrics_keys = ["LV", "MYO", "MI", "MVO"]
    metric_labels = ["LV Dice", "MYO Dice", "MI Dice", "MVO Dice"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for variant in ALL_VARIANTS:
        if variant not in all_cv:
            continue
        folds = all_cv[variant].get("folds", [])
        values = []
        for cls in metrics_keys:
            fold_dices = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_dices.append(summary[cls]["dice"]["mean"])
            values.append(np.mean(fold_dices) if fold_dices else 0)

        values += values[:1]  # Close
        linewidth = 3 if variant == "M5" else 1.5
        alpha = 0.3 if variant == "M5" else 0.1
        ax.plot(
            angles, values, "o-",
            linewidth=linewidth,
            color=ALL_COLORS.get(variant, "#64748b"),
            label=DISPLAY_NAMES.get(variant, variant),
        )
        ax.fill(angles, values, alpha=alpha, color=ALL_COLORS.get(variant, "#64748b"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12, color="#e2e8f0")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="#94a3b8")
    ax.yaxis.grid(True, color="#475569", alpha=0.4)
    ax.xaxis.grid(True, color="#475569", alpha=0.4)
    ax.spines["polar"].set_color("#475569")

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.1),
        fontsize=9, framealpha=0.8
    )
    ax.set_title(
        "Model Comparison — Dice Scores Across All Classes",
        fontsize=14, fontweight="bold", color="#f8fafc", pad=30
    )

    plt.tight_layout()
    save_path = FIGURES_DIR / "radar_comparison.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 6: Box Plots (Per-Fold Variance)
# ---------------------------------------------------------------------------
def plot_box_plots(all_cv: dict):
    """Box plots showing per-fold Dice distribution for each model × class."""
    for cls in CLASSES:
        fig, ax = plt.subplots(figsize=(16, 8))

        data_list = []
        labels = []
        positions = []
        colors_list = []

        pos = 0
        for v in ALL_VARIANTS:
            if v not in all_cv:
                continue
            folds = all_cv[v].get("folds", [])
            fold_dices = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_dices.append(summary[cls]["dice"]["mean"])

            if fold_dices:
                data_list.append(fold_dices)
                labels.append(DISPLAY_NAMES.get(v, v))
                positions.append(pos)
                colors_list.append(ALL_COLORS.get(v, "#64748b"))
                pos += 1

        if not data_list:
            plt.close(fig)
            continue

        bp = ax.boxplot(
            data_list, positions=positions, widths=0.6,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="#22d3ee", markeredgecolor="#22d3ee", markersize=6),
            medianprops=dict(color="#f43f5e", linewidth=2),
            whiskerprops=dict(color="#94a3b8"),
            capprops=dict(color="#94a3b8"),
            flierprops=dict(marker="o", markerfacecolor="#f43f5e", markersize=5, alpha=0.5),
        )

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("white")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Dice Score", fontsize=12)
        ax.set_title(
            f"{cls} Dice — Per-Fold Distribution Across Models",
            fontsize=14, fontweight="bold", color=CLASS_COLORS[cls]
        )

        plt.tight_layout()
        save_path = FIGURES_DIR / f"boxplot_{cls}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 7: Metrics Heatmap
# ---------------------------------------------------------------------------
def plot_metrics_heatmap(all_cv: dict):
    """Heatmap of mean Dice scores across all models and classes."""
    variants_present = [v for v in ALL_VARIANTS if v in all_cv]
    if not variants_present:
        return

    metrics = ["LV", "MYO", "MI", "MI_pathological", "MVO"]
    metric_labels = ["LV Dice", "MYO Dice", "MI Dice (all)", "MI Dice (path-only)", "MVO Dice"]

    matrix = np.zeros((len(variants_present), len(metrics)))
    for i, v in enumerate(variants_present):
        folds = all_cv[v].get("folds", [])
        for j, cls in enumerate(metrics):
            fold_dices = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_dices.append(summary[cls]["dice"]["mean"])
            matrix[i, j] = np.mean(fold_dices) if fold_dices else 0

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(variants_present)))
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticklabels(
        [DISPLAY_NAMES.get(v, v) for v in variants_present], fontsize=10
    )

    # Annotate
    for i in range(len(variants_present)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            text_color = "white" if val < 0.4 or val > 0.85 else "#1e293b"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=text_color
            )

    # Highlight best per column
    for j in range(len(metrics)):
        best_row = int(np.argmax(matrix[:, j]))
        ax.add_patch(plt.Rectangle(
            (j - 0.5, best_row - 0.5), 1, 1,
            fill=False, edgecolor="#22d3ee", linewidth=3
        ))

    ax.set_title(
        "Mean Dice Scores — All Models × All Classes (5-Fold CV)",
        fontsize=14, fontweight="bold", color="#f8fafc", pad=15
    )

    plt.colorbar(im, ax=ax, shrink=0.8, label="Dice Score")
    plt.tight_layout()
    save_path = FIGURES_DIR / "metrics_heatmap.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 8: Ablation Progression (Training Curve Style)
# ---------------------------------------------------------------------------
def plot_ablation_progression(all_cv: dict):
    """
    Line plot showing how each metric improves across ablation stages M1→M5.
    This is the closest to a 'training curve' from the available data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Ablation Study Progression (M1 → M5) — Dice Scores",
        fontsize=16, fontweight="bold", color="#f8fafc", y=0.98
    )

    ablation_present = [v for v in ABLATION_VARIANTS if v in all_cv]
    if not ablation_present:
        plt.close(fig)
        return

    for idx, cls in enumerate(CLASSES):
        ax = axes[idx // 2, idx % 2]
        means = []
        stds = []
        variant_labels = []

        for v in ablation_present:
            folds = all_cv[v].get("folds", [])
            fold_dices = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary and "dice" in summary[cls]:
                    fold_dices.append(summary[cls]["dice"]["mean"])
            if fold_dices:
                means.append(np.mean(fold_dices))
                stds.append(np.std(fold_dices))
                variant_labels.append(v)

        if not means:
            ax.set_visible(False)
            continue

        x = np.arange(len(variant_labels))
        colors_abl = [ABLATION_COLORS.get(v, "#6366f1") for v in variant_labels]

        # Line with error band
        ax.fill_between(
            x, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
            alpha=0.2, color=CLASS_COLORS[cls]
        )
        ax.plot(x, means, "o-", color=CLASS_COLORS[cls], linewidth=2.5, markersize=10, zorder=5)

        # Mark individual points with variant colors
        for xi, (m, v) in enumerate(zip(means, variant_labels)):
            ax.scatter([xi], [m], s=120, color=ABLATION_COLORS[v], zorder=6, edgecolors="white", linewidth=1.5)
            ax.annotate(
                f"{m:.3f}",
                (xi, m), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, color="#e2e8f0", fontweight="bold"
            )

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(v, v) for v in variant_labels], fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Dice Score")
        ax.set_title(f"{cls} Dice — Ablation Progression", fontsize=13, color=CLASS_COLORS[cls])

        # Add improvement arrows
        if len(means) >= 2:
            delta = means[-1] - means[0]
            sign = "+" if delta > 0 else ""
            ax.text(
                0.98, 0.05,
                f"Δ(M1→M5): {sign}{delta:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#22d3ee", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#475569")
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = FIGURES_DIR / "ablation_progression.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 9: Precision-Recall per model (pathology classes)
# ---------------------------------------------------------------------------
def plot_precision_recall(all_cv: dict):
    """Scatter plot of precision vs recall for MI and MVO across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Precision vs Recall — Pathology Classes (5-Fold CV Mean)",
        fontsize=14, fontweight="bold", color="#f8fafc", y=1.02
    )

    for idx, cls in enumerate(["MI", "MVO"]):
        ax = axes[idx]

        for v in ALL_VARIANTS:
            if v not in all_cv:
                continue
            folds = all_cv[v].get("folds", [])
            precisions = []
            recalls = []
            for fold_info in folds:
                summary = fold_info.get("summary", {})
                if cls in summary:
                    precisions.append(summary[cls].get("precision", {}).get("mean", 0))
                    recalls.append(summary[cls].get("recall", {}).get("mean", 0))

            if precisions and recalls:
                p = np.mean(precisions)
                r = np.mean(recalls)
                size = 200 if v == "M5" else 100
                ax.scatter(
                    r, p, s=size, color=ALL_COLORS.get(v, "#64748b"),
                    edgecolors="white", linewidth=1.5, zorder=5,
                    label=DISPLAY_NAMES.get(v, v)
                )
                ax.annotate(
                    v, (r, p), textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color="#e2e8f0"
                )

        # F1 iso-lines
        for f1_val in [0.2, 0.4, 0.6, 0.8]:
            r_range = np.linspace(0.01, 1, 100)
            p_range = f1_val * r_range / (2 * r_range - f1_val)
            valid = p_range > 0
            ax.plot(
                r_range[valid], p_range[valid], "--",
                color="#475569", alpha=0.4, linewidth=0.8
            )
            # Label
            label_idx = len(r_range[valid]) // 2
            if label_idx < len(r_range[valid]):
                ax.text(
                    r_range[valid][label_idx], p_range[valid][label_idx] + 0.02,
                    f"F1={f1_val}", fontsize=7, color="#64748b", alpha=0.6
                )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{cls}", fontsize=13, color=CLASS_COLORS[cls])
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    save_path = FIGURES_DIR / "precision_recall.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Plot 10: Combined Summary Dashboard
# ---------------------------------------------------------------------------
def plot_summary_dashboard(all_cv: dict, all_fold_data: dict):
    """A single dashboard figure with key metrics."""
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        "EMIDEC Heart Segmentation — Results Dashboard",
        fontsize=20, fontweight="bold", color="#f8fafc", y=0.99
    )

    # Subplot 1: Ablation bar chart for MI pathological
    ax1 = fig.add_subplot(2, 3, 1)
    ablation = [v for v in ABLATION_VARIANTS if v in all_cv]
    mi_path_means = []
    mi_path_stds = []
    for v in ablation:
        folds = all_cv[v].get("folds", [])
        vals = []
        for fold_info in folds:
            s = fold_info.get("summary", {})
            if "MI_pathological" in s:
                vals.append(s["MI_pathological"]["dice"]["mean"])
        mi_path_means.append(np.mean(vals) if vals else 0)
        mi_path_stds.append(np.std(vals) if vals else 0)

    x = np.arange(len(ablation))
    colors_abl = [ABLATION_COLORS[v] for v in ablation]
    bars = ax1.bar(x, mi_path_means, yerr=mi_path_stds, color=colors_abl, alpha=0.9,
                   edgecolor="white", linewidth=0.5, capsize=3)
    for b, val in zip(bars, mi_path_means):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ablation, fontsize=9)
    ax1.set_ylabel("Dice Score")
    ax1.set_title("MI Pathological Dice (Ablation)", fontsize=12, color="#f59e0b")

    # Subplot 2: Baseline bar chart for MI
    ax2 = fig.add_subplot(2, 3, 2)
    baselines = [v for v in BASELINE_VARIANTS if v in all_cv]
    mi_means_bl = []
    mi_stds_bl = []
    for v in baselines:
        folds = all_cv[v].get("folds", [])
        vals = []
        for fold_info in folds:
            s = fold_info.get("summary", {})
            if "MI" in s:
                vals.append(s["MI"]["dice"]["mean"])
        mi_means_bl.append(np.mean(vals) if vals else 0)
        mi_stds_bl.append(np.std(vals) if vals else 0)

    # Add M5 for comparison
    if "M5" in all_cv:
        baselines_ext = baselines + ["M5"]
        folds = all_cv["M5"].get("folds", [])
        vals = [fold_info["summary"]["MI"]["dice"]["mean"]
                for fold_info in folds if "MI" in fold_info.get("summary", {})]
        mi_means_bl.append(np.mean(vals) if vals else 0)
        mi_stds_bl.append(np.std(vals) if vals else 0)
    else:
        baselines_ext = baselines

    x2 = np.arange(len(baselines_ext))
    colors_bl = [BASELINE_COLORS.get(v, ABLATION_COLORS.get(v, "#64748b")) for v in baselines_ext]
    bars2 = ax2.bar(x2, mi_means_bl, yerr=mi_stds_bl, color=colors_bl, alpha=0.9,
                    edgecolor="white", linewidth=0.5, capsize=3)
    for b, val in zip(bars2, mi_means_bl):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(baselines_ext, fontsize=9)
    ax2.set_ylabel("Dice Score")
    ax2.set_title("MI Dice — Baselines vs AFDD-Net (M5)", fontsize=12, color="#f59e0b")

    # Subplot 3: LV + MYO comparison
    ax3 = fig.add_subplot(2, 3, 3)
    present = [v for v in ALL_VARIANTS if v in all_cv]
    lv_means = []
    myo_means = []
    for v in present:
        folds = all_cv[v].get("folds", [])
        lv_vals = [fold_info["summary"]["LV"]["dice"]["mean"]
                   for fold_info in folds if "LV" in fold_info.get("summary", {})]
        myo_vals = [fold_info["summary"]["MYO"]["dice"]["mean"]
                    for fold_info in folds if "MYO" in fold_info.get("summary", {})]
        lv_means.append(np.mean(lv_vals) if lv_vals else 0)
        myo_means.append(np.mean(myo_vals) if myo_vals else 0)

    x3 = np.arange(len(present))
    w = 0.35
    ax3.bar(x3 - w / 2, lv_means, w, label="LV", color="#3b82f6", alpha=0.85)
    ax3.bar(x3 + w / 2, myo_means, w, label="MYO", color="#10b981", alpha=0.85)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(present, fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Dice Score")
    ax3.set_title("LV & MYO Dice — All Models", fontsize=12)
    ax3.legend(fontsize=9)

    # Subplot 4: Disease accuracy (M5 per fold)
    ax4 = fig.add_subplot(2, 3, 4)
    if "M5" in all_cv:
        folds = all_cv["M5"].get("folds", [])
        d_accs = []
        fold_nums = []
        for fold_info in folds:
            s = fold_info.get("summary", {})
            d_acc = s.get("disease_acc", None)
            if d_acc is not None:
                d_accs.append(d_acc)
                fold_nums.append(fold_info["fold"])

        if d_accs:
            x4 = np.arange(len(fold_nums))
            bars4 = ax4.bar(x4, d_accs, color="#8b5cf6", alpha=0.85, edgecolor="white")
            for b, val in zip(bars4, d_accs):
                ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#e2e8f0")
            mean_acc = np.mean(d_accs)
            ax4.axhline(mean_acc, color="#f43f5e", linestyle="--", linewidth=1.5)
            ax4.text(len(fold_nums) - 0.5, mean_acc + 0.005,
                     f"Mean: {mean_acc:.3f}", fontsize=9, color="#f43f5e", ha="right")
            ax4.set_xticks(x4)
            ax4.set_xticklabels([f"Fold {i}" for i in fold_nums])
            ax4.set_ylabel("Accuracy")
            ax4.set_title("M5 Disease Classification Accuracy", fontsize=12, color="#8b5cf6")
            ax4.set_ylim(0, 1.05)

    # Subplot 5: Parameters vs MI Dice
    ax5 = fig.add_subplot(2, 3, 5)
    for v in ALL_VARIANTS:
        if v not in all_cv:
            continue
        folds = all_cv[v].get("folds", [])
        params = None
        mi_vals = []
        for fold_info in folds:
            s = fold_info.get("summary", {})
            if params is None:
                params = s.get("params_M", None)
            if "MI" in s:
                mi_vals.append(s["MI"]["dice"]["mean"])
        if params and mi_vals:
            size = 250 if v == "M5" else 120
            ax5.scatter(
                params, np.mean(mi_vals), s=size,
                color=ALL_COLORS.get(v, "#64748b"),
                edgecolors="white", linewidth=1.5, zorder=5
            )
            ax5.annotate(
                v, (params, np.mean(mi_vals)),
                textcoords="offset points", xytext=(8, 5),
                fontsize=9, color="#e2e8f0"
            )

    ax5.set_xlabel("Parameters (M)")
    ax5.set_ylabel("MI Dice Score")
    ax5.set_title("Efficiency: Parameters vs MI Dice", fontsize=12, color="#22d3ee")

    # Subplot 6: Inference time vs MI Dice
    ax6 = fig.add_subplot(2, 3, 6)
    for v in ALL_VARIANTS:
        if v not in all_cv:
            continue
        folds = all_cv[v].get("folds", [])
        inf_times = []
        mi_vals = []
        for fold_info in folds:
            s = fold_info.get("summary", {})
            t = s.get("inference_ms_mean", None)
            if t:
                inf_times.append(t)
            if "MI" in s:
                mi_vals.append(s["MI"]["dice"]["mean"])
        if inf_times and mi_vals:
            size = 250 if v == "M5" else 120
            ax6.scatter(
                np.mean(inf_times), np.mean(mi_vals), s=size,
                color=ALL_COLORS.get(v, "#64748b"),
                edgecolors="white", linewidth=1.5, zorder=5
            )
            ax6.annotate(
                v, (np.mean(inf_times), np.mean(mi_vals)),
                textcoords="offset points", xytext=(8, 5),
                fontsize=9, color="#e2e8f0"
            )

    ax6.set_xlabel("Inference Time (ms)")
    ax6.set_ylabel("MI Dice Score")
    ax6.set_title("Speed vs Accuracy", fontsize=12, color="#22d3ee")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = FIGURES_DIR / "summary_dashboard.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EMIDEC Results Visualization — Generating All Plots")
    print("=" * 60)

    setup_style()

    # Load data
    print("\n📂 Loading results...")
    all_cv = load_all_cv_data()
    all_fold_data = load_all_fold_test_data()
    print(f"  Loaded CV metrics for: {list(all_cv.keys())}")
    print(f"  Loaded fold test data for: {list(all_fold_data.keys())}")

    if not all_cv:
        print("❌ No CV metrics found! Check results directory.")
        sys.exit(1)

    # Generate plots
    print("\n📊 Generating per-fold Dice bar charts...")
    plot_per_fold_dice(all_cv)

    print("\n📊 Generating CV comparison bar charts...")
    plot_cv_comparison_bars(all_cv)

    print("\n📊 Generating ablation progression plots...")
    plot_ablation_progression(all_cv)

    print("\n📊 Generating confusion matrices...")
    confusion_data = compute_confusion_from_per_case(all_fold_data)
    plot_confusion_matrices(confusion_data)

    print("\n📊 Generating disease classification confusion matrices...")
    plot_disease_classification_cm(all_fold_data)

    print("\n📊 Generating radar comparison chart...")
    plot_radar_comparison(all_cv)

    print("\n📊 Generating box plots...")
    plot_box_plots(all_cv)

    print("\n📊 Generating metrics heatmap...")
    plot_metrics_heatmap(all_cv)

    print("\n📊 Generating precision-recall plots...")
    plot_precision_recall(all_cv)

    print("\n📊 Generating summary dashboard...")
    plot_summary_dashboard(all_cv, all_fold_data)

    print(f"\n✅ All plots saved to: {FIGURES_DIR}")
    print(f"   Total figures: {len(list(FIGURES_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
