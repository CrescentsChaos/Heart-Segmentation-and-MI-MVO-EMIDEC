"""
Generate all paper-ready metric graphs and comparison tables for AFDD-Net.
Outputs under figures/paper/ and results/paper/:
  - Training curves (loss, Dice, HD95, Recall) per variant
  - Ablation bar charts (Dice / HD95 / Recall / IoU)
  - SOTA comparison vs verified EMIDEC-only published methods
  - Efficiency plot (params vs MI Dice)
  - IEEE-style Markdown + CSV comparison tables
Usage:
  python -m src.paper_figures
  python -m src.paper_figures --split test
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from model_identity import (
    ABLATION_VARIANTS,
    BASELINE_VARIANTS,
    MODEL_CITE,
    MODEL_FULL_NAME,
    MODEL_NAME,
    MODEL_YEAR,
    SOTA_BENCHMARKS,
    TARGET_MI_DICE,
    TARGET_MI_DICE_LABEL,
    VARIANT_NAMES,
    VARIANT_SHORT,
    is_multiclass_variant,
)
# Style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)
PALETTE = {
    "M1": "#4C78A8",
    "M2": "#F58518",
    "M3": "#54A24B",
    "M4": "#E45756",
    "M5": "#B279A2",
    "UNET": "#4C78A8",
    "SEGRESNET": "#72B7B2",
    "SWINUNETR": "#F58518",
    "DYNUNET": "#54A24B",
    "DYNUNET_RES": "#D67195",
    "NNUNET": "#1F77B4",
    "sota": "#72B7B2",
    "ours": "#B279A2",
}

def _ensure_dirs():
    fig = cfg.FIGURES_DIR / "paper"
    res = cfg.RESULTS_DIR / "paper"
    fig.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return fig, res

def _dice_key(variant: str) -> str:
    return "MI"


def _mi_path_key(variant: str) -> str:
    return "MI_pathological"

def load_history(variant: str) -> Optional[List[Dict]]:
    path = cfg.RESULTS_DIR / f"{variant}_history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def load_test_summary(variant: str, split: str = "test") -> Optional[Dict]:
    path = cfg.RESULTS_DIR / f"{variant}_{split}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]

def _metric(summary: Dict, region: str, name: str) -> Optional[float]:
    if region not in summary or name not in summary[region]:
        return None
    return float(summary[region][name]["mean"])

def _metric_std(summary: Dict, region: str, name: str) -> Optional[float]:
    if region not in summary or name not in summary[region]:
        return None
    return float(summary[region][name]["std"])
# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(variant: str, history: List[Dict], out_dir: Path):
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    mi_key = _dice_key(variant)
    def series(region, metric="dice"):
        vals = []
        for h in history:
            v = h.get("val", {}).get(region, {}).get(metric, {})
            vals.append(v.get("mean", np.nan) if isinstance(v, dict) else np.nan)
        return vals
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{VARIANT_NAMES[variant]} - training curves", fontweight="bold")
    axes[0, 0].plot(epochs, losses, color=PALETTE.get(variant, "#333"), lw=2)
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    for region, label in [("LV", "LV"), ("MYO", "MYO"), (mi_key, "MI/Infarct")]:
        if not is_multiclass_variant(variant) and region == "MVO":
            continue
        y = series(region, "dice")
        if np.all(np.isnan(y)):
            continue
        axes[0, 1].plot(epochs, y, lw=2, label=label)
    if not is_multiclass_variant(variant):
        y = series("MVO", "dice")
        if not np.all(np.isnan(y)):
            axes[0, 1].plot(epochs, y, lw=2, label="MVO", ls="--")
    axes[0, 1].set_title("Validation Dice")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=9)
    for region, label in [("LV", "LV"), ("MYO", "MYO"), (mi_key, "MI/Infarct")]:
        y = series(region, "hd95")
        if np.all(np.isnan(y)):
            continue
        axes[1, 0].plot(epochs, y, lw=2, label=label)
    axes[1, 0].set_title("Validation HD95 (mm)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("HD95 (mm)")
    axes[1, 0].legend(fontsize=9)
    for region, label in [("LV", "LV"), ("MYO", "MYO"), (mi_key, "MI/Infarct")]:
        y = series(region, "recall")
        if np.all(np.isnan(y)):
            continue
        axes[1, 1].plot(epochs, y, lw=2, label=label)
    axes[1, 1].set_title("Validation Recall (sensitivity)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / f"training_curves_{variant}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_all_variants_overlay(histories: Dict[str, List[Dict]], out_dir: Path):
    """Overlay MI/Infarct Dice learning curves across ablation variants."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for v, hist in histories.items():
        key = _dice_key(v)
        epochs = [h["epoch"] for h in hist]
        y = []
        for h in hist:
            m = h.get("val", {}).get(key, {}).get("dice", {})
            y.append(m.get("mean", np.nan) if isinstance(m, dict) else np.nan)
        ax.plot(epochs, y, lw=2, color=PALETTE.get(v, None), label=VARIANT_SHORT[v])
    ax.axhline(
        TARGET_MI_DICE,
        color="gray",
        ls="--",
        lw=1.5,
        label=f"{TARGET_MI_DICE_LABEL} ({TARGET_MI_DICE})",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MI / Infarct Dice")
    ax.set_title(f"{MODEL_NAME} ablation - infarct Dice learning curves")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "ablation_learning_curves_MI.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
# ---------------------------------------------------------------------------
# Ablation / test metric bar charts
# ---------------------------------------------------------------------------

def collect_ablation_rows(split: str = "test") -> List[Dict[str, Any]]:
    rows = []
    for v in list(ABLATION_VARIANTS) + list(BASELINE_VARIANTS):
        s = load_test_summary(v, split)
        if s is None:
            continue
        mi_key = _dice_key(v)
        multi = is_multiclass_variant(v)
        rows.append(
            {
                "variant": v,
                "display": VARIANT_SHORT[v],
                "full_name": VARIANT_NAMES[v],
                "LV_dice": _metric(s, "LV", "dice"),
                "MYO_dice": _metric(s, "MYO", "dice"),
                "MI_dice": _metric(s, mi_key, "dice"),
                "MI_path_dice": _metric(s, "MI_pathological", "dice"),
                "MVO_dice": _metric(s, "MVO", "dice"),
                "LV_hd95": _metric(s, "LV", "hd95"),
                "MYO_hd95": _metric(s, "MYO", "hd95"),
                "MI_hd95": _metric(s, mi_key, "hd95"),
                "MI_recall": _metric(s, mi_key, "recall"),
                "LV_iou": _metric(s, "LV", "iou"),
                "MYO_iou": _metric(s, "MYO", "iou"),
                "MI_iou": _metric(s, mi_key, "iou"),
                "params_M": s.get("params_M"),
                "inference_ms": s.get("inference_ms_mean"),
                "LV_dice_std": _metric_std(s, "LV", "dice"),
                "MYO_dice_std": _metric_std(s, "MYO", "dice"),
                "MI_dice_std": _metric_std(s, mi_key, "dice"),
            }
        )
    return rows

def plot_grouped_bars(rows: List[Dict], metric_keys, title, ylabel, out_path: Path, ylim=None):
    if not rows:
        return None
    labels = [r["display"] for r in rows]
    x = np.arange(len(labels))
    width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4C78A8", "#54A24B", "#E45756", "#F2CF5B"]
    for i, (key, leg) in enumerate(metric_keys):
        vals = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        stds = [r.get(key + "_std") if r.get(key + "_std") is not None else 0 for r in rows]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            yerr=stds if any(stds) else None,
            capsize=3,
            label=leg,
            color=colors[i % len(colors)],
            alpha=0.9,
        )
        for b, v in zip(bars, vals):
            if v is not None and not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_sota_comparison(rows: List[Dict], out_dir: Path):
    """Bar chart: published methods + AFDD-Net (M5 if available else best ours)."""
    ours = None
    for prefer in ("M5", "M4", "M3", "M2", "M1"):
        for r in rows:
            if r["variant"] == prefer:
                ours = r
                break
        if ours:
            break
    methods = []
    lv, myo, mi = [], [], []
    for b in SOTA_BENCHMARKS:
        methods.append(f"{b['method']}\n({b['year']})")
        lv.append(b["LV"] if b["LV"] is not None else np.nan)
        myo.append(b["MYO"] if b["MYO"] is not None else np.nan)
        mi.append(b["MI"] if b["MI"] is not None else np.nan)
    if ours:
        methods.append(f"{MODEL_NAME}\n({MODEL_YEAR})")
        lv.append(ours["LV_dice"] if ours["LV_dice"] is not None else np.nan)
        myo.append(ours["MYO_dice"] if ours["MYO_dice"] is not None else np.nan)
        mi.append(ours["MI_dice"] if ours["MI_dice"] is not None else np.nan)
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, 1.15 * len(methods)), 6.0))
    ax.bar(x - width, lv, width, label="LV Dice", color="#4C78A8")
    ax.bar(x, myo, width, label="MYO Dice", color="#54A24B")
    ax.bar(x + width, mi, width, label="MI Dice", color="#E45756")
    ax.axhline(
        TARGET_MI_DICE,
        color="#E45756",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"{TARGET_MI_DICE_LABEL}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Dice Similarity Coefficient")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"EMIDEC-only comparison - {MODEL_NAME} vs published methods")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "sota_comparison_dice.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # MI-only focused figure
    n_mi = len(SOTA_BENCHMARKS) + (1 if ours and ours.get("MI_dice") is not None else 0)
    fig2, ax2 = plt.subplots(figsize=(10, max(5.0, 0.45 * n_mi)))
    mi_methods = []
    mi_vals = []
    for b in SOTA_BENCHMARKS:
        proto = b.get("protocol") or ""
        short = proto.split("(")[0].strip() if proto else ""
        label = f"{b['method']} ({b['year']})"
        if short:
            label = f"{label} [{short}]"
        mi_methods.append(label)
        mi_vals.append(b["MI"])
    if ours and ours["MI_dice"] is not None:
        mi_methods.append(MODEL_CITE)
        mi_vals.append(ours["MI_dice"])
    colors = ["#72B7B2"] * (len(mi_vals) - (1 if ours else 0)) + (["#B279A2"] if ours else [])
    bars = ax2.barh(mi_methods, mi_vals, color=colors)
    ax2.axvline(
        TARGET_MI_DICE,
        color="gray",
        ls="--",
        label=f"{TARGET_MI_DICE_LABEL} ({TARGET_MI_DICE})",
    )
    for b, v in zip(bars, mi_vals):
        ax2.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Myocardial Infarction Dice")
    ax2.set_title(f"MI Dice - {MODEL_NAME} vs verified EMIDEC results")
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "sota_comparison_MI.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    return path, path2

def plot_efficiency(rows: List[Dict], out_dir: Path):
    usable = [r for r in rows if r.get("params_M") and r.get("MI_dice") is not None]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in usable:
        ax.scatter(
            r["params_M"],
            r["MI_dice"],
            s=120,
            color=PALETTE.get(r["variant"], "#333"),
            label=r["display"],
            zorder=3,
        )
        ax.annotate(r["display"], (r["params_M"], r["MI_dice"]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("MI / Infarct Dice (test)")
    ax.set_title(f"{MODEL_NAME} - accuracy vs model size")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "efficiency_params_vs_dice.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_radar_m5(rows: List[Dict], out_dir: Path):
    ours = next((r for r in rows if r["variant"] == "M5"), None)
    if ours is None:
        ours = next((r for r in rows if r["variant"] in ("M4", "M3")), None)
    if ours is None:
        return None
    labels = ["LV Dice", "MYO Dice", "MI Dice", "MI Recall", "1/(1+HD95/50)"]
    mi_hd = ours.get("MI_hd95") or 50
    values = [
        ours.get("LV_dice") or 0,
        ours.get("MYO_dice") or 0,
        ours.get("MI_dice") or 0,
        ours.get("MI_recall") or 0,
        1.0 / (1.0 + float(mi_hd) / 50.0),
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_c, values_c, color=PALETTE["M5"], lw=2)
    ax.fill(angles_c, values_c, color=PALETTE["M5"], alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"{MODEL_NAME} metric profile", y=1.08, fontweight="bold")
    path = out_dir / "metric_radar_AFDDNet.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
# ---------------------------------------------------------------------------
# Tables (JSON / CSV / Markdown)
# ---------------------------------------------------------------------------

def write_tables(rows: List[Dict], out_res: Path, split: str):
    # Ablation CSV
    abl_csv = out_res / "ablation_metrics.csv"
    fields = [
        "variant", "display", "full_name",
        "LV_dice", "MYO_dice", "MI_dice", "MI_path_dice", "MVO_dice",
        "LV_hd95", "MYO_hd95", "MI_hd95", "MI_recall",
        "LV_iou", "MYO_iou", "MI_iou",
        "params_M", "inference_ms",
    ]
    with abl_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # SOTA comparison CSV
    sota_csv = out_res / "sota_comparison.csv"
    with sota_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Method", "Citation", "Year", "Architecture",
            "LV Dice", "MYO Dice", "MI Dice", "MVO Dice", "Dataset", "Protocol",
        ])
        for b in SOTA_BENCHMARKS:
            w.writerow([
                b["method"], b["citation"], b["year"], b["architecture"],
                b["LV"], b["MYO"], b["MI"], b["MVO"], b["dataset"], b.get("protocol", ""),
            ])
        ours = next((r for r in rows if r["variant"] == "M5"), None) or (rows[-1] if rows else None)
        if ours:
            w.writerow([
                MODEL_NAME, "this work", MODEL_YEAR, MODEL_FULL_NAME,
                ours.get("LV_dice"), ours.get("MYO_dice"), ours.get("MI_dice"),
                ours.get("MVO_dice"), "EMIDEC", f"Stratified {split} split",
            ])
    # Markdown report
    md = out_res / "PAPER_COMPARISON.md"
    lines = [
        f"# {MODEL_NAME} - Paper Comparison Report",
        "",
        f"**Full name:** {MODEL_FULL_NAME}",
        "",
        f"**Dataset:** EMIDEC only (non-EMIDEC papers excluded from Table 4.7)",
        f"**Split evaluated:** `{split}`",
        f"**Primary target:** MI Dice > {TARGET_MI_DICE} ({TARGET_MI_DICE_LABEL})",
        f"**Stretch target:** MI Dice > 0.783 (ICPIU-Net 5-fold CV)",
        "",
        "> Comparison uses verified EMIDEC MI/scar Dice only. "
        "Isensee et al. 2021 nnU-Net (private LGE cohort) and non-EMIDEC 2025–2026 papers are excluded. "
        "Protocols differ (official test vs 5-fold CV); interpret MI Dice accordingly.",
        "",
        "## Ablation study (methodology Table 4.5 / 4.6)",
        "",
        "| Variant | Model | LV Dice | MYO Dice | **MI_path** | MI (all) | MVO Dice | Params (M) | Infer (ms) |",
        "|---------|-------|--------:|---------:|-----------:|---------:|---------:|-----------:|-----------:|",
    ]
    for r in rows:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "-"
        lines.append(
            f"| {r['variant']} | {r['display']} | {fmt(r['LV_dice'])} | {fmt(r['MYO_dice'])} | "
            f"**{fmt(r.get('MI_path_dice'))}** | {fmt(r['MI_dice'])} | {fmt(r['MVO_dice'])} | "
            f"{fmt(r['params_M']) if r.get('params_M') else '-'} | "
            f"{fmt(r['inference_ms']) if r.get('inference_ms') else '-'} |"
        )
    lines += [
        "",
        "> **PRIMARY: MI_path** = pure MI Dice (EMIDEC label 3) on pathological cases only. "
        "Multiclass models (M1/M2/baselines) now predict MI and MVO as separate classes — "
        "not merged infarct. MI_all includes healthy empty–empty = 1.0; do not cite as scar metric. "
        "NNUNET row is real nnU-Net v2; DYNUNET_RES is MONAI residual DynUNet. "
        "**MI (all)** is secondary. Disease classifier (M5 only) + voxel suppression reduce healthy FPs. "
        "Train best is on **val**; this table is **test**.",
        "",
        "## Comparison with state-of-the-art (methodology Table 4.7, EMIDEC-only)",
        "",
        "| Method | Year | Protocol | MYO Dice | MI Dice |",
        "|--------|-----:|----------|---------:|--------:|",
    ]
    for b in SOTA_BENCHMARKS:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "-"
        lines.append(
            f"| {b['method']} ({b['citation']}) | {b['year']} | "
            f"{b.get('protocol', '-')} | {fmt(b['MYO'])} | {fmt(b['MI'])} |"
        )
    if rows:
        ours = next((r for r in rows if r["variant"] == "M5"), rows[-1])
        if ours.get("MI_dice") is not None:
            mi_path = ours.get("MI_path_dice")
            mi_path_s = f"{mi_path:.3f}" if isinstance(mi_path, float) else "-"
            lines.append(
                f"| **{MODEL_NAME} (this work)** | {MODEL_YEAR} | "
                f"Stratified {split} split | "
                f"**{ours['MYO_dice']:.3f}** | **{ours['MI_dice']:.3f}** "
                f"(MI_path **{mi_path_s}**) |"
            )
        else:
            lines.append(
                f"| **{MODEL_NAME} (this work)** | {MODEL_YEAR} | "
                f"Stratified {split} split | - | - |"
            )
    lines += [
        "",
        "## Notes on protocol fairness",
        "",
        "- Official test (50 cases): Zhang 0.712, ICPIU-Net 0.734",
        "- 5-fold CV (100 cases): Schwab 0.760 (current best), ICPIU-Net 0.783",
        "- Expert inter-observer MI Dice on EMIDEC: 0.69 (honest ceiling reference)",
        "",
        "## Recommended thesis comparison paragraph",
        "",
        "> Quantitative comparison was performed exclusively on the EMIDEC dataset "
        "(Lalande et al., 2020), the standard benchmark for LGE-MRI myocardial infarction "
        "segmentation. Methods were compared using Dice Similarity Coefficient (DSC) for "
        "left ventricle myocardium (MYO) and myocardial infarction (MI), following the "
        "official EMIDEC evaluation protocol. On the EMIDEC test set, the challenge-winning "
        "cascaded 2D–3D nnU-Net (Zhang, 2021) achieved MYO DSC of 0.879 and MI DSC of 0.712. "
        "ICPIU-Net (Brahim et al., 2022) improved MI DSC to 0.734. The current best published "
        "result using 5-fold cross-validation is Schwab et al. (2025) with MI DSC of 0.760. "
        "For context, expert inter-observer MI DSC on EMIDEC is 0.69. Our models are compared "
        "on a stratified EMIDEC split using the same structures (LV, MYO, MI, MVO).",
        "",
        "## How to cite this model",
        "",
        f"> {MODEL_NAME}: {MODEL_FULL_NAME}. "
        "Joint anatomy (LV, MYO) and pathology (MI, MVO) segmentation on EMIDEC LGE-MRI "
        "using anisotropic factorized 3D convolutions, dual-decoder MYO soft gating, "
        "Focal Tversky loss, and topology consistency loss.",
        "",
        "## Figures",
        "",
        "All graphs are saved under `figures/paper/`.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "year": MODEL_YEAR,
        "split": split,
        "target_mi_dice": TARGET_MI_DICE,
        "target_mi_dice_label": TARGET_MI_DICE_LABEL,
        "ablation": rows,
        "sota": SOTA_BENCHMARKS,
        "note": (
            "EMIDEC-only SOTA table. Removed Isensee 2021 (private dataset). "
            "Schwab MYO/MI corrected to 0.860/0.760 (5-fold). "
            "ICPIU listed as 0.734 (test) and 0.783 (5-fold)."
        ),
    }
    (out_res / "paper_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return abl_csv, sota_csv, md

def render_table_figure(rows: List[Dict], out_dir: Path):
    """Render ablation + SOTA tables as PNG for thesis slides."""
    fig, ax = plt.subplots(figsize=(14, 3 + 0.45 * max(len(rows), 1)))
    ax.axis("off")
    col_labels = ["Variant", "Model", "LV", "MYO", "MI", "MVO", "Params(M)", "ms"]
    cell = []
    for r in rows:
        def fmt(x, nd=3):
            return f"{x:.{nd}f}" if isinstance(x, float) else "-"
        cell.append([
            r["variant"], r["display"],
            fmt(r["LV_dice"]), fmt(r["MYO_dice"]), fmt(r["MI_dice"]), fmt(r["MVO_dice"]),
            fmt(r["params_M"], 2) if r.get("params_M") else "-",
            fmt(r["inference_ms"], 1) if r.get("inference_ms") else "-",
        ])
    if not cell:
        cell = [["-"] * len(col_labels)]
    table = ax.table(cellText=cell, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(f"{MODEL_NAME} ablation metrics (test)", fontweight="bold", pad=20)
    path = out_dir / "table_ablation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # SOTA table figure
    n_sota = len(SOTA_BENCHMARKS) + (1 if rows else 0)
    fig2, ax2 = plt.subplots(figsize=(14, max(3.8, 0.42 * n_sota + 1.2)))
    ax2.axis("off")
    scol = ["Method", "Year", "Protocol", "MYO Dice", "MI Dice"]
    scell = []
    for b in SOTA_BENCHMARKS:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "-"
        scell.append([
            b["method"],
            str(b["year"]),
            b.get("protocol", "-"),
            fmt(b["MYO"]),
            fmt(b["MI"]),
        ])
    ours = next((r for r in rows if r["variant"] == "M5"), rows[-1] if rows else None)
    if ours:
        scell.append([
            MODEL_NAME,
            str(MODEL_YEAR),
            "Stratified test split",
            f"{ours['MYO_dice']:.3f}" if ours.get("MYO_dice") is not None else "-",
            f"{ours['MI_dice']:.3f}" if ours.get("MI_dice") is not None else "-",
        ])
    t2 = ax2.table(cellText=scell, colLabels=scol, loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.35)
    ax2.set_title(f"EMIDEC-only SOTA comparison including {MODEL_NAME}", fontweight="bold", pad=20)
    path2 = out_dir / "table_sota.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    return path, path2
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(split: str = "test") -> Dict[str, Any]:
    fig_dir, res_dir = _ensure_dirs()
    saved = []
    histories = {}
    for v in list(ABLATION_VARIANTS) + list(BASELINE_VARIANTS):
        h = load_history(v)
        if h:
            histories[v] = h
            saved.append(str(plot_training_curves(v, h, fig_dir)))
    if histories:
        # Ablation overlay (M1-M5 only) + separate baseline overlay when present
        ab_hist = {k: v for k, v in histories.items() if k in ABLATION_VARIANTS}
        if ab_hist:
            saved.append(str(plot_all_variants_overlay(ab_hist, fig_dir)))
        bl_hist = {k: v for k, v in histories.items() if k in BASELINE_VARIANTS}
        if bl_hist:
            fig, ax = plt.subplots(figsize=(10, 5))
            for v, hist in bl_hist.items():
                key = _dice_key(v)
                epochs = [h["epoch"] for h in hist]
                y = []
                for h in hist:
                    m = h.get("val", {}).get(key, {}).get("dice", {})
                    y.append(m.get("mean", np.nan) if isinstance(m, dict) else np.nan)
                ax.plot(epochs, y, lw=2, color=PALETTE.get(v, None), label=VARIANT_SHORT[v])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Infarct Dice")
            ax.set_title("External baselines - infarct Dice learning curves")
            ax.legend(fontsize=9)
            fig.tight_layout()
            bp = fig_dir / "baseline_learning_curves_MI.png"
            fig.savefig(bp, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(bp))
    rows = collect_ablation_rows(split)
    if rows:
        saved.append(str(plot_grouped_bars(
            rows,
            [("LV_dice", "LV"), ("MYO_dice", "MYO"), ("MI_dice", "MI"), ("MVO_dice", "MVO")],
            f"{MODEL_NAME} ablation - Dice ({split})",
            "Dice",
            fig_dir / "ablation_dice.png",
            ylim=(0, 1.05),
        )))
        saved.append(str(plot_grouped_bars(
            rows,
            [("LV_hd95", "LV"), ("MYO_hd95", "MYO"), ("MI_hd95", "MI")],
            f"{MODEL_NAME} ablation - HD95 ({split})",
            "HD95 (mm)",
            fig_dir / "ablation_hd95.png",
        )))
        saved.append(str(plot_grouped_bars(
            rows,
            [("MI_recall", "MI Recall")],
            f"{MODEL_NAME} ablation - infarct recall ({split})",
            "Recall",
            fig_dir / "ablation_recall.png",
            ylim=(0, 1.05),
        )))
        saved.append(str(plot_grouped_bars(
            rows,
            [("LV_iou", "LV"), ("MYO_iou", "MYO"), ("MI_iou", "MI")],
            f"{MODEL_NAME} ablation - IoU ({split})",
            "IoU",
            fig_dir / "ablation_iou.png",
            ylim=(0, 1.05),
        )))
        p1, p2 = plot_sota_comparison(rows, fig_dir)
        saved.extend([str(p1), str(p2)])
        eff = plot_efficiency(rows, fig_dir)
        if eff:
            saved.append(str(eff))
        rad = plot_radar_m5(rows, fig_dir)
        if rad:
            saved.append(str(rad))
        t1, t2 = render_table_figure(rows, fig_dir)
        saved.extend([str(t1), str(t2)])
    csvs = write_tables(rows, res_dir, split)
    saved.extend([str(p) for p in csvs])
    # Master index
    index = {
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "split": split,
        "figures": [s for s in saved if s.endswith(".png")],
        "tables": [s for s in saved if s.endswith((".csv", ".md", ".json"))],
        "ablation_rows": rows,
    }
    index_path = res_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    index = generate_all(args.split)
    print(f"\n{MODEL_NAME} paper figures generated")
    print(f"  Figures: {cfg.FIGURES_DIR / 'paper'}")
    print(f"  Tables:  {cfg.RESULTS_DIR / 'paper'}")
    for p in index.get("figures", []):
        print(f"    - {p}")
    for p in index.get("tables", []):
        print(f"    - {p}")
if __name__ == "__main__":
    main()
