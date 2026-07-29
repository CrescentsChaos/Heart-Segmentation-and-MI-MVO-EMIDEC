"""Generate M5 (AFDD-Net) segmentation images for selected patients.

Produces per-patient LV / MYO / MI / MVO overlay figures and a combined
gallery.  By default picks a few normal + pathological representative cases;
use --patients for manual selection.

For 5-fold CV checkpoints the script automatically picks the fold checkpoint
where each patient was in the *test* set (no data leakage).

Usage:
  # Auto-select 3 normal + 5 pathological patients
  python -m src.generate_segmentations

  # Manual patient list
  python -m src.generate_segmentations --patients P001 P087 N006 P044

  # Pick more/fewer auto patients
  python -m src.generate_segmentations --n-normal 4 --n-patho 6

  # Use CPU
  python -m src.generate_segmentations --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from data.cv_splits import load_folds
from inference import postprocess_pathology
from model_identity import MODEL_NAME, VARIANT_SHORT, is_multiclass_variant
from models.dual_decoder import build_model
from train import _load_state_dict
from visualize_patient import (
    COLORS,
    _legend_handles,
    _rgb_overlay,
    compute_dysfunction_stats,
    find_case_npz,
    load_clinical,
    normalize_case_id,
)


# ─────────────────────────── fold-aware checkpoint ────────────────────────── #

def _case_to_fold(case_id: str) -> Optional[int]:
    """Return the fold index where *case_id* was in the test set, or None."""
    try:
        folds_meta = load_folds()
    except FileNotFoundError:
        return None
    for fold_idx, fold_cases in enumerate(folds_meta["folds"]):
        if case_id in fold_cases:
            return fold_idx
    return None


def _pick_checkpoint(case_id: str, variant: str = "M5") -> Tuple[str, Path, Optional[int]]:
    """Pick the fold checkpoint where *case_id* was held-out (test).

    Falls back to fold0 if the patient isn't in any fold (shouldn't happen
    with official EMIDEC), then to the non-fold best checkpoint.
    """
    v = variant.upper()
    fold = _case_to_fold(case_id)
    if fold is not None:
        ckpt = cfg.CHECKPOINT_DIR / f"{v}_fold{fold}_best.pth"
        if ckpt.exists():
            return v, ckpt, fold
    # fallback: try each fold checkpoint
    for f in range(5):
        ckpt = cfg.CHECKPOINT_DIR / f"{v}_fold{f}_best.pth"
        if ckpt.exists():
            return v, ckpt, f
    # final fallback: non-fold checkpoint
    ckpt = cfg.CHECKPOINT_DIR / f"{v}_best.pth"
    if ckpt.exists():
        return v, ckpt, None
    raise FileNotFoundError(
        f"No M5 checkpoint found in {cfg.CHECKPOINT_DIR}.  "
        f"Expected M5_foldK_best.pth or M5_best.pth."
    )


# ──────────────────────────── auto patient pick ───────────────────────────── #

def _auto_select(n_normal: int = 3, n_patho: int = 5, seed: int = 42) -> List[str]:
    """Pick a stratified random sample from all available cases."""
    import random

    all_dir = cfg.DATASET_DIR / "all"
    if not all_dir.exists():
        all_dir = cfg.DATASET_DIR / "test"
    cases = sorted(p.stem for p in all_dir.glob("Case_*.npz"))
    normals = [c for c in cases if "_N" in c]
    pathos = [c for c in cases if "_P" in c]
    rng = random.Random(seed)
    rng.shuffle(normals)
    rng.shuffle(pathos)
    picked = normals[:n_normal] + pathos[:n_patho]
    return sorted(picked)


# ───────────────────────────── single-case inference ──────────────────────── #

@torch.no_grad()
def _predict_case(
    case_id: str,
    variant: str,
    ckpt: Path,
    device: torch.device,
) -> Dict:
    npz_path = find_case_npz(case_id)
    data = np.load(npz_path)
    image_hw_d = data["image"].astype(np.float32)  # (H, W, D)
    gt_anatomy = data["anatomy"].astype(np.int64) if "anatomy" in data.files else None
    gt_pathology = data["pathology"].astype(np.float32) if "pathology" in data.files else None

    # (1, 1, D, H, W)
    x = (
        torch.from_numpy(image_hw_d)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    model = build_model(
        variant,
        filters=tuple(cfg.BASE_FILTERS),
        use_disease_classifier=(
            variant.upper() == "M5"
            and getattr(cfg, "USE_DISEASE_CLASSIFIER", True)
        ),
        gate_pathology_by_disease=(
            variant.upper() == "M5"
            and getattr(cfg, "GATE_PATHOLOGY_BY_DISEASE", True)
        ),
        disease_threshold=getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    _load_state_dict(model, state["model"], strict=False)
    model.eval()

    out = model(x)
    if is_multiclass_variant(variant):
        mi_cls = int(getattr(cfg, "MULTICLASS_MI", 3))
        mvo_cls = int(getattr(cfg, "MULTICLASS_MVO", 4))
        multi = out["multiclass_logits"].argmax(1)[0]
        if getattr(cfg, "MI_VOXEL_SUPPRESSION", True):
            thr = int(getattr(cfg, "MIN_MI_VOXELS", 50))
            mi_vox = multi == mi_cls
            if mi_vox.sum() < thr:
                multi = multi.clone()
                multi[mi_vox] = 2
        multi = multi.cpu().numpy()
        anatomy = np.zeros_like(multi, dtype=np.uint8)
        anatomy[multi == 1] = 1
        anatomy[np.isin(multi, [2, mi_cls, mvo_cls])] = 2
        pathology = np.zeros((2,) + multi.shape, dtype=np.uint8)
        pathology[0] = (multi == mi_cls).astype(np.uint8)
        pathology[1] = (multi == mvo_cls).astype(np.uint8)
    else:
        anatomy = out["anatomy_logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        pathology = (
            postprocess_pathology(out, hard_mask=True)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    vol = x[0, 0].cpu().numpy()  # (D, H, W)
    return {
        "image": vol,
        "anatomy": anatomy,
        "pathology": pathology,
        "gt_anatomy": (
            None if gt_anatomy is None else np.transpose(gt_anatomy, (2, 0, 1))
        ),
        "gt_pathology": (
            None
            if gt_pathology is None
            else np.transpose(gt_pathology, (3, 2, 0, 1))
        ),
        "variant": variant,
        "checkpoint": str(ckpt),
        "npz_path": str(npz_path),
    }


# ──────────────────────────── per-patient figure ──────────────────────────── #

def _draw_patient_panel(
    case_id: str,
    pred: Dict,
    stats: Dict,
    clinical: Dict,
    out_dir: Path,
    all_slices: bool = False,
) -> List[Path]:
    """Generate a detailed per-patient segmentation figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = pred["image"]  # (D, H, W)
    anatomy = pred["anatomy"]
    pathology = pred["pathology"]
    gt_anatomy = pred["gt_anatomy"]
    gt_pathology = pred["gt_pathology"]
    D = image.shape[0]
    mid = D // 2

    is_patho = "_P" in case_id
    type_tag = "Pathological" if is_patho else "Normal"
    title_variant = VARIANT_SHORT.get(pred["variant"], pred["variant"])

    # ── Main 6-panel figure: GT vs Prediction ──
    has_gt = gt_anatomy is not None and gt_pathology is not None
    ncols = 3 if has_gt else 3
    nrows = 2 if has_gt else 1

    fig = plt.figure(
        figsize=(5 * ncols, 4.5 * nrows + 0.8), facecolor="white"
    )

    if has_gt:
        gs = fig.add_gridspec(
            nrows, ncols, hspace=0.30, wspace=0.12,
            top=0.90, bottom=0.02, left=0.02, right=0.98,
        )
        # Row 0: Ground Truth
        ax_gt_lge = fig.add_subplot(gs[0, 0])
        ax_gt_lge.imshow(image[mid], cmap="gray")
        ax_gt_lge.set_title("LGE MRI", fontsize=11)
        ax_gt_lge.axis("off")

        ax_gt_anat = fig.add_subplot(gs[0, 1])
        gt_anat_rgb = np.zeros((*gt_anatomy[mid].shape, 3))
        gt_anat_rgb[gt_anatomy[mid] == 1] = COLORS["LV"]
        gt_anat_rgb[gt_anatomy[mid] == 2] = COLORS["MYO"]
        ax_gt_anat.imshow(image[mid], cmap="gray")
        ax_gt_anat.imshow(gt_anat_rgb, alpha=0.45)
        ax_gt_anat.set_title("GT: LV + MYO", fontsize=11)
        ax_gt_anat.axis("off")

        ax_gt_full = fig.add_subplot(gs[0, 2])
        ax_gt_full.imshow(
            _rgb_overlay(image[mid], gt_anatomy[mid], gt_pathology[:, mid])
        )
        ax_gt_full.set_title("GT: Full Overlay", fontsize=11)
        ax_gt_full.axis("off")

        # Row 1: Prediction
        ax_pred_lv = fig.add_subplot(gs[1, 0])
        lv_rgb = np.zeros((*anatomy[mid].shape, 3))
        lv_rgb[anatomy[mid] == 1] = COLORS["LV"]
        ax_pred_lv.imshow(image[mid], cmap="gray")
        ax_pred_lv.imshow(lv_rgb, alpha=0.50)
        ax_pred_lv.set_title("Pred: LV Cavity", fontsize=11)
        ax_pred_lv.axis("off")

        ax_pred_myo = fig.add_subplot(gs[1, 1])
        myo_rgb = np.zeros((*anatomy[mid].shape, 3))
        myo_rgb[anatomy[mid] == 2] = COLORS["MYO"]
        # overlay MI + MVO on the MYO panel for context
        myo_rgb[pathology[0, mid].astype(bool)] = COLORS["MI"]
        myo_rgb[pathology[1, mid].astype(bool)] = COLORS["MVO"]
        ax_pred_myo.imshow(image[mid], cmap="gray")
        ax_pred_myo.imshow(myo_rgb, alpha=0.50)
        ax_pred_myo.set_title("Pred: MYO + MI + MVO", fontsize=11)
        ax_pred_myo.axis("off")

        ax_pred_full = fig.add_subplot(gs[1, 2])
        ax_pred_full.imshow(
            _rgb_overlay(image[mid], anatomy[mid], pathology[:, mid])
        )
        ax_pred_full.set_title("Pred: Full Overlay", fontsize=11)
        ax_pred_full.axis("off")
        ax_pred_full.legend(
            handles=_legend_handles(), loc="lower right", fontsize=7, framealpha=0.85
        )
    else:
        gs = fig.add_gridspec(
            1, ncols, hspace=0.30, wspace=0.12,
            top=0.88, bottom=0.02, left=0.02, right=0.98,
        )
        ax_lge = fig.add_subplot(gs[0, 0])
        ax_lge.imshow(image[mid], cmap="gray")
        ax_lge.set_title("LGE MRI", fontsize=11)
        ax_lge.axis("off")

        ax_anat = fig.add_subplot(gs[0, 1])
        anat_rgb = np.zeros((*anatomy[mid].shape, 3))
        anat_rgb[anatomy[mid] == 1] = COLORS["LV"]
        anat_rgb[anatomy[mid] == 2] = COLORS["MYO"]
        ax_anat.imshow(image[mid], cmap="gray")
        ax_anat.imshow(anat_rgb, alpha=0.45)
        ax_anat.set_title("Pred: LV + MYO", fontsize=11)
        ax_anat.axis("off")

        ax_full = fig.add_subplot(gs[0, 2])
        ax_full.imshow(_rgb_overlay(image[mid], anatomy[mid], pathology[:, mid]))
        ax_full.set_title("Pred: Full Overlay", fontsize=11)
        ax_full.axis("off")
        ax_full.legend(
            handles=_legend_handles(), loc="lower right", fontsize=7, framealpha=0.85
        )

    # suptitle with patient info
    mi_pct = stats["percent_of_MYO"]["MI"]
    mvo_pct = stats["percent_of_MYO"]["MVO"]
    dys_pct = stats["percent_of_MYO"]["dysfunction"]
    subtitle = (
        f"MI={mi_pct:.1f}% MVO={mvo_pct:.1f}% Dysfunction={dys_pct:.1f}% of MYO"
    )
    fig.suptitle(
        f"{MODEL_NAME} ({title_variant})  —  {case_id}  [{type_tag}]\n{subtitle}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    panel_path = out_dir / f"{case_id}_segmentation.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved = [panel_path]

    # ── All-slices montage (optional) ──
    if all_slices:
        cols = min(4, D)
        rows = int(np.ceil(D / cols))
        fig2, axes = plt.subplots(
            rows, cols, figsize=(4 * cols, 4 * rows), facecolor="white"
        )
        axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            if i < D:
                ax.imshow(_rgb_overlay(image[i], anatomy[i], pathology[:, i]))
                ax.set_title(f"z={i}", fontsize=9)
            ax.axis("off")
        fig2.suptitle(f"{case_id} — all slices", fontsize=12, fontweight="bold")
        mont_path = out_dir / f"{case_id}_allslices.png"
        fig2.savefig(mont_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        saved.append(mont_path)

    return saved


# ────────────────────────────── gallery figure ────────────────────────────── #

def _draw_gallery(
    patients: List[Dict],
    out_dir: Path,
) -> Path:
    """Combined gallery: one row per patient, columns = LGE | GT | Prediction."""
    n = len(patients)
    has_gt_any = any(p["pred"]["gt_anatomy"] is not None for p in patients)
    ncols = 3 if has_gt_any else 2
    fig, axes = plt.subplots(
        n, ncols, figsize=(5.2 * ncols, 4 * n + 0.6), facecolor="white",
        squeeze=False,
    )

    for row, p in enumerate(patients):
        case_id = p["case_id"]
        pred = p["pred"]
        image = pred["image"]
        anatomy = pred["anatomy"]
        pathology = pred["pathology"]
        gt_anatomy = pred["gt_anatomy"]
        gt_pathology = pred["gt_pathology"]
        mid = image.shape[0] // 2
        is_patho = "_P" in case_id
        type_tag = "P" if is_patho else "N"

        # Column 0: raw LGE
        axes[row, 0].imshow(image[mid], cmap="gray")
        axes[row, 0].set_ylabel(
            f"{case_id}\n[{type_tag}]", fontsize=10, fontweight="bold", rotation=0,
            labelpad=60, va="center",
        )
        if row == 0:
            axes[row, 0].set_title("LGE MRI", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        if has_gt_any:
            # Column 1: GT overlay
            if gt_anatomy is not None and gt_pathology is not None:
                axes[row, 1].imshow(
                    _rgb_overlay(image[mid], gt_anatomy[mid], gt_pathology[:, mid])
                )
            else:
                axes[row, 1].imshow(image[mid], cmap="gray")
                axes[row, 1].text(
                    0.5, 0.5, "No GT", ha="center", va="center",
                    transform=axes[row, 1].transAxes, fontsize=12, color="white",
                )
            if row == 0:
                axes[row, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
            axes[row, 1].axis("off")

            # Column 2: Prediction
            axes[row, 2].imshow(
                _rgb_overlay(image[mid], anatomy[mid], pathology[:, mid])
            )
            if row == 0:
                axes[row, 2].set_title(
                    f"{MODEL_NAME} Prediction", fontsize=12, fontweight="bold",
                )
            axes[row, 2].axis("off")
        else:
            # Column 1: Prediction
            axes[row, 1].imshow(
                _rgb_overlay(image[mid], anatomy[mid], pathology[:, mid])
            )
            if row == 0:
                axes[row, 1].set_title(
                    f"{MODEL_NAME} Prediction", fontsize=12, fontweight="bold",
                )
            axes[row, 1].axis("off")

    # shared legend at bottom
    legend_ax = axes[-1, -1]
    legend_ax.legend(
        handles=_legend_handles(), loc="lower right", fontsize=9, framealpha=0.9,
    )

    fig.suptitle(
        f"{MODEL_NAME} (M5) — Segmentation Gallery",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    gallery_path = out_dir / "segmentation_gallery.png"
    fig.savefig(gallery_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return gallery_path


# ──────────────────────────────── CLI entry ───────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate M5 (AFDD-Net) segmentation images for selected patients. "
            "Uses the fold checkpoint where each patient was in the held-out test set."
        ),
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Patient IDs (e.g. P001 N006 Case_P087). "
        "If omitted, auto-selects a few normal + pathological.",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=3,
        help="Number of normal patients to auto-select (default: 3)",
    )
    parser.add_argument(
        "--n-patho",
        type=int,
        default=5,
        help="Number of pathological patients to auto-select (default: 5)",
    )
    parser.add_argument(
        "--variant",
        default="M5",
        help="Model variant (default: M5)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu (default: auto)",
    )
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="Also generate full-slice montage per patient",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: figures/segmentations/)",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Skip combined gallery figure",
    )
    args = parser.parse_args()

    # ── resolve device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info()
                if free < 1.5 * 1024**3:
                    print(f"GPU memory low ({free / 1024**3:.1f} GB free) — using CPU")
                    device = torch.device("cpu")
            except Exception:
                pass

    # ── resolve patient list ──
    if args.patients:
        case_ids = [normalize_case_id(p) for p in args.patients]
    else:
        case_ids = _auto_select(n_normal=args.n_normal, n_patho=args.n_patho)

    out_dir = Path(args.out) if args.out else cfg.FIGURES_DIR / "segmentations"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant.upper()

    print(f"{'=' * 65}")
    print(f"  {MODEL_NAME} Segmentation Generator")
    print(f"  Variant : {VARIANT_SHORT.get(variant, variant)} ({variant})")
    print(f"  Device  : {device}")
    print(f"  Patients: {len(case_ids)}")
    print(f"  Output  : {out_dir}")
    print(f"{'=' * 65}")

    all_patient_data: List[Dict] = []
    all_saved: List[Path] = []

    for i, case_id in enumerate(case_ids, 1):
        is_patho = "_P" in case_id
        type_tag = "Pathological" if is_patho else "Normal"

        try:
            v, ckpt, fold = _pick_checkpoint(case_id, variant)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(case_ids)}] SKIP {case_id}: {e}")
            continue

        fold_tag = f"fold{fold}" if fold is not None else "single"
        print(
            f"\n  [{i}/{len(case_ids)}] {case_id}  [{type_tag}]"
            f"  ckpt={ckpt.name}  ({fold_tag})"
        )

        pred = _predict_case(case_id, v, ckpt, device)
        stats = compute_dysfunction_stats(pred["anatomy"], pred["pathology"])
        clinical = load_clinical(case_id)

        # per-patient figures
        patient_dir = out_dir / case_id
        saved = _draw_patient_panel(
            case_id, pred, stats, clinical, patient_dir,
            all_slices=args.all_slices,
        )
        all_saved.extend(saved)

        # JSON stats sidecar
        report = {
            "case": case_id,
            "type": type_tag,
            "model": MODEL_NAME,
            "variant": v,
            "fold": fold,
            "checkpoint": str(ckpt),
            "dysfunction_stats": stats,
            "clinical": {k: v for k, v in clinical.items() if not k.startswith("_")},
            "figures": [str(p) for p in saved],
        }
        json_path = patient_dir / f"{case_id}_stats.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        all_saved.append(json_path)

        all_patient_data.append({
            "case_id": case_id,
            "pred": pred,
            "stats": stats,
        })

        print(
            f"    MI={stats['percent_of_MYO']['MI']:.1f}%  "
            f"MVO={stats['percent_of_MYO']['MVO']:.1f}%  "
            f"Dysfunction={stats['percent_of_MYO']['dysfunction']:.1f}% of MYO"
        )
        for s in saved:
            print(f"    → {s}")

    # ── combined gallery ──
    if all_patient_data and not args.no_gallery:
        gallery_path = _draw_gallery(all_patient_data, out_dir)
        all_saved.append(gallery_path)
        print(f"\n  Gallery → {gallery_path}")

    # ── summary index ──
    index = {
        "model": MODEL_NAME,
        "variant": variant,
        "patients": [p["case_id"] for p in all_patient_data],
        "files": [str(p) for p in all_saved],
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"\n{'=' * 65}")
    print(f"  Done — {len(all_saved)} files saved to {out_dir}")
    print(f"  Index → {index_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
