"""Patient-level segmentation figures + dysfunction percentages.
Usage examples:
  python -m src.visualize_patient --case P001
  python -m src.visualize_patient --case Case_P087 --variant M5
  python -m src.visualize_patient --case N006 --all-slices
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from inference import postprocess_pathology
from model_identity import MODEL_NAME, VARIANT_SHORT, is_multiclass_variant
from models.dual_decoder import build_model
from train import _load_state_dict

# ---------------------------- Case resolution --------------------------------

def normalize_case_id(case: str) -> str:
    c = case.strip().replace(" ", "_")
    if c.upper().startswith("CASE_"):
        c = c[5:]
    if c.upper().startswith("CASE"):
        c = c[4:].lstrip("_")
    # Accept P001 / N006 / Case_P001
    if "_" not in c and len(c) >= 2 and c[0].upper() in ("P", "N"):
        return f"Case_{c[0].upper()}{c[1:]}"
    if not c.startswith("Case_"):
        return f"Case_{c}"
    return c

def find_case_npz(case_id: str) -> Path:
    name = f"{case_id}.npz"
    for split in ("all", "test", "val", "train"):
        p = cfg.DATASET_DIR / split / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No preprocessed volume for {case_id}. "
        f"Expected under Dataset/{{all,train,val,test}}/{name}"
    )

def load_clinical(case_id: str) -> Dict[str, str]:
    """Load EMIDEC clinical text if present (workspace or EMIDEC root)."""
    candidates = [
        cfg.ROOT / "Dataset" / f"{case_id.replace('Case_', 'Case ')}.txt",
        cfg.ROOT / "Dataset" / f"{case_id}.txt",
        cfg.EMIDEC_ROOT / f"{case_id.replace('_', ' ')}.txt",
        cfg.EMIDEC_ROOT / f"{case_id}.txt",
    ]
    # Also match "Case P001.txt" style in Dataset/
    short = case_id.replace("Case_", "")
    candidates.append(cfg.ROOT / "Dataset" / f"Case {short}.txt")
    candidates.append(cfg.EMIDEC_ROOT / f"Case {short}.txt")
    info: Dict[str, str] = {}
    for path in candidates:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
            info["_source"] = str(path)
            break
    return info

def pick_checkpoint(variant: Optional[str] = None) -> Tuple[str, Path]:
    """Prefer requested variant; else best available by validation Dice."""
    if variant:
        v = variant.upper()
        path = cfg.CHECKPOINT_DIR / f"{v}_best.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        return v, path
    best_v, best_p, best_score = None, None, -1.0
    for v in ("M5", "M4", "M3", "M2", "M1"):
        path = cfg.CHECKPOINT_DIR / f"{v}_best.pth"
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", {})
            key = "MI" if "MI" in metrics else "Infarct"
            score = float(metrics.get(key, {}).get("dice", {}).get("mean", -1.0))
        except Exception:
            score = -1.0
        if score > best_score:
            best_v, best_p, best_score = v, path, score
    if best_v is None:
        raise FileNotFoundError(f"No checkpoints in {cfg.CHECKPOINT_DIR}")
    print(f"Auto-selected {best_v} (val Dice={best_score:.3f})")
    return best_v, best_p

# ---------------------------- Inference --------------------------------------
@torch.no_grad()

def predict_case(case_id: str, variant: str, ckpt: Path, device: torch.device):
    npz_path = find_case_npz(case_id)
    data = np.load(npz_path)
    image_hw_d = data["image"].astype(np.float32)  # (H,W,D)
    gt_anatomy = data["anatomy"].astype(np.int64) if "anatomy" in data.files else None
    gt_pathology = data["pathology"].astype(np.float32) if "pathology" in data.files else None
    gt_multi = data["multiclass"].astype(np.int64) if "multiclass" in data.files else None
    # (1,1,D,H,W)
    x = torch.from_numpy(image_hw_d).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
    model = build_model(
        variant,
        filters=tuple(cfg.BASE_FILTERS),
        use_disease_classifier=(
            variant.upper() == "M5" and getattr(cfg, "USE_DISEASE_CLASSIFIER", True)
        ),
        gate_pathology_by_disease=(
            variant.upper() == "M5" and getattr(cfg, "GATE_PATHOLOGY_BY_DISEASE", True)
        ),
        disease_threshold=getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    _load_state_dict(model, state["model"], strict=False)
    model.eval()
    out = model(x)
    if is_multiclass_variant(variant):
        multi = out["multiclass_logits"].argmax(1)[0]
        mi_cls = int(getattr(cfg, "MULTICLASS_MI", 3))
        mvo_cls = int(getattr(cfg, "MULTICLASS_MVO", 4))
        if getattr(cfg, "MI_VOXEL_SUPPRESSION", True):
            thr = int(getattr(cfg, "MIN_MI_VOXELS", 50))
            mi_vox = multi == mi_cls
            if mi_vox.sum() < thr:
                multi = multi.clone()
                multi[mi_vox] = 2
        multi = multi.cpu().numpy()  # (D,H,W)
        anatomy = np.zeros_like(multi, dtype=np.uint8)
        anatomy[multi == 1] = 1  # LV
        anatomy[np.isin(multi, [2, mi_cls, mvo_cls])] = 2  # wall
        pathology = np.zeros((2,) + multi.shape, dtype=np.uint8)
        pathology[0] = (multi == mi_cls).astype(np.uint8)  # pure MI
        pathology[1] = (multi == mvo_cls).astype(np.uint8)  # MVO
    else:
        anatomy = out["anatomy_logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        pathology = (
            postprocess_pathology(out, hard_mask=True)[0].cpu().numpy().astype(np.uint8)
        )
    vol = x[0, 0].cpu().numpy()  # (D,H,W)
    return {
        "image": vol,
        "anatomy": anatomy,
        "pathology": pathology,
        "gt_anatomy": None if gt_anatomy is None else np.transpose(gt_anatomy, (2, 0, 1)),
        "gt_pathology": None if gt_pathology is None else np.transpose(gt_pathology, (3, 2, 0, 1)),
        "gt_multi": None if gt_multi is None else np.transpose(gt_multi, (2, 0, 1)),
        "npz_path": str(npz_path),
        "variant": variant,
        "checkpoint": str(ckpt),
    }

# ---------------------------- Percentages ------------------------------------

def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * float(num) / float(den)

def compute_dysfunction_stats(anatomy: np.ndarray, pathology: np.ndarray) -> Dict:
    """
    Percentages of MI / MVO / combined dysfunction inside anatomical regions.
    MYO  = predicted myocardial wall
    LV   = LV cavity only
    LV_total = LV cavity + MYO  (whole left ventricle)
    """
    lv = anatomy == 1
    myo = anatomy == 2
    lv_total = lv | myo
    mi = pathology[0].astype(bool)
    mvo = pathology[1].astype(bool)
    dys = mi | mvo
    # Spatially, infarct lives in MYO; still report LV-cavity intersection (should be ~0)
    stats = {
        "voxel_counts": {
            "LV_cavity": int(lv.sum()),
            "MYO": int(myo.sum()),
            "LV_total": int(lv_total.sum()),
            "MI": int(mi.sum()),
            "MVO": int(mvo.sum()),
            "dysfunction_MI_or_MVO": int(dys.sum()),
        },
        "percent_of_MYO": {
            "MI": round(_pct(int((mi & myo).sum()), int(myo.sum())), 2),
            "MVO": round(_pct(int((mvo & myo).sum()), int(myo.sum())), 2),
            "dysfunction": round(_pct(int((dys & myo).sum()), int(myo.sum())), 2),
        },
        "percent_of_LV_cavity": {
            "MI": round(_pct(int((mi & lv).sum()), int(lv.sum())), 2),
            "MVO": round(_pct(int((mvo & lv).sum()), int(lv.sum())), 2),
            "dysfunction": round(_pct(int((dys & lv).sum()), int(lv.sum())), 2),
        },
        "percent_of_LV_total_cavity_plus_MYO": {
            "MI": round(_pct(int((mi & lv_total).sum()), int(lv_total.sum())), 2),
            "MVO": round(_pct(int((mvo & lv_total).sum()), int(lv_total.sum())), 2),
            "dysfunction": round(_pct(int((dys & lv_total).sum()), int(lv_total.sum())), 2),
        },
        "MVO_as_percent_of_MI": round(_pct(int(mvo.sum()), int(mi.sum())), 2)
        if int(mi.sum()) > 0
        else 0.0,
    }
    return stats

# ---------------------------- Figures ----------------------------------------
COLORS = {
    "LV": (0.15, 0.55, 0.95),
    "MYO": (0.20, 0.80, 0.35),
    "MI": (0.95, 0.15, 0.15),
    "MVO": (0.95, 0.85, 0.10),
}

def _rgb_overlay(base: np.ndarray, anatomy: np.ndarray, pathology: np.ndarray, alpha: float = 0.45):
    """Compose colour overlay on a 2D LGE slice (H,W)."""
    img = base.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    rgb = np.stack([img, img, img], axis=-1)
    def paint(mask, color, a=alpha):
        m = mask.astype(bool)
        for c in range(3):
            rgb[..., c][m] = (1 - a) * rgb[..., c][m] + a * color[c]
    paint(anatomy == 1, COLORS["LV"], 0.35)
    paint(anatomy == 2, COLORS["MYO"], 0.30)
    paint(pathology[0].astype(bool), COLORS["MI"], 0.55)
    paint(pathology[1].astype(bool), COLORS["MVO"], 0.55)
    return np.clip(rgb, 0, 1)

def _legend_handles():
    return [
        mpatches.Patch(color=COLORS["LV"], label="LV cavity"),
        mpatches.Patch(color=COLORS["MYO"], label="Myocardium"),
        mpatches.Patch(color=COLORS["MI"], label="MI (infarct)"),
        mpatches.Patch(color=COLORS["MVO"], label="MVO (no-reflow)"),
    ]

def save_patient_report(
    case_id: str,
    pred: Dict,
    stats: Dict,
    clinical: Dict,
    out_dir: Path,
    all_slices: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = pred["image"]  # (D,H,W)
    anatomy = pred["anatomy"]
    pathology = pred["pathology"]
    D = image.shape[0]
    mid = D // 2
    # ---- Main summary figure ----
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.25, wspace=0.15)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image[mid], cmap="gray")
    ax0.set_title("LGE MRI (mid slice)")
    ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1])
    anat_rgb = np.zeros((*anatomy[mid].shape, 3))
    anat_rgb[anatomy[mid] == 1] = COLORS["LV"]
    anat_rgb[anatomy[mid] == 2] = COLORS["MYO"]
    ax1.imshow(image[mid], cmap="gray")
    ax1.imshow(anat_rgb, alpha=0.45)
    ax1.set_title("Anatomy: LV + MYO")
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(_rgb_overlay(image[mid], anatomy[mid], pathology[:, mid]))
    ax2.set_title("Full: LV / MYO / MI / MVO")
    ax2.axis("off")
    ax2.legend(handles=_legend_handles(), loc="lower right", fontsize=8, framealpha=0.9)
    # Pathology-only panel
    ax3 = fig.add_subplot(gs[1, 0])
    path_rgb = np.zeros((*image[mid].shape, 3))
    path_rgb[pathology[0, mid].astype(bool)] = COLORS["MI"]
    path_rgb[pathology[1, mid].astype(bool)] = COLORS["MVO"]
    ax3.imshow(image[mid], cmap="gray")
    ax3.imshow(path_rgb, alpha=0.55)
    ax3.set_title("Pathology: MI + MVO")
    ax3.axis("off")
    # Stats text panel
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis("off")
    lines = [
        f"Patient: {case_id}   |   Model: {MODEL_NAME} ({VARIANT_SHORT.get(pred['variant'], pred['variant'])})   |   Slice {mid + 1}/{D}",
        "",
        "Dysfunction burden (% of myocardium / MYO wall):",
        f"  MI   = {stats['percent_of_MYO']['MI']:.2f}% of MYO",
        f"  MVO  = {stats['percent_of_MYO']['MVO']:.2f}% of MYO",
        f"  Any dysfunction (MI U MVO) = {stats['percent_of_MYO']['dysfunction']:.2f}% of MYO",
        "",
        "Dysfunction burden (% of LV cavity):",
        f"  MI   = {stats['percent_of_LV_cavity']['MI']:.2f}% of LV cavity",
        f"  MVO  = {stats['percent_of_LV_cavity']['MVO']:.2f}% of LV cavity",
        f"  (Expected ~0%: infarct is confined to the myocardial wall)",
        "",
        "Dysfunction burden (% of whole LV = cavity + MYO):",
        f"  MI   = {stats['percent_of_LV_total_cavity_plus_MYO']['MI']:.2f}% of LV total",
        f"  MVO  = {stats['percent_of_LV_total_cavity_plus_MYO']['MVO']:.2f}% of LV total",
        f"  Dysfunction = {stats['percent_of_LV_total_cavity_plus_MYO']['dysfunction']:.2f}% of LV total",
        "",
        f"Voxel counts - LV:{stats['voxel_counts']['LV_cavity']}  "
        f"MYO:{stats['voxel_counts']['MYO']}  "
        f"MI:{stats['voxel_counts']['MI']}  "
        f"MVO:{stats['voxel_counts']['MVO']}",
    ]
    if clinical:
        age = clinical.get("Age", clinical.get("Age ", "?"))
        sex = clinical.get("Sex", "?")
        troponin = clinical.get("Troponin", "?")
        fevg = clinical.get("FEVG", "?")
        lines.append("")
        lines.append(f"Clinical - Sex:{sex}  Age:{age}  Troponin:{troponin}  LVEF(FEVG):{fevg}%")
    ax4.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax4.transAxes,
    )
    fig.suptitle(f"{MODEL_NAME} Segmentation Report - {case_id}", fontsize=14, fontweight="bold")
    summary_path = out_dir / f"{case_id}_report.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ---- Separate anatomy / pathology panels ----
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="white")
    titles = ["LGE", "LV", "MYO", "MI + MVO"]
    axes[0].imshow(image[mid], cmap="gray")
    lv_only = np.zeros((*image[mid].shape, 3))
    lv_only[anatomy[mid] == 1] = COLORS["LV"]
    axes[1].imshow(image[mid], cmap="gray")
    axes[1].imshow(lv_only, alpha=0.5)
    myo_only = np.zeros((*image[mid].shape, 3))
    myo_only[anatomy[mid] == 2] = COLORS["MYO"]
    axes[2].imshow(image[mid], cmap="gray")
    axes[2].imshow(myo_only, alpha=0.5)
    axes[3].imshow(_rgb_overlay(image[mid], np.zeros_like(anatomy[mid]), pathology[:, mid]))
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.axis("off")
    fig2.suptitle(f"{case_id} - structure & dysfunction")
    panels_path = out_dir / f"{case_id}_panels.png"
    fig2.savefig(panels_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    saved = [summary_path, panels_path]
    # ---- Optional all-slice montage ----
    if all_slices:
        cols = min(4, D)
        rows = int(np.ceil(D / cols))
        fig3, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor="white")
        axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            if i < D:
                ax.imshow(_rgb_overlay(image[i], anatomy[i], pathology[:, i]))
                ax.set_title(f"z={i}")
            ax.axis("off")
        fig3.suptitle(f"{case_id} - all slices")
        mont_path = out_dir / f"{case_id}_allslices.png"
        fig3.savefig(mont_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig3)
        saved.append(mont_path)
    # JSON sidecar
    report = {
        "case": case_id,
        "model": MODEL_NAME, "variant": pred["variant"],
        "checkpoint": pred["checkpoint"],
        "source_npz": pred["npz_path"],
        "dysfunction_stats": stats,
        "clinical": {k: v for k, v in clinical.items() if not k.startswith("_")},
        "figures": [str(p) for p in saved],
    }
    json_path = out_dir / f"{case_id}_stats.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    saved.append(json_path)
    return report, saved

# ---------------------------- CLI --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize LV/MYO/MI/MVO for one EMIDEC patient")
    parser.add_argument("--case", required=True, help="Patient id, e.g. P001, Case_P001, N006")
    parser.add_argument(
        "--variant",
        default=None,
        help="M1-M5 or registered PyTorch baseline (default: best available)",
    )
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if free else cpu)")
    parser.add_argument("--all-slices", action="store_true", help="Also save full-slice montage")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()
    case_id = normalize_case_id(args.case)
    variant, ckpt = pick_checkpoint(args.variant)
    if args.device:
        device = torch.device(args.device)
    else:
        # Prefer CPU if another training job holds most of the GPU VRAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info()
                if free < 1.5 * 1024**3:  # <1.5 GB free
                    print(f"GPU memory low ({free/1024**3:.1f} GB free) - using CPU")
                    device = torch.device("cpu")
            except Exception:
                pass
    print(f"Case: {case_id}")
    print(f"Model: {MODEL_NAME} / {VARIANT_SHORT.get(variant, variant)}  ({ckpt.name})  device={device}")
    pred = predict_case(case_id, variant, ckpt, device)
    stats = compute_dysfunction_stats(pred["anatomy"], pred["pathology"])
    clinical = load_clinical(case_id)
    out_dir = Path(args.out) if args.out else cfg.FIGURES_DIR / "patients" / case_id
    report, saved = save_patient_report(case_id, pred, stats, clinical, out_dir, all_slices=args.all_slices)
    print("\n=== Dysfunction percentages ===")
    print(f"  of MYO wall : MI {stats['percent_of_MYO']['MI']}% | "
          f"MVO {stats['percent_of_MYO']['MVO']}% | "
          f"any {stats['percent_of_MYO']['dysfunction']}%")
    print(f"  of LV cavity: MI {stats['percent_of_LV_cavity']['MI']}% | "
          f"MVO {stats['percent_of_LV_cavity']['MVO']}%")
    print(f"  of LV total : MI {stats['percent_of_LV_total_cavity_plus_MYO']['MI']}% | "
          f"MVO {stats['percent_of_LV_total_cavity_plus_MYO']['MVO']}% | "
          f"any {stats['percent_of_LV_total_cavity_plus_MYO']['dysfunction']}%")
    print("\nSaved:")
    for p in saved:
        print(f"  {p}")

if __name__ == "__main__":
    main()
