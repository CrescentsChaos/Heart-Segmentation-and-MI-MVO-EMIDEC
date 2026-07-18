"""Evaluate checkpoints on EMIDEC (single split or 5-fold CV)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from data.preprocess import EMIDECDataset
from inference import postprocess_pathology
from metrics import binary_metrics, summarize
from model_identity import ABLATION_VARIANTS, BASELINE_VARIANTS, is_multiclass_variant
from models.dual_decoder import build_model, count_parameters
from train import _load_state_dict, collate


def _is_pathological_case(name: str, gt_mi: np.ndarray) -> bool:
    """EMIDEC pathological cases are Case_P*; also trust GT MI>0."""
    if gt_mi.sum() > 0:
        return True
    return name.upper().startswith("CASE_P") or name.upper().startswith("P")


def _dice_mean(summary: dict, key: str) -> Optional[float]:
    block = summary.get(key)
    if not isinstance(block, dict):
        return None
    dice = block.get("dice")
    if not isinstance(dice, dict):
        return None
    return dice.get("mean")


def aggregate_fold_summaries(fold_summaries: List[dict]) -> dict:
    """Mean ± std across folds of each structure's Dice (and disease_acc)."""
    if not fold_summaries:
        return {}
    # Keys that look like metric blocks with dice.mean
    metric_keys = set()
    for s in fold_summaries:
        for k, v in s.items():
            if isinstance(v, dict) and "dice" in v and isinstance(v["dice"], dict):
                metric_keys.add(k)

    out: Dict = {"n_folds": len(fold_summaries)}
    for key in sorted(metric_keys):
        vals = []
        for s in fold_summaries:
            m = _dice_mean(s, key)
            if m is not None and not (isinstance(m, float) and np.isnan(m)):
                vals.append(float(m))
        if vals:
            out[key] = {
                "dice": {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    "n": len(vals),
                    "per_fold": vals,
                }
            }

    # Optional scalar disease accuracy
    dvals = []
    for s in fold_summaries:
        if "disease_acc" in s and s["disease_acc"] is not None:
            dvals.append(float(s["disease_acc"]))
    if dvals:
        out["disease_acc"] = {
            "mean": float(np.mean(dvals)),
            "std": float(np.std(dvals, ddof=1) if len(dvals) > 1 else 0.0),
            "per_fold": dvals,
        }

    # Carry params / timing averages
    params = [s["params_M"] for s in fold_summaries if "params_M" in s]
    if params:
        out["params_M"] = float(np.mean(params))
    ms = [s["inference_ms_mean"] for s in fold_summaries if "inference_ms_mean" in s]
    if ms:
        out["inference_ms_mean"] = float(np.mean(ms))
        out["inference_ms_std"] = float(np.std(ms, ddof=1) if len(ms) > 1 else 0.0)
    return out


@torch.no_grad()
def run_eval(
    variant: str,
    ckpt_path: Path,
    split: str = "test",
    save_figs: bool = True,
    fold: Optional[int] = None,
    case_names: Optional[List[str]] = None,
):
    from data.cv_splits import get_fold_splits, metrics_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if case_names is not None:
        ds = EMIDECDataset(case_names=case_names, augment=False)
        split_label = f"fold{fold}_test" if fold is not None else split
    elif fold is not None:
        splits = get_fold_splits(fold)
        ds = EMIDECDataset(case_names=splits["test"], augment=False)
        split_label = f"fold{fold}_test"
    else:
        ds = EMIDECDataset(cfg.DATASET_DIR / split, augment=False)
        split_label = split

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = build_model(
        variant,
        filters=tuple(cfg.BASE_FILTERS),
        detach_myo_gate=getattr(cfg, "DETACH_MYO_GATE", True),
        soft_myo_restrict=True,
        use_disease_classifier=getattr(cfg, "USE_DISEASE_CLASSIFIER", True),
        gate_pathology_by_disease=getattr(cfg, "GATE_PATHOLOGY_BY_DISEASE", True),
        disease_threshold=getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _load_state_dict(model, ckpt["model"], strict=False)
    model.eval()
    n_params = count_parameters(model)
    anat_scores = {"LV": [], "MYO": []}
    path_scores = {"MI": [], "MVO": []}
    path_only = {"MI": [], "MVO": []}
    multi_scores = {"LV": [], "MYO": [], "Infarct": []}
    multi_path_only = {"Infarct": []}
    per_case = []
    times = []
    class_correct = 0
    class_total = 0
    hard_mask = bool(getattr(cfg, "HARD_MYO_MASK_AT_INFER", True))
    fig_dir = cfg.FIGURES_DIR / f"{variant}_{split_label}"
    if save_figs:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        x = batch["image"].to(device)
        name = batch["name"][0]
        t0 = time.time()
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        img = batch["image"][0, 0].numpy()
        mid = img.shape[0] // 2
        case_row = {"case": name, "fold": fold}
        if is_multiclass_variant(variant):
            pred = out["multiclass_logits"].argmax(1)[0]
            if getattr(cfg, "MI_VOXEL_SUPPRESSION", True):
                thr = int(getattr(cfg, "MIN_MI_VOXELS", 50))
                infarct = pred == 3
                if infarct.sum() < thr:
                    pred = pred.clone()
                    pred[infarct] = 2
            pred = pred.cpu().numpy()
            gt = batch["multiclass"][0].numpy()
            for cls, key in [(1, "LV"), (2, "MYO"), (3, "Infarct")]:
                m = binary_metrics(pred == cls, gt == cls, spacing=cfg.TARGET_SPACING)
                multi_scores[key].append(m)
                case_row[key] = m
            if (gt == 3).sum() > 0:
                multi_path_only["Infarct"].append(case_row["Infarct"])
            case_row["pathological"] = bool((gt == 3).sum() > 0)
            if save_figs:
                _save_overlay(
                    img[mid],
                    pred[mid],
                    gt[mid],
                    fig_dir / f"{name}.png",
                    title=f"{variant} {name}",
                    mode="multi",
                )
        else:
            anat = out["anatomy_logits"].argmax(1)[0].cpu().numpy()
            path = postprocess_pathology(out, hard_mask=hard_mask)[0].cpu().numpy()
            gt_a = batch["anatomy"][0].numpy()
            gt_p = batch["pathology"][0].numpy()
            for cls, key in [(1, "LV"), (2, "MYO")]:
                m = binary_metrics(anat == cls, gt_a == cls, spacing=cfg.TARGET_SPACING)
                anat_scores[key].append(m)
                case_row[key] = m
            for c, key in enumerate(["MI", "MVO"]):
                m = binary_metrics(path[c], gt_p[c], spacing=cfg.TARGET_SPACING)
                path_scores[key].append(m)
                case_row[key] = m
            is_path = _is_pathological_case(name, gt_p[0])
            case_row["pathological"] = is_path
            if "disease_prob" in out:
                d_prob = float(out["disease_prob"][0].item())
                case_row["disease_prob"] = d_prob
                case_row["disease_pred"] = int(
                    d_prob > getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5)
                )
                gt_c = 1 if is_path else 0
                class_correct += int(case_row["disease_pred"] == gt_c)
                class_total += 1
            if is_path:
                path_only["MI"].append(case_row["MI"])
                path_only["MVO"].append(case_row["MVO"])
            if save_figs:
                _save_dual(
                    img[mid],
                    anat[mid],
                    path[:, mid],
                    gt_a[mid],
                    gt_p[:, mid],
                    fig_dir / f"{name}.png",
                    title=f"{variant} {name}",
                )
        per_case.append(case_row)

    if is_multiclass_variant(variant):
        summary = {k: summarize(v) for k, v in multi_scores.items()}
        if multi_path_only["Infarct"]:
            summary["Infarct_pathological"] = summarize(multi_path_only["Infarct"])
    else:
        summary = {
            "LV": summarize(anat_scores["LV"]),
            "MYO": summarize(anat_scores["MYO"]),
            "MI": summarize(path_scores["MI"]),
            "MVO": summarize(path_scores["MVO"]),
        }
        if path_only["MI"]:
            summary["MI_pathological"] = summarize(path_only["MI"])
        if path_only["MVO"]:
            summary["MVO_pathological"] = summarize(path_only["MVO"])
        if class_total > 0:
            summary["disease_acc"] = class_correct / class_total

    summary["params_M"] = n_params / 1e6
    summary["inference_ms_mean"] = float(np.mean(times)) if times else 0.0
    summary["inference_ms_std"] = float(np.std(times)) if times else 0.0
    summary["variant"] = variant
    summary["split"] = split_label
    summary["fold"] = fold
    summary["n_cases"] = len(per_case)
    summary["checkpoint"] = str(ckpt_path)
    summary["hard_myo_mask"] = hard_mask
    summary["mi_voxel_suppression"] = bool(getattr(cfg, "MI_VOXEL_SUPPRESSION", True))
    summary["primary_metric"] = "MI_pathological"
    summary["protocol"] = "5-fold-cv" if fold is not None else "single-split"

    out_path = cfg.RESULTS_DIR / metrics_name(variant, "test", fold)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_case": per_case}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    mi_all = _dice_mean(summary, "MI") or _dice_mean(summary, "Infarct")
    mi_path = _dice_mean(summary, "MI_pathological") or _dice_mean(
        summary, "Infarct_pathological"
    )
    print("-" * 60)
    tag = f"{variant}" + (f" fold{fold}" if fold is not None else "")
    print(f"{tag} [{split_label}] PRIMARY = MI_path (pathological only)")
    if mi_path is not None:
        print(f"  MI_path: {mi_path:.4f}")
    if mi_all is not None:
        print(f"  MI_all:  {mi_all:.4f}")
    if "disease_acc" in summary:
        print(f"  Disease classification accuracy: {summary['disease_acc']:.4f}")
    print(f"Saved -> {out_path}")
    return summary


def run_cv_eval(
    variant: str,
    folds: Optional[List[int]] = None,
    save_figs: bool = False,
) -> dict:
    """Evaluate all folds for one variant and write aggregated mean±std."""
    from data.cv_splits import ckpt_name, cv_metrics_name, ensure_folds

    folds_meta = ensure_folds(overwrite=False)
    n_folds = int(folds_meta["n_folds"])
    fold_list = folds if folds is not None else list(range(n_folds))
    fold_summaries = []
    fold_payload = []
    for fold in fold_list:
        ckpt = cfg.CHECKPOINT_DIR / ckpt_name(variant, fold)
        if not ckpt.exists():
            print(f"Skip {variant} fold{fold}: missing {ckpt}")
            continue
        summary = run_eval(
            variant, ckpt, fold=fold, save_figs=save_figs
        )
        fold_summaries.append(summary)
        fold_payload.append({"fold": fold, "summary": summary})

    aggregate = aggregate_fold_summaries(fold_summaries)
    out = {
        "variant": variant.upper(),
        "protocol": "5-fold stratified CV (same folds for all models)",
        "n_folds_requested": len(fold_list),
        "n_folds_evaluated": len(fold_summaries),
        "seed": folds_meta.get("seed"),
        "primary_metric": "MI_pathological",
        "folds": fold_payload,
        "aggregate": aggregate,
    }
    out_path = cfg.RESULTS_DIR / cv_metrics_name(variant)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"{variant} 5-fold CV aggregate (mean ± std across folds)")
    for key in ("LV", "MYO", "MI_pathological", "MI", "MVO", "Infarct", "Infarct_pathological"):
        block = aggregate.get(key, {}).get("dice")
        if block:
            print(f"  {key}: {block['mean']:.4f} ± {block['std']:.4f}")
    print(f"Saved -> {out_path}")
    return out


def _save_overlay(img, pred, gt, path, title, mode="multi"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("LGE")
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(gt, cmap="tab10", alpha=0.45, vmin=0, vmax=4)
    axes[1].set_title("GT")
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(pred, cmap="tab10", alpha=0.45, vmin=0, vmax=4)
    axes[2].set_title("Pred")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_dual(img, anat_p, path_p, anat_g, path_g, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("LGE")
    axes[0, 1].imshow(img, cmap="gray")
    axes[0, 1].imshow(anat_g, cmap="tab10", alpha=0.45, vmin=0, vmax=3)
    axes[0, 1].set_title("GT Anatomy")
    axes[0, 2].imshow(img, cmap="gray")
    axes[0, 2].imshow(anat_p, cmap="tab10", alpha=0.45, vmin=0, vmax=3)
    axes[0, 2].set_title("Pred Anatomy")
    mi_g = np.zeros((*img.shape, 3))
    mi_g[..., 0] = path_g[0]
    mi_g[..., 1] = path_g[1]
    mi_p = np.zeros((*img.shape, 3))
    mi_p[..., 0] = path_p[0]
    mi_p[..., 1] = path_p[1]
    axes[1, 0].imshow(img, cmap="gray")
    axes[1, 0].set_title("LGE")
    axes[1, 1].imshow(img, cmap="gray")
    axes[1, 1].imshow(mi_g, alpha=0.5)
    axes[1, 1].set_title("GT MI(red)/MVO(green)")
    axes[1, 2].imshow(img, cmap="gray")
    axes[1, 2].imshow(mi_p, alpha=0.5)
    axes[1, 2].set_title("Pred MI/MVO")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    from data.cv_splits import ckpt_name
    from data.preprocess import sync_all_pool_from_splits

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="M5",
        help="M1-M5 / UNET|SEGRESNET|SWINUNETR|NNUNET|DYNUNET / comma-list",
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--all", action="store_true", help="Evaluate ablation M1-M5")
    parser.add_argument("--baselines", action="store_true", help="Evaluate MONAI baselines")
    parser.add_argument("--cv", action="store_true", help="Evaluate 5-fold CV and aggregate")
    parser.add_argument("--fold", type=int, default=None, help="Evaluate a single fold")
    parser.add_argument("--no-figs", action="store_true", help="Skip overlay figures")
    args = parser.parse_args()

    if args.all and args.baselines:
        variants = list(ABLATION_VARIANTS) + list(BASELINE_VARIANTS)
    elif args.all:
        variants = list(ABLATION_VARIANTS)
    elif args.baselines:
        variants = list(BASELINE_VARIANTS)
    else:
        variants = [v.strip().upper() for v in args.variant.split(",") if v.strip()]

    save_figs = not args.no_figs
    table = {}

    if args.cv or args.fold is not None:
        sync_all_pool_from_splits()
        fold_list = [args.fold] if args.fold is not None and not args.cv else None
        # If --fold without --cv, still eval that fold only (no full aggregate needed beyond 1)
        for v in variants:
            if args.cv:
                table[v] = run_cv_eval(v, folds=fold_list, save_figs=save_figs)
            else:
                ckpt = (
                    Path(args.ckpt)
                    if args.ckpt
                    else cfg.CHECKPOINT_DIR / ckpt_name(v, args.fold)
                )
                if not ckpt.exists():
                    print(f"Skip {v}: missing {ckpt}")
                    continue
                table[v] = run_eval(
                    v, ckpt, fold=args.fold, save_figs=save_figs
                )
        out_name = "comparison_cv.json" if args.cv else f"comparison_fold{args.fold}.json"
    else:
        for v in variants:
            ckpt = Path(args.ckpt) if args.ckpt else cfg.CHECKPOINT_DIR / ckpt_name(v, None)
            if not ckpt.exists():
                print(f"Skip {v}: missing {ckpt}")
                continue
            table[v] = run_eval(v, ckpt, split=args.split, save_figs=save_figs)
        out_name = f"comparison_{args.split}.json"

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / out_name).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"Comparison -> {cfg.RESULTS_DIR / out_name}")


if __name__ == "__main__":
    main()
