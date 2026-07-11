"""Evaluate checkpoints on EMIDEC test set with paper-comparable metrics."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from data.preprocess import EMIDECDataset
from metrics import binary_metrics, summarize
from models.dual_decoder import build_model, count_parameters
from train import collate
@torch.no_grad()

def run_eval(variant: str, ckpt_path: Path, split: str = "test", save_figs: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EMIDECDataset(cfg.DATASET_DIR / split, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = build_model(variant, filters=tuple(cfg.BASE_FILTERS)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = count_parameters(model)
    anat_scores = {"LV": [], "MYO": []}
    path_scores = {"MI": [], "MVO": []}
    multi_scores = {"LV": [], "MYO": [], "Infarct": []}
    per_case = []
    times = []
    fig_dir = cfg.FIGURES_DIR / f"{variant}_{split}"
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
        img = batch["image"][0, 0].numpy()  # (D,H,W)
        mid = img.shape[0] // 2
        case_row = {"case": name}
        if variant in ("M1", "M2"):
            pred = out["multiclass_logits"].argmax(1)[0].cpu().numpy()
            gt = batch["multiclass"][0].numpy()
            for cls, key in [(1, "LV"), (2, "MYO"), (3, "Infarct")]:
                m = binary_metrics(pred == cls, gt == cls, spacing=cfg.TARGET_SPACING)
                multi_scores[key].append(m)
                case_row[key] = m
            if save_figs:
                _save_overlay(img[mid], pred[mid], gt[mid], fig_dir / f"{name}.png",
                              title=f"{variant} {name}", mode="multi")
        else:
            anat = out["anatomy_logits"].argmax(1)[0].cpu().numpy()
            path = (out["pathology_prob"][0] > 0.5).cpu().numpy()
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
            if save_figs:
                _save_dual(img[mid], anat[mid], path[:, mid], gt_a[mid], gt_p[:, mid],
                           fig_dir / f"{name}.png", title=f"{variant} {name}")
        per_case.append(case_row)
    if variant in ("M1", "M2"):
        summary = {k: summarize(v) for k, v in multi_scores.items()}
    else:
        summary = {
            "LV": summarize(anat_scores["LV"]),
            "MYO": summarize(anat_scores["MYO"]),
            "MI": summarize(path_scores["MI"]),
            "MVO": summarize(path_scores["MVO"]),
        }
    summary["params_M"] = n_params / 1e6
    summary["inference_ms_mean"] = float(np.mean(times))
    summary["inference_ms_std"] = float(np.std(times))
    summary["variant"] = variant
    summary["split"] = split
    summary["checkpoint"] = str(ckpt_path)
    out_path = cfg.RESULTS_DIR / f"{variant}_{split}_metrics.json"
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved ? {out_path}")
    return summary

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
    # path_*: (2, H, W)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="M5")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    variants = ["M1", "M2", "M3", "M4", "M5"] if args.all else [args.variant.upper()]
    table = {}
    for v in variants:
        ckpt = Path(args.ckpt) if args.ckpt else cfg.CHECKPOINT_DIR / f"{v}_best.pth"
        if not ckpt.exists():
            print(f"Skip {v}: missing {ckpt}")
            continue
        table[v] = run_eval(v, ckpt, split=args.split)
    (cfg.RESULTS_DIR / f"comparison_{args.split}.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
