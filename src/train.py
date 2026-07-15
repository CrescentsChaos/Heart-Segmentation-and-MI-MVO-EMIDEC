"""Train ablation variants M1-M5 on EMIDEC (revised methodology)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from data.preprocess import EMIDECDataset
from inference import hard_myo_mask_pathology
from losses.joint_loss import JointLoss
from metrics import binary_metrics, summarize
from model_identity import MODEL_NAME, VARIANT_SHORT
from models.dual_decoder import build_model, count_parameters


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "anatomy": torch.stack([b["anatomy"] for b in batch]),
        "pathology": torch.stack([b["pathology"] for b in batch]),
        "multiclass": torch.stack([b["multiclass"] for b in batch]),
        "name": [b["name"] for b in batch],
    }


def topo_lambda_for_epoch(epoch: int) -> float:
    """Curriculum: 0 during warmup, then linear ramp to LAMBDA_TOPO."""
    warm = int(getattr(cfg, "TOPO_WARMUP_EPOCHS", 40))
    ramp = int(getattr(cfg, "TOPO_RAMP_EPOCHS", 20))
    target = float(getattr(cfg, "LAMBDA_TOPO", 0.05))
    if epoch <= warm:
        return 0.0
    if ramp <= 0:
        return target
    t = min(1.0, (epoch - warm) / float(ramp))
    return target * t


def _pathology_prediction(out, hard_mask: bool) -> torch.Tensor:
    path = out["pathology_prob"]
    if hard_mask and "anatomy_logits" in out:
        path = hard_myo_mask_pathology(out["anatomy_logits"], path)
    return path > 0.5


@torch.no_grad()
def evaluate_epoch(model, loader, device, variant: str, hard_mask: bool = True):
    model.eval()
    anat_scores = {"LV": [], "MYO": []}
    path_scores = {"MI": [], "MVO": []}
    path_only_mi = []
    multi_scores = {"LV": [], "MYO": [], "Infarct": []}
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        if variant in ("M1", "M2"):
            pred = out["multiclass_logits"].argmax(1).cpu().numpy()
            gt = batch["multiclass"].numpy()
            for b in range(pred.shape[0]):
                for cls, name in [(1, "LV"), (2, "MYO"), (3, "Infarct")]:
                    multi_scores[name].append(
                        binary_metrics(pred[b] == cls, gt[b] == cls, spacing=cfg.TARGET_SPACING)
                    )
        else:
            anat = out["anatomy_logits"].argmax(1).cpu().numpy()
            path = _pathology_prediction(out, hard_mask=hard_mask).cpu().numpy()
            gt_a = batch["anatomy"].numpy()
            gt_p = batch["pathology"].numpy()
            for b in range(anat.shape[0]):
                anat_scores["LV"].append(
                    binary_metrics(anat[b] == 1, gt_a[b] == 1, spacing=cfg.TARGET_SPACING)
                )
                anat_scores["MYO"].append(
                    binary_metrics(anat[b] == 2, gt_a[b] == 2, spacing=cfg.TARGET_SPACING)
                )
                mi_m = binary_metrics(path[b, 0], gt_p[b, 0], spacing=cfg.TARGET_SPACING)
                mvo_m = binary_metrics(path[b, 1], gt_p[b, 1], spacing=cfg.TARGET_SPACING)
                path_scores["MI"].append(mi_m)
                path_scores["MVO"].append(mvo_m)
                # Pathological-only: GT has any MI voxels
                if gt_p[b, 0].sum() > 0:
                    path_only_mi.append(mi_m)
    if variant in ("M1", "M2"):
        return {k: summarize(v) for k, v in multi_scores.items()}
    out_m = {
        "LV": summarize(anat_scores["LV"]),
        "MYO": summarize(anat_scores["MYO"]),
        "MI": summarize(path_scores["MI"]),
        "MVO": summarize(path_scores["MVO"]),
    }
    if path_only_mi:
        out_m["MI_pathological"] = summarize(path_only_mi)
    return out_m


def primary_score(metrics: dict, variant: str) -> float:
    """
    Checkpoint selection: prefer pathological-only MI Dice when available
    (matches EMIDEC reporting practice; avoids FP-on-normal zeros).
    """
    if variant not in ("M1", "M2") and "MI_pathological" in metrics:
        return float(metrics["MI_pathological"]["dice"]["mean"])
    key = "Infarct" if variant in ("M1", "M2") else "MI"
    if key not in metrics or "dice" not in metrics[key]:
        return -1.0
    return float(metrics[key]["dice"]["mean"])


def _maybe_warm_start(model, variant: str, init_from: str | None, device):
    if not init_from:
        return
    src = init_from.upper()
    ckpt_path = cfg.CHECKPOINT_DIR / f"{src}_best.pth"
    if not ckpt_path.exists():
        print(f"  [warn] warm-start skipped: missing {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(
        f"  [ok] warm-started {variant} from {ckpt_path.name} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def train_variant(
    variant: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    init_from: str | None = None,
):
    variant = variant.upper()
    print(f"\n{'=' * 70}\nTraining {variant} ({VARIANT_SHORT.get(variant, variant)}) | {MODEL_NAME}\n{'=' * 70}")
    train_ds = EMIDECDataset(cfg.DATASET_DIR / "train", augment=True)
    val_ds = EMIDECDataset(cfg.DATASET_DIR / "val", augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=collate
    )
    model = build_model(
        variant,
        filters=tuple(cfg.BASE_FILTERS),
        in_ch=cfg.IN_CHANNELS,
        detach_myo_gate=getattr(cfg, "DETACH_MYO_GATE", True),
        soft_myo_restrict=True,
    ).to(device)
    # M5 default: warm-start from M4 so topology fine-tunes a good MI head
    if variant == "M5" and init_from is None:
        init_from = "M4"
    _maybe_warm_start(model, variant, init_from, device)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params / 1e6:.3f} M")
    criterion = JointLoss(
        variant=variant,
        num_anatomy=cfg.NUM_ANATOMY_CLASSES,
        anatomy_weights=torch.tensor(cfg.ANATOMY_CE_WEIGHTS, dtype=torch.float32),
        lambda_ftl=cfg.LAMBDA_FTL,
        lambda_topo=cfg.LAMBDA_TOPO,
        ftl_alpha=cfg.FTL_ALPHA,
        ftl_beta=cfg.FTL_BETA,
        ftl_gamma=cfg.FTL_GAMMA,
        use_gt_myo_for_topo=getattr(cfg, "USE_GT_MYO_FOR_TOPO", True),
        mi_channel_weight=getattr(cfg, "MI_CHANNEL_WEIGHT", 1.5),
        mvo_channel_weight=getattr(cfg, "MVO_CHANNEL_WEIGHT", 0.75),
    ).to(device)
    optim = Adam(model.parameters(), lr=cfg.LR)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.MIN_LR)
    ckpt_dir = cfg.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{variant}_best.pth"
    history = []
    best_mi = -1.0
    hard_mask = bool(getattr(cfg, "HARD_MYO_MASK_AT_INFER", True))

    for epoch in range(1, epochs + 1):
        model.train()
        # Curriculum for M5 topology (prevents early wall-painting collapse)
        lam = topo_lambda_for_epoch(epoch) if variant == "M5" else 0.0
        criterion.set_lambda_topo(lam)

        t0 = time.time()
        losses = []
        for batch in train_loader:
            batch_dev = {
                "anatomy": batch["anatomy"].to(device),
                "pathology": batch["pathology"].to(device),
                "multiclass": batch["multiclass"].to(device),
            }
            x = batch["image"].to(device)
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss_dict = criterion(out, batch_dev)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss_dict["loss"].item()))
        sched.step()
        val_metrics = evaluate_epoch(model, val_loader, device, variant, hard_mask=hard_mask)
        mi = primary_score(val_metrics, variant)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "lr": float(sched.get_last_lr()[0]),
            "lambda_topo": lam,
            "val": val_metrics,
            "primary_mi_dice": mi,
            "sec": time.time() - t0,
        }
        history.append(row)
        if variant in ("M1", "M2"):
            msg = (
                f"[{variant}] ep {epoch:03d}/{epochs}  loss={row['loss']:.4f}  "
                f"LV={val_metrics['LV']['dice']['mean']:.3f}  "
                f"MYO={val_metrics['MYO']['dice']['mean']:.3f}  "
                f"Infarct={val_metrics['Infarct']['dice']['mean']:.3f}  "
                f"({row['sec']:.1f}s)"
            )
        else:
            path_mi = val_metrics.get("MI_pathological", {}).get("dice", {}).get("mean", float("nan"))
            msg = (
                f"[{variant}] ep {epoch:03d}/{epochs}  loss={row['loss']:.4f}  "
                f"λtopo={lam:.3f}  "
                f"LV={val_metrics['LV']['dice']['mean']:.3f}  "
                f"MYO={val_metrics['MYO']['dice']['mean']:.3f}  "
                f"MI={val_metrics['MI']['dice']['mean']:.3f}  "
                f"MI_path={path_mi:.3f}  "
                f"MVO={val_metrics['MVO']['dice']['mean']:.3f}  "
                f"({row['sec']:.1f}s)"
            )
        print(msg)
        if mi > best_mi and not np.isnan(mi):
            best_mi = mi
            torch.save(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "metrics": val_metrics,
                    "params": n_params,
                    "lambda_topo": lam,
                },
                best_path,
            )
            print(f"  [ok] saved best -> {best_path.name} (primary MI_path Dice={best_mi:.4f} | val pathological)")
    hist_path = cfg.RESULTS_DIR / f"{variant}_history.json"
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"History -> {hist_path}")
    return best_path, best_mi, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="M5", help="M1|M2|M3|M4|M5 or all")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Warm-start weights from another variant (default M5<-M4)",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable default M5<-M4 warm-start",
    )
    args = parser.parse_args()
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model family: {MODEL_NAME}")
    print(f"Device: {device}")
    if args.variant.lower() == "all":
        variants = ["M1", "M2", "M3", "M4", "M5"]
    else:
        variants = [v.strip().upper() for v in args.variant.split(",")]
    summary = {}
    for v in variants:
        init_from = None if args.no_warm_start else args.init_from
        path, score, n_params = train_variant(
            v, args.epochs, args.batch_size, device, init_from=init_from
        )
        summary[v] = {"best_ckpt": str(path), "best_mi_dice": score, "params": n_params}
    sum_path = cfg.RESULTS_DIR / "train_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {sum_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
