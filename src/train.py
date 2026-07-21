"""Train AFDD-Net ablation (M1-M5) and PyTorch baselines on EMIDEC."""
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
from inference import postprocess_pathology
from losses.joint_loss import JointLoss
from metrics import binary_metrics, summarize
from model_identity import (
    ABLATION_VARIANTS,
    MODEL_NAME,
    PYTORCH_BASELINE_VARIANTS,
    VARIANT_SHORT,
    is_multiclass_variant,
    is_real_nnunet,
)
from models.dual_decoder import build_model, count_parameters


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "anatomy": torch.stack([b["anatomy"] for b in batch]),
        "pathology": torch.stack([b["pathology"] for b in batch]),
        "multiclass": torch.stack([b["multiclass"] for b in batch]),
        "name": [b["name"] for b in batch],
    }
    if "pathological" in batch[0]:
        out["pathological"] = torch.stack([b["pathological"] for b in batch])
    return out


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
    """Binary pathology for train-val scoring (no sparse-MI suppress — checkpoint safe)."""
    return postprocess_pathology(
        out, hard_mask=hard_mask, use_voxel_suppress=False
    )


def _load_state_dict(
    model,
    state: dict,
    strict: bool = False,
    disable_missing_classifier: bool = True,
):
    """
    Load weights; classifier may be missing from older checkpoints.
    disable_missing_classifier: if True (eval), turn off disease gating when
    classifier weights are absent. If False (warm-start training), keep the
    randomly initialised classifier so it can be trained.
    """
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if (
            disable_missing_classifier
            and any("classifier" in k for k in missing)
            and hasattr(model, "use_disease_classifier")
        ):
            model.use_disease_classifier = False
            model.gate_pathology_by_disease = False
            print("  [warn] disease classifier missing in checkpoint — gating disabled")
        elif any("classifier" in k for k in missing):
            print("  [ok] classifier randomly initialised (warm-start); will be trained")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    return missing, unexpected


@torch.no_grad()
def evaluate_epoch(model, loader, device, variant: str, hard_mask: bool = True):
    model.eval()
    anat_scores = {"LV": [], "MYO": []}
    path_scores = {"MI": [], "MVO": []}
    path_only_mi = []
    path_only_mvo = []
    multi_scores = {"LV": [], "MYO": [], "MI": [], "MVO": [], "Infarct": []}
    multi_path_only_mi = []
    multi_path_only_mvo = []
    class_correct = 0
    class_total = 0
    # Do NOT apply sparse-MI suppression while selecting checkpoints — it can
    # freeze all-empty infarct Dice at ~N_normal/N_val and save dead weights
    # (SegResNet fold0 collapse). Suppression remains for final test eval.
    apply_suppress = False
    mi_cls = int(getattr(cfg, "MULTICLASS_MI", 3))
    mvo_cls = int(getattr(cfg, "MULTICLASS_MVO", 4))
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        if is_multiclass_variant(variant):
            pred = out["multiclass_logits"].argmax(1)
            if apply_suppress and getattr(cfg, "MI_VOXEL_SUPPRESSION", True):
                thr = int(getattr(cfg, "MIN_MI_VOXELS", 50))
                pred = pred.clone()
                for b in range(pred.shape[0]):
                    mi_vox = pred[b] == mi_cls
                    if mi_vox.sum() < thr:
                        pred[b][mi_vox] = 2
            pred_np = pred.cpu().numpy()
            gt = batch["multiclass"].numpy()
            names = batch.get("name", [""] * pred_np.shape[0])
            for b in range(pred_np.shape[0]):
                lv_m = binary_metrics(pred_np[b] == 1, gt[b] == 1, spacing=cfg.TARGET_SPACING)
                myo_m = binary_metrics(pred_np[b] == 2, gt[b] == 2, spacing=cfg.TARGET_SPACING)
                mi_m = binary_metrics(
                    pred_np[b] == mi_cls, gt[b] == mi_cls, spacing=cfg.TARGET_SPACING
                )
                mvo_m = binary_metrics(
                    pred_np[b] == mvo_cls, gt[b] == mvo_cls, spacing=cfg.TARGET_SPACING
                )
                inf_m = binary_metrics(
                    np.isin(pred_np[b], [mi_cls, mvo_cls]),
                    np.isin(gt[b], [mi_cls, mvo_cls]),
                    spacing=cfg.TARGET_SPACING,
                )
                multi_scores["LV"].append(lv_m)
                multi_scores["MYO"].append(myo_m)
                multi_scores["MI"].append(mi_m)
                multi_scores["MVO"].append(mvo_m)
                multi_scores["Infarct"].append(inf_m)
                is_path = bool((gt[b] == mi_cls).sum() > 0 or (gt[b] == mvo_cls).sum() > 0)
                if "pathological" in batch:
                    is_path = is_path or bool(batch["pathological"][b].item() > 0.5)
                name = names[b] if b < len(names) else ""
                if name:
                    u = str(name).upper()
                    is_path = is_path or u.startswith("CASE_P") or u.startswith("P")
                if is_path:
                    multi_path_only_mi.append(mi_m)
                    multi_path_only_mvo.append(mvo_m)
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
                # Pathological-only: GT has any MI voxels OR Case_P*
                is_path = gt_p[b, 0].sum() > 0
                if "pathological" in batch:
                    is_path = is_path or bool(batch["pathological"][b].item() > 0.5)
                if is_path:
                    path_only_mi.append(mi_m)
                    path_only_mvo.append(mvo_m)
            # Disease classification accuracy
            if "disease_prob" in out and "pathological" in batch:
                pred_c = (out["disease_prob"].view(-1) > getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5)).float()
                gt_c = batch["pathological"].to(device).view(-1)
                class_correct += int((pred_c == gt_c).sum().item())
                class_total += int(gt_c.numel())
    if is_multiclass_variant(variant):
        out_m = {k: summarize(v) for k, v in multi_scores.items()}
        if multi_path_only_mi:
            out_m["MI_pathological"] = summarize(multi_path_only_mi)
        if multi_path_only_mvo:
            out_m["MVO_pathological"] = summarize(multi_path_only_mvo)
        return out_m
    out_m = {
        "LV": summarize(anat_scores["LV"]),
        "MYO": summarize(anat_scores["MYO"]),
        "MI": summarize(path_scores["MI"]),
        "MVO": summarize(path_scores["MVO"]),
    }
    if path_only_mi:
        out_m["MI_pathological"] = summarize(path_only_mi)
    if path_only_mvo:
        out_m["MVO_pathological"] = summarize(path_only_mvo)
    if class_total > 0:
        out_m["disease_acc"] = class_correct / class_total
    return out_m


def primary_score(metrics: dict, variant: str) -> float:
    """
    Checkpoint selection metric.

    Prefer pathological-only *pure MI* Dice (thesis primary), but add a small
    LV+MYO bonus so all-background models cannot lock on empty–empty Dice
    (normals → 1.0) while anatomy is still zero — the SegResNet fold-0 bug.
    """
    lv = float(metrics.get("LV", {}).get("dice", {}).get("mean", 0.0) or 0.0)
    myo = float(metrics.get("MYO", {}).get("dice", {}).get("mean", 0.0) or 0.0)
    anat_bonus = 0.05 * (lv + myo)

    if "MI_pathological" in metrics:
        mi_p = float(metrics["MI_pathological"]["dice"]["mean"])
        return mi_p + anat_bonus

    # Legacy fallbacks
    if "Infarct_pathological" in metrics:
        return float(metrics["Infarct_pathological"]["dice"]["mean"]) + anat_bonus
    key = "MI" if "MI" in metrics else ("Infarct" if "Infarct" in metrics else None)
    if key is None or "dice" not in metrics[key]:
        return -1.0
    return float(metrics[key]["dice"]["mean"]) + anat_bonus


def _maybe_warm_start(
    model,
    variant: str,
    init_from: str | None,
    device,
    fold: int | None = None,
):
    if not init_from:
        return
    from data.cv_splits import ckpt_name

    src = init_from.upper()
    ckpt_path = cfg.CHECKPOINT_DIR / ckpt_name(src, fold)
    if not ckpt_path.exists():
        print(f"  [warn] warm-start skipped: missing {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = _load_state_dict(
        model, ckpt["model"], strict=False, disable_missing_classifier=False
    )
    print(
        f"  [ok] warm-started {variant} from {ckpt_path.name} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _default_batch_size(variant: str, cli_batch: int | None) -> int:
    if cli_batch is not None:
        return cli_batch
    per_model = getattr(cfg, "BASELINE_BATCH_SIZES", {})
    if variant in per_model:
        return int(per_model[variant])
    if variant in PYTORCH_BASELINE_VARIANTS:
        return int(getattr(cfg, "BASELINE_BATCH_SIZE", cfg.BATCH_SIZE))
    return int(cfg.BATCH_SIZE)


def train_variant(
    variant: str,
    epochs: int,
    batch_size: int | None,
    device: torch.device,
    init_from: str | None = None,
    fold: int | None = None,
):
    from data.cv_splits import (
        ckpt_name,
        get_fold_splits,
        history_name,
    )

    variant = variant.upper()
    batch_size = _default_batch_size(variant, batch_size)
    fold_tag = f" fold{fold}" if fold is not None else ""
    print(
        f"\n{'=' * 70}\nTraining {variant}{fold_tag} "
        f"({VARIANT_SHORT.get(variant, variant)}) | {MODEL_NAME}\n{'=' * 70}"
    )

    if fold is not None:
        splits = get_fold_splits(fold)
        train_ds = EMIDECDataset(case_names=splits["train"], augment=True)
        val_ds = EMIDECDataset(case_names=splits["val"], augment=False)
        print(
            f"  CV fold{fold}: train={len(splits['train'])}  "
            f"val={len(splits['val'])}  test={len(splits['test'])} (held out)"
        )
    else:
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
        # Disease classifier is M5-only (build_model also enforces this)
        use_disease_classifier=(
            variant == "M5" and getattr(cfg, "USE_DISEASE_CLASSIFIER", True)
        ),
        gate_pathology_by_disease=(
            variant == "M5" and getattr(cfg, "GATE_PATHOLOGY_BY_DISEASE", True)
        ),
        disease_threshold=getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5),
    ).to(device)
    # M5 default: warm-start from M4 (same fold under CV)
    if variant == "M5" and init_from is None:
        init_from = "M4"
    _maybe_warm_start(model, variant, init_from, device, fold=fold)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params / 1e6:.3f} M  |  batch_size={batch_size}  |  epochs={epochs}")
    criterion = JointLoss(
        variant=variant,
        num_anatomy=cfg.NUM_ANATOMY_CLASSES,
        anatomy_weights=torch.tensor(cfg.ANATOMY_CE_WEIGHTS, dtype=torch.float32),
        lambda_ftl=cfg.LAMBDA_FTL,
        lambda_topo=cfg.LAMBDA_TOPO,
        lambda_class=getattr(cfg, "LAMBDA_CLASS", 0.5),
        ftl_alpha=cfg.FTL_ALPHA,
        ftl_beta=cfg.FTL_BETA,
        ftl_gamma=cfg.FTL_GAMMA,
        use_gt_myo_for_topo=getattr(cfg, "USE_GT_MYO_FOR_TOPO", True),
        mi_channel_weight=getattr(cfg, "MI_CHANNEL_WEIGHT", 1.5),
        mvo_channel_weight=getattr(cfg, "MVO_CHANNEL_WEIGHT", 0.75),
        path_loss_on_pathological_only=getattr(cfg, "PATH_LOSS_ON_PATHOLOGICAL_ONLY", True),
    ).to(device)
    optim = Adam(model.parameters(), lr=cfg.LR)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.MIN_LR)
    ckpt_dir = cfg.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / ckpt_name(variant, fold)
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
            if "pathological" in batch:
                batch_dev["pathological"] = batch["pathological"].to(device)
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
            "fold": fold,
            "loss": float(np.mean(losses)),
            "lr": float(sched.get_last_lr()[0]),
            "lambda_topo": lam,
            "val": val_metrics,
            "primary_mi_dice": mi,
            "sec": time.time() - t0,
        }
        history.append(row)
        tag = f"{variant}" + (f"/f{fold}" if fold is not None else "")
        if is_multiclass_variant(variant):
            path_mi = val_metrics.get("MI_pathological", {}).get("dice", {}).get("mean", float("nan"))
            msg = (
                f"[{tag}] ep {epoch:03d}/{epochs}  loss={row['loss']:.4f}  "
                f"LV={val_metrics['LV']['dice']['mean']:.3f}  "
                f"MYO={val_metrics['MYO']['dice']['mean']:.3f}  "
                f"MI={val_metrics['MI']['dice']['mean']:.3f}  "
                f"MI_path={path_mi:.3f}  "
                f"MVO={val_metrics['MVO']['dice']['mean']:.3f}  "
                f"({row['sec']:.1f}s)"
            )
        else:
            path_mi = val_metrics.get("MI_pathological", {}).get("dice", {}).get("mean", float("nan"))
            d_acc = val_metrics.get("disease_acc", float("nan"))
            d_acc_s = f"{d_acc:.3f}" if isinstance(d_acc, float) and not np.isnan(d_acc) else "n/a"
            msg = (
                f"[{tag}] ep {epoch:03d}/{epochs}  loss={row['loss']:.4f}  "
                f"λtopo={lam:.3f}  "
                f"LV={val_metrics['LV']['dice']['mean']:.3f}  "
                f"MYO={val_metrics['MYO']['dice']['mean']:.3f}  "
                f"MI={val_metrics['MI']['dice']['mean']:.3f}  "
                f"MI_path={path_mi:.3f}  "
                f"MVO={val_metrics['MVO']['dice']['mean']:.3f}  "
                f"ClsAcc={d_acc_s}  "
                f"({row['sec']:.1f}s)"
            )
        print(msg)
        if mi > best_mi and not np.isnan(mi):
            best_mi = mi
            torch.save(
                {
                    "variant": variant,
                    "fold": fold,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "metrics": val_metrics,
                    "params": n_params,
                    "lambda_topo": lam,
                    "use_disease_classifier": (
                        variant == "M5" and getattr(cfg, "USE_DISEASE_CLASSIFIER", True)
                    ),
                    "protocol": "5-fold-cv" if fold is not None else "single-split",
                    "primary_metric": "MI_pathological + 0.05*(LV+MYO)",
                },
                best_path,
            )
            print(
                f"  [ok] saved best -> {best_path.name} "
                f"(primary={best_mi:.4f} | MI_path + anat bonus)"
            )
    hist_path = cfg.RESULTS_DIR / history_name(variant, fold)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"History -> {hist_path}")
    return best_path, best_mi, n_params


def _resolve_variants(spec: str) -> list[str]:
    s = spec.strip().lower()
    if s == "all":
        return list(ABLATION_VARIANTS)
    if s in ("baselines", "baseline"):
        # Native PyTorch only here; real nnU-Net via src.nnunet_emidec.
        return list(PYTORCH_BASELINE_VARIANTS)
    if s == "everything":
        return list(ABLATION_VARIANTS) + list(PYTORCH_BASELINE_VARIANTS)
    variants = [v.strip().upper() for v in spec.split(",") if v.strip()]
    if any(is_real_nnunet(v) for v in variants):
        raise SystemExit(
            "NNUNET (real nnU-Net v2) is trained via:\n"
            "  python -m src.nnunet_emidec prepare\n"
            "  python -m src.nnunet_emidec train --cv\n"
            "  python -m src.nnunet_emidec eval --cv\n"
            "MONAI DynUNet-Res is variant DYNUNET_RES."
        )
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="M5",
        help="M1-M5 or a registered PyTorch baseline; comma-list, 'all' "
        "(ablation), 'baselines', or 'everything'. "
        "Real nnU-Net: python -m src.nnunet_emidec",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs (CV default: config.CV_EPOCHS for ALL models; "
        "single-split: EPOCHS / BASELINE_EPOCHS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override model-specific default batch size",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Warm-start weights from another variant (default M5<-M4, same fold)",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable default M5<-M4 warm-start",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="5-fold CV: same folds + same epochs for every model",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single fold (0..N_FOLDS-1). Implies CV protocol for that fold.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="CV only: skip jobs whose history already has CV_EPOCHS epochs "
        "(safe resume after crash / power loss)",
    )
    args = parser.parse_args()
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model family: {MODEL_NAME}")
    print(f"Device: {device}")

    use_cv = bool(args.cv or args.fold is not None)
    variants = _resolve_variants(args.variant)
    summary: dict = {}

    if use_cv:
        from data.cv_splits import ckpt_name, ensure_folds, history_name
        from data.preprocess import sync_all_pool_from_splits

        sync_all_pool_from_splits()
        folds_meta = ensure_folds(
            n_folds=getattr(cfg, "N_FOLDS", 5),
            seed=cfg.SEED,
            overwrite=False,
        )
        n_folds = int(folds_meta["n_folds"])
        fold_list = [args.fold] if args.fold is not None else list(range(n_folds))
        for f in fold_list:
            if f < 0 or f >= n_folds:
                raise SystemExit(f"--fold must be in 0..{n_folds - 1}, got {f}")
        # Identical epoch budget for ALL models under CV (fair comparison)
        epochs = int(args.epochs if args.epochs is not None else getattr(cfg, "CV_EPOCHS", cfg.EPOCHS))
        print(
            f"5-fold CV protocol | folds={fold_list} | epochs={epochs} (all variants) | "
            f"seed={folds_meta['seed']}"
            + (" | skip-done" if args.skip_done else "")
        )
        # Outer fold, inner variant → M4_foldk exists before M5_foldk warm-start
        for fold in fold_list:
            for v in variants:
                if args.skip_done:
                    hist_path = cfg.RESULTS_DIR / history_name(v, fold)
                    if hist_path.exists():
                        try:
                            hist = json.loads(hist_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            hist = []
                        if len(hist) >= epochs:
                            print(
                                f"[skip] {v} fold{fold}: already complete "
                                f"({len(hist)}/{epochs} epochs) -> {hist_path.name}"
                            )
                            summary[f"{v}_fold{fold}"] = {
                                "best_ckpt": str(cfg.CHECKPOINT_DIR / ckpt_name(v, fold)),
                                "best_mi_dice": float(hist[-1].get("primary_mi_dice", -1)),
                                "params": None,
                                "epochs": epochs,
                                "fold": fold,
                                "protocol": "5-fold-cv",
                                "skipped": True,
                            }
                            continue
                init_from = None if args.no_warm_start else args.init_from
                path, score, n_params = train_variant(
                    v,
                    epochs,
                    args.batch_size,
                    device,
                    init_from=init_from,
                    fold=fold,
                )
                summary[f"{v}_fold{fold}"] = {
                    "best_ckpt": str(path),
                    "best_mi_dice": score,
                    "params": n_params,
                    "epochs": epochs,
                    "fold": fold,
                    "protocol": "5-fold-cv",
                }
        sum_path = cfg.RESULTS_DIR / "train_summary_cv.json"
    else:
        for v in variants:
            if args.epochs is not None:
                epochs = args.epochs
            elif v in PYTORCH_BASELINE_VARIANTS:
                epochs = int(getattr(cfg, "BASELINE_EPOCHS", 80))
            else:
                epochs = int(cfg.EPOCHS)
            init_from = None if args.no_warm_start else args.init_from
            path, score, n_params = train_variant(
                v, epochs, args.batch_size, device, init_from=init_from, fold=None
            )
            summary[v] = {
                "best_ckpt": str(path),
                "best_mi_dice": score,
                "params": n_params,
                "epochs": epochs,
                "protocol": "single-split",
            }
        sum_path = cfg.RESULTS_DIR / "train_summary.json"

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {sum_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
