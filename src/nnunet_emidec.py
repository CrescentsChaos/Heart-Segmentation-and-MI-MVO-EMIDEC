# -*- coding: utf-8 -*-
"""
Real nnU-Net v2 baseline on EMIDEC (variant NNUNET).

This is NOT the MONAI DynUNet-Res formerly mislabeled as nnU-Net.
That model is now DYNUNET_RES.

Pipeline (same folds + 80 epochs as AFDD CV protocol):
  python -m src.nnunet_emidec prepare
  python -m src.nnunet_emidec train --cv
  python -m src.nnunet_emidec eval --cv

Requires: pip install nnunetv2
Env dirs are created under config.NNUNET_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from data.cv_splits import ensure_folds, get_fold_splits, cv_metrics_name, metrics_name
from metrics import binary_metrics, summarize
from evaluate import aggregate_fold_summaries, _is_pathological_case


def _setup_nnunet_env() -> Dict[str, Path]:
    root = Path(getattr(cfg, "NNUNET_ROOT", ROOT / "nnunet_data"))
    raw = root / "nnUNet_raw"
    pre = root / "nnUNet_preprocessed"
    res = root / "nnUNet_results"
    for p in (raw, pre, res):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(res)
    # Custom 80-epoch trainer lives in repo (not site-packages)
    ext = str(ROOT / "nnunet_trainers")
    prev = os.environ.get("nnUNet_extTrainer", "")
    if ext not in prev.split(os.pathsep):
        os.environ["nnUNet_extTrainer"] = (
            ext if not prev else ext + os.pathsep + prev
        )
    os.environ.setdefault("nnUNet_n_proc_DA", "2")
    return {"root": root, "raw": raw, "preprocessed": pre, "results": res}


def _dataset_folder_name() -> str:
    did = int(getattr(cfg, "NNUNET_DATASET_ID", 501))
    name = getattr(cfg, "NNUNET_DATASET_NAME", "EMIDEC")
    return f"Dataset{did:03d}_{name}"


def _require_nnunet():
    try:
        import nnunetv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "nnunetv2 is not installed. Install with:\n"
            "  pip install nnunetv2\n"
            f"({exc})"
        ) from exc


def prepare(overwrite: bool = False) -> Path:
    """
    Convert EMIDEC NIfTI → nnU-Net raw Dataset (5 labels: BG/LV/MYO/MI/MVO).
    Labels match official EMIDEC integers 0–4 (pure MI = 3).
    """
    _require_nnunet()
    paths = _setup_nnunet_env()
    ds_name = _dataset_folder_name()
    ds_dir = paths["raw"] / ds_name
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    if ds_dir.exists() and not overwrite:
        print(f"[ok] raw dataset exists: {ds_dir} (pass --overwrite to rebuild)")
        return ds_dir

    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    emidec = Path(cfg.EMIDEC_ROOT)
    cases = sorted(
        [
            p.name
            for p in emidec.iterdir()
            if p.is_dir() and p.name.startswith("Case_")
        ]
    )
    if not cases:
        raise FileNotFoundError(f"No Case_* folders under {emidec}")

    n_ok = 0
    for name in cases:
        img_p = emidec / name / "Images" / f"{name}.nii.gz"
        lbl_p = emidec / name / "Contours" / f"{name}.nii.gz"
        if not img_p.exists() or not lbl_p.exists():
            print(f"  [skip] missing files for {name}")
            continue
        # nnU-Net expects {CASE}_0000.nii.gz for channel 0
        shutil.copy2(img_p, images_tr / f"{name}_0000.nii.gz")
        shutil.copy2(lbl_p, labels_tr / f"{name}.nii.gz")
        n_ok += 1

    dataset_json = {
        "channel_names": {"0": "LGE"},
        "labels": {
            "background": 0,
            "LV": 1,
            "MYO": 2,
            "MI": 3,
            "MVO": 4,
        },
        "numTraining": n_ok,
        "file_ending": ".nii.gz",
        "name": getattr(cfg, "NNUNET_DATASET_NAME", "EMIDEC"),
        "description": (
            "EMIDEC LGE-MRI for AFDD-Net thesis baseline. "
            "Pure MI (label 3) and MVO (label 4) kept separate."
        ),
        "reference": "Lalande et al. Data 2020; AFDD-Net folds.json",
        "licence": "EMIDEC challenge terms",
        "tensorImageSize": "3D",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    print(f"[ok] wrote {n_ok} cases -> {ds_dir}")

    # Plan + preprocess
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )

    dataset_id = int(getattr(cfg, "NNUNET_DATASET_ID", 501))
    print("[..] extracting fingerprint / planning / preprocessing (this can take a while)...")
    extract_fingerprints([dataset_id], num_processes=2, check_dataset_integrity=True)
    plan_experiments([dataset_id])
    preprocess(
        [dataset_id],
        configurations=[getattr(cfg, "NNUNET_CONFIGURATION", "3d_fullres")],
        num_processes=[2],
    )

    write_splits_from_folds()
    return ds_dir


def write_splits_from_folds() -> Path:
    """
    Map AFDD folds.json → nnU-Net splits_final.json.

    For each fold k: nnU-Net val = our held-out test fold (so CV Dice matches).
    nnU-Net train = remaining cases (our train+val pool).
    """
    paths = _setup_nnunet_env()
    ds_name = _dataset_folder_name()
    pre_dir = paths["preprocessed"] / ds_name
    pre_dir.mkdir(parents=True, exist_ok=True)

    meta = ensure_folds(overwrite=False)
    splits = []
    for fold_idx in range(int(meta["n_folds"])):
        sp = get_fold_splits(fold_idx)
        # Hold out the same test cases our other models use
        val_ids = list(sp["test"])
        train_ids = list(sp["train"]) + list(sp["val"])
        splits.append({"train": train_ids, "val": val_ids})
        print(
            f"  fold{fold_idx}: nnUNet train={len(train_ids)}  "
            f"val/test={len(val_ids)}"
        )

    out = pre_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    # Also copy next to raw for some nnU-Net versions
    raw_copy = paths["raw"] / ds_name / "splits_final.json"
    if (paths["raw"] / ds_name).exists():
        raw_copy.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    return out


def train_fold(fold: int, continue_training: bool = False) -> None:
    _require_nnunet()
    paths = _setup_nnunet_env()
    write_splits_from_folds()

    from nnunetv2.run.run_training import run_training

    dataset_id = str(int(getattr(cfg, "NNUNET_DATASET_ID", 501)))
    configuration = getattr(cfg, "NNUNET_CONFIGURATION", "3d_fullres")
    trainer_name = getattr(cfg, "NNUNET_TRAINER", "nnUNetTrainerAFDD80")

    print(
        f"\n=== nnU-Net v2 train fold {fold} | {trainer_name} "
        f"| epochs={getattr(cfg, 'CV_EPOCHS', 80)} ==="
    )
    run_training(
        dataset_name_or_id=dataset_id,
        configuration=configuration,
        fold=fold,
        trainer_class_name=trainer_name,
        continue_training=continue_training,
    )
    print(f"[ok] fold {fold} training finished. Results under {paths['results']}")


def train_cv(folds: Optional[List[int]] = None, continue_training: bool = False) -> None:
    meta = ensure_folds(overwrite=False)
    fold_list = folds if folds is not None else list(range(int(meta["n_folds"])))
    for f in fold_list:
        train_fold(f, continue_training=continue_training)


def _model_folder() -> Path:
    paths = _setup_nnunet_env()
    ds_name = _dataset_folder_name()
    configuration = getattr(cfg, "NNUNET_CONFIGURATION", "3d_fullres")
    # Default layout: RESULTS/DatasetXXX/Trainer__Plans__config/fold_k
    trainer = "nnUNetTrainerAFDD80"
    plans = "nnUNetPlans"
    return paths["results"] / ds_name / f"{trainer}__{plans}__{configuration}"


def _predict_fold(fold: int, out_dir: Path) -> Path:
    _require_nnunet()
    _setup_nnunet_env()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = _model_folder()
    fold_dir = model_folder / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Missing trained fold: {fold_dir}")

    sp = get_fold_splits(fold)
    test_cases = sp["test"]
    paths = _setup_nnunet_env()
    ds_name = _dataset_folder_name()
    images_tr = paths["raw"] / ds_name / "imagesTr"

    list_of_lists = []
    case_ids = []
    for name in test_cases:
        img = images_tr / f"{name}_0000.nii.gz"
        if not img.exists():
            raise FileNotFoundError(img)
        list_of_lists.append([str(img)])
        case_ids.append(name)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=__import__("torch").device(
            "cuda" if __import__("torch").cuda.is_available() else "cpu"
        ),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        list_of_lists,
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return out_dir


def eval_fold(fold: int, save_figs: bool = False) -> dict:
    """Score nnU-Net predictions with the same pure-MI Dice as AFDD."""
    pred_dir = cfg.RESULTS_DIR / "nnunet_preds" / f"fold{fold}"
    _predict_fold(fold, pred_dir)

    paths = _setup_nnunet_env()
    ds_name = _dataset_folder_name()
    labels_tr = paths["raw"] / ds_name / "labelsTr"
    sp = get_fold_splits(fold)

    mi_cls = int(getattr(cfg, "MULTICLASS_MI", 3))
    mvo_cls = int(getattr(cfg, "MULTICLASS_MVO", 4))
    scores = {"LV": [], "MYO": [], "MI": [], "MVO": [], "Infarct": []}
    path_only = {"MI": [], "MVO": []}
    per_case = []

    for name in sp["test"]:
        pred_p = pred_dir / f"{name}.nii.gz"
        # nnU-Net may write with or without .nii.gz depending on version
        if not pred_p.exists():
            alt = list(pred_dir.glob(f"{name}*"))
            if not alt:
                print(f"  [warn] missing prediction for {name}")
                continue
            pred_p = alt[0]
        gt_p = labels_tr / f"{name}.nii.gz"
        pred = np.asanyarray(nib.load(str(pred_p)).dataobj).astype(np.int16)
        gt = np.asanyarray(nib.load(str(gt_p)).dataobj).astype(np.int16)
        # Spacing from nifti zooms (x,y,z) → our metrics expect (sx,sy,sz)
        zooms = nib.load(str(gt_p)).header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        row = {"case": name, "fold": fold}
        for cls, key in [(1, "LV"), (2, "MYO"), (mi_cls, "MI"), (mvo_cls, "MVO")]:
            m = binary_metrics(pred == cls, gt == cls, spacing=spacing)
            scores[key].append(m)
            row[key] = m
        inf = binary_metrics(
            np.isin(pred, [mi_cls, mvo_cls]),
            np.isin(gt, [mi_cls, mvo_cls]),
            spacing=spacing,
        )
        scores["Infarct"].append(inf)
        row["Infarct"] = inf
        is_path = _is_pathological_case(name, (gt == mi_cls).astype(np.float32))
        is_path = is_path or bool((gt == mvo_cls).sum() > 0)
        row["pathological"] = is_path
        if is_path:
            path_only["MI"].append(row["MI"])
            path_only["MVO"].append(row["MVO"])
        per_case.append(row)

    summary = {k: summarize(v) for k, v in scores.items() if v}
    if path_only["MI"]:
        summary["MI_pathological"] = summarize(path_only["MI"])
    if path_only["MVO"]:
        summary["MVO_pathological"] = summarize(path_only["MVO"])
    summary["params_M"] = None  # filled from plans if available
    summary["inference_ms_mean"] = None
    summary["variant"] = "NNUNET"
    summary["fold"] = fold
    summary["n_cases"] = len(per_case)
    summary["protocol"] = "5-fold-cv"
    summary["primary_metric"] = "MI_pathological"
    summary["note"] = "Real nnU-Net v2; pure MI Dice (label 3)"

    # Rough param count from plans if present
    plans_p = _model_folder() / "plans.json"
    if plans_p.exists():
        try:
            plans = json.loads(plans_p.read_text(encoding="utf-8"))
            summary["nnunet_plans"] = {
                "configurations": list(plans.get("configurations", {}).keys())
            }
        except Exception:
            pass

    out_path = cfg.RESULTS_DIR / metrics_name("NNUNET", "test", fold)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_case": per_case}, indent=2),
        encoding="utf-8",
    )
    mi_p = summary.get("MI_pathological", {}).get("dice", {}).get("mean")
    print(f"NNUNET fold{fold} MI_path={mi_p}  -> {out_path}")
    return summary


def eval_cv(folds: Optional[List[int]] = None) -> dict:
    meta = ensure_folds(overwrite=False)
    fold_list = folds if folds is not None else list(range(int(meta["n_folds"])))
    fold_summaries = []
    fold_payload = []
    for fold in fold_list:
        try:
            s = eval_fold(fold)
        except FileNotFoundError as exc:
            print(f"Skip fold{fold}: {exc}")
            continue
        fold_summaries.append(s)
        fold_payload.append({"fold": fold, "summary": s})

    aggregate = aggregate_fold_summaries(fold_summaries)
    out = {
        "variant": "NNUNET",
        "protocol": "5-fold stratified CV (real nnU-Net v2, same test folds)",
        "n_folds_requested": len(fold_list),
        "n_folds_evaluated": len(fold_summaries),
        "seed": meta.get("seed"),
        "primary_metric": "MI_pathological",
        "epochs": int(getattr(cfg, "CV_EPOCHS", 80)),
        "folds": fold_payload,
        "aggregate": aggregate,
    }
    out_path = cfg.RESULTS_DIR / cv_metrics_name("NNUNET")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    block = aggregate.get("MI_pathological", {}).get("dice")
    if block:
        print(f"NNUNET CV MI_path: {block['mean']:.4f} ± {block['std']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Real nnU-Net v2 EMIDEC baseline")
    parser.add_argument(
        "command",
        choices=["prepare", "splits", "train", "eval"],
        help="prepare=convert+plan; splits=write folds; train; eval",
    )
    parser.add_argument("--cv", action="store_true", help="All folds")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue", dest="continue_training", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(overwrite=args.overwrite)
    elif args.command == "splits":
        _setup_nnunet_env()
        write_splits_from_folds()
    elif args.command == "train":
        if args.cv or args.fold is None:
            folds = [args.fold] if args.fold is not None else None
            train_cv(folds=folds, continue_training=args.continue_training)
        else:
            train_fold(args.fold, continue_training=args.continue_training)
    elif args.command == "eval":
        if args.cv or args.fold is None:
            folds = [args.fold] if args.fold is not None else None
            eval_cv(folds=folds)
        else:
            eval_fold(args.fold)


if __name__ == "__main__":
    main()
