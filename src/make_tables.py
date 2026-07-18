"""Build methodology tables from single-split or 5-fold CV evaluation JSON."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from model_identity import (
    ABLATION_VARIANTS,
    BASELINE_VARIANTS,
    MODEL_NAME,
    MODEL_FULL_NAME,
    SOTA_BENCHMARKS,
    TARGET_MI_DICE,
    TARGET_MI_DICE_LABEL,
    VARIANT_SHORT,
    is_multiclass_variant,
)


def _dice(summary, key):
    if key not in summary or "dice" not in summary[key]:
        return None
    return summary[key]["dice"]["mean"]


def _dice_std(summary, key):
    if key not in summary or "dice" not in summary[key]:
        return None
    return summary[key]["dice"].get("std")


def load_variant(variant: str, split: str = "test"):
    path = cfg.RESULTS_DIR / f"{variant}_{split}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def load_cv_variant(variant: str):
    path = cfg.RESULTS_DIR / f"{variant}_cv_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("aggregate")


def _collect_rows(variants: list[str], split: str = "test", use_cv: bool = False):
    rows = []
    for v in variants:
        s = load_cv_variant(v) if use_cv else load_variant(v, split)
        if s is None:
            rows.append(
                {
                    "variant": v,
                    "status": "missing",
                    "display": VARIANT_SHORT.get(v, v),
                    "protocol": "5-fold-cv" if use_cv else "single-split",
                }
            )
            continue
        multi = is_multiclass_variant(v)
        mi_key = "Infarct" if multi else "MI"
        mi_path_key = "Infarct_pathological" if multi else "MI_pathological"
        rows.append(
            {
                "variant": v,
                "display": VARIANT_SHORT.get(v, v),
                "LV": _dice(s, "LV"),
                "LV_std": _dice_std(s, "LV") if use_cv else None,
                "MYO": _dice(s, "MYO"),
                "MYO_std": _dice_std(s, "MYO") if use_cv else None,
                "MI": _dice(s, mi_key),
                "MI_std": _dice_std(s, mi_key) if use_cv else None,
                "MI_path": _dice(s, mi_path_key),
                "MI_path_std": _dice_std(s, mi_path_key) if use_cv else None,
                "MVO": None if multi else _dice(s, "MVO"),
                "MVO_std": None if multi else (_dice_std(s, "MVO") if use_cv else None),
                "params_M": s.get("params_M"),
                "inference_ms": s.get("inference_ms_mean"),
                "protocol": "5-fold-cv" if use_cv else "single-split",
            }
        )
    return rows


def _fmt(mean, std=None):
    if not isinstance(mean, float):
        return "   -  "
    if isinstance(std, float):
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean:.3f}"


def _print_table(title: str, rows: list[dict], use_cv: bool = False):
    print(f"\n{title}")
    if use_cv:
        print(
            f"{'Var':<10} {'Display':<24} {'LV':>11} {'MYO':>11} "
            f"{'MI_path':>11} {'MI_all':>11} {'MVO':>11} {'Params':>8}"
        )
    else:
        print(
            f"{'Var':<10} {'Display':<24} {'LV':>7} {'MYO':>7} "
            f"{'MI_path':>7} {'MI_all':>7} {'MVO':>7} {'Params':>8} {'ms':>7}"
        )
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['variant']:<10} {r['display']:<24}  (not evaluated yet)")
            continue
        params = r.get("params_M")
        params_s = f"{params:>7.2f}M" if isinstance(params, (int, float)) else "   -   "
        if use_cv:
            print(
                f"{r['variant']:<10} {r['display']:<24} "
                f"{_fmt(r['LV'], r.get('LV_std')):>11} "
                f"{_fmt(r['MYO'], r.get('MYO_std')):>11} "
                f"{_fmt(r.get('MI_path'), r.get('MI_path_std')):>11} "
                f"{_fmt(r['MI'], r.get('MI_std')):>11} "
                f"{_fmt(r['MVO'], r.get('MVO_std')):>11} "
                f"{params_s}"
            )
        else:
            ms = r.get("inference_ms")
            ms_s = f"{ms:>6.1f}" if isinstance(ms, (int, float)) else "   -  "
            print(
                f"{r['variant']:<10} {r['display']:<24} {_fmt(r['LV']):>7} {_fmt(r['MYO']):>7} "
                f"{_fmt(r.get('MI_path')):>7} {_fmt(r['MI']):>7} {_fmt(r['MVO']):>7} "
                f"{params_s} {ms_s}"
            )


def _write_csv(path: Path, rows: list[dict], use_cv: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_cv:
        fields = [
            "variant",
            "display",
            "protocol",
            "LV",
            "LV_std",
            "MYO",
            "MYO_std",
            "MI_path",
            "MI_path_std",
            "MI",
            "MI_std",
            "MVO",
            "MVO_std",
            "params_M",
            "inference_ms",
        ]
    else:
        fields = [
            "variant",
            "display",
            "protocol",
            "LV",
            "MYO",
            "MI_path",
            "MI",
            "MVO",
            "params_M",
            "inference_ms",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r.get("status") == "missing":
                continue
            w.writerow(r)


def _write_per_fold_csv(path: Path, variants: list[str]) -> int:
    """One row per variant x fold from *_cv_metrics.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "fold",
        "LV",
        "MYO",
        "MI_path",
        "MI_all",
        "MVO",
        "params_M",
        "n_cases",
    ]
    rows_out = []
    for v in variants:
        p = cfg.RESULTS_DIR / f"{v}_cv_metrics.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        multi = is_multiclass_variant(v)
        for item in data.get("folds", []):
            s = item.get("summary", {})
            mi_key = "Infarct" if multi else "MI"
            mi_path_key = "Infarct_pathological" if multi else "MI_pathological"
            rows_out.append(
                {
                    "variant": v,
                    "fold": item.get("fold"),
                    "LV": _dice(s, "LV"),
                    "MYO": _dice(s, "MYO"),
                    "MI_path": _dice(s, mi_path_key),
                    "MI_all": _dice(s, mi_key),
                    "MVO": None if multi else _dice(s, "MVO"),
                    "params_M": s.get("params_M"),
                    "n_cases": s.get("n_cases"),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return len(rows_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use *_cv_metrics.json (5-fold mean±std). Preferred for SOTA comparison.",
    )
    parser.add_argument("--split", default="test", help="Single-split metrics suffix")
    args = parser.parse_args()

    use_cv = bool(args.cv)
    ablation_rows = _collect_rows(list(ABLATION_VARIANTS), args.split, use_cv=use_cv)
    baseline_rows = _collect_rows(list(BASELINE_VARIANTS), args.split, use_cv=use_cv)
    all_rows = ablation_rows + baseline_rows

    protocol = (
        "5-fold stratified CV (same folds + same epochs for all models)"
        if use_cv
        else f"Single stratified split ({args.split})"
    )
    out = {
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "protocol": protocol,
        "ablation_table": ablation_rows,
        "baseline_table": baseline_rows,
        "sota_targets": SOTA_BENCHMARKS,
        "target_mi_dice": TARGET_MI_DICE,
        "primary_metric": "MI_path",
        "note": (
            f"EMIDEC-only SOTA. Target: {MODEL_NAME} MI_path Dice > {TARGET_MI_DICE} "
            f"({TARGET_MI_DICE_LABEL}); stretch > 0.783 (ICPIU-Net 5-fold). "
            f"Protocol: {protocol}. "
            "PRIMARY: MI_path (pathological only). "
            "Under --cv, values are mean±std across 5 folds. "
            "All models share Dataset/folds.json and CV_EPOCHS."
        ),
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paper_dir = cfg.RESULTS_DIR / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    fname = "paper_tables_cv.json" if use_cv else "paper_tables.json"
    path = cfg.RESULTS_DIR / fname
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if use_cv:
        csv_abl = paper_dir / "cv_ablation_metrics.csv"
        csv_base = paper_dir / "cv_baseline_metrics.csv"
        csv_all = paper_dir / "cv_all_metrics.csv"
        csv_folds = paper_dir / "cv_per_fold_metrics.csv"
        _write_csv(csv_abl, ablation_rows, use_cv=True)
        _write_csv(csv_base, baseline_rows, use_cv=True)
        _write_csv(csv_all, all_rows, use_cv=True)
        n_fold_rows = _write_per_fold_csv(
            csv_folds, list(ABLATION_VARIANTS) + list(BASELINE_VARIANTS)
        )
        csv_paths = [csv_abl, csv_base, csv_all, csv_folds]
    else:
        csv_all = paper_dir / f"{args.split}_all_metrics.csv"
        _write_csv(csv_all, all_rows, use_cv=False)
        csv_paths = [csv_all]
        n_fold_rows = 0

    print(f"{MODEL_NAME} results | {protocol}")
    print("PRIMARY thesis metric = MI_path (pathological only).")
    _print_table(f"{MODEL_NAME} ablation", ablation_rows, use_cv=use_cv)
    _print_table("External baselines (MONAI)", baseline_rows, use_cv=use_cv)
    print(f"\nSaved JSON -> {path}")
    for p in csv_paths:
        print(f"Saved CSV  -> {p}")
    if use_cv:
        print(f"Per-fold rows written: {n_fold_rows}")


if __name__ == "__main__":
    main()
