"""Build methodology Tables 4.6 / 4.7 from evaluation JSON results."""
from __future__ import annotations

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


def load_variant(variant: str, split: str = "test"):
    path = cfg.RESULTS_DIR / f"{variant}_{split}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def _collect_rows(variants: list[str], split: str = "test"):
    rows = []
    for v in variants:
        s = load_variant(v, split)
        if s is None:
            rows.append({"variant": v, "status": "missing", "display": VARIANT_SHORT.get(v, v)})
            continue
        multi = is_multiclass_variant(v)
        mi_key = "Infarct" if multi else "MI"
        mi_path_key = "Infarct_pathological" if multi else "MI_pathological"
        rows.append(
            {
                "variant": v,
                "display": VARIANT_SHORT.get(v, v),
                "LV": _dice(s, "LV"),
                "MYO": _dice(s, "MYO"),
                "MI": _dice(s, mi_key),
                "MI_path": _dice(s, mi_path_key),
                "MVO": None if multi else _dice(s, "MVO"),
                "params_M": s.get("params_M"),
                "inference_ms": s.get("inference_ms_mean"),
            }
        )
    return rows


def _print_table(title: str, rows: list[dict]):
    print(f"\n{title}")
    print(
        f"{'Var':<10} {'Display':<24} {'LV':>7} {'MYO':>7} {'MI':>7} {'MI_path':>7} {'MVO':>7} {'Params':>8} {'ms':>7}"
    )
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['variant']:<10} {r['display']:<24}  (not evaluated yet)")
            continue

        def f(x):
            return f"{x:.3f}" if isinstance(x, float) else "   -  "

        params = r.get("params_M")
        ms = r.get("inference_ms")
        params_s = f"{params:>7.2f}M" if isinstance(params, (int, float)) else "   -   "
        ms_s = f"{ms:>6.1f}" if isinstance(ms, (int, float)) else "   -  "
        print(
            f"{r['variant']:<10} {r['display']:<24} {f(r['LV']):>7} {f(r['MYO']):>7} "
            f"{f(r['MI']):>7} {f(r.get('MI_path')):>7} {f(r['MVO']):>7} "
            f"{params_s} {ms_s}"
        )


def main():
    ablation_rows = _collect_rows(list(ABLATION_VARIANTS), "test")
    baseline_rows = _collect_rows(list(BASELINE_VARIANTS), "test")

    out = {
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "ablation_table": ablation_rows,
        "baseline_table": baseline_rows,
        "sota_targets": SOTA_BENCHMARKS,
        "target_mi_dice": TARGET_MI_DICE,
        "note": (
            f"EMIDEC-only SOTA. Target: {MODEL_NAME} MI Dice > {TARGET_MI_DICE} "
            f"({TARGET_MI_DICE_LABEL}); stretch > 0.783 (ICPIU-Net 5-fold). "
            "MI = all-case nanmean (FP on normals → 0). "
            "MI_path = pathological cases only (matches train 'primary MI Dice' / EMIDEC practice). "
            "Baselines are 4-class multiclass (Infarct = MI+MVO)."
        ),
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.RESULTS_DIR / "paper_tables.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} results (test Dice)")
    print(
        "Note: train 'saved best primary MI Dice' = MI_path (pathological only, usually val). "
        "MI column below = all cases on TEST (normals with FPs drag it down)."
    )
    _print_table(f"{MODEL_NAME} ablation", ablation_rows)
    _print_table("External baselines (MONAI)", baseline_rows)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
