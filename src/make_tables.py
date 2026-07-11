"""Build methodology Tables 4.6 / 4.7 from evaluation JSON results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg
from model_identity import MODEL_NAME, MODEL_FULL_NAME, SOTA_BENCHMARKS, TARGET_MI_DICE, VARIANT_SHORT


def _dice(summary, key):
    if key not in summary or "dice" not in summary[key]:
        return None
    return summary[key]["dice"]["mean"]


def load_variant(variant: str, split: str = "test"):
    path = cfg.RESULTS_DIR / f"{variant}_{split}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def main():
    rows = []
    for v in ["M1", "M2", "M3", "M4", "M5"]:
        s = load_variant(v, "test")
        if s is None:
            rows.append({"variant": v, "status": "missing", "display": VARIANT_SHORT[v]})
            continue
        mi_key = "Infarct" if v in ("M1", "M2") else "MI"
        rows.append(
            {
                "variant": v,
                "display": VARIANT_SHORT[v],
                "LV": _dice(s, "LV"),
                "MYO": _dice(s, "MYO"),
                "MI": _dice(s, mi_key),
                "MVO": _dice(s, "MVO") if v not in ("M1", "M2") else None,
                "params_M": s.get("params_M"),
                "inference_ms": s.get("inference_ms_mean"),
            }
        )

    out = {
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "ablation_table": rows,
        "sota_targets": SOTA_BENCHMARKS,
        "target_mi_dice": TARGET_MI_DICE,
        "note": f"Target: {MODEL_NAME} (M5) infarct Dice should exceed ICPIU-Net ({TARGET_MI_DICE}).",
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.RESULTS_DIR / "paper_tables.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} ablation (test Dice)")
    print(f"{'Var':<4} {'Display':<22} {'LV':>7} {'MYO':>7} {'MI':>7} {'MVO':>7} {'Params':>8} {'ms':>7}")
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['variant']:<4} {r['display']:<22}  (not evaluated yet)")
            continue

        def f(x):
            return f"{x:.3f}" if isinstance(x, float) else "   -  "

        print(
            f"{r['variant']:<4} {r['display']:<22} {f(r['LV']):>7} {f(r['MYO']):>7} {f(r['MI']):>7} {f(r['MVO']):>7} "
            f"{r['params_M']:>7.2f}M {r['inference_ms']:>6.1f}"
        )
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
