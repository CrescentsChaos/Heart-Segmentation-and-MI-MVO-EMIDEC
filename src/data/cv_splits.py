"""Stratified 5-fold CV splits shared by ALL models (AFDD-Net + baselines).

Protocol (matches EMIDEC SOTA reporting style, e.g. Schwab / ICPIU-Net):
  - 5 folds over all 100 labeled cases, stratified by Case_N* / Case_P*
  - Fold k: test = fold k; train_pool = other 4 folds
  - Inner val = stratified fraction of train_pool (checkpoint selection only)
  - Same folds.json for every variant - never regenerate mid-experiment
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


def is_normal_case(name: str) -> bool:
    return "_N" in name


def is_path_case(name: str) -> bool:
    return "_P" in name


def make_stratified_folds(
    cases: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[str]]:
    """
    Partition cases into n_folds, stratified by normal vs pathological.
    Round-robin after shuffle ? fold sizes differ by at most 1 within each class.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    normal = [c for c in cases if is_normal_case(c)]
    path = [c for c in cases if is_path_case(c)]
    other = [c for c in cases if c not in normal and c not in path]
    if other:
        raise ValueError(f"Cases must be Case_N* or Case_P*; unexpected: {other[:5]}")

    rng = random.Random(seed)
    normal = list(normal)
    path = list(path)
    rng.shuffle(normal)
    rng.shuffle(path)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, c in enumerate(normal):
        folds[i % n_folds].append(c)
    for i, c in enumerate(path):
        folds[i % n_folds].append(c)
    for f in folds:
        f.sort()
    return folds


def _stratified_val_split(
    pool: Sequence[str],
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split pool into train / inner-val, stratified by N/P."""
    normal = [c for c in pool if is_normal_case(c)]
    path = [c for c in pool if is_path_case(c)]
    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(path)

    def take(lst: List[str]) -> Tuple[List[str], List[str]]:
        if not lst:
            return [], []
        n_va = max(1, int(round(len(lst) * val_frac))) if len(lst) > 1 else 0
        # Keep at least 1 train sample per class when possible
        if n_va >= len(lst) and len(lst) > 1:
            n_va = len(lst) - 1
        return lst[:n_va], lst[n_va:]

    nv, nt = take(normal)
    pv, pt = take(path)
    return sorted(nt + pt), sorted(nv + pv)


def fold_train_val_test(
    folds: Sequence[Sequence[str]],
    fold_idx: int,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    For outer fold `fold_idx`:
      test  = folds[fold_idx]
      pool  = union of other folds
      train / val = stratified split of pool (val only for ckpt selection)
    """
    n = len(folds)
    if fold_idx < 0 or fold_idx >= n:
        raise IndexError(f"fold_idx={fold_idx} out of range [0, {n})")
    test = sorted(folds[fold_idx])
    pool = sorted(c for i, f in enumerate(folds) if i != fold_idx for c in f)
    # Deterministic but fold-specific inner val
    train, val = _stratified_val_split(pool, val_frac=val_frac, seed=seed + 1000 + fold_idx)
    # Sanity: no leakage
    tset, vset, teset = set(train), set(val), set(test)
    assert not (tset & vset), "train/val overlap"
    assert not (tset & teset), "train/test overlap"
    assert not (vset & teset), "val/test overlap"
    assert tset | vset | teset == set(c for f in folds for c in f), "cases missing from fold split"
    return {"train": train, "val": val, "test": test}


def discover_case_names(dataset_dir: Optional[Path] = None) -> List[str]:
    """Collect Case_* names from Dataset/{all,train,val,test}/*.npz."""
    root = Path(dataset_dir or cfg.DATASET_DIR)
    names = set()
    for sub in ("all", "train", "val", "test"):
        d = root / sub
        if d.is_dir():
            for p in d.glob("Case_*.npz"):
                names.add(p.stem)
    if not names:
        raise FileNotFoundError(
            f"No Case_*.npz under {root}/{{all,train,val,test}}. Run preprocess first."
        )
    return sorted(names)


def build_case_index(dataset_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Map case name ? npz path. Prefer Dataset/all, then train/val/test."""
    root = Path(dataset_dir or cfg.DATASET_DIR)
    index: Dict[str, Path] = {}
    # Later dirs do not overwrite earlier (all has priority)
    for sub in ("test", "val", "train", "all"):
        d = root / sub
        if d.is_dir():
            for p in d.glob("*.npz"):
                index[p.stem] = p
    return index


def resolve_case_files(
    case_names: Sequence[str],
    dataset_dir: Optional[Path] = None,
) -> List[Path]:
    index = build_case_index(dataset_dir)
    missing = [n for n in case_names if n not in index]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} npz files (e.g. {missing[:5]}). "
            "Run: python -m src.data.preprocess"
        )
    return [index[n] for n in case_names]


def folds_path(dataset_dir: Optional[Path] = None) -> Path:
    return Path(dataset_dir or cfg.DATASET_DIR) / "folds.json"


def save_folds(
    folds: Sequence[Sequence[str]],
    dataset_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    n_folds: Optional[int] = None,
) -> Path:
    path = folds_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_folds": int(n_folds if n_folds is not None else len(folds)),
        "seed": int(seed if seed is not None else cfg.SEED),
        "val_frac": float(getattr(cfg, "CV_INNER_VAL_FRAC", 0.15)),
        "protocol": (
            "Stratified 5-fold CV on EMIDEC (Case_N / Case_P). "
            "Same folds for every model. Per fold: test=held-out fold; "
            "train/val=stratified split of remaining cases."
        ),
        "folds": [list(f) for f in folds],
        "counts": [
            {
                "fold": i,
                "n": len(f),
                "n_normal": sum(1 for c in f if is_normal_case(c)),
                "n_pathological": sum(1 for c in f if is_path_case(c)),
            }
            for i, f in enumerate(folds)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_folds(dataset_dir: Optional[Path] = None) -> Dict:
    path = folds_path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.data.preprocess --folds-only"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_folds(
    dataset_dir: Optional[Path] = None,
    n_folds: Optional[int] = None,
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> Dict:
    """Load folds.json or create it once from discovered cases."""
    path = folds_path(dataset_dir)
    if path.exists() and not overwrite:
        return load_folds(dataset_dir)
    n = int(n_folds if n_folds is not None else getattr(cfg, "N_FOLDS", 5))
    s = int(seed if seed is not None else cfg.SEED)
    cases = discover_case_names(dataset_dir)
    folds = make_stratified_folds(cases, n_folds=n, seed=s)
    save_folds(folds, dataset_dir=dataset_dir, seed=s, n_folds=n)
    return load_folds(dataset_dir)


def get_fold_splits(
    fold_idx: int,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    meta = ensure_folds(dataset_dir)
    val_frac = float(meta.get("val_frac", getattr(cfg, "CV_INNER_VAL_FRAC", 0.15)))
    seed = int(meta.get("seed", cfg.SEED))
    return fold_train_val_test(meta["folds"], fold_idx, val_frac=val_frac, seed=seed)


def ckpt_name(variant: str, fold: Optional[int] = None) -> str:
    v = variant.upper()
    if fold is None:
        return f"{v}_best.pth"
    return f"{v}_fold{fold}_best.pth"


def history_name(variant: str, fold: Optional[int] = None) -> str:
    v = variant.upper()
    if fold is None:
        return f"{v}_history.json"
    return f"{v}_fold{fold}_history.json"


def metrics_name(variant: str, split: str = "test", fold: Optional[int] = None) -> str:
    v = variant.upper()
    if fold is None:
        return f"{v}_{split}_metrics.json"
    return f"{v}_fold{fold}_{split}_metrics.json"


def cv_metrics_name(variant: str) -> str:
    return f"{variant.upper()}_cv_metrics.json"
