"""Dataset I/O and EMIDEC preprocessing with anatomy + pathology targets."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
import config as cfg
from src.augmentations import augment_sample

def resample_volume(volume, zooms, target_spacing=cfg.TARGET_SPACING, order=3):
    scale = [zooms[i] / target_spacing[i] for i in range(3)]
    return ndimage.zoom(volume, scale, order=order)

def normalize_volume(volume):
    mean, std = volume.mean(), volume.std()
    if std < 1e-8:
        return volume - mean
    return (volume - mean) / (std + 1e-8)

def resize_volume(volume, target_shape=cfg.TARGET_SHAPE, order=3):
    scale = [target_shape[i] / volume.shape[i] for i in range(3)]
    return ndimage.zoom(volume, scale, order=order)

def is_pathological_case_name(name: str) -> bool:
    """EMIDEC: Case_P* = pathological, Case_N* = normal/healthy."""
    u = name.upper()
    return u.startswith("CASE_P") or u.startswith("P")


def build_targets(raw_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    EMIDEC raw labels → training targets.
    Anatomy (no RV in EMIDEC):
      0 BG, 1 LV cavity, 2 full MYO wall (raw 2 ∪ 3 ∪ 4)
    Pathology (multi-label):
      channel 0 = MI (raw 3), channel 1 = MVO (raw 4)
    Multiclass (M1/M2 + MONAI baselines) — pure MI for SOTA comparison:
      0 BG, 1 LV, 2 healthy MYO (raw 2), 3 MI (raw 3), 4 MVO (raw 4)
    """
    anatomy = np.zeros(raw_mask.shape, dtype=np.uint8)
    anatomy[raw_mask == cfg.RAW_LV] = cfg.ANAT_LV
    anatomy[np.isin(raw_mask, [cfg.RAW_MYO, cfg.RAW_MI, cfg.RAW_MVO])] = cfg.ANAT_MYO
    pathology = np.zeros(raw_mask.shape + (2,), dtype=np.float32)
    pathology[..., 0] = (raw_mask == cfg.RAW_MI).astype(np.float32)
    pathology[..., 1] = (raw_mask == cfg.RAW_MVO).astype(np.float32)
    multiclass = np.zeros(raw_mask.shape, dtype=np.uint8)
    multiclass[raw_mask == cfg.RAW_LV] = cfg.MULTICLASS_LV
    multiclass[raw_mask == cfg.RAW_MYO] = cfg.MULTICLASS_MYO
    multiclass[raw_mask == cfg.RAW_MI] = cfg.MULTICLASS_MI
    multiclass[raw_mask == cfg.RAW_MVO] = cfg.MULTICLASS_MVO
    return {
        "anatomy": anatomy,
        "pathology": pathology,
        "multiclass": multiclass,
        "raw": raw_mask.astype(np.uint8),
    }

# Note: training-time augmentation (flips, in-plane rotation/scale/elastic,
# intensity jitter) now lives in augmentations.py (AFDDAugmentor) and is
# applied in EMIDECDataset.__getitem__ below, on the torch (C, D, H, W)
# tensors after the numpy (H, W, D) -> torch permute. See that module's
# docstring for why geometric ops are restricted to the in-plane (H, W)
# axes given EMIDEC's ~1.5x1.5x10mm anisotropic spacing.

# ----------------------------- Preprocess CLI --------------------------------

def stratified_split(cases: List[str], seed: int = 42):
    normal = [c for c in cases if "_N" in c]
    path = [c for c in cases if "_P" in c]
    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(path)
    def split(lst, tr=0.7, va=0.15):
        n = len(lst)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        return lst[:n_tr], lst[n_tr : n_tr + n_va], lst[n_tr + n_va :]
    nt, nv, nte = split(normal)
    pt, pv, pte = split(path)
    return sorted(nt + pt), sorted(nv + pv), sorted(nte + pte)

def process_case(case_dir: Path) -> Dict[str, np.ndarray]:
    name = case_dir.name
    img = nib.load(str(case_dir / "Images" / f"{name}.nii.gz"))
    msk = nib.load(str(case_dir / "Contours" / f"{name}.nii.gz"))
    img_data = img.get_fdata().astype(np.float32)
    mask_data = np.rint(msk.get_fdata()).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    img_r = resample_volume(img_data, zooms, order=3)
    msk_r = resample_volume(mask_data, zooms, order=0)
    img_n = normalize_volume(img_r)
    img_f = resize_volume(img_n, order=3).astype(np.float32)
    msk_f = np.rint(resize_volume(msk_r.astype(np.float32), order=0)).astype(np.uint8)
    targets = build_targets(msk_f)
    return {
        "image": img_f,
        "anatomy": targets["anatomy"],
        "pathology": targets["pathology"],
        "multiclass": targets["multiclass"],
        "raw": targets["raw"],
        "spacing": np.array(cfg.TARGET_SPACING, dtype=np.float32),
    }

def run_preprocess(emidec_root: Optional[Path] = None, out_dir: Optional[Path] = None):
    emidec_root = Path(emidec_root or cfg.EMIDEC_ROOT)
    out_dir = Path(out_dir or cfg.DATASET_DIR)
    for s in ("train", "val", "test", "all"):
        (out_dir / s).mkdir(parents=True, exist_ok=True)
    cases = sorted(
        [d for d in emidec_root.iterdir() if d.is_dir() and (d.name.startswith("Case_N") or d.name.startswith("Case_P"))]
    )
    case_names = [c.name for c in cases]
    train, val, test = stratified_split(case_names, seed=cfg.SEED)
    split_map = {n: "train" for n in train}
    split_map.update({n: "val" for n in val})
    split_map.update({n: "test" for n in test})
    meta = {
        "train": train,
        "val": val,
        "test": test,
        "target_shape": cfg.TARGET_SHAPE,
        "spacing": cfg.TARGET_SPACING,
        "note": "Legacy single split (70/15/15). Prefer Dataset/folds.json for 5-fold CV.",
    }
    (out_dir / "split.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for case_dir in cases:
        data = process_case(case_dir)
        split = split_map[case_dir.name]
        out = out_dir / split / f"{case_dir.name}.npz"
        np.savez_compressed(out, **data)
        # Flat pool for fold-based loading (no duplication of preprocess)
        all_out = out_dir / "all" / f"{case_dir.name}.npz"
        np.savez_compressed(all_out, **data)
        print(
            f"Saved {out.name} -> {split}  shape={data['image'].shape}  "
            f"MI={int(data['pathology'][..., 0].sum())} MVO={int(data['pathology'][..., 1].sum())}"
        )
    # Keep existing folds.json (do not reshuffle mid-experiment)
    from data.cv_splits import ensure_folds

    folds_meta = ensure_folds(
        out_dir, n_folds=getattr(cfg, "N_FOLDS", 5), seed=cfg.SEED, overwrite=False
    )
    print(f"Done. train={len(train)} val={len(val)} test={len(test)}")
    print(f"5-fold CV -> {out_dir / 'folds.json'}  ({folds_meta['n_folds']} folds, seed={folds_meta['seed']})")
    for c in folds_meta["counts"]:
        print(f"  fold{c['fold']}: n={c['n']} (N={c['n_normal']}, P={c['n_pathological']})")


def sync_all_pool_from_splits(out_dir: Optional[Path] = None) -> int:
    """Copy train/val/test npz into Dataset/all/ (for existing preprocess without re-run)."""
    import shutil

    out_dir = Path(out_dir or cfg.DATASET_DIR)
    all_dir = out_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in ("train", "val", "test"):
        d = out_dir / sub
        if not d.is_dir():
            continue
        for src in d.glob("*.npz"):
            dst = all_dir / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
                n += 1
    return n


# ----------------------------- Torch Dataset ---------------------------------

class EMIDECDataset(Dataset):
    """
    EMIDEC volumes from a split directory OR an explicit file / case-name list.
    Prefer case_names + Dataset/all for 5-fold CV (same pool, different folds).
    """

    def __init__(
        self,
        split_dir: Optional[Path] = None,
        augment: bool = False,
        files: Optional[List[Path]] = None,
        case_names: Optional[List[str]] = None,
    ):
        self.augment = augment
        if files is not None:
            self.files = [Path(f) for f in files]
        elif case_names is not None:
            from data.cv_splits import resolve_case_files

            self.files = resolve_case_files(case_names)
        elif split_dir is not None:
            self.files = sorted(Path(split_dir).glob("*.npz"))
        else:
            raise ValueError("Provide split_dir, files, or case_names")
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files for dataset (split_dir={split_dir}). Run preprocess first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        last_err = None
        image = anatomy = pathology = multiclass = None
        for attempt in range(3):
            try:
                with np.load(f, allow_pickle=False) as d:
                    image = d["image"].astype(np.float32).copy()
                    anatomy = d["anatomy"].astype(np.int64).copy()
                    pathology = d["pathology"].astype(np.float32).copy()
                    multiclass = d["multiclass"].astype(np.int64).copy()
                break
            except (EOFError, OSError, ValueError) as err:
                last_err = err
                if attempt == 2:
                    raise RuntimeError(f"Failed to load {f}") from last_err

        # numpy (H, W, D) -> torch (C, D, H, W)
        image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        anatomy_t = torch.from_numpy(anatomy).long().permute(2, 0, 1)
        multiclass_t = torch.from_numpy(multiclass).long().permute(2, 0, 1)
        pathology_t = torch.from_numpy(pathology).float().permute(3, 2, 0, 1)

        pathological = 1.0 if is_pathological_case_name(f.stem) else 0.0

        sample = {
            "image": image_t,
            "anatomy": anatomy_t,
            "pathology": pathology_t,
            "multiclass": multiclass_t,
            "pathological": torch.tensor(pathological, dtype=torch.float32),
            "name": f.stem,
        }

        if self.augment:
            sample = augment_sample(sample)

        return sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EMIDEC preprocess + 5-fold definition")
    parser.add_argument(
        "--folds-only",
        action="store_true",
        help="Only (re)write Dataset/folds.json from existing npz; skip NIfTI preprocess",
    )
    parser.add_argument(
        "--overwrite-folds",
        action="store_true",
        help="Replace existing folds.json (do not use mid-experiment)",
    )
    args = parser.parse_args()
    if args.folds_only:
        from data.cv_splits import ensure_folds

        n = sync_all_pool_from_splits()
        meta = ensure_folds(
            n_folds=getattr(cfg, "N_FOLDS", 5),
            seed=cfg.SEED,
            overwrite=bool(args.overwrite_folds),
        )
        print(f"Synced {n} files into Dataset/all/")
        print(f"Wrote {cfg.DATASET_DIR / 'folds.json'}")
        for c in meta["counts"]:
            print(f"  fold{c['fold']}: n={c['n']} (N={c['n_normal']}, P={c['n_pathological']})")
    else:
        run_preprocess()