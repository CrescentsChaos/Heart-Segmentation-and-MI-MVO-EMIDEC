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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

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

def build_targets(raw_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    EMIDEC raw labels ? training targets.
    Anatomy (no RV in EMIDEC):
      0 BG, 1 LV cavity, 2 full MYO wall (raw 2 ? 3 ? 4)
    Pathology (multi-label):
      channel 0 = MI (raw 3), channel 1 = MVO (raw 4)
    Multiclass baseline (M1/M2):
      0 BG, 1 LV, 2 healthy MYO (raw 2 only), 3 infarct?MVO (raw 3?4)
    """
    anatomy = np.zeros(raw_mask.shape, dtype=np.uint8)
    anatomy[raw_mask == cfg.RAW_LV] = cfg.ANAT_LV
    anatomy[np.isin(raw_mask, [cfg.RAW_MYO, cfg.RAW_MI, cfg.RAW_MVO])] = cfg.ANAT_MYO
    pathology = np.zeros(raw_mask.shape + (2,), dtype=np.float32)
    pathology[..., 0] = (raw_mask == cfg.RAW_MI).astype(np.float32)
    pathology[..., 1] = (raw_mask == cfg.RAW_MVO).astype(np.float32)
    multiclass = np.zeros(raw_mask.shape, dtype=np.uint8)
    multiclass[raw_mask == cfg.RAW_LV] = 1
    multiclass[raw_mask == cfg.RAW_MYO] = 2
    multiclass[np.isin(raw_mask, [cfg.RAW_MI, cfg.RAW_MVO])] = 3
    return {
        "anatomy": anatomy,
        "pathology": pathology,
        "multiclass": multiclass,
        "raw": raw_mask.astype(np.uint8),
    }

# ----------------------------- Augmentations ---------------------------------

def rotate_volume(image, anatomy, pathology, multiclass, angle_range=(-15, 15)):
    angle = np.random.uniform(*angle_range)
    kw = dict(axes=(0, 1), reshape=False, mode="constant")
    image = ndimage.rotate(image, angle, order=3, cval=0.0, **kw)
    anatomy = ndimage.rotate(anatomy, angle, order=0, cval=0, **kw)
    multiclass = ndimage.rotate(multiclass, angle, order=0, cval=0, **kw)
    p0 = ndimage.rotate(pathology[..., 0], angle, order=0, cval=0, **kw)
    p1 = ndimage.rotate(pathology[..., 1], angle, order=0, cval=0, **kw)
    pathology = np.stack([p0, p1], axis=-1)
    return image, anatomy, pathology, multiclass

def flip_volume(image, anatomy, pathology, multiclass):
    if np.random.rand() > 0.5:
        image = np.flip(image, 0)
        anatomy = np.flip(anatomy, 0)
        multiclass = np.flip(multiclass, 0)
        pathology = np.flip(pathology, 0)
    if np.random.rand() > 0.5:
        image = np.flip(image, 1)
        anatomy = np.flip(anatomy, 1)
        multiclass = np.flip(multiclass, 1)
        pathology = np.flip(pathology, 1)
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(anatomy),
        np.ascontiguousarray(pathology),
        np.ascontiguousarray(multiclass),
    )

def scale_volume(image, anatomy, pathology, multiclass, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    h, w, d = image.shape
    img_s = ndimage.zoom(image, (scale, scale, 1.0), order=3)
    anat_s = ndimage.zoom(anatomy, (scale, scale, 1.0), order=0)
    multi_s = ndimage.zoom(multiclass, (scale, scale, 1.0), order=0)
    p0 = ndimage.zoom(pathology[..., 0], (scale, scale, 1.0), order=0)
    p1 = ndimage.zoom(pathology[..., 1], (scale, scale, 1.0), order=0)
    def crop_or_pad(arr, fill):
        out = np.full((h, w, d), fill, dtype=arr.dtype)
        src, tgt = [], []
        for i in range(3):
            s_len, t_len = arr.shape[i], (h, w, d)[i]
            if s_len >= t_len:
                start = (s_len - t_len) // 2
                src.append(slice(start, start + t_len))
                tgt.append(slice(None))
            else:
                start = (t_len - s_len) // 2
                src.append(slice(None))
                tgt.append(slice(start, start + s_len))
        out[tuple(tgt)] = arr[tuple(src)]
        return out
    image = crop_or_pad(img_s, 0.0)
    anatomy = crop_or_pad(anat_s, 0)
    multiclass = crop_or_pad(multi_s, 0)
    pathology = np.stack([crop_or_pad(p0, 0), crop_or_pad(p1, 0)], axis=-1)
    return image, anatomy, pathology, multiclass

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
    for s in ("train", "val", "test"):
        (out_dir / s).mkdir(parents=True, exist_ok=True)
    cases = sorted(
        [d for d in emidec_root.iterdir() if d.is_dir() and (d.name.startswith("Case_N") or d.name.startswith("Case_P"))]
    )
    case_names = [c.name for c in cases]
    train, val, test = stratified_split(case_names, seed=cfg.SEED)
    split_map = {n: "train" for n in train}
    split_map.update({n: "val" for n in val})
    split_map.update({n: "test" for n in test})
    meta = {"train": train, "val": val, "test": test, "target_shape": cfg.TARGET_SHAPE, "spacing": cfg.TARGET_SPACING}
    (out_dir / "split.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for case_dir in cases:
        data = process_case(case_dir)
        split = split_map[case_dir.name]
        out = out_dir / split / f"{case_dir.name}.npz"
        np.savez_compressed(out, **data)
        print(f"Saved {out.name} ? {split}  shape={data['image'].shape}  "
              f"MI={int(data['pathology'][...,0].sum())} MVO={int(data['pathology'][...,1].sum())}")
    print(f"Done. train={len(train)} val={len(val)} test={len(test)}")

# ----------------------------- Torch Dataset ---------------------------------

class EMIDECDataset(Dataset):
    def __init__(self, split_dir: Path, augment: bool = False):
        self.files = sorted(Path(split_dir).glob("*.npz"))
        self.augment = augment
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {split_dir}. Run preprocess first.")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        last_err = None
        image = anatomy = pathology = multiclass = None
        for attempt in range(3):
            try:
                with np.load(f, allow_pickle=False) as d:
                    image = d['image'].astype(np.float32).copy()
                    anatomy = d['anatomy'].astype(np.int64).copy()
                    pathology = d['pathology'].astype(np.float32).copy()
                    multiclass = d['multiclass'].astype(np.int64).copy()
                break
            except (EOFError, OSError, ValueError) as err:
                last_err = err
                if attempt == 2:
                    raise RuntimeError(f'Failed to load {f}') from last_err

        if self.augment:
            if np.random.rand() > 0.5:
                image, anatomy, pathology, multiclass = rotate_volume(image, anatomy, pathology, multiclass)
            if np.random.rand() > 0.5:
                image, anatomy, pathology, multiclass = flip_volume(image, anatomy, pathology, multiclass)
            if np.random.rand() > 0.5:
                image, anatomy, pathology, multiclass = scale_volume(image, anatomy, pathology, multiclass)

        # numpy (H, W, D) -> torch (C, D, H, W)
        image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        anatomy_t = torch.from_numpy(anatomy).long().permute(2, 0, 1)
        multiclass_t = torch.from_numpy(multiclass).long().permute(2, 0, 1)
        pathology_t = torch.from_numpy(pathology).float().permute(3, 2, 0, 1)

        return {
            'image': image_t,
            'anatomy': anatomy_t,
            'pathology': pathology_t,
            'multiclass': multiclass_t,
            'name': f.stem,
        }


if __name__ == "__main__":
    run_preprocess()
