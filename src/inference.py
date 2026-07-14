"""Inference utilities: hard MYO masking (Fix 2) + test-time augmentation (Fix 5)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import config as cfg


def hard_myo_mask_pathology(
    anatomy_logits: torch.Tensor,
    pathology_prob: torch.Tensor,
    myo_class: int = cfg.ANAT_MYO,
) -> torch.Tensor:
    """Zero pathology probability outside predicted MYO wall."""
    anat_pred = anatomy_logits.argmax(dim=1)  # (B, D, H, W)
    myo_mask = (anat_pred == myo_class).float().unsqueeze(1)  # (B, 1, D, H, W)
    return pathology_prob * myo_mask


def _invert_aug(pred: torch.Tensor, name: str) -> torch.Tensor:
    """Invert spatial augmentations applied to (B, C, D, H, W) predictions."""
    if name == "id":
        return pred
    if name == "flip_w":
        return torch.flip(pred, dims=[-1])
    if name == "flip_h":
        return torch.flip(pred, dims=[-2])
    if name == "flip_d":
        return torch.flip(pred, dims=[-3])
    if name == "rot90":
        return torch.rot90(pred, k=3, dims=[-2, -1])  # inverse of +90
    if name == "rot180":
        return torch.rot90(pred, k=2, dims=[-2, -1])
    if name == "rot270":
        return torch.rot90(pred, k=1, dims=[-2, -1])  # inverse of +270
    if name == "rot90_flip_w":
        # forward: rot90 then flip W; inverse: flip W then rot270
        return torch.rot90(torch.flip(pred, dims=[-1]), k=3, dims=[-2, -1])
    raise ValueError(name)


def _apply_aug(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return x
    if name == "flip_w":
        return torch.flip(x, dims=[-1])
    if name == "flip_h":
        return torch.flip(x, dims=[-2])
    if name == "flip_d":
        return torch.flip(x, dims=[-3])
    if name == "rot90":
        return torch.rot90(x, k=1, dims=[-2, -1])
    if name == "rot180":
        return torch.rot90(x, k=2, dims=[-2, -1])
    if name == "rot270":
        return torch.rot90(x, k=3, dims=[-2, -1])
    if name == "rot90_flip_w":
        return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])
    raise ValueError(name)


AUG_NAMES = ["id", "flip_w", "flip_h", "flip_d", "rot90", "rot180", "rot270", "rot90_flip_w"]


@torch.no_grad()
def predict_dual(
    model: torch.nn.Module,
    volume: torch.Tensor,
    use_tta: bool = True,
    hard_mask: bool = True,
    n_augs: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Dual-decoder inference with optional TTA and hard MYO masking.
    volume: (B, 1, D, H, W)
    """
    model.eval()
    names = AUG_NAMES[: max(1, min(n_augs, len(AUG_NAMES)))] if use_tta else ["id"]

    anat_logits_acc = []
    path_prob_acc = []

    for name in names:
        aug_vol = _apply_aug(volume, name)
        out = model(aug_vol)
        anat_logits_acc.append(_invert_aug(out["anatomy_logits"], name))
        path_prob_acc.append(_invert_aug(out["pathology_prob"], name))

    anatomy_logits = torch.stack(anat_logits_acc, dim=0).mean(dim=0)
    pathology_prob = torch.stack(path_prob_acc, dim=0).mean(dim=0)
    anatomy_prob = F.softmax(anatomy_logits, dim=1)
    myo_soft = anatomy_prob[:, cfg.ANAT_MYO : cfg.ANAT_MYO + 1]

    if hard_mask:
        pathology_prob = hard_myo_mask_pathology(anatomy_logits, pathology_prob)

    pathology_bin = (pathology_prob > 0.5).float()
    anatomy_pred = anatomy_logits.argmax(dim=1)

    return {
        "anatomy_logits": anatomy_logits,
        "anatomy_prob": anatomy_prob,
        "anatomy_pred": anatomy_pred,
        "myo_mask": myo_soft,
        "pathology_prob": pathology_prob,
        "pathology_bin": pathology_bin,
    }


@torch.no_grad()
def predict_multiclass(
    model: torch.nn.Module,
    volume: torch.Tensor,
    use_tta: bool = True,
    n_augs: int = 8,
) -> Dict[str, torch.Tensor]:
    """Single-decoder (M1/M2) inference with optional TTA."""
    model.eval()
    names = AUG_NAMES[: max(1, min(n_augs, len(AUG_NAMES)))] if use_tta else ["id"]
    acc = []
    for name in names:
        out = model(_apply_aug(volume, name))
        logits = out["multiclass_logits"]
        acc.append(_invert_aug(logits, name))
    logits = torch.stack(acc, dim=0).mean(dim=0)
    return {"multiclass_logits": logits, "multiclass_pred": logits.argmax(dim=1)}
