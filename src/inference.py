"""Inference utilities: hard MYO mask, disease gate, voxel suppression, TTA."""
from __future__ import annotations

from typing import Dict, Optional

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


def suppress_sparse_pathology(
    pathology_bin: torch.Tensor,
    min_mi_voxels: Optional[int] = None,
) -> torch.Tensor:
    """
    Inference-only rule (Solution 3): if predicted MI voxels < threshold,
    treat the case as healthy and zero all pathology channels.
    pathology_bin: (B, C, D, H, W) binary
    """
    thr = int(min_mi_voxels if min_mi_voxels is not None else getattr(cfg, "MIN_MI_VOXELS", 50))
    out = pathology_bin.clone()
    # Channel 0 = MI
    for b in range(out.shape[0]):
        if out[b, 0].sum() < thr:
            out[b] = 0
    return out


def apply_disease_gate(
    pathology_prob: torch.Tensor,
    disease_prob: torch.Tensor,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Zero pathology when classifier predicts healthy (P(pathological) <= thr)."""
    thr = float(
        threshold if threshold is not None else getattr(cfg, "DISEASE_CLASS_THRESHOLD", 0.5)
    )
    gate = (disease_prob > thr).float().view(-1, 1, 1, 1, 1)
    return pathology_prob * gate


def postprocess_pathology(
    out: Dict[str, torch.Tensor],
    hard_mask: bool = True,
    use_disease_gate: Optional[bool] = None,
    use_voxel_suppress: Optional[bool] = None,
    min_mi_voxels: Optional[int] = None,
) -> torch.Tensor:
    """
    Full dual-decoder pathology post-process → binary (B, C, D, H, W).
    Order: optional disease gate → hard MYO mask → threshold → voxel suppress.
    """
    path = out["pathology_prob"]
    # Disease gate may already be applied inside DualDecoderNet.eval(); re-apply if present
    do_gate = (
        getattr(cfg, "GATE_PATHOLOGY_BY_DISEASE", True)
        if use_disease_gate is None
        else use_disease_gate
    )
    if do_gate and "disease_prob" in out and "disease_gate" not in out:
        path = apply_disease_gate(path, out["disease_prob"])

    if hard_mask and "anatomy_logits" in out:
        path = hard_myo_mask_pathology(out["anatomy_logits"], path)

    path_bin = (path > 0.5).float()

    do_suppress = (
        getattr(cfg, "MI_VOXEL_SUPPRESSION", True)
        if use_voxel_suppress is None
        else use_voxel_suppress
    )
    if do_suppress:
        path_bin = suppress_sparse_pathology(path_bin, min_mi_voxels=min_mi_voxels)
    return path_bin


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
    use_voxel_suppress: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Dual-decoder inference with optional TTA, disease gate, hard MYO mask,
    and sparse-MI voxel suppression.
    volume: (B, 1, D, H, W)
    """
    model.eval()
    names = AUG_NAMES[: max(1, min(n_augs, len(AUG_NAMES)))] if use_tta else ["id"]

    anat_logits_acc = []
    path_prob_acc = []
    disease_acc = []

    for name in names:
        aug_vol = _apply_aug(volume, name)
        out = model(aug_vol)
        anat_logits_acc.append(_invert_aug(out["anatomy_logits"], name))
        # Use ungated logits path via pathology_prob before averaging; model may gate in eval
        path_prob_acc.append(_invert_aug(out["pathology_prob"], name))
        if "disease_prob" in out:
            disease_acc.append(out["disease_prob"])

    anatomy_logits = torch.stack(anat_logits_acc, dim=0).mean(dim=0)
    pathology_prob = torch.stack(path_prob_acc, dim=0).mean(dim=0)
    anatomy_prob = F.softmax(anatomy_logits, dim=1)
    myo_soft = anatomy_prob[:, cfg.ANAT_MYO : cfg.ANAT_MYO + 1]

    result: Dict[str, torch.Tensor] = {
        "anatomy_logits": anatomy_logits,
        "anatomy_prob": anatomy_prob,
        "anatomy_pred": anatomy_logits.argmax(dim=1),
        "myo_mask": myo_soft,
        "pathology_prob": pathology_prob,
    }
    if disease_acc:
        result["disease_prob"] = torch.stack(disease_acc, dim=0).mean(dim=0)

    pathology_bin = postprocess_pathology(
        result,
        hard_mask=hard_mask,
        use_voxel_suppress=use_voxel_suppress,
    )
    result["pathology_bin"] = pathology_bin
    result["pathology_prob"] = pathology_prob  # keep soft probs pre-binarize for inspection
    if hard_mask:
        result["pathology_prob"] = hard_myo_mask_pathology(anatomy_logits, pathology_prob)
    return result


@torch.no_grad()
def predict_multiclass(
    model: torch.nn.Module,
    volume: torch.Tensor,
    use_tta: bool = True,
    n_augs: int = 8,
    use_voxel_suppress: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """Single-decoder (M1/M2) inference with optional TTA + infarct voxel suppress."""
    model.eval()
    names = AUG_NAMES[: max(1, min(n_augs, len(AUG_NAMES)))] if use_tta else ["id"]
    acc = []
    for name in names:
        out = model(_apply_aug(volume, name))
        logits = out["multiclass_logits"]
        acc.append(_invert_aug(logits, name))
    logits = torch.stack(acc, dim=0).mean(dim=0)
    pred = logits.argmax(dim=1)
    do_suppress = (
        getattr(cfg, "MI_VOXEL_SUPPRESSION", True)
        if use_voxel_suppress is None
        else use_voxel_suppress
    )
    if do_suppress:
        thr = int(getattr(cfg, "MIN_MI_VOXELS", 50))
        # class 3 = infarct∪MVO; sparse → reclassify as healthy MYO (2)
        pred = pred.clone()
        for b in range(pred.shape[0]):
            infarct = pred[b] == 3
            if infarct.sum() < thr:
                pred[b][infarct] = 2
    return {"multiclass_logits": logits, "multiclass_pred": pred}
