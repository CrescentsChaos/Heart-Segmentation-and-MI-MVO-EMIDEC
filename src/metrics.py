"""Segmentation metrics comparable to EMIDEC / SOTA papers (Dice, HD95, Recall)."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from scipy.spatial.distance import cdist

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * inter / s)

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)

def precision_recall(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    if pred.sum() == 0:
        prec = float("nan") if gt.sum() == 0 else 0.0
    else:
        prec = float(inter / pred.sum())
    if gt.sum() == 0:
        rec = float("nan")
    else:
        rec = float(inter / gt.sum())
    return prec, rec

def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.5, 1.5, 10.0), max_samples=400) -> float:
    """Hausdorff-95 in millimetres. pred/gt are binary volumes in (D, H, W)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        # Large penalty when one side empty (methodology secondary metric)
        return float(np.linalg.norm(np.array(pred.shape) * np.array(spacing[::-1])))
    # Coordinates in (D, H, W) ? physical mm with spacing (sd, sh, sw)
    sd, sh, sw = spacing[2], spacing[0], spacing[1]
    p_idx = np.argwhere(pred).astype(np.float64)
    g_idx = np.argwhere(gt).astype(np.float64)
    p_idx *= np.array([sd, sh, sw])
    g_idx *= np.array([sd, sh, sw])
    if len(p_idx) > max_samples:
        p_idx = p_idx[np.random.choice(len(p_idx), max_samples, replace=False)]
    if len(g_idx) > max_samples:
        g_idx = g_idx[np.random.choice(len(g_idx), max_samples, replace=False)]
    dists = cdist(p_idx, g_idx, metric="euclidean")
    d_pg = dists.min(axis=1)
    d_gp = dists.min(axis=0)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))

def binary_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1.5, 1.5, 10.0)) -> Dict[str, float]:
    prec, rec = precision_recall(pred, gt)
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": prec,
        "recall": rec,
        "hd95": hd95(pred, gt, spacing=spacing),
    }

def summarize(metric_list):
    """Nan-aware mean +/- std over a list of dicts."""
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=np.float64)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "n": int(np.sum(~np.isnan(vals))),
        }
    return out
