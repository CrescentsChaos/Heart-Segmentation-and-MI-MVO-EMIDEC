from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.losses.joint_loss import PaperForegroundDiceLoss
from src.metrics import (
    dice_score,
    emidec_dual_masks,
    emidec_multiclass_masks,
)
from src.model_identity import PYTORCH_BASELINE_VARIANTS
from src.models.dual_decoder import build_model
from src.train import _resolve_variants, primary_score
from src.visualize_patient import (
    compute_dysfunction_stats,
    save_patient_report,
)


def test_official_emidec_multiclass_unions():
    labels = np.array([0, 1, 2, 3, 4])
    masks = emidec_multiclass_masks(labels)
    assert masks["LV"].tolist() == [False, True, False, False, False]
    assert masks["MYO"].tolist() == [False, False, True, True, True]
    assert masks["MI"].tolist() == [False, False, False, True, True]
    assert masks["Pure_MI"].tolist() == [False, False, False, True, False]
    assert masks["MVO"].tolist() == [False, False, False, False, True]


def test_dual_masks_match_official_unions():
    anatomy = np.array([0, 1, 2, 2])
    pathology = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8
    )
    masks = emidec_dual_masks(anatomy, pathology)
    assert masks["MYO"].tolist() == [False, False, True, True]
    assert masks["MI"].tolist() == [False, False, True, True]
    assert masks["Pure_MI"].tolist() == [False, False, True, False]
    assert masks["MVO"].tolist() == [False, False, False, True]


def test_paper_dice_empty_case_and_positive_case():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0
    gt = empty.copy()
    gt[0, 0] = 1
    assert dice_score(empty, gt) == 0.0


def test_squared_foreground_dice_formula_and_backward():
    prediction = torch.tensor([[[[0.5, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = PaperForegroundDiceLoss(multiclass=False)(
        prediction, target
    )
    expected = -(2 * 0.5 + 1) / (0.5**2 + 1.0**2 + 1.0 + 1)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert prediction.grad is not None


def test_checkpoint_score_is_foreground_mean():
    metrics = {
        key: {"dice": {"mean": value}}
        for key, value in zip(("LV", "MYO", "MI", "MVO"), (0.8, 0.6, 0.4, 0.2))
    }
    assert primary_score(metrics, "M5") == pytest.approx(0.5)


def test_native_nnunet_registration_and_forward():
    assert "NNUNET" in PYTORCH_BASELINE_VARIANTS
    assert _resolve_variants("NNUNET") == ["NNUNET"]
    model = build_model("NNUNET", in_ch=1, num_classes=5).train()
    x = torch.zeros(1, 1, 16, 32, 32)
    logits = model(x)["multiclass_logits"]
    assert logits.shape == (1, 5, 16, 32, 32)
    assert torch.isfinite(logits).all()
    target = torch.zeros(1, 16, 32, 32, dtype=torch.long)
    loss = PaperForegroundDiceLoss(num_classes=5)(
        logits, target
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_external_nnunet_pipeline_removed():
    assert not (ROOT / "src" / "nnunet_emidec.py").exists()
    assert not (
        ROOT / "nnunet_trainers" / "nnUNetTrainerAFDD80.py"
    ).exists()


def test_patient_report_renders_paper_dice(tmp_path):
    image = np.zeros((2, 8, 8), dtype=np.float32)
    anatomy = np.zeros((2, 8, 8), dtype=np.uint8)
    anatomy[:, 2:6, 2:6] = 2
    anatomy[:, 3:5, 3:5] = 1
    pathology = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    pathology[0, 0, 2, 2] = 1
    pathology[1, 0, 2, 3] = 1
    pred = {
        "image": image,
        "anatomy": anatomy,
        "pathology": pathology,
        "gt_anatomy": anatomy.copy(),
        "gt_pathology": pathology.copy(),
        "variant": "NNUNET",
        "checkpoint": "synthetic.pth",
        "npz_path": "synthetic.npz",
    }
    stats = compute_dysfunction_stats(anatomy, pathology)
    report, saved = save_patient_report(
        "Case_Test", pred, stats, {}, tmp_path
    )
    assert report["segmentation_dice"]["MI"] == 1.0
    assert report["segmentation_dice"]["MVO"] == 1.0
    assert any(path.name.endswith("_report.png") for path in saved)
    assert all(path.exists() for path in saved)
