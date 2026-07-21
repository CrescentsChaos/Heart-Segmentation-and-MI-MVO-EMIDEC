from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from src.losses.joint_loss import JointLoss
from src.model_identity import MULTICLASS_VARIANTS, PYTORCH_BASELINE_VARIANTS
from src.models.baselines import BASELINE_BUILDERS
from src.models.dual_decoder import build_model
from src.train import _resolve_variants


MODERN = {"MEDNEXT", "UXNET3D", "SWINUNETR_V2", "UMAMBA_ENC", "SEGMAMBA"}


def test_modern_variants_registered():
    assert MODERN <= set(BASELINE_BUILDERS)
    assert MODERN <= set(PYTORCH_BASELINE_VARIANTS)
    assert MODERN <= set(MULTICLASS_VARIANTS)


def test_model_specific_batch_sizes():
    configured = set(cfg.BASELINE_BATCH_SIZES)
    assert MODERN <= configured
    assert all(cfg.BASELINE_BATCH_SIZES[name] == 1 for name in MODERN)


def test_cli_baseline_expansion():
    assert _resolve_variants("baselines") == list(PYTORCH_BASELINE_VARIANTS)
    assert set(_resolve_variants("everything")) >= MODERN


def test_missing_optional_checkout_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "THIRD_PARTY_MODEL_DIR", tmp_path)
    with pytest.raises(ImportError, match="setup_modern_baselines.py umamba"):
        build_model("UMAMBA_ENC", in_ch=1, num_classes=5)


@pytest.mark.parametrize("variant", sorted(MODERN))
def test_forward_contract_when_dependency_available(variant: str):
    try:
        model = build_model(variant, in_ch=1, num_classes=5)
    except ImportError as exc:
        pytest.skip(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x = torch.zeros(1, 1, 16, 128, 128, device=device)
    with torch.no_grad():
        logits = model(x)["multiclass_logits"]
    assert logits.shape == (1, 5, 16, 128, 128)
    assert torch.isfinite(logits).all()


def test_multiclass_loss_backward_for_lightweight_contract():
    # Contract-level loss test avoids allocating every large architecture.
    logits = torch.randn(1, 5, 4, 8, 8, requires_grad=True)
    target = torch.zeros(1, 4, 8, 8, dtype=torch.long)
    criterion = JointLoss("MEDNEXT")
    out = criterion({"multiclass_logits": logits}, {"multiclass": target})
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert logits.grad is not None


def test_optional_checkout_is_not_committed():
    assert Path(cfg.THIRD_PARTY_MODEL_DIR).name == "third_party"
