"""Losses: Dice+WCE, Focal Tversky, topology consistency, disease BCE, joint objective."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_identity import is_multiclass_variant


class DiceCELoss(nn.Module):
    """Generalised Dice + class-weighted CE (anatomy head, Sec. 4.4.1)."""

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.class_weights.to(logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, weight=w)
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * one_hot, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return ce + (1.0 - dice.mean())


class FocalTverskyLoss(nn.Module):
    """Focal Tversky on multi-label sigmoid outputs (Sec. 4.4.2)."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        eps: float = 1e-5,
        channel_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        if channel_weights is None:
            self.channel_weights = None
        else:
            self.register_buffer("channel_weights", channel_weights.float())

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs, targets: (B, C, D, H, W) in [0,1]
        dims = (0, 2, 3, 4)
        tp = torch.sum(probs * targets, dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        ti = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss_c = (1.0 - ti) ** self.gamma
        if self.channel_weights is not None:
            w = self.channel_weights.to(loss_c.device)
            return torch.sum(w * loss_c) / torch.clamp(w.sum(), min=1e-6)
        return torch.mean(loss_c)


class TopologyConsistencyLoss(nn.Module):
    """
    L_topo: penalise pathology probability outside the myocardial wall.

    Critical fix vs original: ALWAYS use detached gate (predicted soft MYO
    or GT wall). Without detach, L_topo = mean(path * (1-myo)) is trivially
    minimized by expanding path ≈ myo (paint whole wall as MI), which caused
    M5 MI Dice collapse (0.13 vs M4 0.36).
    """

    def forward(self, path_prob: torch.Tensor, myo_mask: torch.Tensor) -> torch.Tensor:
        # path_prob: (B, C, D, H, W); myo_mask: (B, 1, D, H, W) in [0,1]
        myo = myo_mask.detach()
        outside = 1.0 - myo
        # Emphasize MI channel (index 0) more than MVO — MI is the thesis target
        if path_prob.shape[1] >= 2:
            w = path_prob.new_tensor([1.5, 0.5]).view(1, -1, 1, 1, 1)
            return torch.mean(w * path_prob * outside)
        return torch.mean(path_prob * outside)


class JointLoss(nn.Module):
    """
    L_total = L_anat + λ_ftl * L_FTL + λ_topo * L_topo + λ_class * L_class

    For M1/M2 and MONAI baselines (UNET, SEGRESNET, SWINUNETR, DYNUNET, DYNUNET_RES):
        Dice+WCE on 5-class multiclass (BG/LV/MYO/MI/MVO).
    For M3: anatomy + soft Dice-like pathology (α=β=0.5, γ=1).
    For M4/M5: FTL on pathology; M5 enables L_topo (curriculum-controlled).
    Disease BCE (L_class) when outputs contain disease_logits.
    Optionally restrict L_FTL to pathological cases only (mixed N/P batches).
    """

    def __init__(
        self,
        variant: str,
        num_anatomy: int = 3,
        anatomy_weights: Optional[torch.Tensor] = None,
        lambda_ftl: float = 1.0,
        lambda_topo: float = 0.05,
        lambda_class: float = 0.5,
        ftl_alpha: float = 0.65,
        ftl_beta: float = 0.35,
        ftl_gamma: float = 0.75,
        use_gt_myo_for_topo: bool = True,
        mi_channel_weight: float = 1.5,
        mvo_channel_weight: float = 0.75,
        path_loss_on_pathological_only: bool = True,
    ):
        super().__init__()
        self.variant = variant.upper()
        self.lambda_ftl = lambda_ftl
        self.lambda_class = float(lambda_class)
        self.base_lambda_topo = float(lambda_topo) if self.variant == "M5" else 0.0
        self.lambda_topo = self.base_lambda_topo
        self.use_gt_myo_for_topo = use_gt_myo_for_topo
        self.path_loss_on_pathological_only = path_loss_on_pathological_only
        self.anat_loss = DiceCELoss(num_anatomy, anatomy_weights)
        # 5-class: BG / LV / MYO / MI / MVO (pure MI — matches EMIDEC SOTA reporting)
        try:
            import config as _cfg

            n_multi = int(getattr(_cfg, "NUM_MULTICLASS_CLASSES", 5))
            w_multi = list(getattr(_cfg, "MULTICLASS_CE_WEIGHTS", [0.1, 1.0, 1.0, 2.5, 2.0]))
        except Exception:  # pragma: no cover
            n_multi, w_multi = 5, [0.1, 1.0, 1.0, 2.5, 2.0]
        self.multi_loss = DiceCELoss(n_multi, torch.tensor(w_multi, dtype=torch.float32))
        ch_w = torch.tensor([mi_channel_weight, mvo_channel_weight], dtype=torch.float32)
        if self.variant == "M3":
            self.path_loss = FocalTverskyLoss(
                alpha=0.5, beta=0.5, gamma=1.0, channel_weights=ch_w
            )
        else:
            self.path_loss = FocalTverskyLoss(
                alpha=ftl_alpha,
                beta=ftl_beta,
                gamma=ftl_gamma,
                channel_weights=ch_w,
            )
        self.topo_loss = TopologyConsistencyLoss()

    def set_lambda_topo(self, value: float) -> None:
        """Curriculum schedule: keep at 0 during warmup, then ramp."""
        if self.variant != "M5":
            self.lambda_topo = 0.0
            return
        self.lambda_topo = float(max(0.0, value))

    def _gt_myo_mask(self, anatomy: torch.Tensor) -> torch.Tensor:
        # anatomy: (B, D, H, W) with MYO wall = class 2 (healthy+MI+MVO)
        return (anatomy == 2).float().unsqueeze(1)

    def _pathological_mask(self, batch: Dict[str, torch.Tensor], bsz: int, device) -> torch.Tensor:
        """Boolean (B,) — prefer explicit label; else GT MI/MVO > 0."""
        if "pathological" in batch:
            return batch["pathological"].to(device).bool().view(-1)
        path = batch["pathology"]
        return path.reshape(bsz, -1).sum(dim=1) > 0

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if is_multiclass_variant(self.variant):
            loss = self.multi_loss(outputs["multiclass_logits"], batch["multiclass"])
            return {"loss": loss, "L_multi": loss.detach()}

        L_anat = self.anat_loss(outputs["anatomy_logits"], batch["anatomy"])
        path_prob = outputs["pathology_prob"]
        bsz = path_prob.shape[0]
        device = path_prob.device

        # Pathology FTL — optionally only on pathological patients (avoids FP pressure on N*)
        if self.path_loss_on_pathological_only:
            y_path = self._pathological_mask(batch, bsz, device)
            if y_path.any():
                L_ftl = self.path_loss(path_prob[y_path], batch["pathology"][y_path])
            else:
                L_ftl = path_prob.new_zeros(())
        else:
            L_ftl = self.path_loss(path_prob, batch["pathology"])

        if self.lambda_topo > 0:
            if self.use_gt_myo_for_topo:
                myo_for_topo = self._gt_myo_mask(batch["anatomy"])
            else:
                myo_for_topo = outputs["myo_mask"]
            L_topo = self.topo_loss(path_prob, myo_for_topo)
        else:
            L_topo = path_prob.new_zeros(())

        # Disease classification BCE (normal=0, pathological=1)
        if "disease_logits" in outputs and self.lambda_class > 0:
            y_path = self._pathological_mask(batch, bsz, device).float()
            L_class = F.binary_cross_entropy_with_logits(
                outputs["disease_logits"].view(-1),
                y_path.view(-1),
            )
        else:
            L_class = path_prob.new_zeros(())

        total = (
            L_anat
            + self.lambda_ftl * L_ftl
            + self.lambda_topo * L_topo
            + self.lambda_class * L_class
        )
        return {
            "loss": total,
            "L_anat": L_anat.detach(),
            "L_ftl": L_ftl.detach() if torch.is_tensor(L_ftl) else L_ftl,
            "L_topo": L_topo.detach() if torch.is_tensor(L_topo) else L_topo,
            "L_class": L_class.detach() if torch.is_tensor(L_class) else L_class,
            "lambda_topo": path_prob.new_tensor(self.lambda_topo),
        }
