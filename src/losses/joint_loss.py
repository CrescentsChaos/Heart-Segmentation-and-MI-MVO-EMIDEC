"""Losses: Dice+WCE, Focal Tversky, topology consistency, joint objective."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """Generalised Dice + class-weighted CE (anatomy head, Sec. 4.4.1)."""
    def __init__(self, num_classes: int, class_weights: Optional[torch.Tensor] = None, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        self.ce = None  # built in forward to stay on correct device
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
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs, targets: (B, C, H, W, D) in [0,1]
        dims = (0, 2, 3, 4)
        tp = torch.sum(probs * targets, dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        ti = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return torch.mean((1.0 - ti) ** self.gamma)

class TopologyConsistencyLoss(nn.Module):
    """L_topo: penalise pathology probability outside predicted MYO (Sec. 4.4.3)."""
    def forward(self, path_prob: torch.Tensor, myo_mask: torch.Tensor) -> torch.Tensor:
        # path_prob: (B, C, H, W, D); myo_mask: (B, 1, H, W, D)
        outside = 1.0 - myo_mask
        # Average over pathology channels and voxels
        return torch.mean(path_prob * outside)

class JointLoss(nn.Module):
    """
    L_total = L_anat + ?1 * L_FTL + ?2 * L_topo   (Sec. 4.4.4)
    For M1/M2 (single decoder): Dice+WCE on 4-class multiclass only.
    For M3: Dice+WCE anatomy + Dice+WCE-style pathology (use FTL with ??1, ?=?=0.5 via flag)
    For M4/M5: FTL on pathology; M5 also enables L_topo.
    """
    def __init__(
        self,
        variant: str,
        num_anatomy: int = 3,
        anatomy_weights: Optional[torch.Tensor] = None,
        lambda_ftl: float = 1.0,
        lambda_topo: float = 0.5,
        ftl_alpha: float = 0.7,
        ftl_beta: float = 0.3,
        ftl_gamma: float = 0.75,
    ):
        super().__init__()
        self.variant = variant.upper()
        self.lambda_ftl = lambda_ftl
        self.lambda_topo = lambda_topo if self.variant == "M5" else 0.0
        self.anat_loss = DiceCELoss(num_anatomy, anatomy_weights)
        self.multi_loss = DiceCELoss(4, torch.tensor([0.1, 1.0, 1.0, 2.0]))
        if self.variant in ("M3",):
            # Pathology with standard Tversky (?=?=0.5, ?=1) ? Dice soft
            self.path_loss = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
        else:
            self.path_loss = FocalTverskyLoss(alpha=ftl_alpha, beta=ftl_beta, gamma=ftl_gamma)
        self.topo_loss = TopologyConsistencyLoss()
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.variant in ("M1", "M2"):
            loss = self.multi_loss(outputs["multiclass_logits"], batch["multiclass"])
            return {"loss": loss, "L_multi": loss.detach()}
        L_anat = self.anat_loss(outputs["anatomy_logits"], batch["anatomy"])
        L_ftl = self.path_loss(outputs["pathology_prob"], batch["pathology"])
        L_topo = self.topo_loss(outputs["pathology_prob"], outputs["myo_mask"])
        total = L_anat + self.lambda_ftl * L_ftl + self.lambda_topo * L_topo
        return {
            "loss": total,
            "L_anat": L_anat.detach(),
            "L_ftl": L_ftl.detach(),
            "L_topo": L_topo.detach(),
        }
