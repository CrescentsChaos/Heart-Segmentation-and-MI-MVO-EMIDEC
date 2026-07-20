"""Dual-decoder 3D network and ablation variants M1-M5 (methodology Ch. 4)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import AttentionGate3D, DoubleConvBlock, make_conv_block

class SharedEncoder(nn.Module):
    """Four-stage encoder ? bottleneck at 1/16 in-plane resolution."""
    def __init__(self, in_ch: int = 1, filters=(32, 64, 128, 256), factorized: bool = True):
        super().__init__()
        f1, f2, f3, f4 = filters
        self.enc1 = DoubleConvBlock(in_ch, f1, factorized)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConvBlock(f1, f2, factorized)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConvBlock(f2, f3, factorized)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConvBlock(f3, f4, factorized)
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConvBlock(f4, f4 * 2, factorized)
        self.out_channels = (f1, f2, f3, f4, f4 * 2)
    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        z = self.bottleneck(self.pool4(e4))
        return z, (e1, e2, e3, e4)

class AnatomyDecoder(nn.Module):
    """Softmax decoder: BG / LV / MYO (+ optional RV if num_classes=4)."""
    def __init__(
        self,
        filters=(32, 64, 128, 256),
        bottleneck_ch: int = 512,
        num_classes: int = 3,
        factorized: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        f1, f2, f3, f4 = filters
        self.use_attention = use_attention
        self.up4 = nn.ConvTranspose3d(bottleneck_ch, f4, kernel_size=2, stride=2)
        self.att4 = AttentionGate3D(f4, f4) if use_attention else nn.Identity()
        self.dec4 = DoubleConvBlock(f4 * 2, f4, factorized)
        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(f3, f3) if use_attention else nn.Identity()
        self.dec3 = DoubleConvBlock(f3 * 2, f3, factorized)
        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(f2, f2) if use_attention else nn.Identity()
        self.dec2 = DoubleConvBlock(f2 * 2, f2, factorized)
        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(f1, f1) if use_attention else nn.Identity()
        self.dec1 = DoubleConvBlock(f1 * 2, f1, factorized)
        self.head = nn.Conv3d(f1, num_classes, kernel_size=1)
    def _attend(self, gate, skip, att):
        if isinstance(att, AttentionGate3D):
            return att(gate, skip)
        return skip
    def forward(self, z, skips):
        e1, e2, e3, e4 = skips
        x = self.up4(z)
        if x.shape[2:] != e4.shape[2:]:
            x = F.interpolate(x, size=e4.shape[2:], mode="trilinear", align_corners=False)
        s4 = self._attend(x, e4, self.att4)
        x = self.dec4(torch.cat([x, s4], dim=1))
        x = self.up3(x)
        if x.shape[2:] != e3.shape[2:]:
            x = F.interpolate(x, size=e3.shape[2:], mode="trilinear", align_corners=False)
        s3 = self._attend(x, e3, self.att3)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode="trilinear", align_corners=False)
        s2 = self._attend(x, e2, self.att2)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode="trilinear", align_corners=False)
        s1 = self._attend(x, e1, self.att1)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.head(x)

class PathologyDecoder(nn.Module):
    """Sigmoid decoder for MI / MVO with MYO soft-mask gating (Sec. 4.3.3)."""
    def __init__(
        self,
        filters=(32, 64, 128, 256),
        bottleneck_ch: int = 512,
        num_classes: int = 2,
        factorized: bool = True,
        use_myo_gate: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        f1, f2, f3, f4 = filters
        self.use_myo_gate = use_myo_gate
        self.use_attention = use_attention
        gate_extra = 1 if use_myo_gate else 0
        self.up4 = nn.ConvTranspose3d(bottleneck_ch, f4, kernel_size=2, stride=2)
        self.att4 = AttentionGate3D(f4, f4) if use_attention else nn.Identity()
        self.dec4 = DoubleConvBlock(f4 * 2 + gate_extra, f4, factorized)
        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(f3, f3) if use_attention else nn.Identity()
        self.dec3 = DoubleConvBlock(f3 * 2 + gate_extra, f3, factorized)
        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(f2, f2) if use_attention else nn.Identity()
        self.dec2 = DoubleConvBlock(f2 * 2 + gate_extra, f2, factorized)
        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(f1, f1) if use_attention else nn.Identity()
        self.dec1 = DoubleConvBlock(f1 * 2 + gate_extra, f1, factorized)
        self.head = nn.Conv3d(f1, num_classes, kernel_size=1)
    def _attend(self, gate, skip, att):
        if isinstance(att, AttentionGate3D):
            return att(gate, skip)
        return skip
    def _cat_gate(self, feats, myo_mask):
        if not self.use_myo_gate or myo_mask is None:
            return feats
        g = F.interpolate(myo_mask, size=feats.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([feats, g], dim=1)
    def forward(self, z, skips, myo_mask: Optional[torch.Tensor] = None):
        e1, e2, e3, e4 = skips
        x = self.up4(z)
        if x.shape[2:] != e4.shape[2:]:
            x = F.interpolate(x, size=e4.shape[2:], mode="trilinear", align_corners=False)
        s4 = self._attend(x, e4, self.att4)
        x = self.dec4(self._cat_gate(torch.cat([x, s4], dim=1), myo_mask))
        x = self.up3(x)
        if x.shape[2:] != e3.shape[2:]:
            x = F.interpolate(x, size=e3.shape[2:], mode="trilinear", align_corners=False)
        s3 = self._attend(x, e3, self.att3)
        x = self.dec3(self._cat_gate(torch.cat([x, s3], dim=1), myo_mask))
        x = self.up2(x)
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode="trilinear", align_corners=False)
        s2 = self._attend(x, e2, self.att2)
        x = self.dec2(self._cat_gate(torch.cat([x, s2], dim=1), myo_mask))
        x = self.up1(x)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode="trilinear", align_corners=False)
        s1 = self._attend(x, e1, self.att1)
        x = self.dec1(self._cat_gate(torch.cat([x, s1], dim=1), myo_mask))
        return self.head(x)

class SingleDecoderUNet3D(nn.Module):
    """M1/M2 single-decoder multiclass baseline (BG, LV, MYO, MI?MVO)."""
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 4,
        filters=(32, 64, 128, 256),
        factorized: bool = False,
    ):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, filters, factorized=factorized)
        f1, f2, f3, f4 = filters
        bn = f4 * 2
        self.decoder = AnatomyDecoder(
            filters=filters,
            bottleneck_ch=bn,
            num_classes=num_classes,
            factorized=factorized,
            use_attention=True,
        )
        self.num_classes = num_classes
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, skips = self.encoder(x)
        logits = self.decoder(z, skips)
        return {"multiclass_logits": logits}

class DualDecoderNet(nn.Module):
    """Joint anatomy + pathology network (M3/M4/M5) with optional disease head."""

    def __init__(
        self,
        in_ch: int = 1,
        num_anatomy: int = 3,
        num_pathology: int = 2,
        filters=(32, 64, 128, 256),
        factorized: bool = True,
        use_myo_gate: bool = True,
        myo_class_index: int = 2,
        detach_myo_gate: bool = True,
        soft_myo_restrict: bool = True,
        use_disease_classifier: bool = True,
        gate_pathology_by_disease: bool = True,
        disease_threshold: float = 0.5,
    ):
        super().__init__()
        self.myo_class_index = myo_class_index
        self.detach_myo_gate = detach_myo_gate
        self.soft_myo_restrict = soft_myo_restrict
        self.use_disease_classifier = use_disease_classifier
        self.gate_pathology_by_disease = gate_pathology_by_disease
        self.disease_threshold = float(disease_threshold)
        self.encoder = SharedEncoder(in_ch, filters, factorized=factorized)
        f4 = filters[-1]
        bn = f4 * 2
        self.anatomy_decoder = AnatomyDecoder(
            filters=filters,
            bottleneck_ch=bn,
            num_classes=num_anatomy,
            factorized=factorized,
            use_attention=True,
        )
        self.pathology_decoder = PathologyDecoder(
            filters=filters,
            bottleneck_ch=bn,
            num_classes=num_pathology,
            factorized=factorized,
            use_myo_gate=use_myo_gate,
            use_attention=True,
        )
        # Lightweight normal-vs-pathological prior (ICPIU-Net motivation, linear head)
        if use_disease_classifier:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(bn, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, skips = self.encoder(x)
        anat_logits = self.anatomy_decoder(z, skips)
        anat_prob = F.softmax(anat_logits, dim=1)
        # Soft MYO probability for gating (B,1,D,H,W)
        myo_mask = anat_prob[:, self.myo_class_index : self.myo_class_index + 1]
        gate = myo_mask.detach() if self.detach_myo_gate else myo_mask
        path_logits = self.pathology_decoder(z, skips, myo_mask=gate)
        path_prob = torch.sigmoid(path_logits)
        # Soft anatomical restriction: MI/MVO mass must live inside MYO.
        # Detached so pathology cannot inflate MYO to "legalize" wall-wide MI.
        if self.soft_myo_restrict:
            path_prob = path_prob * myo_mask.detach()

        out: Dict[str, torch.Tensor] = {
            "anatomy_logits": anat_logits,
            "anatomy_prob": anat_prob,
            "myo_mask": myo_mask,
            "pathology_logits": path_logits,
            "pathology_prob": path_prob,
        }

        if self.classifier is not None:
            disease_logits = self.classifier(z)  # (B, 1) — P(pathological) logits
            disease_prob = torch.sigmoid(disease_logits)
            out["disease_logits"] = disease_logits
            out["disease_prob"] = disease_prob
            # Inference only: zero pathology for patients classified as healthy
            if (
                not self.training
                and self.gate_pathology_by_disease
                and self.use_disease_classifier
            ):
                disease_gate = (disease_prob > self.disease_threshold).float()
                disease_gate = disease_gate.view(-1, 1, 1, 1, 1)
                out["pathology_prob"] = path_prob * disease_gate
                out["disease_gate"] = disease_gate

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant: str, **kwargs) -> nn.Module:
    """
    Ablation variants (methodology Table 4.5):
      M1 - 3D U-Net baseline: isotropic 3x3x3, single decoder, 5-class
      M2 - + factorized convs in encoder/decoder
      M3 - + dual decoder with MYO soft gating (Dice+WCE both heads)
      M4 - + Focal Tversky on pathology (loss-side; architecture = M3)
      M5 - + topology consistency loss (loss-side; architecture = M3)
          + disease classification head (normal vs pathological prior)
            NOTE: classifier is M5-only (not enabled for M3/M4).

    External MONAI baselines (5-class BG/LV/MYO/MI/MVO):
      UNET, SEGRESNET, SWINUNETR, DYNUNET, DYNUNET_RES
    Real nnU-Net v2: variant NNUNET via src/nnunet_emidec.py
    """
    import config as cfg

    variant = variant.upper()
    filters = kwargs.get("filters", (32, 64, 128, 256))
    in_ch = kwargs.get("in_ch", 1)
    n_multi = int(kwargs.get("num_classes", getattr(cfg, "NUM_MULTICLASS_CLASSES", 5)))
    if variant == "M1":
        return SingleDecoderUNet3D(
            in_ch=in_ch, num_classes=n_multi, filters=filters, factorized=False
        )
    if variant == "M2":
        return SingleDecoderUNet3D(
            in_ch=in_ch, num_classes=n_multi, filters=filters, factorized=True
        )
    if variant in ("M3", "M4", "M5"):
        # Disease classifier is an M5 contribution only (ablation integrity)
        want_cls = bool(kwargs.get("use_disease_classifier", True))
        use_cls = want_cls and variant == "M5"
        want_gate = bool(kwargs.get("gate_pathology_by_disease", True))
        return DualDecoderNet(
            in_ch=in_ch,
            num_anatomy=kwargs.get("num_anatomy", 3),
            num_pathology=kwargs.get("num_pathology", 2),
            filters=filters,
            factorized=True,
            use_myo_gate=True,
            detach_myo_gate=kwargs.get("detach_myo_gate", True),
            soft_myo_restrict=kwargs.get("soft_myo_restrict", True),
            use_disease_classifier=use_cls,
            gate_pathology_by_disease=use_cls and want_gate,
            disease_threshold=kwargs.get("disease_threshold", 0.5),
        )
    from .baselines import BASELINE_BUILDERS, build_baseline

    if variant == "NNUNET":
        raise ValueError(
            "NNUNET is real nnU-Net v2. Use: python -m src.nnunet_emidec prepare|train|eval"
        )
    if variant in BASELINE_BUILDERS:
        return build_baseline(variant, in_ch=in_ch)
    raise ValueError(
        f"Unknown variant {variant}. Expected M1-M5 or "
        f"{', '.join(list(BASELINE_BUILDERS) + ['NNUNET'])}."
    )
