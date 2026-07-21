# -*- coding: utf-8 -*-
"""External baseline segmentors (MONAI) for fair comparison with AFDD-Net.

All MONAI baselines share the M1/M2 interface: 5-class multiclass head
  BG / LV / MYO / MI / MVO   (pure MI — matches EMIDEC / published SOTA)

Variants:
  UNET        - MONAI 3D UNet
  SEGRESNET   - MONAI SegResNet
  SWINUNETR   - MONAI SwinUNETR (depth padded 16->32 for /32 constraint)
  DYNUNET     - MONAI DynUNet (non-residual, filters 32..512)
  DYNUNET_RES - MONAI residual DynUNet (filters 32..320; formerly mislabeled nnU-Net)

Real nnU-Net v2 is NOT built here — see src/nnunet_emidec.py (variant NNUNET).
"""
from __future__ import annotations

import inspect
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import DynUNet, SegResNet, SwinUNETR, UNet
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Baseline models require MONAI. Install with: pip install monai>=1.3"
    ) from exc

import config as cfg

NUM_MULTICLASS = int(getattr(cfg, "NUM_MULTICLASS_CLASSES", 5))


class MulticlassWrapper(nn.Module):
    """Wrap a MONAI net so forward returns {'multiclass_logits': ...}."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"multiclass_logits": self.net(x)}


class SwinUNETRPadded(nn.Module):
    """SwinUNETR needs spatial dims divisible by 32; pad D=16 -> 32, crop back."""

    def __init__(self, net: nn.Module, pad_d: int = 32):
        super().__init__()
        self.net = net
        self.pad_d = pad_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D, H, W)
        d = x.shape[2]
        if d < self.pad_d:
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad_d - d))
        logits = self.net(x)
        return logits[:, :, :d]


def _dynunet_strides_for_emidec():
    """5-stage DynUNet compatible with 16 x 128 x 128."""
    kernels = [[3, 3, 3]] * 5
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return kernels, strides


def build_unet(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    net = UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    )
    return MulticlassWrapper(net)


def build_segresnet(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    net = SegResNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=num_classes,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    return MulticlassWrapper(net)


def build_swinunetr(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    # MONAI <=1.4 requires img_size; newer releases removed that argument.
    kwargs = dict(
        in_channels=in_ch,
        out_channels=num_classes,
        feature_size=24,
        spatial_dims=3,
        use_checkpoint=False,
    )
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = (32, 128, 128)
    net = SwinUNETR(**kwargs)
    return MulticlassWrapper(SwinUNETRPadded(net, pad_d=32))


def build_dynunet_res(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """MONAI residual DynUNet (nnU-Net-inspired filter ladder — not real nnU-Net)."""
    kernels, strides = _dynunet_strides_for_emidec()
    net = DynUNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=num_classes,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=[32, 64, 128, 256, 320],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )
    return MulticlassWrapper(net)


def build_dynunet(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """MONAI DynUNet without residual blocks."""
    kernels, strides = _dynunet_strides_for_emidec()
    net = DynUNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=num_classes,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=[32, 64, 128, 256, 512],
        norm_name="instance",
        deep_supervision=False,
        res_block=False,
    )
    return MulticlassWrapper(net)


# Backward-compatible alias (old code / checkpoints named NNUNET were this net)
build_nnunet = build_dynunet_res


BASELINE_BUILDERS = {
    "UNET": build_unet,
    "SEGRESNET": build_segresnet,
    "SWINUNETR": build_swinunetr,
    "DYNUNET": build_dynunet,
    "DYNUNET_RES": build_dynunet_res,
}


def build_baseline(variant: str, in_ch: int = 1, **_kwargs) -> nn.Module:
    key = variant.upper()
    if key == "NNUNET":
        raise ValueError(
            "NNUNET is real nnU-Net v2 — train/eval via: python -m src.nnunet_emidec ..."
        )
    if key not in BASELINE_BUILDERS:
        raise ValueError(f"Unknown baseline {variant}. Expected one of {list(BASELINE_BUILDERS)}")
    return BASELINE_BUILDERS[key](in_ch=in_ch)
