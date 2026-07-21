# -*- coding: utf-8 -*-
"""External 3D segmentors for fair comparison with AFDD-Net.

All baselines share the M1/M2 interface: 5-class multiclass head
  BG / LV / MYO / MI / MVO   (pure MI — matches EMIDEC / published SOTA)

Variants:
  UNET        - MONAI 3D UNet
  SEGRESNET   - MONAI SegResNet
  SWINUNETR   - MONAI SwinUNETR (depth padded 16->32 for /32 constraint)
  SWINUNETR_V2 - MONAI SwinUNETR with stage residual convolutions
  DYNUNET     - MONAI DynUNet (non-residual, filters 32..512)
  DYNUNET_RES - MONAI residual DynUNet (filters 32..320; formerly mislabeled nnU-Net)
  MEDNEXT     - official MIC-DKFZ MedNeXt-S
  UXNET3D     - official MASILab 3D UX-Net
  UMAMBA_ENC  - official U-Mamba encoder architecture
  SEGMAMBA    - official SegMamba architecture

Real nnU-Net v2 is NOT built here — see src/nnunet_emidec.py (variant NNUNET).
"""
from __future__ import annotations

import inspect
import importlib.util
import sys
from pathlib import Path
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

    def __init__(self, net: nn.Module, num_classes: int = NUM_MULTICLASS):
        super().__init__()
        self.net = net
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        if isinstance(logits, (list, tuple)):
            # Deep supervision is disabled in our builders. This fallback keeps
            # the public contract stable if an upstream implementation changes.
            logits = logits[0]
        expected = (x.shape[0], self.num_classes, *x.shape[2:])
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"{type(self.net).__name__} returned {tuple(logits.shape)}; "
                f"expected {expected}"
            )
        return {"multiclass_logits": logits}


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
    return MulticlassWrapper(net, num_classes)


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
    return MulticlassWrapper(net, num_classes)


def _build_swinunetr(
    in_ch: int,
    num_classes: int,
    *,
    use_v2: bool,
) -> nn.Module:
    # MONAI <=1.4 requires img_size; newer releases removed that argument.
    kwargs = dict(
        in_channels=in_ch,
        out_channels=num_classes,
        feature_size=24,
        spatial_dims=3,
        use_checkpoint=False,
    )
    if "use_v2" not in inspect.signature(SwinUNETR).parameters and use_v2:
        raise ImportError(
            "SWINUNETR_V2 requires a MONAI release exposing SwinUNETR(use_v2=True). "
            "Upgrade with: python -m pip install --upgrade monai"
        )
    if "use_v2" in inspect.signature(SwinUNETR).parameters:
        kwargs["use_v2"] = use_v2
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = (32, 128, 128)
    net = SwinUNETR(**kwargs)
    return MulticlassWrapper(SwinUNETRPadded(net, pad_d=32), num_classes)


def build_swinunetr(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    return _build_swinunetr(in_ch, num_classes, use_v2=False)


def build_swinunetr_v2(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    return _build_swinunetr(in_ch, num_classes, use_v2=True)


def _third_party_root(repo_dir: str, import_hint: str) -> Path:
    root = Path(getattr(cfg, "THIRD_PARTY_MODEL_DIR", Path(cfg.ROOT) / "third_party"))
    repo = root / repo_dir
    if not repo.exists():
        raise ImportError(
            f"Missing official {repo_dir} checkout at {repo}. "
            f"Run: python scripts/setup_modern_baselines.py {import_hint}"
        )
    repo_s = str(repo.resolve())
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    return repo


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_mednext(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """Official MIC-DKFZ MedNeXt-v1 Small, kernel 3, no deep supervision."""
    _third_party_root("MedNeXt", "mednext")
    try:
        from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
            create_mednext_v1,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import the official MedNeXt-S implementation. "
            "Run: python scripts/setup_modern_baselines.py mednext --install"
        ) from exc
    net = create_mednext_v1(
        num_input_channels=in_ch,
        num_classes=num_classes,
        model_id="S",
        kernel_size=3,
        deep_supervision=False,
    )
    return MulticlassWrapper(net, num_classes)


def build_uxnet3d(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """Official MASILab 3D UX-Net reference configuration."""
    _third_party_root("3DUX-Net", "uxnet3d")
    try:
        from networks.UXNet_3D.network_backbone import UXNET
    except ImportError as exc:
        raise ImportError(
            "Could not import the official 3D UX-Net implementation. "
            "Run: python scripts/setup_modern_baselines.py uxnet3d --install"
        ) from exc
    net = UXNET(
        in_chans=in_ch,
        out_chans=num_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    return MulticlassWrapper(net, num_classes)


def build_umamba_enc(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """Official U-Mamba Enc 3D architecture under the shared AFDD protocol."""
    repo = _third_party_root("U-Mamba", "umamba")
    bundled_root = str((repo / "umamba").resolve())
    if bundled_root not in sys.path:
        sys.path.insert(0, bundled_root)
    source = repo / "umamba" / "nnunetv2" / "nets" / "UMambaEnc_3d.py"
    try:
        module = _load_module_from_file("afdd_umamba_enc_3d", source)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "U-Mamba requires Linux/WSL2 CUDA plus mamba-ssm and "
            "causal-conv1d. See docs/MODERN_BASELINES.md."
        ) from exc

    # EMIDEC-aware anisotropic plan: preserve depth in the first two stages.
    net = module.UMambaEnc(
        input_size=(16, 128, 128),
        input_channels=in_ch,
        n_stages=5,
        features_per_stage=[32, 64, 128, 256, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[(3, 3, 3)] * 5,
        strides=[
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ],
        n_conv_per_stage=[2, 2, 2, 2, 2],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    return MulticlassWrapper(net, num_classes)


def build_segmamba(in_ch: int = 1, num_classes: int = NUM_MULTICLASS) -> nn.Module:
    """Official ge-xing SegMamba reference configuration (external checkout)."""
    repo = _third_party_root("SegMamba", "segmamba")
    bundled_mamba = str((repo / "mamba").resolve())
    if bundled_mamba not in sys.path:
        sys.path.insert(0, bundled_mamba)
    try:
        from model_segmamba.segmamba import SegMamba
    except ImportError as exc:
        raise ImportError(
            "SegMamba requires its official CUDA Mamba extension on Linux/WSL2. "
            "See docs/MODERN_BASELINES.md."
        ) from exc
    net = SegMamba(
        in_chans=in_ch,
        out_chans=num_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    )
    return MulticlassWrapper(net, num_classes)


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
    return MulticlassWrapper(net, num_classes)


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
    return MulticlassWrapper(net, num_classes)


# Backward-compatible alias (old code / checkpoints named NNUNET were this net)
build_nnunet = build_dynunet_res


BASELINE_BUILDERS = {
    "UNET": build_unet,
    "SEGRESNET": build_segresnet,
    "SWINUNETR": build_swinunetr,
    "SWINUNETR_V2": build_swinunetr_v2,
    "DYNUNET": build_dynunet,
    "DYNUNET_RES": build_dynunet_res,
    "MEDNEXT": build_mednext,
    "UXNET3D": build_uxnet3d,
    "UMAMBA_ENC": build_umamba_enc,
    "SEGMAMBA": build_segmamba,
}


def build_baseline(
    variant: str,
    in_ch: int = 1,
    num_classes: int = NUM_MULTICLASS,
    **_kwargs,
) -> nn.Module:
    key = variant.upper()
    if key == "NNUNET":
        raise ValueError(
            "NNUNET is real nnU-Net v2 — train/eval via: python -m src.nnunet_emidec ..."
        )
    if key not in BASELINE_BUILDERS:
        raise ValueError(f"Unknown baseline {variant}. Expected one of {list(BASELINE_BUILDERS)}")
    return BASELINE_BUILDERS[key](in_ch=in_ch, num_classes=num_classes)
