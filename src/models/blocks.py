"""Building blocks: anisotropic factorized 3D convs + attention gates.
Tensor layout throughout: (N, C, D, H, W) with D = through-plane (slice) axis.
EMIDEC spacing ? 10 mm (D) vs 1.5 mm (H, W).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedConv3D(nn.Module):
    """Anisotropic factorized 3D convolution (methodology Sec. 4.2).
    In-plane:      Conv3d kernel (1, 3, 3) - independent per axial slice
    Through-plane: Conv3d kernel (3, 1, 1) - aggregates neighbouring slices
    Residual add when in_ch == out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.inplane = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.through = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.through(self.inplane(x)) + self.shortcut(x)

class StandardConv3D(nn.Module):
    """Isotropic 3x3x3 residual conv block (baseline / ablation M1)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)

def make_conv_block(in_ch: int, out_ch: int, factorized: bool) -> nn.Module:
    return FactorizedConv3D(in_ch, out_ch) if factorized else StandardConv3D(in_ch, out_ch)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factorized: bool):
        super().__init__()
        self.conv1 = make_conv_block(in_ch, out_ch, factorized)
        self.conv2 = make_conv_block(out_ch, out_ch, factorized)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))

class AttentionGate3D(nn.Module):
    """Additive attention gate (Oktay et al., 2018)."""
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int | None = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(skip_ch // 2, 1)
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)
        a = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(a)
