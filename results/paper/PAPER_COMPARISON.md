# AFDD-Net - Paper Comparison Report

**Full name:** Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating and Topology Consistency

**Dataset:** EMIDEC (100 cases; stratified train/val/test)
**Split evaluated:** `test`
**Primary target:** MI Dice > 0.783 (ICPIU-Net, Brahim et al., 2022)

## Ablation study (methodology Table 4.5 / 4.6)

| Variant | Model | LV Dice | MYO Dice | MI Dice | MVO Dice | Params (M) | Infer (ms) |
|---------|-------|--------:|---------:|--------:|---------:|-----------:|-----------:|
| M1 | Baseline 3D U-Net | 0.922 | 0.767 | 0.407 | - | 46.580 | 135.824 |
| M5 | AFDD-Net | 0.919 | 0.784 | 0.367 | 0.091 | 16.059 | 114.832 |

## Comparison with state-of-the-art (methodology Table 4.7)

| Method | Year | LV Dice | MYO Dice | MI Dice |
|--------|-----:|--------:|---------:|--------:|
| nnU-Net (Isensee et al.) | 2021 | 0.941 | 0.856 | 0.720 |
| ICPIU-Net (Brahim et al.) | 2022 | 0.932 | 0.895 | 0.783 |
| 2D-3D Cascade (Schwab et al.) | 2025 | - | 0.830 | 0.720 |
| **AFDD-Net (this work)** | 2026 | **0.919** | **0.784** | **0.367** |

## How to cite this model

> AFDD-Net: Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating and Topology Consistency. Joint anatomy (LV, MYO) and pathology (MI, MVO) segmentation on EMIDEC LGE-MRI using anisotropic factorized 3D convolutions, dual-decoder MYO soft gating, Focal Tversky loss, and topology consistency loss.

## Figures

All graphs are saved under `figures/paper/`.
