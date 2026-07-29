# AFDD-Net - Paper Comparison Report

**Full name:** Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior

**Dataset:** EMIDEC only (non-EMIDEC papers excluded from Table 4.7)
**Split evaluated:** `test`
**Primary target:** MI Dice > 0.76 (Schwab 2025 (EMIDEC best))
**Stretch target:** MI Dice > 0.783 (ICPIU-Net 5-fold CV)

> Comparison uses verified EMIDEC MI/scar Dice only. Isensee et al. 2021 nnU-Net (private LGE cohort) and non-EMIDEC 2025–2026 papers are excluded. Protocols differ (official test vs 5-fold CV); interpret MI Dice accordingly.

## Ablation study (methodology Table 4.5 / 4.6)

| Variant | Model | LV Dice | MYO Dice | **MI_path** | MI (all) | MVO Dice | Params (M) | Infer (ms) |
|---------|-------|--------:|---------:|-----------:|---------:|---------:|-----------:|-----------:|

> **PRIMARY: MI_path** = pure MI Dice (EMIDEC label 3) on pathological cases only. Multiclass models (M1/M2/baselines) now predict MI and MVO as separate classes — not merged infarct. MI_all includes healthy empty–empty = 1.0; do not cite as scar metric. **MI (all)** is secondary. Disease classifier (M5 only) + voxel suppression reduce healthy FPs. Train best is on **val**; this table is **test**.

## Comparison with state-of-the-art (methodology Table 4.7, EMIDEC-only)

| Method | Year | Protocol | MYO Dice | MI Dice |
|--------|-----:|----------|---------:|--------:|
| Zhang (cascaded nnU-Net) (Zhang et al.) | 2021 | Official test (50 cases) | 0.879 | 0.712 |
| ICPIU-Net (Brahim et al.) | 2022 | Official test (50 cases) | 0.877 | 0.734 |
| ICPIU-Net (5-fold) (Brahim et al.) | 2022 | 5-fold CV (100 cases) | 0.895 | 0.783 |
| 3D nnU-Net (EMIDEC) (nnU-Net baseline) | 2021 | 5-fold CV | 0.872 | 0.688 |
| 2D nnU-Net (EMIDEC) (nnU-Net baseline) | 2021 | 5-fold CV | 0.851 | 0.509 |
| GAN-aug. cascade (Lustermans et al.) | 2022 | Test / per-slice (see paper) | 0.840 | 0.720 |
| CLAIM (Ramzan et al.) | 2025 | 10 held-out cases | - | 0.635 |
| 2D-3D Cascade (EcorC) (Schwab et al.) | 2025 | 5-fold CV (100 cases) | 0.860 | 0.760 |
| Expert (inter-observer) (Lalande et al.) | 2020 | Inter-observer (Data 2020) | 0.830 | 0.690 |

## Notes on protocol fairness

- Official test (50 cases): Zhang 0.712, ICPIU-Net 0.734
- 5-fold CV (100 cases): Schwab 0.760 (current best), ICPIU-Net 0.783
- Expert inter-observer MI Dice on EMIDEC: 0.69 (honest ceiling reference)

## Recommended thesis comparison paragraph

> Quantitative comparison was performed exclusively on the EMIDEC dataset (Lalande et al., 2020), the standard benchmark for LGE-MRI myocardial infarction segmentation. Methods were compared using Dice Similarity Coefficient (DSC) for left ventricle myocardium (MYO) and myocardial infarction (MI), following the official EMIDEC evaluation protocol. On the EMIDEC test set, the challenge-winning cascaded 2D–3D nnU-Net (Zhang, 2021) achieved MYO DSC of 0.879 and MI DSC of 0.712. ICPIU-Net (Brahim et al., 2022) improved MI DSC to 0.734. The current best published result using 5-fold cross-validation is Schwab et al. (2025) with MI DSC of 0.760. For context, expert inter-observer MI DSC on EMIDEC is 0.69. Our models are compared on a stratified EMIDEC split using the same structures (LV, MYO, MI, MVO).

## How to cite this model

> AFDD-Net: Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior. Joint anatomy (LV, MYO) and pathology (MI, MVO) segmentation on EMIDEC LGE-MRI using anisotropic factorized 3D convolutions, dual-decoder MYO soft gating, Focal Tversky loss, and topology consistency loss.

## Figures

All graphs are saved under `figures/paper/`.
