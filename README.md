# AFDD-Net — Cardiac Segmentation on EMIDEC LGE-MRI

**AFDD-Net** (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior*)  
Joint anatomy (LV, MYO) and pathology (MI, MVO) segmentation on the [EMIDEC](http://emidec.com/) LGE-MRI challenge dataset.

---

## Table of Contents

- [Overview](#overview)
- [Primary Metric](#primary-metric)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [5-Fold Cross-Validation (Recommended)](#5-fold-cross-validation-recommended)
  - [Single Split (Legacy)](#single-split-legacy)
- [Ablation Study (M1 → M5)](#ablation-study-m1--m5)
- [External Baselines](#external-baselines)
  - [MONAI Baselines](#monai-baselines)
  - [Modern Baselines (Optional)](#modern-baselines-optional)
  - [Real nnU-Net v2](#real-nnu-net-v2)
- [Evaluation & Reporting](#evaluation--reporting)
- [Patient Visualisation](#patient-visualisation)
- [Technical Details](#technical-details)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

AFDD-Net addresses the challenging task of **myocardial infarction (MI)** and **microvascular obstruction (MVO)** segmentation in Late Gadolinium Enhanced (LGE) cardiac MRI. The architecture employs:

- **Anisotropic factorized 3D convolutions** — tailored to EMIDEC's anisotropic resolution (in-plane ~1.5 mm, through-plane ~10 mm)
- **Dual-decoder with MYO soft-gating** — anatomy decoder (BG/LV/MYO) and pathology decoder (MI/MVO) with myocardial wall gating
- **Focal Tversky loss** — recall-oriented loss for sparse pathology voxels
- **Topology consistency loss** — curriculum-scheduled constraint keeping MI inside the myocardial wall
- **Disease classification prior** — lightweight head suppressing MI/MVO predictions on healthy patients

**Label protocol (EMIDEC official):** `0=Background, 1=LV cavity, 2=Normal MYO, 3=MI, 4=MVO`

---

## Primary Metric

> **MI_path** = pure MI Dice (EMIDEC label 3) on **pathological cases only**.

This matches the reporting convention in published EMIDEC SOTA papers (Schwab 2025 / ICPIU-Net / nnU-Net).

**Do not cite** `MI_all` — healthy empty–empty Dice = 1.0 inflates it. The merged `Infarct` column (MI∪MVO) is reported only as a secondary metric.

---

## Project Structure

```
.
├── config.py                        # Centralised hyperparameters and paths
├── requirements.txt                 # Core dependencies
├── requirements-modern-baselines.txt # Optional modern baseline deps
├── requirements-dev.txt             # Dev (includes pytest)
├── THIRD_PARTY_MODELS.md            # Attribution and licenses
│
├── src/
│   ├── data/
│   │   ├── preprocess.py            # EMIDEC → npz conversion, EMIDECDataset
│   │   └── cv_splits.py             # Stratified 5-fold split generation
│   ├── models/
│   │   ├── dual_decoder.py          # AFDD-Net (M1–M5) + modern baseline builders
│   │   ├── baselines.py             # MONAI baseline wrappers (UNet, SegResNet, …)
│   │   └── blocks.py                # Factorised conv blocks
│   ├── losses/
│   │   └── joint_loss.py            # Multi-component loss (CE, FTL, topo, class.)
│   ├── train.py                     # Training loop (ablation + baselines, CV + single)
│   ├── evaluate.py                  # Test-set evaluation with full metrics
│   ├── inference.py                 # Postprocessing (MYO hard mask, voxel suppress)
│   ├── metrics.py                   # Dice, IoU, HD95, Precision, Recall
│   ├── make_tables.py               # Generate paper-ready CSV/MD tables
│   ├── paper_figures.py             # Generate publication figures
│   ├── architecture_figure.py       # Network diagram generation
│   ├── model_identity.py            # Canonical model names, SOTA benchmarks
│   ├── augmentations.py             # 3D data augmentation pipeline
│   └── visualize_patient.py         # Per-patient overlay visualisation
│
├── scripts/
│   └── setup_modern_baselines.py    # Clone pinned third-party repos
│
├── tests/
│   ├── test_paper_protocol.py       # Protocol / reproducibility guards
│   └── test_modern_baselines.py     # Modern baseline integration tests
│
├── docs/
│   ├── AVAILABLE_MODELS.md          # Detailed model descriptions
│   └── MODERN_BASELINES.md          # Setup guide for modern baselines
│
├── Dataset/                         # EMIDEC data (gitignored)
├── checkpoints/                     # Trained weights (gitignored)
├── results/                         # Metrics JSON + paper CSVs
│   └── paper/                       # cv_all_metrics.csv, PAPER_COMPARISON.md, …
├── figures/                         # Generated figures (gitignored)
│   └── paper/                       # Publication-ready plots
└── third_party/                     # Cloned external repos (gitignored)
```

---

## Requirements

**Python ≥ 3.10** with CUDA-enabled PyTorch recommended.

### Core dependencies

```bash
pip install -r requirements.txt
```

| Package     | Version   | Purpose                        |
|-------------|-----------|--------------------------------|
| `torch`     | ≥ 2.0     | Training framework             |
| `monai`     | ≥ 1.3     | Medical image transforms / baselines |
| `nibabel`   | ≥ 5.0     | NIfTI I/O                      |
| `numpy`     | ≥ 1.23, <2.0 | Array operations           |
| `scipy`     | ≥ 1.10    | HD95 distance computation      |
| `matplotlib`| ≥ 3.7     | Figures                        |
| `tqdm`      | ≥ 4.65    | Progress bars                  |
| `einops`    | ≥ 0.6     | Tensor rearrangements          |
| `nnunetv2`  | ≥ 2.5     | nnU-Net v2 baseline (optional) |

### Modern baselines (optional)

```bash
pip install -r requirements-modern-baselines.txt
```

### Development

```bash
pip install -r requirements-dev.txt    # includes pytest ≥ 9
pytest tests/
```

---

## Quick Start

### Data Preparation

1. Place raw EMIDEC data under `Dataset/` (or set `EMIDEC_ROOT` in `config.py`).
2. Preprocess into standardised npz volumes:

```bash
python -m src.data.preprocess                  # Full rebuild (5-class MI/MVO)
python -m src.data.preprocess --folds-only     # Keep existing folds.json
```

### 5-Fold Cross-Validation (Recommended)

All models share the **same** `Dataset/folds.json`, the **same** epoch budget (`CV_EPOCHS=80`), and the **same** checkpoint metric for fair comparison.

```bash
# Train all ablation variants (M1–M5) + all installed PyTorch baselines
python -m src.train --variant everything --cv

# Evaluate all
python -m src.evaluate --all --baselines --cv --no-figs

# Generate paper-ready tables
python -m src.make_tables --cv

# Generate publication figures
python -m src.paper_figures
```

### Single Split (Legacy)

70 / 15 / 15 train / val / test split:

```bash
python -m src.train --variant M5 --epochs 150
python -m src.evaluate --variant M5 --split test
```

---

## Ablation Study (M1 → M5)

Progressive ablation leading to the full AFDD-Net (methodology Table 4.5):

| Variant | Name | What It Adds |
|---------|------|-------------|
| **M1** | Baseline 3D U-Net | Isotropic 3×3×3 convs, single decoder, 5-class (BG/LV/MYO/MI/MVO) |
| **M2** | AFDD-Net-F | + Anisotropic factorized convolutions |
| **M3** | AFDD-Net-D | + Dual decoder + MYO soft gate (no disease classifier) |
| **M4** | AFDD-Net-T | + Focal Tversky loss (no disease classifier) |
| **M5** | **AFDD-Net** | + Topology curriculum + disease classification prior |

M5 warm-starts from the M4 checkpoint (same fold). All ablation models are defined in `src/models/dual_decoder.py`.

### Train a single variant / fold

```bash
python -m src.train --variant M5 --cv --fold 0
python -m src.evaluate --variant M5 --cv
```

---

## External Baselines

### MONAI Baselines

Built-in baselines using the MONAI library, all trained on the same 5-class target:

| Key | Architecture | Source |
|-----|-------------|--------|
| `UNET` | 3D U-Net | `monai.networks.nets.UNet` |
| `SEGRESNET` | SegResNet (residual) | `monai.networks.nets.SegResNet` |
| `SWINUNETR` | SwinUNETR (transformer) | `monai.networks.nets.SwinUNETR` |
| `DYNUNET` | DynUNet (non-residual) | `monai.networks.nets.DynUNet` |

### Modern Baselines (Optional)

Additional state-of-the-art architectures from official repositories. These require setup before training:

```bash
# Cross-platform (MedNeXt, 3D UX-Net — also unlocks SwinUNETR-V2)
python scripts/setup_modern_baselines.py mednext uxnet3d --install

# Linux/WSL2 only (Mamba CUDA extensions required)
python scripts/setup_modern_baselines.py umamba segmamba
```

| Key | Architecture | Source | Platform |
|-----|-------------|--------|----------|
| `MEDNEXT` | MedNeXt-S | MIC-DKFZ/MedNeXt | Cross-platform |
| `UXNET3D` | 3D UX-Net | MASILab/3DUX-Net | Cross-platform |
| `SWINUNETR_V2` | SwinUNETR v2 | MONAI (use_v2=True) | Cross-platform |
| `UMAMBA_ENC` | U-Mamba Enc | bowang-lab/U-Mamba | Linux/WSL2 |
| `SEGMAMBA` | SegMamba | ge-xing/SegMamba | Linux/WSL2 |

Train individually:

```bash
python -m src.train --variant MEDNEXT --cv
python -m src.train --variant UXNET3D --cv
python -m src.train --variant SWINUNETR_V2 --cv
python -m src.train --variant UMAMBA_ENC --cv
python -m src.train --variant SEGMAMBA --cv
```

> See [`docs/MODERN_BASELINES.md`](docs/MODERN_BASELINES.md) for pinned revisions, Mamba CUDA setup, and [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md) for attribution and licenses.

### Real nnU-Net v2

A genuine nnU-Net v2 pipeline (not MONAI DynUNet) with matched 80-epoch budget:

```bash
pip install nnunetv2
python -m src.nnunet_emidec prepare       # EMIDEC → Dataset501 + plan
python -m src.nnunet_emidec train --cv    # 80 epochs / fold, same test folds
python -m src.nnunet_emidec eval --cv     # pure MI_path mean ± std
python -m src.make_tables --cv
```

---

## Evaluation & Reporting

### Metrics

Computed per-structure (Dice, IoU, Precision, Recall, HD95 in mm):

| Metric | Description |
|--------|-------------|
| **MI_path** | MI Dice on pathological cases only (**primary**) |
| LV / MYO / MI / MVO Dice | Per-structure Dice coefficients |
| HD95 | 95th-percentile Hausdorff distance (mm) |
| Disease Acc | Normal vs pathological classification accuracy (M5 only) |

### Checkpoint selection

Validation checkpointing uses **MI_path + 0.05 × (LV + MYO)**.  
This prevents all-background models from locking on empty–empty Infarct Dice ≈ N_normal / N_val (the SegResNet fold-0 collapse). Sparse-MI voxel suppression is **not** applied during training validation — only at final test.

### Generated outputs

| File | Contents |
|------|----------|
| `results/paper/cv_all_metrics.csv` | All models — mean ± std |
| `results/paper/cv_ablation_metrics.csv` | M1–M5 ablation |
| `results/paper/cv_baseline_metrics.csv` | PyTorch baselines + nnU-Net |
| `results/paper/cv_per_fold_metrics.csv` | Per-fold breakdown |
| `results/paper/sota_comparison.csv` | Literature SOTA comparison |
| `results/paper/PAPER_COMPARISON.md` | Formatted comparison report |
| `figures/paper/` | Publication-ready plots |

---

## Patient Visualisation

Generate per-patient overlay visualisations:

```bash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
```

---

## Technical Details

### Volume geometry

- **Target spacing:** 1.5 × 1.5 × 10.0 mm (in-plane × through-plane)
- **Target shape:** 128 × 128 × 16 (H × W × D)
- **Depth padded/resized to 16** so four stride-2 pools reach a valid bottleneck

### Training configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimiser | Adam | lr=1e-4, cosine → 1e-6 |
| Batch size | 2 | 1 for SwinUNETR / modern baselines |
| CV epochs | 80 | Same for all models |
| Single-split epochs | 150 | Ablation; 80 for baselines |
| FTL (α, β, γ) | 0.65, 0.35, 0.75 | Reduced FN-obsession |
| Topo λ | 0.05 | Curriculum: warmup 40ep, ramp 20ep |
| Disease λ | 0.5 | Binary classification loss weight |
| Seed | 42 | Reproducible splits and init |

### CLI options

```
python -m src.train --help

  --variant       M1-M5, baseline name, comma-list, 'all', 'baselines', 'everything'
  --epochs        Override epoch count (CV default: CV_EPOCHS for ALL models)
  --batch-size    Override model-specific default batch size
  --cv            5-fold CV mode
  --fold N        Run single fold (0..4); implies CV
  --init-from     Warm-start from another variant
  --no-warm-start Disable default M5←M4 warm-start
  --skip-done     CV: skip folds with completed history (safe crash resume)
```

---

## Results

### 5-Fold CV — Ablation (MI_path Dice, pathological cases only)

| Variant | LV | MYO | **MI_path** | MVO | Params (M) |
|---------|----:|-----:|----------:|-----:|-----------:|
| M1 | 0.911 | 0.739 | 0.440 | 0.383 | 46.6 |
| M2 | 0.908 | 0.729 | 0.410 | 0.355 | 11.5 |
| M3 | 0.905 | 0.761 | 0.423 | 0.367 | 16.1 |
| M4 | 0.906 | 0.759 | 0.433 | 0.309 | 16.1 |
| **M5** | **0.912** | **0.776** | **0.401** | **0.426** | **16.1** |

### 5-Fold CV — Baselines

| Variant | LV | MYO | **MI_path** | MVO | Params (M) |
|---------|----:|-----:|----------:|-----:|-----------:|
| UNet | 0.880 | 0.665 | 0.272 | 0.560 | 19.2 |
| SegResNet | 0.903 | 0.700 | 0.298 | 0.601 | 4.7 |
| SwinUNETR | 0.811 | 0.680 | 0.297 | 0.476 | 15.7 |
| DynUNet | 0.897 | 0.689 | 0.376 | 0.528 | 22.6 |

### EMIDEC Literature SOTA

| Method | Year | Protocol | MYO | MI |
|--------|-----:|----------|----:|---:|
| Zhang (cascaded nnU-Net) | 2021 | Official test (50) | 0.879 | 0.712 |
| ICPIU-Net | 2022 | Official test (50) | 0.877 | 0.734 |
| ICPIU-Net (5-fold) | 2022 | 5-fold CV (100) | 0.895 | 0.783 |
| Schwab (EcorC) | 2025 | 5-fold CV (100) | 0.860 | **0.760** |
| Expert inter-observer | 2020 | EMIDEC Data 2020 | 0.830 | 0.690 |

> SOTA MI Dice target: **0.760** (Schwab 2025, 5-fold). Stretch: **0.783** (ICPIU-Net 5-fold).

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/AVAILABLE_MODELS.md`](docs/AVAILABLE_MODELS.md) | All model variants with architecture details |
| [`docs/MODERN_BASELINES.md`](docs/MODERN_BASELINES.md) | Setup guide for MedNeXt, UX-Net, U-Mamba, SegMamba |
| [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md) | Third-party model provenance, revisions, licenses |
| [`results/paper/PAPER_COMPARISON.md`](results/paper/PAPER_COMPARISON.md) | Formatted SOTA comparison for thesis |

---

## Citation

> **AFDD-Net:** Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior. Joint anatomy (LV, MYO) and pathology (MI, MVO) segmentation on EMIDEC LGE-MRI using anisotropic factorized 3D convolutions, dual-decoder MYO soft gating, Focal Tversky loss, and topology consistency loss.

Raw EMIDEC path: `config.py` → `EMIDEC_ROOT`.
