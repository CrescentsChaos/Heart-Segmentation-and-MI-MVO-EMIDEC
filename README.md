# AFDD-Net on EMIDEC (Revised Methodology)

**AFDD-Net** (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior*)
segments LV + MYO first, then MI + MVO on EMIDEC LGE-MRI.

## Primary metric

**MI_path** = pure MI Dice (EMIDEC label 3) on pathological cases only.  
This matches published EMIDEC SOTA (Schwab / ICPIU / nnU-Net). Do **not** cite MI_all (healthy empty–empty Dice = 1.0 inflates it).  
`Infarct` (MI∪MVO) is reported only as a secondary multiclass column.

## 5-fold CV (preferred for SOTA comparison)

All models share the **same** `Dataset/folds.json` and **same** epoch budget (`CV_EPOCHS=80`).

```bash
pip install -r requirements.txt
python -m src.data.preprocess                 # rebuild npz (5-class MI/MVO)
python -m src.data.preprocess --folds-only    # keep existing folds.json
python -m src.train --variant everything --cv # M1–M5 + installed PyTorch baselines
python -m src.evaluate --all --baselines --cv --no-figs
python -m src.make_tables --cv
python -m src.paper_figures
```

### Real nnU-Net v2 (not MONAI DynUNet)

```bash
pip install nnunetv2
python -m src.nnunet_emidec prepare           # EMIDEC → Dataset501 + plan
python -m src.nnunet_emidec train --cv        # 80 epochs / fold, same test folds
python -m src.nnunet_emidec eval --cv         # pure MI_path mean±std
python -m src.make_tables --cv
```

CSVs under `results/paper/`:
- `cv_all_metrics.csv` — all models (mean ± std)
- `cv_ablation_metrics.csv` — M1–M5
- `cv_baseline_metrics.csv` — PyTorch baselines + nnU-Net
- `cv_per_fold_metrics.csv`

## Ablation notes (important)

| Variant | What it adds |
|---------|----------------|
| M1 | Isotropic 3D U-Net, 5-class (BG/LV/MYO/**MI**/MVO) |
| M2 | + anisotropic factorized convs |
| M3 | + dual decoder + MYO soft gate (**no** disease classifier) |
| M4 | + Focal Tversky (**no** disease classifier) |
| M5 | + topology curriculum + **disease classifier / gate** |

## External baselines

| Key | What it is |
|-----|------------|
| `UNET` / `SEGRESNET` / `SWINUNETR` / `DYNUNET` | MONAI nets, 5-class pure MI |
| `DYNUNET_RES` | MONAI residual DynUNet (formerly mislabeled “nnU-Net”) |
| `SWINUNETR_V2` | MONAI SwinUNETR with stage residual convolutions |
| `MEDNEXT` | Official MIC-DKFZ MedNeXt-S |
| `UXNET3D` | Official MASILab 3D UX-Net |
| `UMAMBA_ENC` | Official U-Mamba Enc (Linux/WSL2 CUDA) |
| `SEGMAMBA` | Official external SegMamba (Linux/WSL2 CUDA) |
| `NNUNET` | **Real** nnU-Net v2 (`src/nnunet_emidec.py`) |

Under `--cv`, every PyTorch model trains **80 epochs**. Real nnU-Net uses `nnUNetTrainerAFDD80` (also 80 epochs).

Prepare the official optional model sources before training:

```bash
python scripts/setup_modern_baselines.py mednext uxnet3d --install
python scripts/setup_modern_baselines.py umamba segmamba
```

See `docs/MODERN_BASELINES.md` for pinned revisions and Mamba CUDA setup.

## Checkpoint selection (SegResNet fix)

Val checkpointing uses **MI_path + 0.05×(LV+MYO)**.  
This prevents all-background models from locking on empty–empty Infarct Dice ≈ N_normal/N_val (the SegResNet fold-0 collapse). Sparse-MI voxel suppression is **not** applied during training validation (only at final test).

## Train / eval one fold

```bash
python -m src.train --variant M5 --cv --fold 0
python -m src.evaluate --variant M5 --cv
```

## Legacy single split (70/15/15)

```bash
python -m src.data.preprocess
python -m src.train --variant M5 --epochs 150
python -m src.evaluate --variant M5 --split test
```

## Patient report

```bash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
```

Raw EMIDEC path: `config.py` → `EMIDEC_ROOT`.  
SOTA MI Dice target: **0.760** (Schwab 2025, 5-fold). Stretch: **0.783** (ICPIU-Net 5-fold).
