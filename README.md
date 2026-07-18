# AFDD-Net on EMIDEC (Revised Methodology)

**AFDD-Net** (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior*)
segments LV + MYO first, then MI + MVO on EMIDEC LGE-MRI.

## Quick start (5-fold CV — preferred for SOTA comparison)

All models (M1–M5 **and** baselines) share the **same** `Dataset/folds.json` and the **same** epoch budget (`CV_EPOCHS=80`, aligned with nnU-Net 2021 5-fold EMIDEC reporting).

```bash
pip install -r requirements.txt
python -m src.data.preprocess --folds-only          # freeze folds.json (once)
python -m src.train --variant everything --cv       # 5 folds × all models, same epochs
python -m src.evaluate --all --baselines --cv --no-figs
python -m src.make_tables --cv
python -m src.paper_figures
```

CSVs written under `results/paper/`:
- `cv_all_metrics.csv` — all models (mean ± std)
- `cv_ablation_metrics.csv` — M1–M5
- `cv_baseline_metrics.csv` — MONAI baselines
- `cv_per_fold_metrics.csv` — 50 rows (10 models × 5 folds)
- `ablation_metrics.csv` / `sota_comparison.csv` — paper figures export

## Train / eval one fold

```bash
python -m src.train --variant M5 --cv --fold 0
python -m src.evaluate --variant M5 --cv
```

Artifacts: `checkpoints/M5_fold{k}_best.pth`, `results/M5_fold{k}_test_metrics.json`, `results/M5_cv_metrics.json` (mean±std).

## Legacy single split (70/15/15)

```bash
python -m src.data.preprocess
python -m src.train --variant M5 --epochs 150
python -m src.evaluate --variant M5 --split test
python -m src.paper_figures
```

Full ablation (single split):

```bash
python -m src.train --variant all --epochs 150
python -m src.evaluate --all --split test
python -m src.paper_figures
```

## External baselines

Under **`--cv`**, every model (M1–M5 + 5 baselines) trains for **80 epochs** (`CV_EPOCHS`) on the same folds.

Without `--cv`, baselines still default to **80** epochs (`BASELINE_EPOCHS`); ablation defaults to `EPOCHS` (150) unless you pass `--epochs 80`.

```bash
python -m src.train --variant baselines --epochs 80
python -m src.evaluate --baselines --split test
```

Notes:

- Baselines are **4-class multiclass** (BG / LV / MYO / Infarct); MI+MVO are merged as Infarct (same protocol as M1/M2).
- `NNUNET` here is an **nnU-Net-style residual DynUNet** in MONAI (not the full nnU-Net auto-config planner).
- `SWINUNETR` pads depth 16→32 internally and defaults to `batch_size=1`.
- **Do not regenerate `folds.json` mid-experiment** (use `--overwrite-folds` only if starting fresh).
- Per fold: test = held-out fold; val = stratified 15% of the other 4 folds; M5 warm-starts from **same-fold** M4.

## Patient report

```bash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
python -m src.visualize_patient --case P001 --variant UNET
```

## Paper comparison outputs

After evaluation, `python -m src.paper_figures` / `python -m src.make_tables --cv` write thesis tables.

SOTA MI Dice target: **0.760** (Schwab 2025, 5-fold). Stretch: **0.783** (ICPIU-Net 5-fold).

Raw EMIDEC path: `config.py` → `EMIDEC_ROOT`.
