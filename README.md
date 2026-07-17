# AFDD-Net on EMIDEC (Revised Methodology)

**AFDD-Net** (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating and Topology Consistency*)
segments LV + MYO first, then MI + MVO on EMIDEC LGE-MRI.

## Quick start

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.train --variant M5 --epochs 150
python -m src.evaluate --variant M5 --split test
python -m src.paper_figures
```

Full ablation:

```bash
python -m src.train --variant all --epochs 150
python -m src.evaluate --all --split test
python -m src.paper_figures
```

## External baselines (80 epochs)

Compare against MONAI **UNet**, **SegResNet**, **SwinUNETR**, **nnU-Net-style DynUNet**, and **DynUNet** (same EMIDEC preprocess + Dice+CE multiclass loss as M1/M2).

Train all baselines (default **80** epochs):

```bash
python -m src.train --variant baselines --epochs 80
```

Or one at a time:

```bash
python -m src.train --variant UNET --epochs 80
python -m src.train --variant SEGRESNET --epochs 80
python -m src.train --variant SWINUNETR --epochs 80
python -m src.train --variant NNUNET --epochs 80
python -m src.train --variant DYNUNET --epochs 80
```

Evaluate baselines:

```bash
python -m src.evaluate --baselines --split test
# or individually:
python -m src.evaluate --variant UNET,SEGRESNET,SWINUNETR,NNUNET,DYNUNET --split test
```

Train ablation + baselines together:

```bash
python -m src.train --variant everything
python -m src.evaluate --all --baselines --split test
python -m src.make_tables
python -m src.paper_figures
```

Notes:

- Baselines are **4-class multiclass** (BG / LV / MYO / Infarct); MI+MVO are merged as Infarct (same protocol as M1/M2).
- `NNUNET` here is an **nnU-Net-style residual DynUNet** in MONAI (not the full nnU-Net auto-config planner).
- `SWINUNETR` pads depth 16→32 internally (Swin requires spatial size divisible by 32) and defaults to `batch_size=1`.
- Omitting `--epochs` for a baseline variant also uses `config.BASELINE_EPOCHS` (80).

## Patient report

```bash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
python -m src.visualize_patient --case P001 --variant UNET
```

## Paper comparison outputs

After evaluation, `python -m src.paper_figures` writes:

| Path | Content |
|------|---------|
| `figures/paper/training_curves_*.png` | Loss / Dice / HD95 / Recall curves |
| `figures/paper/ablation_*.png` | Ablation Dice, HD95, Recall, IoU |
| `figures/paper/baseline_learning_curves_MI.png` | Baseline infarct Dice curves |
| `figures/paper/sota_comparison_*.png` | vs verified EMIDEC-only methods (Zhang, ICPIU-Net, Schwab, …) |
| `figures/paper/table_*.png` | Rendered thesis tables |
| `results/paper/PAPER_COMPARISON.md` | Markdown comparison report |
| `results/paper/*.csv` | Ablation + SOTA CSVs |

SOTA MI Dice target: **0.760** (Schwab 2025, current EMIDEC best). Stretch: **0.783** (ICPIU-Net 5-fold).

Raw EMIDEC path: `config.py` → `EMIDEC_ROOT`.
