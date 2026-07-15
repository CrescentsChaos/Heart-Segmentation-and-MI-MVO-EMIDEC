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

## Patient report

```bash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
```

## Paper comparison outputs

After evaluation, `python -m src.paper_figures` writes:

| Path | Content |
|------|---------|
| `figures/paper/training_curves_*.png` | Loss / Dice / HD95 / Recall curves |
| `figures/paper/ablation_*.png` | Ablation Dice, HD95, Recall, IoU |
| `figures/paper/sota_comparison_*.png` | vs verified EMIDEC-only methods (Zhang, ICPIU-Net, Schwab, …) |
| `figures/paper/table_*.png` | Rendered thesis tables |
| `results/paper/PAPER_COMPARISON.md` | Markdown comparison report |
| `results/paper/*.csv` | Ablation + SOTA CSVs |

SOTA MI Dice target: **0.760** (Schwab 2025, current EMIDEC best). Stretch: **0.783** (ICPIU-Net 5-fold).

Raw EMIDEC path: `config.py` → `EMIDEC_ROOT`.
