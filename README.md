# EMIDEC Dual-Decoder Segmentation (Revised Methodology)



Jointly-trained 3D network for EMIDEC LGE-MRI:



1. **Anatomy decoder** - LV cavity + myocardium (RV omitted: EMIDEC has no RV mask)

2. **Pathology decoder** - MI + MVO with MYO soft gating

3. **Innovations** - anisotropic factorized 3D convs, dual decoder + MYO gate, Focal Tversky, topology consistency loss \(L_{topo}\)



## Quick start



```bash

pip install -r requirements.txt

python -m src.data.preprocess

python -m src.train --variant M5 --epochs 150

python -m src.evaluate --variant M5 --split test

```



Ablation (methodology Table 4.5):



```bash

python -m src.train --variant all --epochs 150

python -m src.evaluate --all --split test

```



Raw EMIDEC NIfTI path is set in `config.py` (`EMIDEC_ROOT`).



## Variants



| ID | Change |

|----|--------|

| M1 | Isotropic 3x3x3 single decoder (4-class) |

| M2 | + Factorized anisotropic convs |

| M3 | + Dual decoder + MYO soft gate |

| M4 | + Focal Tversky on pathology |

| M5 | + \(L_{topo}\) (full proposed) |



## Metrics (paper-comparable)



LV / MYO / MI / MVO Dice, IoU, Recall, HD95 (mm), parameter count, inference ms.

SOTA targets from methodology: ICPIU-Net MI Dice 0.783, nnU-Net 0.720.

## Patient report (figures + dysfunction %)

`ash
python -m src.visualize_patient --case P001
python -m src.visualize_patient --case Case_P087 --all-slices
`

Outputs in figures/patients/<Case_ID>/: report PNG, panels PNG, stats JSON.
Reports MI/MVO/dysfunction as % of MYO, % of LV cavity, and % of LV total (cavity+MYO).
