# AFDD-Net — Innovations, Failure Analysis, and Fixes

**Model:** AFDD-Net (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating and Topology Consistency*)  
**Thesis topic:** Efficient 3D deep segmentation of myocardial infarction (blockage indicator) on cardiac LGE-MRI  
**Dataset:** EMIDEC  
**Primary success metric:** MI Dice (target **> 0.78**; current published EMIDEC best ? **0.760** Schwab 2025 / stretch **0.783** ICPIU-Net 5-fold)

---

## 1. What this system segments

| Structure | Meaning | Head |
|-----------|---------|------|
| **LV** | Left-ventricular cavity | Anatomy decoder (softmax) |
| **MYO** | Full myocardial wall (healthy + scar + MVO) | Anatomy decoder |
| **MI** | Myocardial infarction (blockage scar) | Pathology decoder (sigmoid) |
| **MVO** | Microvascular obstruction / no-reflow | Pathology decoder |

EMIDEC labels: `0=BG, 1=LV, 2=healthy MYO, 3=MI, 4=MVO`.  
Anatomy MYO wall = `{2,3,4}`. Pathology is two independent channels for MI and MVO.

---

## 2. Ablation ladder (innovations added step-by-step)

| Variant | Name | Innovation added | Architecture | Loss on MI |
|---------|------|------------------|--------------|------------|
| **M1** | Baseline 3D U-Net | Isotropic `3×3×3` single-decoder multiclass | Single decoder | Dice + weighted CE |
| **M2** | AFDD-Net-F | **Anisotropic factorized convolutions** (`3×3×1` + `1×1×3`) matching EMIDEC spacing ? 1.5×1.5×10 mm | Single decoder | Same |
| **M3** | AFDD-Net-D | **Dual decoder** + **MYO soft-gating** into pathology features at every scale | Dual decoder | Soft Dice / Tversky ?=?=0.5 |
| **M4** | AFDD-Net-T | **Focal Tversky** on pathology (FN-sensitive for tiny infarcts) | Same as M3 | FTL ?=0.65, ?=0.35, ?=0.75 |
| **M5** | **AFDD-Net (full)** | **Topology consistency** `L_topo` + curriculum + hard MYO restrict | Same as M3 | FTL + small `L_topo` |

**Important:** M3 / M4 / M5 share the **same network weights topology**. M4 and M5 differences are loss / constraint / training schedule only.

### Novel claims (for thesis wording)

1. **Anisotropy-aware 3D blocks** — native factorization for thick-slice LGE instead of 2D?3D cascade (Zhang / Schwab).  
2. **Joint anatomy–pathology dual decoder with MYO soft-gating** — MI decoded only in myocardial context; imbalance mitigated (~1:200 ? wall-local).  
3. **Topology consistency** — MI / MVO mass must lie inside the myocardial wall (clinical prior: blockage scar ? MYO).  
4. **Efficiency** — full model ? **16M** params vs baseline ? **47M**, with competitive anatomy Dice.

---

## 3. Why M5 MI Dice collapsed to **0.132**

Observed ablation (your run):

| Variant | LV | MYO | **MI** | MVO |
|---------|---:|----:|-------:|----:|
| M1 | 0.915 | 0.728 | 0.321 | — |
| M2 | 0.905 | 0.730 | 0.325 | — |
| M3 | 0.907 | 0.758 | 0.340 | 0.028 |
| M4 | 0.906 | 0.761 | **0.358** | 0.035 |
| M5 (broken) | 0.913 | 0.770 | **0.132** | 0.016 |

### Root cause

M5’s **only** extra vs M4 was:

```text
L_topo = mean( path_prob × (1 ? soft_MYO) )    with ?_topo = 0.5
```

Problems:

1. **?_topo = 0.5 was far too strong** vs `L_FTL`.  
2. Soft MYO was **not detached** ? gradients could expand `path ? MYO` (paint the whole wall as infarct).  
3. Combined with **recall-biased Focal Tversky** (old ?=0.7), the cheap optimum was **high recall, disastrous precision** (wall-wide MI).  
4. Evaluation averaged **false positives on normal cases** (Dice = 0), further dragging the reported mean.

Training symptoms matched this: M5 val MI stuck ? **0.16** with precision ? **0.09**, while M4 reached ? **0.44**.

---

## 4. Code fixes applied (must retrain M5)

### 4.1 Topology loss (`src/losses/joint_loss.py`)

- `myo_mask` is **always detached** inside `TopologyConsistencyLoss`.  
- Prefer **GT myocardial wall** for `L_topo` (`USE_GT_MYO_FOR_TOPO=True`) — teaches “MI ? wall” without coupled MYO inflation.  
- Extra weight on MI channel in the outside penalty.

### 4.2 Soft restrict + detached gate (`src/models/dual_decoder.py`)

- Soft MYO fed to pathology decoder is **detached**.  
- Pathology probability is multiplied by detached soft MYO (`path *= myo.detach()`), so infarct mass cannot legally live outside the wall.

### 4.3 Safer hyperparameters (`config.py`)

| Setting | Old (broken) | New |
|---------|-------------:|----:|
| `LAMBDA_TOPO` | 0.5 | **0.05** |
| `TOPO_WARMUP_EPOCHS` | — | **40** (train like M4 first) |
| `TOPO_RAMP_EPOCHS` | — | **20** (linear ramp) |
| `FTL_ALPHA / BETA` | 0.7 / 0.3 | **0.65 / 0.35** |
| `MI_CHANNEL_WEIGHT` | — | **1.5** |
| `HARD_MYO_MASK_AT_INFER` | unused in eval | **True** |

### 4.4 Training curriculum (`src/train.py`)

- Epochs 1–40: `?_topo = 0` (behaves as M4).  
- Epochs 41–60: ramp `?_topo ? 0.05`.  
- **Default warm-start M5 ? M4** (`M4_best.pth`) so topology fine-tunes a healthy MI head.  
- Checkpoint selection uses **pathological-only MI Dice** when available.

### 4.5 Evaluation (`src/evaluate.py`)

- Applies **hard MYO mask** before thresholding (MYO-first protocol, as in SOTA).  
- Reports both overall and **`MI_pathological`** means (EMIDEC-comparable).

---

## 5. How to retrain and check (required)

> Existing `checkpoints/M5_best.pth` was trained with the broken topology loss. **Do not cite it.** Retrain.

```bash
# 1) Ensure M4 exists (best current MI among dual models)
python -m src.train --variant M4 --epochs 150

# 2) Retrain fixed M5 (auto warm-starts from M4)
python -m src.train --variant M5 --epochs 150

# 3) Evaluate all
python -m src.evaluate --all --split test

# 4) Refresh tables / figures
python -m src.make_tables
python -m src.paper_figures --split test
```

**Watch during training:**

- Warmup: `MI_path` should track M4 (? ~0.40 val).  
- After topo ramp: MI must **not** crater; precision should stay reasonable.  
- If MI drops sharply when `?_topo` turns on ? lower `LAMBDA_TOPO` to `0.02`.

---

## 6. Realistic path to MI Dice > 0.78

Target **0.78** is above current published EMIDEC 5-fold best (**0.760** Schwab). Code fixes stop the collapse; reaching SOTA still needs a full experimental protocol:

| Stage | Expected MI Dice | Notes |
|------:|-----------------:|-------|
| Broken M5 (before fix) | ~0.13 | Topology collapse — discarded |
| Fixed M5 (stratified test, path-only) | **0.45–0.65** | After retrain with curriculum |
| Competitive (match Zhang 0.712) | **? 0.70** | Needs 150+ epochs, strong aug, hard mask |
| Claim vs Schwab / ICPIU | **? 0.76 / 0.78** | Prefer **5-fold CV on 100 labeled cases** + official EMIDEC metrics |

### Recommended next experiments (after fixed M5 converges)

1. Report **pathological-only MI Dice** + FP rate on normals separately.  
2. Match SOTA preprocessing more closely (?96×96 crop, per-volume z-score — already partially done).  
3. Run **5-fold CV** for fair comparison with Schwab / ICPIU.  
4. Optional boosters (pick one): late MI boundary / Hausdorff surrogate; mild TTA (already in `inference.py`); longer fine-tune of pathology head with frozen anatomy.

Until MI ? ~0.65–0.70, frame the thesis contribution as:

> addressing class collapse and topology-aware blockage localization under extreme imbalance, with progressive ablation proving each component,

not as having surpassed EMIDEC SOTA yet.

---

## 7. EMIDEC-only comparison targets (use these in Table 4.7)

| Method | Protocol | MYO | MI |
|--------|----------|----:|---:|
| Zhang (cascaded nnU-Net) | Official test | 0.879 | **0.712** |
| ICPIU-Net | Official test | 0.877 | **0.734** |
| ICPIU-Net | 5-fold CV | 0.895 | **0.783** |
| Schwab EcorC | 5-fold CV | 0.860 | **0.760** |
| 3D nnU-Net (EMIDEC) | 5-fold | 0.872 | 0.688 |
| Expert inter-observer | Data 2020 | 0.830 | 0.690 |

**Do not cite** Isensee 2021 nnU-Net MI 0.72 as EMIDEC — that number is from a private LGE cohort.

Source of truth in code: `src/model_identity.py` ? `SOTA_BENCHMARKS`.

---

## 8. File map

| Path | Role |
|------|------|
| `config.py` | Hyperparameters, topo curriculum |
| `src/models/dual_decoder.py` | M1–M5 architectures, MYO gate + soft restrict |
| `src/losses/joint_loss.py` | Dice/CE, Focal Tversky, fixed `L_topo` |
| `src/train.py` | Training, warm-start, curriculum, path-only selection |
| `src/evaluate.py` | Test metrics, hard MYO mask, pathological-only |
| `src/inference.py` | Hard mask + TTA utilities |
| `src/model_identity.py` | Names + verified EMIDEC SOTA table |
| `src/paper_figures.py` | Thesis figures / comparison report |

---

## 9. Bottom line

- **M5 was not “worse architecture”** — it was **broken topology training** (`?=0.5`, undetached MYO, paint-the-wall).  
- **Fixes are in the code**; **retrain M5** (preferably after a solid M4) before updating thesis numbers.  
- Target **> 0.78** is a stretch goal above published EMIDEC best; treat **? 0.70** as the first publication-ready milestone, then push with 5-fold CV.
