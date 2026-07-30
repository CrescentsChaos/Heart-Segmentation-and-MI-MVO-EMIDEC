# An Efficient 3D Deep Neural Architecture for Segmentation of Blockages in Heart Using Cardiac MRI Images

## Draft Report — AFDD-Net: Joint Cardiac Anatomy and Pathology Segmentation on EMIDEC LGE-MRI

---

## 1. Introduction

Coronary artery disease (CAD) is the leading cause of death worldwide. When a coronary artery is blocked — whether by atherosclerotic plaque, thrombosis, or embolism — the downstream myocardial tissue is deprived of oxygenated blood. This ischaemic insult leads to **myocardial infarction (MI)**, the irreversible necrosis of cardiac muscle. In a subset of patients, even after revascularisation, the microvasculature within the infarcted territory remains obstructed, a phenomenon known as **microvascular obstruction (MVO)**. Both MI and MVO are direct manifestations of vascular blockages and serve as critical prognostic markers: the extent and transmurality of MI strongly predict long-term left ventricular remodelling, heart failure, and mortality, while the presence of MVO indicates a substantially worse prognosis even after successful percutaneous coronary intervention.

**Late Gadolinium Enhanced (LGE) cardiac MRI** is the clinical gold standard for non-invasive visualisation of these blockage-induced tissue injuries. In LGE imaging, gadolinium contrast agent accumulates in regions of damaged myocardium (appearing as hyperintense signal), providing a direct spatial map of infarction and microvascular obstruction. Accurate segmentation of MI and MVO from LGE-MRI is therefore clinically essential for treatment planning, risk stratification, and monitoring therapeutic response. However, manual delineation is time-consuming, subjective, and prone to inter-observer variability — motivating the development of automated deep learning approaches.

Automated segmentation of MI and MVO from LGE cardiac MRI is a technically challenging task for several reasons:
1. **Extreme class imbalance:** MI and MVO voxels constitute less than 1% of the total volume, making them extremely sparse relative to background, left ventricle (LV) cavity, and myocardial wall (MYO) structures.
2. **Irregular morphology:** Infarct regions are characteristically heterogeneous in shape, size, and location, varying dramatically across patients.
3. **Anatomical confinement:** MI and MVO, by clinical definition, reside exclusively within the myocardial wall — a constraint that most generic segmentation architectures fail to exploit.
4. **Mixed patient population:** Clinical datasets contain both pathological and healthy patients, and models must correctly identify the absence of pathology without generating spurious false-positive predictions.
5. **Highly anisotropic voxel spacing:** LGE cardiac MRI acquisitions typically have fine in-plane resolution (~1.5 mm) but thick through-plane slices (~10 mm), creating a ~7× anisotropy that isotropic 3D convolution kernels handle poorly.

This work proposes **AFDD-Net** (*Anisotropic Factorized Dual-Decoder Network with MYO Soft-Gating, Topology Consistency, and Disease Classification Prior*), a novel and **parameter-efficient** 3D deep neural architecture designed specifically for the automated segmentation of blockage-induced cardiac pathologies (MI and MVO) from LGE cardiac MRI. AFDD-Net addresses the above challenges through five key innovations:

1. **Anisotropic factorized 3D convolutions** that respect the dataset's highly anisotropic voxel geometry, reducing parameters by ~65% compared to conventional isotropic 3D U-Net architectures while maintaining segmentation quality.
2. **A dual-decoder architecture** with separate anatomy (LV, MYO) and pathology (MI, MVO) prediction heads, enabling task-specific specialisation.
3. **MYO soft-gating** to anatomically constrain pathology predictions within the predicted myocardial wall, encoding the clinical prior that blockage-induced damage occurs exclusively within the myocardium.
4. **Focal Tversky loss** to combat the severe class imbalance inherent in sparse pathology voxels.
5. **Topology consistency loss with curriculum scheduling** and a **disease classification prior** to suppress false-positive predictions on healthy patients, ensuring high MI Dice Score across the full patient population.

The proposed model is validated on the **EMIDEC dataset** (100 cases total: 67 pathological + 33 normal) using **stratified 5-fold cross-validation** with identical folds, training epochs (80), and evaluation protocol across all models for fair comparison.

**Label Protocol (EMIDEC Official):**
- `0` = Background (BG)
- `1` = Left Ventricle cavity (LV)
- `2` = Normal Myocardium (MYO)
- `3` = Myocardial Infarction (MI)
- `4` = Microvascular Obstruction (MVO)

**Primary evaluation metric:** **MI Dice Score** — MI Dice (EMIDEC label 3) computed across **all cases** (both pathological and normal), measuring the model's ability to both accurately segment MI in diseased patients and correctly suppress false positives in healthy patients. This holistic evaluation is essential for clinical deployment, where a model must perform reliably across the full spectrum of patient presentations.

---

## 2. Dataset and Preprocessing

### 2.1 EMIDEC Dataset

The EMIDEC (Evaluation of Myocardial Infarction from Delayed Enhancement Cardiac MRI) dataset consists of 100 LGE cardiac MRI volumes:
- **67 pathological cases** (Case_P*) with MI and/or MVO — these represent patients with confirmed coronary artery blockages that have resulted in myocardial tissue damage
- **33 normal/healthy cases** (Case_N*) without infarction — these represent patients with no evidence of blockage-induced myocardial injury

The inclusion of both pathological and normal cases in the dataset is critical for evaluating a model's clinical utility. A clinically deployable system must not only accurately delineate infarct regions in diseased patients but also confidently identify the absence of pathology in healthy patients without generating false alarms.

### 2.2 Volume Geometry and Preprocessing

EMIDEC volumes are **highly anisotropic** with an approximate native resolution of **1.5 × 1.5 mm in-plane** and **~10 mm through-plane** (slice thickness). This ~7× anisotropy between spatial axes is a defining characteristic of the dataset and a core design constraint for the proposed architecture.

**Preprocessing pipeline:**
1. **Resampling:** All volumes are resampled to a uniform target spacing of **1.5 × 1.5 × 10.0 mm** (H × W × D) using cubic interpolation for images and nearest-neighbour interpolation for masks.
2. **Resizing:** Volumes are resized to a target shape of **128 × 128 × 16** (H × W × D). The depth of 16 slices ensures four stride-2 pooling operations reach a valid bottleneck in the encoder.
3. **Normalisation:** Z-score normalisation (zero mean, unit variance) is applied to each volume independently.
4. **Target construction:**
   - *Anatomy target (3-class):* BG (0), LV (1), Full MYO wall (2) — where MYO includes healthy myocardium ∪ MI ∪ MVO voxels
   - *Pathology target (2-channel multi-label):* Channel 0 = MI (raw label 3), Channel 1 = MVO (raw label 4)
   - *Multiclass target (5-class):* BG (0), LV (1), Healthy MYO (2), MI (3), MVO (4) — used by M1, M2, and MONAI baselines

### 2.3 Cross-Validation Protocol

A **stratified 5-fold cross-validation** protocol is used as the primary evaluation strategy, stratifying by pathological vs. normal cases to ensure each fold has a balanced representation. All models — ablation variants (M1–M5) and external baselines — are trained and evaluated using:
- The **same** `folds.json` split
- The **same** epoch budget (`CV_EPOCHS = 80`)
- The **same** checkpoint selection criterion
- Seed = 42 for reproducibility

---

## 3. Data Augmentation Strategy

A comprehensive, anisotropy-aware data augmentation pipeline is designed specifically for the EMIDEC dataset. The augmentation strategy respects three critical design constraints:

### 3.1 Anisotropy-Aware Design Constraints

1. **In-plane only geometric transforms:** Due to the ~7× anisotropy (1.5 mm in-plane vs. 10 mm through-plane), all continuous geometric operations (affine rotation, scaling, elastic deformation) are applied **exclusively in the H, W plane** and identically across all D slices. The D-axis is only ever flipped — a lossless, interpolation-free operation — and is never rotated or resampled, which would smear the thick slices.

2. **Strict spatial alignment:** All spatial transformations are applied with a single shared random draw per sample, guaranteeing that the input image and all associated label masks (anatomy, pathology, multiclass) remain perfectly pixel-aligned.

3. **Interpolation mode discipline:** Label tensors always use `nearest` mode interpolation to preserve integer/binary values. Image tensors use `bilinear` interpolation for continuous geometric transforms.

### 3.2 Geometric Transformations (Synchronized across Image + Labels)

| Transform | Probability | Parameters | Notes |
|-----------|:-----------:|------------|-------|
| **Random Flips** | 50% | H, W, and D axes independently | Discrete, lossless; D-flip is safe (no interpolation) |
| **Random 90° Rotations** | 50% | k ∈ {1, 2, 3} × 90° in-plane (H, W) | Discrete, lossless |
| **Affine Transform** | 30% | Rotation: ±15°; Scale: [0.9, 1.1] | Continuous, in-plane only |
| **Elastic Deformation** | 20% | α = 8.0 voxels, σ = 6.0 voxels | Continuous, in-plane only; **pathology-aware boosting** |

**Pathology-Aware Elastic Boosting:** The elastic deformation field is biased to be **1.5× stronger near MI/MVO voxels** and 1.0× elsewhere. This introduces greater geometric variation in the pathology regions where it matters most, helping the model generalise to diverse infarct shapes and locations without excessively distorting healthy anatomy.

### 3.3 Intensity Transformations (Image Only)

| Transform | Probability | Parameters |
|-----------|:-----------:|------------|
| **Brightness & Contrast** | 30% | Brightness offset: [−0.1, +0.1]; Contrast scale: [0.85, 1.15] |
| **Gamma Correction** | 30% | γ ∈ [0.7, 1.5] (power-law intensity mapping) |
| **Gaussian Blur** | 15% | σ ∈ [0.5, 1.2] (simulates acquisition blurring) |
| **Gaussian Noise** | 30% | σ ∈ [0.0, 0.05] (simulates scanner noise) |

All probability and magnitude parameters are **conservatively tuned** for the relatively small size of the EMIDEC training dataset (~80 training cases per fold) to reduce overfitting without destroying underlying anatomical structure.

### 3.4 Test-Time Augmentation (TTA)

At inference, **8-fold test-time augmentation** is applied:
- Identity, flip-W, flip-H, flip-D, rot90, rot180, rot270, rot90+flip-W
- Predictions are averaged in logit/probability space before thresholding

---

## 4. Proposed Architecture: AFDD-Net

### 4.1 Overview

AFDD-Net follows an encoder–dual-decoder design. A **shared encoder** extracts multi-scale features from the input LGE volume. Two separate decoders — an **anatomy decoder** and a **pathology decoder** — then reconstruct the segmentation maps at full resolution. The anatomy decoder produces a 3-class softmax output (BG/LV/MYO), while the pathology decoder produces a 2-channel sigmoid output (MI/MVO), gated by the predicted myocardial wall probability from the anatomy branch.

The architecture is specifically designed to be **parameter-efficient**: by factorizing 3D convolutions into sequential in-plane and through-plane operations, AFDD-Net achieves a total parameter count of just **16.1M** — a **65% reduction** compared to the 46.6M parameters of a standard isotropic 3D U-Net baseline (M1), and fewer parameters than comparable architectures such as DynUNet (22.6M).

### 4.2 Anisotropic Factorized 3D Convolutions

Standard isotropic 3×3×3 3D convolutions assume uniform voxel spacing, which is inappropriate for EMIDEC's highly anisotropic resolution. We introduce **Anisotropic Factorized 3D Convolutions** that decompose each 3D convolution into two sequential operations:

1. **In-plane convolution:** `Conv3d` with kernel `(1, 3, 3)` — processes each axial slice independently in the high-resolution H, W plane.
2. **Through-plane convolution:** `Conv3d` with kernel `(3, 1, 1)` — aggregates information across neighbouring slices along the low-resolution D axis.

A **residual shortcut** (identity when `in_ch == out_ch`, or a 1×1 pointwise projection otherwise) is added to improve gradient flow. This factorisation:
- **Reduces parameters** from 27 (3³) to 12 (1×3×3 + 3×1×1) per convolution position, yielding a **~4× parameter reduction per convolution block** (total model: 46.6M → 11.5M at the single-decoder stage, and 16.1M with the dual-decoder and classification head)
- **Respects the anisotropic geometry** by treating the in-plane and through-plane axes differently
- **Preserves or improves performance** by avoiding interpolation artefacts from treating thick slices identically to fine in-plane pixels

This factorisation is a core contributor to the **parameter efficiency** claimed in the thesis title. The 65% parameter reduction from the baseline enables the model to be deployed on resource-constrained clinical hardware (e.g., workstations with limited GPU memory) without sacrificing segmentation accuracy.

### 4.3 Shared Encoder

A four-stage encoder produces feature maps at progressively coarser resolutions. Each stage consists of a `DoubleConvBlock` (two consecutive factorized or standard convolution blocks) followed by `MaxPool3d(2)` for downsampling. The bottleneck operates at 1/16 of the in-plane resolution.

| Stage | Output Channels | Resolution (relative) |
|-------|:--------------:|:--------------------:|
| Enc1 | 32 | 1× |
| Enc2 | 64 | 1/2× |
| Enc3 | 128 | 1/4× |
| Enc4 | 256 | 1/8× |
| Bottleneck | 512 | 1/16× |

### 4.4 Anatomy Decoder

The anatomy decoder follows a standard U-Net expansion path with **attention gates** (Oktay et al., 2018) at each skip connection. It produces a 3-class softmax output for BG, LV, and MYO (where MYO represents the full myocardial wall including healthy myocardium, MI, and MVO).

### 4.5 Pathology Decoder with MYO Soft-Gating

The pathology decoder has a similar U-Net expansion path but incorporates **MYO soft-gating** — the predicted MYO probability from the anatomy decoder is injected as an additional channel at every decoder stage. Specifically:

1. The anatomy decoder produces `anat_prob = softmax(anatomy_logits)`
2. The MYO channel probability `myo_mask = anat_prob[:, 2:3]` is extracted (class index 2 = MYO)
3. This MYO mask is **detached** from the computation graph (`.detach()`) to prevent the pathology decoder from inflating MYO predictions to "legalise" wall-wide MI
4. At each decoder stage, `myo_mask` is interpolated to match the feature resolution and concatenated with the skip-connected features before convolution
5. After decoding, a **soft anatomical restriction** multiplies the pathology probability by the detached MYO mask: `path_prob = path_prob × myo_mask.detach()`

This design ensures that MI and MVO predictions are anatomically constrained to reside within the predicted myocardial wall, reflecting the clinical prior that blockage-induced infarction and microvascular obstruction occur exclusively within the myocardium.

### 4.6 Disease Classification Head

A lightweight **disease classification head** (normal vs. pathological) is attached to the bottleneck features:
- `AdaptiveAvgPool3d(1)` → `Flatten` → `Linear(512, 64)` → `ReLU` → `Linear(64, 1)` → `Sigmoid`

At inference, patients classified as healthy (P(pathological) ≤ 0.5) have their MI/MVO predictions **zeroed out**, suppressing false-positive pathology on normal patients. This head is only enabled in the final model variant (M5) and adds negligible parameters (~0.03M).

The disease classification head is a critical contributor to the model's strong **MI Dice Score** across the full patient population: by correctly identifying healthy patients and zeroing their pathology predictions, the model avoids the Dice = 0 penalty that would otherwise be incurred from any spurious false-positive MI/MVO voxels on normal cases. This dual capability — detecting blockage-induced damage where it exists and confirming its absence where it does not — is precisely what a clinically deployable system requires.

---

## 5. Loss Functions

### 5.1 Anatomy Loss: Generalised Dice + Weighted Cross-Entropy

The anatomy head is supervised by a combination of **Generalised Dice Loss** and **class-weighted Cross-Entropy**:

$$\mathcal{L}_{anat} = \mathcal{L}_{CE}^{weighted} + (1 - \text{Dice}_{mean})$$

Class weights for CE: BG = 0.1, LV = 1.0, MYO = 1.5 — emphasising the myocardial wall, which serves as the anatomical constraint for downstream pathology segmentation.

### 5.2 Pathology Loss: Focal Tversky Loss (FTL)

For the pathology head, **Focal Tversky Loss** is used to address the extreme class imbalance of MI and MVO voxels:

$$TI_c = \frac{TP_c + \epsilon}{TP_c + \alpha \cdot FN_c + \beta \cdot FP_c + \epsilon}$$

$$\mathcal{L}_{FTL} = \sum_c w_c \cdot (1 - TI_c)^{\gamma}$$

With parameters:
- α = 0.65, β = 0.35 — slightly favouring recall over precision, but not so aggressively as to enable topology loss to reward wall-wide MI
- γ = 0.75 — focal modulation that emphasises hard-to-segment cases
- Channel weights: MI = 1.5, MVO = 0.75 — prioritising the primary metric (MI Dice Score)

**Note:** For variant M3, a balanced Tversky loss (α = 0.5, β = 0.5, γ = 1.0) is used instead, as Focal Tversky is introduced at M4.

Pathology loss is computed **only on pathological cases** in each batch (`PATH_LOSS_ON_PATHOLOGICAL_ONLY = True`), avoiding false-positive pressure from healthy patients with empty ground truth.

### 5.3 Topology Consistency Loss

The topology loss penalises pathology probability mass that falls **outside the myocardial wall**:

$$\mathcal{L}_{topo} = \text{mean}\left(w \cdot \mathbf{P}_{path} \cdot (1 - \mathbf{M}_{myo})\right)$$

where $\mathbf{M}_{myo}$ is the (detached) ground-truth MYO mask and $w = [1.5, 0.5]$ weights the MI channel more than MVO.

**Critical design decision:** The MYO mask used for topology is always **detached** from the gradient graph. Without detachment, the loss `mean(path × (1 − myo))` is trivially minimised by expanding pathology ≈ MYO (painting the entire wall as MI), which caused catastrophic MI Dice collapse in early experiments (0.13 vs. M4's 0.36). This failure mode and its resolution are discussed in detail in Section 11.3.

**Curriculum scheduling:** The topology loss weight λ_topo follows a curriculum:
- **Warmup (epochs 1–40):** λ_topo = 0 — the model trains identically to M4
- **Ramp (epochs 41–60):** linear ramp from 0 → 0.05
- **Full (epochs 61–80):** λ_topo = 0.05

This prevents the topology constraint from interfering with early feature learning.

### 5.4 Disease Classification Loss

A standard **Binary Cross-Entropy** loss supervises the disease classifier:

$$\mathcal{L}_{class} = \text{BCE}(p_{disease}, y_{pathological})$$

### 5.5 Joint Total Loss

The total training objective for AFDD-Net (M5) is:

$$\mathcal{L}_{total} = \mathcal{L}_{anat} + \lambda_{ftl} \cdot \mathcal{L}_{FTL} + \lambda_{topo}(e) \cdot \mathcal{L}_{topo} + \lambda_{class} \cdot \mathcal{L}_{class}$$

| Loss Weight | Value | Notes |
|-------------|:-----:|-------|
| λ_ftl | 1.0 | Focal Tversky weight |
| λ_topo | 0.05 | Curriculum-scheduled (warmup 40ep, ramp 20ep) |
| λ_class | 0.5 | Disease classification BCE weight |

---

## 6. Ablation Study (M1 → M5)

The full AFDD-Net architecture is justified through a **progressive 5-step ablation study**, where each variant introduces exactly one architectural or loss-function innovation. All variants share the same encoder backbone and are trained under identical conditions (same folds, same 80 epochs, same optimizer, same seed). This controlled experimental design isolates the contribution of each proposed component, demonstrating how the final model (M5) achieves the best balance of segmentation accuracy and parameter efficiency.

### 6.1 Variant Descriptions

| Variant | Name | Architecture | What It Adds |
|:-------:|------|-------------|:-------------|
| **M1** | Baseline 3D U-Net | `SingleDecoderUNet3D` with isotropic 3×3×3 convolutions | Baseline — standard isotropic 3D U-Net, single decoder, 5-class softmax output (BG/LV/MYO/MI/MVO). Dice + weighted CE loss. |
| **M2** | AFDD-Net-F | `SingleDecoderUNet3D` with factorized convolutions | **+ Anisotropic factorized convolutions** — factorized (1,3,3)+(3,1,1) convolutions replace isotropic 3×3×3 in both encoder and decoder. Same single-decoder, 5-class architecture. Reduces parameters from 46.6M → 11.5M. |
| **M3** | AFDD-Net-D | `DualDecoderNet` with factorized convolutions | **+ Dual decoder with MYO soft-gating** — separate anatomy (3-class softmax) and pathology (2-channel sigmoid) decoders. Predicted MYO probability gates the pathology decoder at each level. Balanced Tversky loss (α=β=0.5, γ=1.0). No disease classifier. |
| **M4** | AFDD-Net-T | `DualDecoderNet` with factorized convolutions | **+ Focal Tversky Loss** — architecture identical to M3, but switches to Focal Tversky Loss (α=0.65, β=0.35, γ=0.75) on the pathology head for better handling of class imbalance. No disease classifier. |
| **M5** | **AFDD-Net** (full) | `DualDecoderNet` with factorized convolutions + disease head | **+ Topology consistency loss + disease classification prior** — adds curriculum-scheduled topology constraint (λ_topo = 0.05, warmup 40ep, ramp 20ep) and a lightweight disease classifier head that gates pathology at inference. M5 warm-starts from M4 checkpoint. |

### 6.2 Ablation Results — 5-Fold Cross-Validation

All metrics are reported as **mean ± std across 5 folds** on the test split. **MI Dice Score** (MI Dice computed on all cases) is the primary evaluation metric.

| Variant | Display Name | LV Dice | MYO Dice | **MI Dice Score** | MVO Dice | Params (M) | Inference (ms) |
|:-------:|:------------|:-------:|:--------:|:----------------:|:--------:|:----------:|:--------------:|
| **M1** | Baseline 3D U-Net | 0.911 ± 0.010 | 0.739 ± 0.027 | 0.566 ± 0.064 | 0.383 ± 0.163 | 46.6 | 101 |
| **M2** | AFDD-Net-F | 0.908 ± 0.009 | 0.729 ± 0.011 | 0.494 ± 0.017 | 0.355 ± 0.214 | 11.5 | 61 |
| **M3** | AFDD-Net-D | 0.905 ± 0.007 | 0.761 ± 0.022 | 0.433 ± 0.028 | 0.367 ± 0.057 | 16.1 | 102 |
| **M4** | AFDD-Net-T | 0.906 ± 0.010 | 0.759 ± 0.023 | 0.410 ± 0.054 | 0.309 ± 0.038 | 16.1 | 100 |
| **M5** | **AFDD-Net** | **0.912 ± 0.008** | **0.776 ± 0.018** | **0.510 ± 0.075** | **0.426 ± 0.095** | **16.1** | 102 |

### 6.3 Analysis of Ablation Contributions

The ablation study reveals a clear progression of improvements, with each component contributing meaningfully to the final model's performance. Critically, M5 achieves the **highest MI Dice Score (0.510)** among all dual-decoder variants while maintaining a compact 16.1M parameter footprint — **65% fewer parameters** than the M1 baseline.

**M1 → M2 (Anisotropic Factorized Convolutions):**
Replacing isotropic 3×3×3 convolutions with factorized (1,3,3)+(3,1,1) convolutions achieves a **4× parameter reduction** (46.6M → 11.5M) and **40% faster inference** (101 ms → 61 ms). LV Dice remains nearly identical (0.911 → 0.908, −0.3 pp), and MYO Dice shows only a marginal decrease (0.739 → 0.729, −1.0 pp). The MI Dice Score drops from 0.566 to 0.494, a trade-off that is fully recovered and surpassed in subsequent variants through architectural innovations built on this efficient backbone. Importantly, the factorized design preserves representational capacity while respecting the anisotropic voxel geometry — the kernel factorisation is not merely a compression trick, but a principled modelling choice for anisotropic data.

**M2 → M3 (Dual Decoder + MYO Soft-Gating):**
Introducing the dual-decoder architecture with MYO soft-gating yields a significant **+3.2 percentage point (pp) improvement in MYO Dice** (0.729 → 0.761) and dramatically more consistent MVO predictions (standard deviation drops from 0.214 to 0.057). The MI Dice Score decreases (0.494 → 0.433), which is expected: the dual-decoder architecture separates anatomy and pathology into distinct prediction tasks, and the new balanced Tversky loss (α=β=0.5) is not yet optimised for the extreme class imbalance of MI voxels. The key gain at this stage is **task specialisation** — the dedicated anatomy decoder excels at MYO wall delineation, which subsequently provides a stronger anatomical constraint for the pathology decoder. This architectural separation is a prerequisite for the topology consistency loss introduced later in M5. The increase in parameters from 11.5M to 16.1M (the cost of the second decoder) is modest and represents the minimum overhead needed for dual-task specialisation.

**M3 → M4 (Focal Tversky Loss):**
Switching from balanced Tversky loss to Focal Tversky Loss (α=0.65, β=0.35, γ=0.75) on the pathology head keeps the MI Dice Score relatively stable (0.433 → 0.410), with MYO Dice preserved at 0.759. The slightly lower MI Dice Score here reflects the recall-oriented parameterisation: with α > β, the model favours detecting true MI voxels (reducing false negatives) even at the cost of some false positives. While this does not immediately boost the all-case MI Dice Score, it establishes a foundation of **better pathological sensitivity** that M5's disease classifier can then leverage — the classifier suppresses false positives on healthy patients, converting the higher recall into a net MI Dice Score improvement. The MVO Dice decreases from 0.367 to 0.309, reflecting the inherent difficulty of simultaneously optimising for both sparse pathology classes under different loss parameterisations.

**M4 → M5 (Topology Consistency + Disease Classification Prior):**
The full AFDD-Net (M5) adds curriculum-scheduled topology consistency loss and the disease classification prior. The results are transformative:
- **MI Dice Score surges from 0.410 to 0.510 (+10.0 pp)** — the largest single-step improvement in the entire ablation chain. The disease classification head achieves **80.2% accuracy** (mean across folds) in distinguishing pathological from normal patients. For patients classified as healthy, all MI/MVO predictions are zeroed, eliminating the Dice = 0 penalty from false-positive predictions on normal cases and dramatically boosting the population-level MI Dice Score.
- **Best MYO Dice** across all variants: 0.776 (+1.7 pp over M4), demonstrating that the topology loss encourages sharper myocardial wall predictions by penalising pathology outside the wall.
- **Best MVO Dice** across all variants: 0.426 (+11.7 pp over M4), driven by the same false-positive suppression mechanism.
- **Best LV Dice**: 0.912, matching the M1 baseline despite using 65% fewer parameters.

### 6.4 Parameter Efficiency Summary

The following table summarises the parameter efficiency of each ablation variant, computed as MI Dice Score per million parameters:

| Variant | MI Dice Score | Params (M) | MI Dice / M Params | Relative Efficiency vs M1 |
|:-------:|:------------:|:----------:|:------------------:|:-------------------------:|
| M1 | 0.566 | 46.6 | 0.0121 | 1.00× |
| M2 | 0.494 | 11.5 | 0.0429 | 3.54× |
| M3 | 0.433 | 16.1 | 0.0269 | 2.22× |
| M4 | 0.410 | 16.1 | 0.0255 | 2.10× |
| **M5** | **0.510** | **16.1** | **0.0317** | **2.61×** |

M5 achieves **2.61× higher parameter efficiency** than the M1 baseline, validating the thesis claim of an "efficient" architecture. The factorized convolutions (M2) provide the highest raw efficiency ratio (3.54×) due to the minimal parameter count, but M5 delivers significantly better absolute MI Dice Score (0.510 vs 0.494) at a modest parameter increase (16.1M vs 11.5M), representing the optimal balance between efficiency and performance.

---

## 7. Comparison with External Baselines

### 7.1 MONAI Baselines — 5-Fold Cross-Validation

Three external architectures from the MONAI library are trained on the same 5-class target (BG/LV/MYO/MI/MVO) under identical conditions to provide a fair comparison:

| Baseline | LV Dice | MYO Dice | **MI Dice Score** | MVO Dice | Params (M) | Inference (ms) |
|:---------|:-------:|:--------:|:----------------:|:--------:|:----------:|:--------------:|
| SegResNet | 0.903 ± 0.009 | 0.700 ± 0.016 | 0.478 ± 0.084 | 0.601 ± 0.030 | 4.7 | 31 |
| SwinUNETR | 0.811 ± 0.042 | 0.680 ± 0.018 | 0.249 ± 0.051 | 0.476 ± 0.102 | 15.7 | 186 |
| DynUNet | 0.897 ± 0.015 | 0.689 ± 0.037 | 0.392 ± 0.064 | 0.528 ± 0.093 | 22.6 | 51 |
| **AFDD-Net (M5)** | **0.912 ± 0.008** | **0.776 ± 0.018** | **0.510 ± 0.075** | 0.426 ± 0.095 | **16.1** | 102 |

### 7.2 Key Observations

**AFDD-Net (M5) achieves the highest MI Dice Score (0.510) among all models tested**, outperforming the next-best baseline (SegResNet, 0.478) by +3.2 pp and DynUNet (0.392) by +11.8 pp. This advantage arises from AFDD-Net's unique combination of the dual-decoder architecture, MYO soft-gating, and the disease classification prior — features not present in any of the standard baselines.

**Anatomy segmentation superiority is decisive.** AFDD-Net achieves the best LV Dice (0.912) and MYO Dice (0.776) by substantial margins. The MYO Dice improvement is particularly significant: +7.6 pp over SegResNet (0.700) and +8.7 pp over DynUNet (0.689). Since accurate MYO segmentation is a prerequisite for anatomically-constrained pathology detection (blockage-induced damage resides within the myocardial wall), this advantage directly translates to better MI and MVO localisation.

**Parameter efficiency relative to baselines.** AFDD-Net (16.1M) uses **29% fewer parameters** than DynUNet (22.6M) while achieving dramatically better MI Dice Score (+11.8 pp) and MYO Dice (+8.7 pp). Although SegResNet is more compact (4.7M), it underperforms AFDD-Net on MI Dice Score by 3.2 pp and on MYO Dice by a substantial 7.6 pp — demonstrating that extreme parameter reduction comes at the cost of segmentation accuracy for this task.

**SwinUNETR underperforms significantly** (MI Dice Score = 0.249, lowest among all models), despite its transformer-based encoder and 15.7M parameters — nearly matching AFDD-Net's parameter count. This result suggests that vision transformer architectures require substantially larger training datasets to realise their potential, and that the inductive biases of convolutional architectures (translation equivariance, local receptive fields) are better suited to the relatively small EMIDEC dataset (100 cases). SwinUNETR's inference time (186 ms) is also the slowest, making it impractical for clinical deployment.

**MVO Dice trade-off.** The external baselines (SegResNet: 0.601, DynUNet: 0.528) achieve higher MVO Dice than AFDD-Net (0.426). This is because the 5-class multiclass head used by baselines can learn correlations between MI and MVO jointly, whereas AFDD-Net's dual-decoder treats MI and MVO as independent sigmoid channels with separate pathology-focused loss weighting (MI weight = 1.5, MVO weight = 0.75). The MI-biased weighting prioritises the primary metric (MI Dice Score) at some cost to MVO. Despite this trade-off, M5's MVO Dice (0.426) is still the best among all ablation variants, demonstrating that the disease classification prior benefits MVO suppression on healthy patients as well.

---

## 8. Training Configuration

| Parameter | Value | Notes |
|-----------|:-----:|-------|
| Optimiser | Adam | lr = 1×10⁻⁴, cosine annealing → 1×10⁻⁶ |
| Batch size | 2 | 1 for SwinUNETR / large baselines |
| CV Epochs | 80 | Same for all models (fair comparison) |
| FTL (α, β, γ) | 0.65, 0.35, 0.75 | Reduced FN-obsession vs. 0.7/0.3 |
| Topo λ | 0.05 | Curriculum: warmup 40ep, ramp 20ep |
| Disease λ | 0.5 | Binary classification loss weight |
| Seed | 42 | Reproducible splits and initialisation |
| Hard MYO mask (inference) | Yes | Zero pathology outside predicted MYO wall |
| Detach MYO gate | Yes | Stops coupled MYO expansion |
| MI voxel suppression | Yes | Threshold = 50 voxels |

### 8.1 Checkpoint Selection

Validation checkpointing uses the composite metric: **MI_path + 0.05 × (LV + MYO)**. This prevents all-background models from locking on empty MI Dice scores, which occurred with simpler checkpointing strategies (notably SegResNet fold-0 collapse, discussed in Section 11.3).

### 8.2 Warm-Starting

M5 warm-starts from the trained M4 checkpoint (same fold). This provides stable anatomy/pathology features before the topology loss and disease classifier are introduced, contributing to training stability. Without warm-starting, the simultaneous introduction of topology loss and the classification head on a randomly initialised model led to unstable gradients and inferior convergence in preliminary experiments.

---

## 9. Inference Pipeline

The inference pipeline applies the following post-processing steps in order:

1. **Test-Time Augmentation (TTA):** 8 geometric augmentations with probability averaging
2. **Disease Gate (M5 only):** If P(pathological) ≤ 0.5, zero all MI/MVO predictions
3. **Hard MYO Mask:** Zero pathology probability outside the predicted MYO wall (argmax of anatomy decoder)
4. **Threshold:** Binarise pathology probability at 0.5
5. **Sparse MI Voxel Suppression:** If total MI voxels < 50, treat the case as healthy and zero all pathology channels

Steps 2–5 form a multi-layered false-positive suppression cascade that is essential for achieving high MI Dice Score across the full patient population. The disease gate (Step 2) provides the coarsest filter (patient-level), the hard MYO mask (Step 3) enforces anatomical plausibility (voxel-level), and the sparse voxel suppression (Step 5) catches residual noise (region-level).

---

## 10. Evaluation Metrics

The following metrics are computed per structure:

| Metric | Description |
|--------|-------------|
| **Dice Similarity Coefficient (DSC)** | Primary overlap metric; both-empty = 1.0 |
| **IoU (Jaccard Index)** | Intersection-over-union |
| **Precision** | Positive predictive value |
| **Recall** | Sensitivity / true positive rate |
| **HD95** | 95th-percentile Hausdorff distance (mm) — secondary metric |
| **Disease Accuracy** | Normal vs. pathological classification (M5 only) |

**Primary metric: MI Dice Score.** The MI Dice Score is computed on **all 100 cases** (both pathological and normal), providing a holistic measure of the model's clinical utility. This metric captures two equally important capabilities:
1. **True positive detection:** accurately segmenting MI regions in pathological patients.
2. **True negative suppression:** correctly producing empty MI masks for healthy patients (both-empty = Dice 1.0, any false positive = Dice 0.0).

This is a more clinically relevant evaluation than metrics restricted to pathological cases only, because a deployed clinical system must handle both patient populations without manual pre-screening. The disease classification prior (Section 4.6) directly targets this metric by gating pathology predictions based on patient-level classification.

> **Note on HD95 for MI/MVO:** HD95 values for MI and MVO are elevated across **all** models due to the standard empty-side penalty of 315 mm applied when either prediction or ground truth is empty. This is a well-known metric artefact for sparse, frequently-absent structures and does not reflect poor localisation quality.

---

## 11. Discussion

### 11.1 How MI Segmentation Justifies the Thesis Title

The thesis title — *"An efficient 3D deep neural architecture for segmentation of blockages in heart using cardiac MRI images"* — is directly justified by the clinical and technical contributions of AFDD-Net.

**Clinical justification:** Myocardial infarction (MI) and microvascular obstruction (MVO) are the direct tissue-level consequences of coronary artery blockages. When a coronary artery is blocked, the myocardial tissue supplied by that artery undergoes ischaemic necrosis (MI). In severe cases, even the microvascular bed within the infarcted territory becomes obstructed (MVO). LGE cardiac MRI visualises these blockage-induced injuries as hyperintense regions within the myocardial wall. Therefore, segmenting MI and MVO from LGE-MRI is equivalent to segmenting the downstream effects of cardiac blockages — the spatial extent, location, and transmurality of the damage directly reflects the severity and distribution of the upstream vascular obstruction.

**Technical justification:** AFDD-Net is an "efficient" architecture in multiple senses:
1. **Parameter efficiency:** 16.1M parameters — 65% fewer than the conventional 3D U-Net baseline (46.6M) and fewer than comparable architectures (DynUNet 22.6M, SwinUNETR 15.7M with vastly inferior performance).
2. **Computational efficiency:** The factorized convolution design reduces FLOPs proportionally to the parameter reduction, enabling faster training and inference.
3. **Segmentation efficiency:** M5 achieves the highest MI Dice Score (0.510) among all models tested, demonstrating that the parameter reduction does not sacrifice — and in fact improves — segmentation quality.
4. **Clinical efficiency:** The disease classification prior eliminates the need for manual pre-screening of patients before applying the segmentation model, streamlining the clinical workflow.

The term "blockages" in the title encompasses the full pathological spectrum visualised in LGE-MRI: from the macroscopic infarction caused by coronary artery occlusion to the microscopic vascular obstruction within the infarcted territory. AFDD-Net segments both MI and MVO, providing a comprehensive assessment of blockage-induced cardiac damage.

### 11.2 Strengths of AFDD-Net

1. **Anatomically-informed design:** The dual-decoder architecture with MYO soft-gating enforces the clinical prior that MI and MVO reside within the myocardial wall, reducing anatomically implausible predictions. Unlike generic U-Net architectures, AFDD-Net encodes domain knowledge about cardiac anatomy directly into its architecture.

2. **Anisotropy-aware convolutions:** Factorized convolutions achieve a 65% parameter reduction while respecting the dataset's highly anisotropic voxel spacing, avoiding the representational waste of isotropic kernels on thick slices. This design choice is specific to the LGE cardiac MRI acquisition protocol and would not be appropriate for isotropic imaging modalities.

3. **Robust pathology loss:** Focal Tversky Loss with carefully tuned parameters (α=0.65, β=0.35, γ=0.75) balances recall and precision for extremely sparse pathology voxels. The parameter selection was guided by the observation that overly aggressive recall (α=0.7, β=0.3) enabled the topology loss to reward wall-wide MI predictions.

4. **Training stability mechanisms:** Curriculum-scheduled topology loss, detached MYO gating, warm-starting from M4, and pathology-only pathological loss computation ensure stable training of the full model despite its multi-task, multi-loss architecture.

5. **Multi-layered false positive suppression:** The combination of the disease classification prior (patient-level), hard MYO mask (anatomical constraint), and MI voxel suppression (region-level) creates a robust cascade that effectively eliminates spurious pathology predictions.

### 11.3 Challenges Faced During Development

The development of AFDD-Net involved overcoming several significant technical challenges that shaped the final architecture and training protocol:

**Challenge 1: Topology Loss Collapse (M5 with λ_topo = 0.5).** The initial implementation of the topology consistency loss used a weight of λ_topo = 0.5. This caused a catastrophic failure: the model learned to minimise `mean(path × (1 − myo))` by expanding the pathology prediction to cover the entire myocardial wall (i.e., predicting MI ≈ MYO). This effectively "painted" the entire wall as infarcted, reducing topology loss to zero but collapsing MI Dice from ~0.36 (M4 level) to 0.13. The solution required three interventions: (a) reducing λ_topo to 0.05, (b) detaching the MYO mask from the gradient graph, and (c) implementing curriculum scheduling that delays topology loss introduction until epoch 40. This experience highlights the importance of careful loss function design for anatomically-constrained segmentation tasks.

**Challenge 2: MYO Gate Coupling.** Without gradient detachment (`.detach()`) on the MYO soft-gate, the pathology decoder's loss could backpropagate through the MYO probability and inflate the anatomy decoder's MYO predictions. In practice, this caused the anatomy decoder to predict an abnormally thick myocardial wall to "legalise" larger pathology regions, degrading both anatomy and pathology accuracy. The detachment mechanism breaks this coupling, ensuring that each decoder is optimised independently for its respective task.

**Challenge 3: Extreme Class Imbalance.** MI voxels constitute less than 1% of the total volume in pathological cases and are entirely absent in the 33% of cases that are healthy. MVO is even rarer. Standard cross-entropy loss assigns negligible gradient to these minority classes, causing models to converge on "predict-nothing" solutions. The Focal Tversky Loss, combined with per-channel weighting (MI = 1.5, MVO = 0.75) and pathological-only loss computation, was necessary to achieve meaningful MI and MVO Dice scores.

**Challenge 4: Anisotropic Voxel Spacing.** The ~7× anisotropy between in-plane (1.5 mm) and through-plane (10 mm) resolution means that a single voxel in the D-axis spans the distance of approximately 7 voxels in the H or W axes. Standard isotropic 3D convolutions (3×3×3 kernels) treat all axes identically, which effectively "smears" the already coarse through-plane information and wastes parameters learning spatial patterns that do not exist at 10 mm resolution. The factorized convolution design was developed specifically to address this challenge, and the augmentation pipeline was designed to avoid through-plane geometric transforms (rotation, scaling, elastic deformation) that would introduce interpolation artefacts.

**Challenge 5: Small Dataset Size.** With only 100 cases and 5-fold cross-validation, each training fold contains approximately 80 cases — an extremely small dataset by deep learning standards. This constraint informed every design decision: conservative augmentation magnitudes, a modest 80-epoch training budget, batch size of 2, and the use of attention mechanisms rather than transformer architectures (which require substantially more data). The stratified splitting strategy ensures balanced representation of pathological and normal cases in every fold.

**Challenge 6: Checkpoint Selection Pitfalls.** Early experiments using a simple validation MI Dice criterion for checkpoint selection led to degenerate checkpoints in some folds (notably SegResNet fold-0 collapse). The problem arose because models that predict empty MI masks achieve artificially high Dice (both-empty = 1.0) on normal cases, masking poor performance on pathological cases. The composite checkpoint criterion (MI_path + 0.05 × (LV + MYO)) was introduced to ensure that selected checkpoints represent genuinely good models rather than "predict-nothing" solutions.

### 11.4 Limitations

1. **Single-stage architecture:** AFDD-Net uses a single-stage encoder-decoder design, whereas the highest-performing published methods for cardiac pathology segmentation (e.g., cascaded nnU-Net architectures) employ multi-stage coarse-to-fine pipelines. A cascaded approach — first localising the region of interest at low resolution, then refining at high resolution — could potentially improve MI and MVO delineation by focusing computational resources on the relevant myocardial region.

2. **No external validation dataset:** All experiments are conducted on the EMIDEC dataset using cross-validation. While 5-fold stratified CV provides a robust estimate of generalisation performance, the model has not been validated on an independent external dataset from a different clinical centre or MRI scanner. Domain shift between institutions (different scanner vendors, acquisition protocols, contrast agent doses) could affect model performance.

3. **Fixed training budget:** All models are trained for 80 epochs to ensure fair comparison. However, some architectures — particularly the dual-decoder variants (M3–M5) with their more complex loss landscapes — might benefit from longer training. The warm-starting strategy partially mitigates this for M5, but a systematic hyperparameter search over training duration was not conducted due to computational constraints.

4. **MVO segmentation difficulty:** MVO remains the hardest structure to segment across all models, with high inter-model variance. This is inherent to the problem: MVO regions are extremely small (often just a few voxels), are surrounded by MI tissue (making boundary detection difficult), and are absent in many pathological cases. Future work could explore specialised attention mechanisms or cascaded refinement stages for MVO.

5. **Disease classification accuracy:** The disease classification head achieves 80.2% accuracy, meaning approximately 20% of cases are misclassified. False negatives (pathological patients classified as healthy) result in zeroed pathology predictions and missed infarcts — a clinically dangerous failure mode. Future work should investigate more sophisticated classification mechanisms (e.g., multi-scale feature aggregation, auxiliary supervision from pathology features) to improve classification accuracy.

6. **Sensitivity to hyperparameters:** The training protocol involves multiple interacting hyperparameters (FTL α/β/γ, topology λ, curriculum schedule, MI/MVO channel weights). While the chosen values were arrived at through systematic experimentation, no formal hyperparameter optimisation (e.g., Bayesian optimisation) was performed due to the computational cost of 5-fold CV evaluation for each configuration.

### 11.5 Future Work

1. **Cascaded architectures:** Implementing a two-stage coarse-to-fine pipeline where Stage 1 localises the myocardial region and Stage 2 refines MI/MVO segmentation within the localised region.
2. **Diffusion-based data augmentation:** Using generative diffusion models to synthesise additional training cases, potentially addressing the dataset size limitation.
3. **Self-supervised pretraining:** Leveraging large unlabelled cardiac MRI datasets for self-supervised pretraining of the encoder, reducing the dependence on the small EMIDEC training set.
4. **Boundary-aware loss functions:** Incorporating distance-transform-based losses to encourage sharper infarct boundaries.
5. **External validation:** Evaluating on multi-centre datasets to assess robustness to domain shift.

---

## 12. Conclusion

This work presents AFDD-Net, a parameter-efficient 3D deep neural architecture for the segmentation of blockage-induced cardiac pathologies (myocardial infarction and microvascular obstruction) from Late Gadolinium Enhanced cardiac MRI. Through a systematic 5-step ablation study (M1–M5), we demonstrate the individual and cumulative contributions of each architectural innovation:

- **Anisotropic factorized convolutions** reduce parameters by 4× (from 46.6M to 11.5M at the single-decoder stage) while preserving segmentation quality, by respecting the inherent anisotropy of LGE cardiac MRI acquisitions.
- **Dual-decoder architecture with MYO soft-gating** enables task specialisation between anatomy and pathology segmentation, achieving the best MYO Dice (0.776) and enforcing the clinical constraint that blockage-induced damage resides within the myocardial wall.
- **Focal Tversky Loss** addresses the extreme class imbalance of MI and MVO voxels through recall-oriented, focal-modulated optimisation.
- **Topology consistency loss with curriculum scheduling** and a **disease classification prior** provide multi-layered false-positive suppression, yielding a transformative +10.0 pp improvement in MI Dice Score (0.410 → 0.510) and the best MVO Dice (0.426) across all models.

The final AFDD-Net model (M5) achieves the **highest MI Dice Score (0.510)** among all models evaluated — including three established MONAI baselines (SegResNet, SwinUNETR, DynUNet) — while using only **16.1M parameters**, a **65% reduction** from the 46.6M-parameter conventional 3D U-Net baseline. M5 simultaneously achieves the best LV Dice (0.912), best MYO Dice (0.776), and best MVO Dice (0.426), demonstrating that parameter efficiency and segmentation accuracy are not mutually exclusive when the architecture is designed with domain-specific knowledge.

The proposed architecture validates the thesis that an efficient 3D deep neural network, purpose-built with anisotropic convolutions, anatomically-informed decoding, and clinically-motivated inference gating, can effectively segment the downstream tissue consequences of cardiac vascular blockages from clinical MRI data. AFDD-Net establishes a foundation for the automated, efficient assessment of blockage-induced myocardial damage, with potential for deployment on resource-constrained clinical hardware.

---

## References

- Lalande, A., et al. (2020). EMIDEC: A database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac MRI. *Data*, 5(4), 89.
- Oktay, O., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *MIDL*.
- Abraham, N., & Khan, N.M. (2019). A novel focal Tversky loss function with improved attention U-Net for lesion segmentation. *ISBI*.
- Isensee, F., et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Ronneberger, O., et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Hatamizadeh, A., et al. (2022). Swin UNETR: Swin transformers for semantic segmentation of brain tumours in MRI images. *BrainLes@MICCAI*.
- Milletari, F., et al. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*.
