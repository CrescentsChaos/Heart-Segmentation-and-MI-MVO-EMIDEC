# Available Models

This document lists all the models available in the project, providing their descriptions, architecture details, and where their code is located.

## Ablation Variants (AFDD-Net Series)

These models represent the step-by-step ablation study leading to the final proposed architecture (AFDD-Net).

* **Code Location:** All ablation models are defined in [`src/models/dual_decoder.py`](file:///e:/Thesis%20Dataset%203/src/models/dual_decoder.py).
* **Builder:** The `build_model(variant="M...", ...)` function in `dual_decoder.py` instantiates these based on the variant string.

| Variant | Description | Architecture Highlights |
| :--- | :--- | :--- |
| **M1** | 3D U-Net Baseline | Isotropic 3x3x3 convolutions, single decoder, 5-class output (BG/LV/MYO/MI/MVO). |
| **M2** | Factorized U-Net | Same as M1, but uses factorized convolutions in both encoder and decoder. |
| **M3** | Dual-Decoder (Basic) | Dual decoder architecture (separate Anatomy and Pathology decoders) with MYO soft-gating. Dice + WCE loss is applied to both heads. |
| **M4** | Dual-Decoder + Focal Tversky | Architecture identical to M3, but uses Focal Tversky loss on the pathology head instead of standard Dice/WCE. |
| **M5** (AFDD-Net) | Final Proposed Architecture | Architecture identical to M3/M4, but adds a **Topology Consistency Loss** and a **Disease Classification Prior** (normal vs. pathological classification head). |

## PyTorch Baselines

These are external, 3D segmentation architectures provided by the MONAI library. They are evaluated on a 5-class target (BG / LV / MYO / MI / MVO) for fair comparison.

* **Code Location:** All PyTorch baselines are defined in [`src/models/baselines.py`](file:///e:/Thesis%20Dataset%203/src/models/baselines.py).
* **Builder:** The `build_baseline(variant="...", ...)` function handles their instantiation.

| Variant | Description | Implementation Source |
| :--- | :--- | :--- |
| **UNET** | MONAI 3D UNet | Provided by `monai.networks.nets.UNet`. Standard, non-residual U-Net. |
| **SEGRESNET** | MONAI SegResNet | Provided by `monai.networks.nets.SegResNet`. Uses residual blocks. |
| **SWINUNETR** | MONAI SwinUNETR | Provided by `monai.networks.nets.SwinUNETR`. Transformer-based encoder with a CNN decoder. Depth is dynamically padded to satisfy divisibility constraints. |
| **DYNUNET** | MONAI DynUNet | Provided by `monai.networks.nets.DynUNet`. A dynamically configured U-Net (non-residual, filters 32..512). |

## Shared Encoders and Decoders

The core components for the AFDD-Net variants (M1-M5) are built in a modular fashion inside [`src/models/dual_decoder.py`](file:///e:/Thesis%20Dataset%203/src/models/dual_decoder.py).

* **`SharedEncoder`**: A four-stage encoder creating a bottleneck at 1/16 in-plane resolution.
* **`AnatomyDecoder`**: A Softmax decoder predicting anatomy (Background, Left Ventricle, Myocardium).
* **`PathologyDecoder`**: A Sigmoid decoder specifically for MI and MVO, optionally utilizing the predicted Myocardium (MYO) mask to "gate" pathology outputs.
