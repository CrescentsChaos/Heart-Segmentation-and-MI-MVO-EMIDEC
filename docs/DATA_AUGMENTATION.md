# Data Augmentation Strategy

This document details the data augmentation pipeline used for training the AFDD-Net model on the EMIDEC dataset, implemented in [`src/augmentations.py`](file:///e:/Thesis%20Dataset%203/src/augmentations.py).

## Design Constraints

The augmentation strategy is specifically tailored for the EMIDEC dataset, taking into account the unique properties of cardiac MRI volumes:

1. **High Anisotropy**: The EMIDEC volumes are highly anisotropic, with the in-plane (H, W) resolution being approximately 7x finer than the through-plane (D) resolution (~1.5 x 1.5 x 10 mm).
   - *Continuous geometric operations* (affine rotation, scaling, elastic deformation) are applied **only** in the H, W plane. The exact same 2D transformation is applied identically across all D slices.
   - The D-axis is only ever modified via *flips*, which is a lossless, interpolation-free operation. The D-axis is never rotated or resampled.
2. **Strict Alignment**: All spatial transformations are applied with one shared random draw per sample. This guarantees that the input image and all associated label masks (anatomy, pathology, multiclass) remain perfectly pixel-aligned.
3. **Interpolation Modes**: 
   - Label tensors always use `nearest` mode interpolation. This ensures they remain integer/binary masks and avoids creating fractional "phantom" labels.
   - Image tensors use `bilinear` interpolation for continuous geometric transforms.
4. **Metadata Preservation**: Case-level non-spatial metadata like `name` and `pathological` flags are passed through untouched.

## Augmentation Operations

The pipeline consists of synchronized geometric transformations followed by image-only intensity transformations.

### 1. Geometric Transformations
These operations affect both the image and all label tensors synchronously.

* **Random Flips (50% probability)**: Discrete, lossless flips applied independently to the H, W, and D dimensions.
* **Random 90-Degree Rotations (50% probability)**: Discrete, lossless 90, 180, or 270-degree in-plane (H, W) rotations.
* **Affine Transform (30% probability)**: Continuous in-plane transformations consisting of:
  * Small-angle rotations between -15° and +15°.
  * Isotropic scaling between 0.9x and 1.1x.
* **Elastic Deformation (20% probability)**: Continuous in-plane non-linear deformation field.
  * *Pathology-Aware Boosting*: The deformation strength is biased to be 1.5x stronger near pathology regions (MI/MVO voxels) to introduce more variation where it matters most, and 1.0x elsewhere.

### 2. Intensity Transformations
These operations alter voxel values and are applied **only** to the image tensor, leaving the label masks untouched.

* **Brightness and Contrast (30% probability)**: Randomly scales contrast by a factor of [0.85, 1.15] and adds a brightness offset of [-0.1, 0.1].
* **Gamma Correction (30% probability)**: Randomly applies power-law gamma correction with a gamma value drawn from [0.7, 1.5] to augment varied lighting/contrast conditions.
* **Gaussian Blur (15% probability)**: Applies a spatial Gaussian blur with a random sigma between [0.5, 1.2] to simulate lower resolution or acquisition blurring.
* **Gaussian Noise (30% probability)**: Adds normally distributed noise with a standard deviation drawn from [0.0, 0.05] to simulate scanner noise.

All probability and magnitude parameters are conservatively tuned for the relatively small size of the EMIDEC training dataset (~100 cases) to reduce overfitting without destroying the underlying anatomical structure.
