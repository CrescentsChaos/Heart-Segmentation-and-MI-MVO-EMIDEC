# -*- coding: utf-8 -*-
"""
Data augmentation for AFDD-Net / EMIDEC training.

Design constraints (do not relax without re-checking spacing):
  - EMIDEC volumes are highly anisotropic: ~1.5 x 1.5 x 10 mm (see
    architecture_figure.py). In-plane (H, W) resolution is ~7x finer than
    through-plane (D). Therefore:
      * Continuous geometric ops (affine rotation, scaling, elastic) are
        applied ONLY in the H, W plane, identically across all D slices.
      * D-axis is only ever flipped (a lossless, interpolation-free op),
        never rotated or resampled.
    This mirrors the TTA convention already used in inference.py
    (AUG_NAMES: flip_w/flip_h/flip_d/rot90/rot180/rot270/rot90_flip_w all
    operate on dims [-2, -1] for rotation, [-1]/[-2]/[-3] for flips).
  - All spatial transforms are applied with ONE shared random draw per
    sample so image / anatomy / pathology / multiclass stay pixel-aligned.
  - Label tensors (anatomy, pathology, multiclass) always use
    mode="nearest" interpolation so they stay integer/binary — never
    bilinear, which would create fractional "phantom" label values.
  - "name" and "pathological" (case-level scalar) are passed through
    untouched.

Usage: see integration notes at the bottom of this file / the chat message.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _rand(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _make_inplane_affine_theta(angle_deg: float, scale: float) -> torch.Tensor:
    """2x3 affine matrix for an in-plane (H, W) rotation + isotropic scale."""
    theta_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(theta_rad), math.sin(theta_rad)
    # inverse scale because affine_grid maps output -> input coordinates
    inv_scale = 1.0 / scale
    mat = torch.tensor(
        [
            [cos_a * inv_scale, -sin_a * inv_scale, 0.0],
            [sin_a * inv_scale, cos_a * inv_scale, 0.0],
        ],
        dtype=torch.float32,
    )
    return mat


def _apply_inplane_affine(
    x: torch.Tensor, theta: torch.Tensor, mode: str
) -> torch.Tensor:
    """
    Apply the same in-plane (H, W) affine transform to every D slice and
    every channel of x. x: (C, D, H, W).
    """
    c, d, h, w = x.shape
    # Fold (C, D) into the batch dimension so grid_sample gets one 2D grid
    # reused for every slice/channel via expand (no extra memory copy).
    x_flat = x.reshape(c * d, 1, h, w)
    grid = F.affine_grid(
        theta.unsqueeze(0), size=(1, 1, h, w), align_corners=False
    )
    grid = grid.expand(c * d, h, w, 2)
    padding_mode = "border" if mode == "bilinear" else "zeros"
    out = F.grid_sample(
        x_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )
    return out.reshape(c, d, h, w)


def _apply_inplane_elastic(
    x: torch.Tensor, disp: torch.Tensor, mode: str
) -> torch.Tensor:
    """
    Apply the same in-plane elastic displacement field to every D slice and
    channel. x: (C, D, H, W). disp: (H, W, 2) in normalized [-1, 1] grid units.
    """
    c, d, h, w = x.shape
    x_flat = x.reshape(c * d, 1, h, w)
    base_grid = F.affine_grid(
        torch.eye(2, 3, dtype=torch.float32).unsqueeze(0),
        size=(1, 1, h, w),
        align_corners=False,
    )[0]  # (H, W, 2)
    grid = (base_grid + disp).unsqueeze(0).expand(c * d, h, w, 2)
    padding_mode = "border" if mode == "bilinear" else "zeros"
    out = F.grid_sample(
        x_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )
    return out.reshape(c, d, h, w)


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _smooth_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (2, H, W) displacement field."""
    radius = max(1, int(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius).to(field.dtype)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    field = field.unsqueeze(0)  # (1, 2, H, W)
    field = F.conv2d(field, kx.expand(2, 1, 1, -1), padding=(0, radius), groups=2)
    field = F.conv2d(field, ky.expand(2, 1, -1, 1), padding=(radius, 0), groups=2)
    return field[0]


# --------------------------------------------------------------------------- #
# Augmentor
# --------------------------------------------------------------------------- #

class AFDDAugmentor:
    """
    Synchronized 3D augmentation for the AFDD-Net EMIDEC sample dict.

    All probabilities/magnitudes are conservative defaults tuned for a small
    dataset (EMIDEC ~100 training cases) to reduce overfitting without
    destroying anatomy. Tune via config.py if desired (see integration note).
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rot90: float = 0.5,
        p_affine: float = 0.3,
        rotate_deg: float = 15.0,
        scale_range=(0.9, 1.1),
        p_elastic: float = 0.2,
        elastic_alpha: float = 8.0,   # displacement magnitude, in voxels
        elastic_sigma: float = 6.0,   # smoothness of the field, in voxels
        p_gauss_noise: float = 0.3,
        gauss_noise_std=(0.0, 0.05),
        p_gamma: float = 0.3,
        gamma_range=(0.7, 1.5),
        p_brightness_contrast: float = 0.3,
        brightness_range=(-0.1, 0.1),
        contrast_range=(0.85, 1.15),
        p_gauss_blur: float = 0.15,
        blur_sigma_range=(0.5, 1.2),
    ):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_affine = p_affine
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_gauss_noise = p_gauss_noise
        self.gauss_noise_std = gauss_noise_std
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_brightness_contrast = p_brightness_contrast
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p_gauss_blur = p_gauss_blur
        self.blur_sigma_range = blur_sigma_range

    # ---- geometric (synchronized across image + all label maps) -------- #

    def _geometric(self, image: torch.Tensor, labels: Dict[str, torch.Tensor]):
        # image: (1, D, H, W) float
        # labels: dict of (Ck, D, H, W) tensors (Ck may be 1 for class maps)

        # 1) discrete, lossless: flips (H, W, and D — D-flip is safe, no
        #    interpolation, unlike rotation which would smear thick slices)
        if _rand(0, 1) < self.p_flip:
            dims = [d for d in (-1, -2, -3) if _rand(0, 1) < 0.5]
            if dims:
                image = torch.flip(image, dims=dims)
                labels = {k: torch.flip(v, dims=dims) for k, v in labels.items()}

        # 2) discrete, lossless: 90-degree in-plane rotations
        if _rand(0, 1) < self.p_rot90:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k=k, dims=(-2, -1))
            labels = {k_: torch.rot90(v, k=k, dims=(-2, -1)) for k_, v in labels.items()}

        # 3) continuous, in-plane only: small-angle rotation + scale
        if _rand(0, 1) < self.p_affine:
            angle = _rand(-self.rotate_deg, self.rotate_deg)
            scale = _rand(*self.scale_range)
            theta = _make_inplane_affine_theta(angle, scale)
            image = _apply_inplane_affine(image, theta, mode="bilinear")
            labels = {
                k_: _apply_inplane_affine(v.float(), theta, mode="nearest").to(v.dtype)
                for k_, v in labels.items()
            }

        # 4) continuous, in-plane only: elastic deformation
        if _rand(0, 1) < self.p_elastic:
            h, w = image.shape[-2:]
            raw = (torch.rand(2, h, w) * 2 - 1)  # in [-1, 1]
            field = _smooth_field(raw, sigma=self.elastic_sigma)
            # normalize to grid units (grid is in [-1, 1] over H/W)
            field = field / max(field.abs().max().item(), 1e-6)
            field = field * (self.elastic_alpha / max(h, w))
            disp = field.permute(1, 2, 0)  # (H, W, 2), order (dx, dy) for grid_sample
            image = _apply_inplane_elastic(image, disp, mode="bilinear")
            labels = {
                k_: _apply_inplane_elastic(v.float(), disp, mode="nearest").to(v.dtype)
                for k_, v in labels.items()
            }

        return image, labels

    # ---- intensity (image only) ----------------------------------------- #

    def _intensity(self, image: torch.Tensor) -> torch.Tensor:
        if _rand(0, 1) < self.p_brightness_contrast:
            brightness = _rand(*self.brightness_range)
            contrast = _rand(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean + brightness

        if _rand(0, 1) < self.p_gamma:
            gamma = _rand(*self.gamma_range)
            lo, hi = image.min(), image.max()
            rng = (hi - lo).clamp_min(1e-6)
            norm = ((image - lo) / rng).clamp(0, 1)
            image = norm.pow(gamma) * rng + lo

        if _rand(0, 1) < self.p_gauss_blur:
            sigma = _rand(*self.blur_sigma_range)
            radius = max(1, int(2 * sigma))
            k = _gaussian_kernel1d(sigma, radius)
            c, d, h, w = image.shape
            flat = image.reshape(c * d, 1, h, w)
            flat = F.conv2d(flat, k.view(1, 1, 1, -1).expand(1, 1, 1, -1),
                             padding=(0, radius))
            flat = F.conv2d(flat, k.view(1, 1, -1, 1).expand(1, 1, -1, 1),
                             padding=(radius, 0))
            image = flat.reshape(c, d, h, w)

        if _rand(0, 1) < self.p_gauss_noise:
            std = _rand(*self.gauss_noise_std)
            image = image + torch.randn_like(image) * std

        return image

    # ---- public entry point --------------------------------------------- #

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        if image.dim() == 3:  # (D, H, W) -> (1, D, H, W)
            image = image.unsqueeze(0)

        label_keys = ("anatomy", "pathology", "multiclass")
        labels = {}
        orig_dims = {}
        for k in label_keys:
            if k in sample:
                v = sample[k]
                orig_dims[k] = v.dim()
                labels[k] = v.unsqueeze(0) if v.dim() == 3 else v  # -> (C, D, H, W)

        image, labels = self._geometric(image, labels)
        image = self._intensity(image)

        out = dict(sample)
        out["image"] = image
        for k, v in labels.items():
            out[k] = v.squeeze(0) if orig_dims[k] == 3 else v

        # "name" and "pathological" (case-level, non-spatial) pass through
        # unchanged automatically since we started from dict(sample).
        return out


# A ready-to-use default instance — import this directly if you don't need
# to tune parameters per-experiment.
default_augmentor = AFDDAugmentor()


def augment_sample(sample: Dict) -> Dict:
    """Functional convenience wrapper around `default_augmentor`."""
    return default_augmentor(sample)