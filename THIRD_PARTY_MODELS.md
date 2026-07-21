# Third-party model provenance

This repository contains integration code only. Official model repositories
are cloned into the ignored `third_party/` directory by
`scripts/setup_modern_baselines.py`.

## MedNeXt

- Source: https://github.com/MIC-DKFZ/MedNeXt
- Revision: `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`
- License: Apache License 2.0
- Configuration: MedNeXt-v1 Small, kernel size 3, no deep supervision

## 3D UX-Net

- Source: https://github.com/MASILab/3DUX-Net
- Revision: `14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d`
- License: MIT
- Configuration: depths `[2,2,2,2]`, features `[48,96,192,384]`

## U-Mamba

- Source: https://github.com/bowang-lab/U-Mamba
- Revision: `28459e33ca03769800dd35e23c6e62491d1925b5`
- License: Apache License 2.0
- Configuration: U-Mamba Enc 3D with EMIDEC-aware anisotropic strides

## SegMamba

- Source: https://github.com/ge-xing/SegMamba
- Revision: `cff35970e0c542ad940b5701267f1ac888298b06`
- License: no explicit software license found at the pinned revision
- Configuration: official four-stage reference network

SegMamba is intentionally not vendored. Public availability of source code is
not equivalent to permission to redistribute or modify it.
