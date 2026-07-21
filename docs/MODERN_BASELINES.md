# Modern 3D baseline setup

All added models use one LGE input channel and five output classes:
BG / LV / MYO / pure MI / MVO. Training uses the same folds, 80 epochs,
loss, and `MI_path` checkpoint metric as the existing PyTorch baselines.

## Reproducible source setup

The setup script clones fixed official revisions into the ignored
`third_party/` directory:

```bash
python scripts/setup_modern_baselines.py mednext uxnet3d --install
python scripts/setup_modern_baselines.py umamba segmamba
```

Then train individual models:

```bash
python -m src.train --variant MEDNEXT --cv
python -m src.train --variant UXNET3D --cv
python -m src.train --variant SWINUNETR_V2 --cv
python -m src.train --variant UMAMBA_ENC --cv
python -m src.train --variant SEGMAMBA --cv
```

`--variant baselines` includes all registered PyTorch baselines, so install
all optional dependencies before using it. Otherwise, run models explicitly.

## U-Mamba and SegMamba

Use Linux or WSL2 with a CUDA-enabled PyTorch environment. Native Windows and
Python 3.12 are not reliable targets for the official `causal-conv1d` /
Mamba CUDA extensions.

U-Mamba:

```bash
python scripts/setup_modern_baselines.py umamba
python -m pip install "causal-conv1d>=1.2.0"
python -m pip install mamba-ssm --no-cache-dir
```

SegMamba uses the customized Mamba sources bundled by its official repository:

```bash
python scripts/setup_modern_baselines.py segmamba
python -m pip install third_party/SegMamba/causal-conv1d
python -m pip install third_party/SegMamba/mamba
```

SegMamba's repository has no explicit software license. Its source is not
copied or redistributed here; this project only supports importing a user's
external checkout. Confirm usage rights before distributing code or results.

## Source revisions

- MedNeXt: `MIC-DKFZ/MedNeXt@0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`
- 3D UX-Net: `MASILab/3DUX-Net@14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d`
- U-Mamba: `bowang-lab/U-Mamba@28459e33ca03769800dd35e23c6e62491d1925b5`
- SegMamba: `ge-xing/SegMamba@cff35970e0c542ad940b5701267f1ac888298b06`

See `THIRD_PARTY_MODELS.md` for attribution and license notes.
