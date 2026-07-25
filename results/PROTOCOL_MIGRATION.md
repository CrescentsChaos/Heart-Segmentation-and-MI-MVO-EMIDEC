# Results protocol migration

Results and checkpoints produced before 2026-07-21 are not comparable with
the paper-aligned protocol now implemented in this repository.

In particular, do not reuse or relabel historical `NNUNET_*` artifacts. The
old key referred either to a residual DynUNet or to the removed external
nnU-Net v2 workflow. It now identifies the native anisotropic 3D nnU-Net
architecture trained by `src.train`.

Regenerate training, evaluation, tables, figures, and patient reports. New
result files report:

- `MYO`: labels 2, 3, and 4
- `MI`: labels 3 and 4
- `Pure_MI`: label 3
- `MVO`: all cases, with empty/empty equal to 1
- `MVO_positive`: only cases containing ground-truth MVO
- `foreground_mean`: mean of LV, MYO, MI, and MVO Dice
