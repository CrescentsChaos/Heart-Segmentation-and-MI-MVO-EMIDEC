# -*- coding: utf-8 -*-
"""One-time helpers after the DynUNet-Res / nnU-Net naming split.

Old checkpoints named NNUNET_fold*_best.pth were MONAI residual DynUNet.
Rename them to DYNUNET_RES_* so they are not confused with real nnU-Net v2.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg


def migrate_dynunet_res(dry_run: bool = True) -> int:
    ckpt = Path(cfg.CHECKPOINT_DIR)
    res = Path(cfg.RESULTS_DIR)
    n = 0
    pairs = []
    for folder in (ckpt, res):
        if not folder.exists():
            continue
        for p in folder.glob("NNUNET*"):
            new_name = p.name.replace("NNUNET", "DYNUNET_RES", 1)
            dest = p.with_name(new_name)
            pairs.append((p, dest))
    for src, dest in pairs:
        print(f"{'[dry] ' if dry_run else ''}{src.name} -> {dest.name}")
        if not dry_run:
            if dest.exists():
                print(f"  skip (exists): {dest}")
                continue
            shutil.move(str(src), str(dest))
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually rename files")
    args = parser.parse_args()
    n = migrate_dynunet_res(dry_run=not args.apply)
    if args.apply:
        print(f"Renamed {n} files.")
    else:
        print("Dry run only. Pass --apply to rename.")


if __name__ == "__main__":
    main()
