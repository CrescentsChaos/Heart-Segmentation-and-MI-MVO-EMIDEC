"""Clone pinned official implementations for optional modern baselines.

Examples:
  python scripts/setup_modern_baselines.py mednext uxnet3d --install
  python scripts/setup_modern_baselines.py all

U-Mamba and SegMamba CUDA extensions should be installed under Linux/WSL2.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"

SOURCES = {
    "mednext": {
        "url": "https://github.com/MIC-DKFZ/MedNeXt.git",
        "dir": "MedNeXt",
        "commit": "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba",
    },
    "uxnet3d": {
        "url": "https://github.com/MASILab/3DUX-Net.git",
        "dir": "3DUX-Net",
        "commit": "14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d",
    },
    "umamba": {
        "url": "https://github.com/bowang-lab/U-Mamba.git",
        "dir": "U-Mamba",
        "commit": "28459e33ca03769800dd35e23c6e62491d1925b5",
    },
    "segmamba": {
        "url": "https://github.com/ge-xing/SegMamba.git",
        "dir": "SegMamba",
        "commit": "cff35970e0c542ad940b5701267f1ac888298b06",
    },
}


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def clone_pinned(key: str) -> Path:
    source = SOURCES[key]
    dest = THIRD_PARTY / source["dir"]
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        run("git", "clone", source["url"], str(dest))
    if not (dest / ".git").exists():
        raise RuntimeError(f"{dest} exists but is not a Git checkout")
    run("git", "-C", str(dest), "fetch", "origin", source["commit"], "--depth", "1")
    run("git", "-C", str(dest), "checkout", "--detach", source["commit"])
    return dest


def install_dependencies(key: str, repo: Path) -> None:
    if key == "mednext":
        run(sys.executable, "-m", "pip", "install", "-e", str(repo))
    elif key == "uxnet3d":
        run(sys.executable, "-m", "pip", "install", "timm")
    elif key == "umamba":
        if sys.platform == "win32":
            raise RuntimeError(
                "Official U-Mamba CUDA kernels are not reliably supported on "
                "native Windows. Use Linux or WSL2; see docs/MODERN_BASELINES.md."
            )
        run(sys.executable, "-m", "pip", "install", "causal-conv1d>=1.2.0")
        run(sys.executable, "-m", "pip", "install", "mamba-ssm", "--no-cache-dir")
    elif key == "segmamba":
        if sys.platform == "win32":
            raise RuntimeError(
                "Official SegMamba CUDA kernels require Linux/WSL2; "
                "see docs/MODERN_BASELINES.md."
            )
        run(sys.executable, "-m", "pip", "install", str(repo / "causal-conv1d"))
        run(sys.executable, "-m", "pip", "install", str(repo / "mamba"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="+",
        choices=["all", *SOURCES],
        help="Official model repositories to prepare",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install each model's Python/CUDA dependencies",
    )
    args = parser.parse_args()
    selected = list(SOURCES) if "all" in args.models else list(dict.fromkeys(args.models))
    for key in selected:
        repo = clone_pinned(key)
        if args.install:
            install_dependencies(key, repo)
        print(f"[ok] {key}: {repo} @ {SOURCES[key]['commit']}")


if __name__ == "__main__":
    main()
