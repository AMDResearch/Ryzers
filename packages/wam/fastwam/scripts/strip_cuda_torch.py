# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip torch/torchvision/torchcodec/numpy pins from FastWAM's pyproject so `pip install -e .`
keeps the base image's ROCm torch and numpy (held via the Dockerfile PIP_CONSTRAINT pin).
"""
import re
import sys
from pathlib import Path

STRIP = ("torch", "torchvision", "torchcodec", "numpy")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml")
    pat = re.compile(r'^\s*"(' + "|".join(STRIP) + r')\s*[=<>!~]')
    kept, removed = [], []
    for line in path.read_text().splitlines():
        if pat.match(line):
            removed.append(line.strip())
        else:
            kept.append(line)
    path.write_text("\n".join(kept) + "\n")
    print("stripped CUDA torch pins:", removed or "(none found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
