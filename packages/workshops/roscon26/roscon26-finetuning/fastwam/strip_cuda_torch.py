# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip the CUDA torch stack + numpy pins from FastWAM's pyproject so `pip install -e .`
cannot pull cu128 wheels over the base image's ROCm torch build, nor force a numpy
downgrade that conflicts with the base image's numpy.

Removes any dependency line for torch / torchvision / torchcodec / numpy; the base image's
versions are then held via the PIP_CONSTRAINT pin in the Dockerfile (see the "Pin the base's
torch + numpy" note there). This lets the one FastWAM layer compose on a numpy-1.26 simulator
base and on a numpy-2.x plain ROCm base identically. Everything else (the exact upstream pins)
is preserved so the direct port stays faithful.
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
