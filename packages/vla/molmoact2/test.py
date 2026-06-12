# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment check for the MolmoAct2 image on Strix Halo (gfx1151).

Verifies ROCm torch on the iGPU and the inference deps at pinned versions, with no
model weights. Exits non-zero on any failure.
"""
import sys


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri and "
            "HSA_OVERRIDE_GFX_VERSION=11.5.1.",
            file=sys.stderr,
        )
        return 1

    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    import transformers
    import accelerate
    import huggingface_hub
    import einops          # noqa: F401
    import fastapi         # noqa: F401
    import json_numpy      # noqa: F401
    import safetensors     # noqa: F401
    import sentencepiece   # noqa: F401

    print(f"transformers     : {transformers.__version__}")
    if not transformers.__version__.startswith("4.57"):
        print(f"FAIL: want transformers 4.57.x, got {transformers.__version__}", file=sys.stderr)
        return 1
    print(f"accelerate       : {accelerate.__version__}")
    print(f"huggingface_hub  : {huggingface_hub.__version__}")
    print("deps import ok   : einops, fastapi, json_numpy, safetensors, sentencepiece")

    print("PASS: MolmoAct2 ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
