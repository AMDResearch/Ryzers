# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the MolmoAct2 Ryzer image on Strix Halo (gfx1151).

Runs inside the built image. No model weights required. Proves the container
has (1) a working ROCm torch on the iGPU and (2) the MolmoAct2 server deps at
the pinned versions, before we pull the ~22 GB checkpoints. Exits non-zero on
any failure so `ryzers run` / CI catches a broken image early.
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
            "FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.",
            file=sys.stderr,
        )
        return 1

    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    # MolmoAct2 server deps (pins from allenai/molmoact2 pyproject.toml).
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
