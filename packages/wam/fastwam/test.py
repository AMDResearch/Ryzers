# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Env sign-of-life for the FastWAM policy image on Strix Halo (gfx1151): checks ROCm torch
on the iGPU and that FastWAM + runtime deps import, with no model weights. Non-zero on failure.
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
        print("FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.", file=sys.stderr)
        return 1

    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    # FastWAM package + core runtime deps.
    import fastwam                       # noqa: F401
    from fastwam.runtime import create_fastwam  # noqa: F401
    import transformers
    import hydra                         # noqa: F401
    import omegaconf                     # noqa: F401
    import einops                        # noqa: F401
    import safetensors                   # noqa: F401
    import sentencepiece                 # noqa: F401

    print(f"transformers     : {transformers.__version__}")
    print("deps import ok   : fastwam, hydra, omegaconf, einops, safetensors, sentencepiece")
    print("PASS: FastWAM ROCm policy env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
