# Vendored FastWAM glue

Everything the `roscon26-finetuning` image needs to run
[FastWAM](https://github.com/yuantianyuan01/FastWAM) — a Wan2.2-TI2V-5B
world-action model (UMT5-XXL text encoder + Wan video VAE + video/action
diffusion transformer) — against the LIBERO simulator on ROCm. Source only, no
binaries, so the Docker build is self-contained from this package directory.

Upstream is cloned at build time and pinned to the commit in
[`docs/UPSTREAM_PIN.commit.txt`](docs/UPSTREAM_PIN.commit.txt). It is a direct
PyTorch port: upstream code runs unmodified on the base image's ROCm torch, and
only its CUDA torch pins are stripped.

| Path | Role |
|------|------|
| `sim_libero/` | Model-agnostic LIBERO harness — env setup, rollout loop, renderer, interactive HTTP server, and the `Policy` seam the adapter implements. Installed to `/opt/sim/sim_libero`. |
| `adapters/fastwam_libero_policy.py` | Implements that `Policy` seam for FastWAM. Selected at run time by `POLICY_FACTORY=fastwam_libero_policy:build_policy`. Installed to `/opt/fastwam-adapters/`. |
| `patches/fastwam_kv_cache.patch` | Exact planning-path caches: memoizes the UMT5-XXL text encode across a replan episode and caches the action cross-attention K/V. Bit-identical output, ~1.45x faster on gfx1151. Default-on; `FASTWAM_TEXT_KV_CACHE=0` restores upstream behaviour for parity checks. |
| `patches/fastwam_public_wan_assets.patch` | Selects Wan's original public T5/VAE files on Hugging Face instead of the converted ModelScope-only redirect. |
| `strip_cuda_torch.py` | Removes FastWAM's `torch`/`torchvision`/`torchcodec`/`numpy` pins (upstream hard-pins `torch==2.7.1+cu128`) before the editable install, so the base image's ROCm torch is never replaced. |
| `sim_libero_compat.pth` | Adds `/opt/sim` to `sys.path` and installs a `torch.load` compatibility shim. Installed into the FastWAM venv's `site-packages` only. |

All of this is confined to `/opt/fastwam-venv`; the MolmoAct2 training
environment in `/opt/train-venv` never sees it.

## References

- Upstream code: <https://github.com/yuantianyuan01/FastWAM>
- Checkpoint: <https://huggingface.co/yuanty/fastwam>
- Replay data: <https://huggingface.co/datasets/yuanty/LIBERO-fastwam>
- Wan2.2-TI2V-5B: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
