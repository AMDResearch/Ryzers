#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Download or verify the WAN 2.2 TI2V-5B checkpoint for runtime use."""

from __future__ import annotations

import argparse
import pathlib

from huggingface_hub import snapshot_download


REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
)


def _has_required_files(path: pathlib.Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WAN 2.2 TI2V-5B weights from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--ckpt_dir", default="/models/Wan2.2-TI2V-5B")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN.")
    parser.add_argument("--check_only", action="store_true", help="Only verify that required files already exist.")
    args = parser.parse_args()

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    if _has_required_files(ckpt_dir):
        print(f"WAN 2.2 TI2V-5B checkpoint already present: {ckpt_dir}", flush=True)
        return

    if args.check_only:
        missing = [name for name in REQUIRED_FILES if not (ckpt_dir / name).is_file()]
        raise SystemExit(f"checkpoint incomplete at {ckpt_dir}; missing: {', '.join(missing)}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {ckpt_dir}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(ckpt_dir),
        token=args.token,
        resume_download=True,
    )
    if not _has_required_files(ckpt_dir):
        missing = [name for name in REQUIRED_FILES if not (ckpt_dir / name).is_file()]
        raise SystemExit(f"download finished but checkpoint is incomplete; missing: {', '.join(missing)}")
    print(f"WAN 2.2 TI2V-5B checkpoint ready: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
