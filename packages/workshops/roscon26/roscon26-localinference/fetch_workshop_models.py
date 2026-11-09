#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stage the workshop's model weights into the image.

Run as ``HF_HOME=<cache> python3 -I fetch_workshop_models.py {perception,lemonade}``.

Both groups are opt-out through the ``BAKE_PERCEPTION_WEIGHTS`` and
``BAKE_LEMONADE_MODELS`` build arguments. They are baked by default because a
conference room is the worst possible place to discover that 13 GB of weights
still have to come down a shared wifi link - and because everything here is
ungated, so there is no Hugging Face token to hand out either.

``allow_patterns`` is what keeps this to 13 GB rather than hundreds:

* the GGUF repositories carry twenty-odd quantizations each, and Lemonade's
  built-in aliases resolve to exactly one of them (Q4_K_M) plus the F16
  multimodal projector that makes the vision path work;
* the perception repositories ship each checkpoint twice, once as
  ``model.safetensors`` and once as a ``.pt``/``.bin``. Transformers reads the
  safetensors, so only that copy is fetched.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

# Notebooks 1-3. E2B is the default chat/vision model; E4B is what notebook 3
# hands to CaP-X for code generation, where the extra capacity shows up as
# programs that actually run.
#
# NOT here: user.Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, the notebook 4 mutation
# model. Notebook 4 replays a recorded evolution and calls no language model at
# all, so 18 GB would buy one opt-in path. The notebook prints the
# `lemonade pull` command that fetches and registers it.
LEMONADE_MODELS = {
    "unsloth/gemma-4-E2B-it-GGUF": [
        "config.json",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        "mmproj-F16.gguf",
    ],
    "unsloth/gemma-4-E4B-it-GGUF": [
        "config.json",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        "mmproj-F16.gguf",
    ],
}

_SAM2_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "*.yaml",
    "video_preprocessor_config.json",
]

_OWLV2_FILES = [
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]

# The default profile (large) and the optional *_fast.yaml profile (small), so
# a session can compare grounding quality against latency without a download.
PERCEPTION_MODELS = {
    "facebook/sam2.1-hiera-large": _SAM2_FILES,
    "google/owlv2-large-patch14-ensemble": _OWLV2_FILES,
    "facebook/sam2.1-hiera-small": _SAM2_FILES,
    "google/owlv2-base-patch16-ensemble": _OWLV2_FILES,
}

GROUPS = {"lemonade": LEMONADE_MODELS, "perception": PERCEPTION_MODELS}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in GROUPS:
        raise SystemExit(f"usage: {argv[0]} {{{'|'.join(GROUPS)}}}")

    cache = os.environ.get("HF_HOME")
    if not cache:
        raise SystemExit(
            "HF_HOME must be set, or the weights land in $HOME/.cache/huggingface - "
            "which the composed lemonade-sdk ryzer bind-mounts from the host at run "
            "time, hiding anything baked underneath it."
        )
    print(f"staging {argv[1]} weights into {cache}")

    for repository, files in GROUPS[argv[1]].items():
        print(f"  {repository}")
        snapshot_download(repository, allow_patterns=files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
