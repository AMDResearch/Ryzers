#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


# Fixed paths
export PATH=/ryzers/llamacpp/build/bin/:$PATH
MODEL="ggml-org/SmolVLM-500M-Instruct-GGUF"

# Fixed prompt
PROMPT="How do magnets work?"

# Run
llama-cli \
    -hf "$MODEL" \
    -p "$PROMPT" \
    -n 64 \
    --no-conversation \
    --single-turn \
    --simple-io \
    </dev/null
