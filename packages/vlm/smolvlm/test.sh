#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


# Fixed paths
export PATH=/ryzers/llamacpp/build/bin/:$PATH
MODEL="ggml-org/SmolVLM-500M-Instruct-GGUF"

# Fixed prompt
PROMPT="How do magnets work?"

# Run
llama-run hf://$MODEL "$PROMPT"
