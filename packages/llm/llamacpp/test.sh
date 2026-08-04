#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

export PATH=/ryzers/llamacpp/build/bin:$PATH
llama-cli \
    -hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M \
    -p "a short story follows:" \
    -n 100 \
    --no-conversation \
    --single-turn \
    --simple-io \
    </dev/null