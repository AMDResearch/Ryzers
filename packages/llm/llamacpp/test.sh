#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

export PATH=/ryzers/llamacpp/build/bin:$PATH
llama-cli -hf MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF  -p "a short story is the follows:" -n 100 -no-cnv