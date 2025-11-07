#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

export PATH=/ryzers/llamacpp/build/bin/:$PATH
llama-server --port 8081 -hf ggml-org/SmolVLM-500M-Instruct-GGUF 
