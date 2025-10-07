#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

if [[ -n "$RYZERS_TESTMODE" ]]; then
    timeout 10s lm-studio --no-sandbox --help 2>/dev/null || echo "LM Studio help command completed (expected for GUI app)"
    exit 0
fi

lm-studio --no-sandbox 
exit 0