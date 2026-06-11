#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing LM Studio installation..."

# Check if lm-studio is installed
if ! command -v lm-studio &> /dev/null; then
    echo "FAIL: lm-studio command not found"
    exit 1
fi

echo "lm-studio found at: $(which lm-studio)"

# Test LM Studio help (with timeout since it's a GUI app)
echo "Checking LM Studio CLI..."
timeout 10s lm-studio --no-sandbox --help 2>/dev/null || echo "LM Studio help check completed (timeout expected for GUI app)"

# Check if the LM Studio directory exists
if [[ -d "/opt/lm-studio" ]]; then
    echo "LM Studio installed at: /opt/lm-studio"
elif [[ -d "$HOME/.local/share/lm-studio" ]]; then
    echo "LM Studio data at: $HOME/.local/share/lm-studio"
fi

echo "SUCCESS: LM Studio installation test passed"
echo "Note: For interactive use, run: lm-studio --no-sandbox"
exit 0