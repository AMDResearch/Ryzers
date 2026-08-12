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

missing_libraries="$(ldd /lm-studio | awk '/not found/')"
if [[ -n "$missing_libraries" ]]; then
    echo "FAIL: missing LM Studio libraries"
    echo "$missing_libraries"
    exit 1
fi

echo "Checking LM Studio CLI..."
set +e
timeout 10s xvfb-run -a lm-studio --no-sandbox >/tmp/lmstudio-test.log 2>&1
status=$?
set -e

if [[ $status -ne 0 && $status -ne 124 ]]; then
    cat /tmp/lmstudio-test.log
    echo "FAIL: LM Studio exited with status $status"
    exit "$status"
fi

# Check if the LM Studio directory exists
if [[ -d "/opt/lm-studio" ]]; then
    echo "LM Studio installed at: /opt/lm-studio"
elif [[ -d "$HOME/.local/share/lm-studio" ]]; then
    echo "LM Studio data at: $HOME/.local/share/lm-studio"
fi

echo "SUCCESS: LM Studio installation test passed"
echo "Note: For interactive use, run: lm-studio --no-sandbox"
exit 0