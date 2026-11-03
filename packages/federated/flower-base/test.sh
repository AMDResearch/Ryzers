#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Running tests for flower-base..."

# Verify all three flower CLI entrypoints are installed
for bin in flower-superlink flower-supernode flower-superexec flwr; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "FAIL: $bin not on PATH"
    exit 1
  fi
  echo "Found: $(command -v "$bin")"
done

# Verify flwr Python package and torch+ROCm
python3 - <<'PY'
import flwr
import torch
print(f"flwr version:  {flwr.__version__}")
print(f"torch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch HIP version:         {torch.version.hip}")
PY

echo "Tests passed!"
