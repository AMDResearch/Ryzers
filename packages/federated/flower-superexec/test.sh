#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Smoke test: validate that the flower-superexec CLI is available, that
# the bundled quickstart-pytorch app imports cleanly, and that PyTorch
# can see the ROCm GPU.

set -e

echo "Running tests for flower-superexec..."

flower-superexec --help >/dev/null
flower-superexec --help 2>&1 | grep -q -- "--plugin-type" \
  || { echo "FAIL: --plugin-type flag missing from help"; exit 1; }

python3 - <<'PY'
import torch
import flwr
from pytorchexample import server_app, client_app

print(f"flwr  version: {flwr.__version__}")
print(f"torch version: {torch.__version__}  (HIP: {torch.version.hip})")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
assert hasattr(server_app, "app"), "pytorchexample.server_app.app missing"
assert hasattr(client_app, "app"), "pytorchexample.client_app.app missing"
print("quickstart-pytorch ServerApp and ClientApp imported successfully.")
PY

echo "Tests passed!"
