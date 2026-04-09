#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "========================================"
echo "  Flower SuperExec (PyTorch Quickstart)"
echo "========================================"
echo ""

# Check PyTorch and GPU availability
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Check Flower installation
echo "Checking Flower installation..."
python3 -c "import flwr; print(f'Flower version: {flwr.__version__}')"
echo ""

# Verify flower-superexec command exists
echo "Checking flower-superexec command..."
which flower-superexec
echo ""

# Verify quickstart app is installed
echo "Checking quickstart app..."
python3 -c "from quickstart.task import Net; print('Quickstart CNN model: OK')"
python3 -c "from quickstart.client_app import app; print('ClientApp: OK')"
python3 -c "from quickstart.server_app import app; print('ServerApp: OK')"
echo ""

echo "SuperExec ready. Usage:"
echo ""
echo "  # Run as ServerApp executor (connects to SuperLink)"
echo "  ryzers run flower-superexec --insecure \\"
echo "    --executor-type serverapp \\"
echo "    --executor-config 'superlink=\"<superlink-host>:9091\"'"
echo ""
echo "  # Run as ClientApp executor (connects to SuperNode)"
echo "  ryzers run flower-superexec --insecure \\"
echo "    --executor-type clientapp \\"
echo "    --executor-config 'supernode=\"<supernode-host>:9094\"'"
