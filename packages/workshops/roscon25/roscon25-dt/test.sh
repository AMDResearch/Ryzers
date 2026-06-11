#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing ROSCon25 DT workshop environment..."

# Check Python
echo "Python version: $(python3 --version)"

# Check JupyterLab
echo "Checking JupyterLab..."
if ! command -v jupyter-lab &> /dev/null; then
    echo "FAIL: jupyter-lab not found"
    exit 1
fi
echo "JupyterLab version: $(jupyter-lab --version)"

# Check key packages
echo "Checking required packages..."
python3 -c "import pandas; print(f'pandas: {pandas.__version__}')"
python3 -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"
python3 -c "import torch; print(f'torch: {torch.__version__}')"

# Check notebooks directory
echo "Checking notebooks..."
if [[ -d "/ryzers/notebooks/Tutorials" ]]; then
    echo "Tutorials found: $(ls /ryzers/notebooks/Tutorials | wc -l) items"
else
    echo "Warning: Tutorials directory not found"
fi

echo "SUCCESS: ROSCon25 DT workshop test passed"
echo "Note: For JupyterLab, run: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
exit 0
