#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

export FLOWER_PROJECT

echo "========================================="
echo "  GPU Diagnostic Information"
echo "========================================="
echo ""

echo "=== Host GPU Information ==="
echo "ROCm devices:"
ls -la /dev/kfd /dev/dri/render* 2>/dev/null || echo "GPU devices not found"
echo ""

echo "Host ROCm version (if installed):"
rocminfo 2>/dev/null | grep -A2 "Marketing Name" || echo "rocminfo not available on host"
echo ""

RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

echo "=== Container GPU Information ==="
echo ""

# Create diagnostic in mounted workspace so container can access it
cat > "$FLOWER_PROJECT/gpu_diagnostic.py" << 'EOF'
import subprocess
import os

print("Environment variables:")
for key in ['HSA_OVERRIDE_GFX_VERSION', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES']:
    print(f"  {key} = {os.environ.get(key, 'not set')}")
print("")

print("ROCm Info:")
try:
    result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
    lines = result.stdout.split('\n')
    in_agent = False
    for line in lines:
        if 'Name:' in line and 'gfx' in line:
            print(f"  {line.strip()}")
        if 'Marketing Name' in line:
            in_agent = True
        if in_agent and ('Marketing' in line or 'Name:' in line or 'gfx' in line):
            print(f"  {line.strip()}")
        if in_agent and line.strip() == '':
            in_agent = False
except Exception as e:
    print(f"  Error running rocminfo: {e}")
print("")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

    # Check what PyTorch thinks the architecture is
    props = torch.cuda.get_device_properties(0)
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
print("")

print("Testing different allocation sizes:")
sizes = [10, 100, 1000, 5000]
for size in sizes:
    try:
        x = torch.randn(size, size).cuda()
        print(f"  ✓ {size}x{size} tensor OK ({x.element_size() * x.nelement() / 1024**2:.2f} MB)")
        del x
    except Exception as e:
        print(f"  ✗ {size}x{size} FAILED: {e}")
        break

print("")
print("Testing model creation (without moving to GPU):")
import torch.nn as nn
try:
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        def forward(self, x):
            return self.fc(x)

    model = SimpleNet()
    print(f"  ✓ Model created on CPU")

    print("  Attempting to move to GPU...")
    model = model.cuda()
    print(f"  ✓ Model moved to GPU successfully")

except Exception as e:
    print(f"  ✗ Model creation/movement failed: {e}")
    import traceback
    traceback.print_exc()
EOF

cd "$RYZERS_ROOT"

# Create wrapper script in mounted volume
cat > "$FLOWER_PROJECT/run_diagnostic.sh" << 'WRAPPER'
#!/bin/bash
python3 /app/gpu_diagnostic.py
WRAPPER
chmod +x "$FLOWER_PROJECT/run_diagnostic.sh"

# Create temp script for interactive mode
TEMP_DIAG_SCRIPT="/tmp/ryzers.diag.sh.tmp"
sed 's/ -d / -it --rm /g' "ryzers.run.$SUPEREXEC_SERVER_NAME.sh" > "$TEMP_DIAG_SCRIPT"
chmod +x "$TEMP_DIAG_SCRIPT"

bash "$TEMP_DIAG_SCRIPT" /app/run_diagnostic.sh

rm -f "$TEMP_DIAG_SCRIPT" "$FLOWER_PROJECT/gpu_diagnostic.py" "$FLOWER_PROJECT/run_diagnostic.sh"

echo ""
echo "========================================="
echo "Diagnostic complete"
echo "========================================="
