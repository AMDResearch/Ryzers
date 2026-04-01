#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

export FLOWER_PROJECT

RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

# Create a test that defines the Net model inline vs importing it
cat > /tmp/test_net_model.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("Testing Net Model Creation")
print("=" * 60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("")

# Test 1: Define model inline (same as Flower's Net)
print("Test 1: Model defined inline...")
try:
    class NetInline(nn.Module):
        def __init__(self):
            super(NetInline, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model1 = NetInline()
    print("  Model created")
    model1.to(device)
    print("  ✓ Model moved to GPU successfully")

    # Test forward pass
    x = torch.randn(1, 3, 32, 32).to(device)
    output = model1(x)
    print(f"  ✓ Forward pass works, output shape: {output.shape}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n" + "=" * 60)
print("Model test passed!")
print("=" * 60)
EOF

echo "Testing Net model directly in container..."
echo ""

cd "$RYZERS_ROOT"
bash "ryzers.run.$SUPEREXEC_SERVER_NAME.sh" "python3 /tmp/test_net_model.py"
TEST_RESULT=$?

rm -f /tmp/test_net_model.py

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✓ Inline model works"
    echo "  Now testing imported model from Flower package..."
    echo ""
    ./scripts/test_flower_training.sh
else
    echo ""
    echo "✗ Even inline model fails"
    echo "  Issue is with the model architecture or PyTorch/ROCm"
fi

exit $TEST_RESULT
