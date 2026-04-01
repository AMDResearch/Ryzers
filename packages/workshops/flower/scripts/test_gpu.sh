#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

export FLOWER_PROJECT

echo "========================================="
echo "  Testing GPU with Realistic Workload"
echo "========================================="
echo ""

RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

# Create a GPU stress test script
cat > /tmp/gpu_stress_test.py << 'EOF'
import torch
import torch.nn as nn
import sys

print("=" * 60)
print("GPU Stress Test")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: No GPU detected")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("")

# Test 1: Simple tensor operations
print("Test 1: Simple tensor operations...")
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✓ Simple operations work")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Larger tensors (like image batches)
print("\nTest 2: Larger tensors (simulating image batch)...")
try:
    batch = torch.randn(32, 3, 32, 32).cuda()  # CIFAR-10 batch size
    print(f"✓ Created batch tensor: {batch.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: CNN operations
print("\nTest 3: CNN operations...")
try:
    conv = nn.Conv2d(3, 16, 3).cuda()
    output = conv(batch)
    print(f"✓ Conv2d output: {output.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Backpropagation
print("\nTest 4: Backpropagation...")
try:
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass works")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Multiple iterations
print("\nTest 5: Multiple iterations (10 iterations)...")
try:
    model = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 10)
    ).cuda()

    for i in range(10):
        batch = torch.randn(32, 3, 32, 32).cuda()
        output = model(batch)
        loss = output.sum()
        loss.backward()
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/10 complete")

    print("✓ Multiple iterations work")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All GPU tests passed!")
print("=" * 60)
EOF

echo "Running GPU stress test in SuperExec container..."
echo ""

cd "$RYZERS_ROOT"
bash "ryzers.run.$SUPEREXEC_SERVER_NAME.sh" "python3 /tmp/gpu_stress_test.py"
TEST_RESULT=$?

rm -f /tmp/gpu_stress_test.py

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✓ GPU stress test PASSED"
    echo "  The GPU can handle training workloads"
else
    echo ""
    echo "✗ GPU stress test FAILED"
    echo "  This is the same type of operation the demo uses"
fi

exit $TEST_RESULT
