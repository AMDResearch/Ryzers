#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

export FLOWER_PROJECT

echo "========================================="
echo "  Testing Flower Project Code Directly"
echo "========================================="
echo ""

if [ ! -d "$FLOWER_PROJECT" ]; then
    echo "Error: Flower project not found at $FLOWER_PROJECT"
    echo "Run ./build_containers.sh first"
    exit 1
fi

RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

# Create a test script that uses the actual Flower project code
cat > "$FLOWER_PROJECT/test_direct.py" << 'EOF'
import sys
sys.path.insert(0, '/app')

import torch
from pytorchexample.task import Net, load_data, train, test

print("=" * 60)
print("Testing Flower Project Code Directly")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("")

# Test 1: Load model
print("Test 1: Loading model...")
try:
    model = Net()
    model.to(device)
    print(f"✓ Model loaded and moved to {device}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Load dataset (this might be where it fails)
print("\nTest 2: Loading CIFAR-10 dataset...")
try:
    trainloader, testloader = load_data(partition_id=0, num_partitions=2, batch_size=32)
    print(f"✓ Dataset loaded")
    print(f"  Train batches: {len(trainloader)}")
    print(f"  Test batches: {len(testloader)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Single training step
print("\nTest 3: Single training step...")
try:
    train_loss = train(model, trainloader, epochs=1, lr=0.01, device=device)
    print(f"✓ Training step completed, loss: {train_loss:.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Evaluation
print("\nTest 4: Evaluation...")
try:
    test_loss, test_acc = test(model, testloader, device)
    print(f"✓ Evaluation completed")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All Flower project tests passed!")
print("=" * 60)
EOF

echo "Running Flower project code test in SuperExec container..."
echo ""

cd "$RYZERS_ROOT"

# Create a wrapper script to avoid quoting issues
cat > "$FLOWER_PROJECT/run_test.sh" << 'WRAPPER'
#!/bin/bash
set -e
cd /app
pip install -q --no-deps -e .
python3 /app/test_direct.py
WRAPPER
chmod +x "$FLOWER_PROJECT/run_test.sh"

# Create interactive version for better output
TEMP_TEST_SCRIPT="/tmp/ryzers.test.flower.sh.tmp"
sed 's/ -d / -it --rm /g' "ryzers.run.$SUPEREXEC_SERVER_NAME.sh" > "$TEMP_TEST_SCRIPT"
chmod +x "$TEMP_TEST_SCRIPT"

bash "$TEMP_TEST_SCRIPT" /app/run_test.sh
TEST_RESULT=$?

rm -f "$TEMP_TEST_SCRIPT" "$FLOWER_PROJECT/test_direct.py" "$FLOWER_PROJECT/run_test.sh"

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✓ Flower project code test PASSED"
    echo "  The issue is in Flower's distributed orchestration"
else
    echo ""
    echo "✗ Flower project code test FAILED"
    echo "  This helps isolate the problem"
fi

exit $TEST_RESULT
