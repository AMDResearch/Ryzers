#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "========================================="
echo "  Testing Flower Ryzers"
echo "========================================="
echo ""

RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

FAILED=0

# Test SuperLink
echo "[1/6] Testing SuperLink..."
if bash "ryzers.run.$SUPERLINK_NAME.sh"; then
    echo "✓ SuperLink test passed"
else
    echo "✗ SuperLink test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test SuperNode 1
echo "[2/6] Testing SuperNode 1..."
if bash "ryzers.run.$SUPERNODE1_NAME.sh"; then
    echo "✓ SuperNode 1 test passed"
else
    echo "✗ SuperNode 1 test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test SuperNode 2
echo "[3/6] Testing SuperNode 2..."
if bash "ryzers.run.$SUPERNODE2_NAME.sh"; then
    echo "✓ SuperNode 2 test passed"
else
    echo "✗ SuperNode 2 test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test SuperExec Server
echo "[4/6] Testing SuperExec Server..."
if bash "ryzers.run.$SUPEREXEC_SERVER_NAME.sh"; then
    echo "✓ SuperExec Server test passed"
else
    echo "✗ SuperExec Server test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test SuperExec Client 1
echo "[5/6] Testing SuperExec Client 1..."
if bash "ryzers.run.$SUPEREXEC_CLIENT1_NAME.sh"; then
    echo "✓ SuperExec Client 1 test passed"
else
    echo "✗ SuperExec Client 1 test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test SuperExec Client 2
echo "[6/6] Testing SuperExec Client 2..."
if bash "ryzers.run.$SUPEREXEC_CLIENT2_NAME.sh"; then
    echo "✓ SuperExec Client 2 test passed"
else
    echo "✗ SuperExec Client 2 test FAILED"
    FAILED=$((FAILED + 1))
fi
echo ""

echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "  All tests passed!"
else
    echo "  $FAILED test(s) FAILED"
fi
echo "========================================="

exit $FAILED
