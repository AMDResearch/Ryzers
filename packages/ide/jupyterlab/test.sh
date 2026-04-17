#!/bin/bash
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Test script for JupyterLab - verifies it can start, then exits

set -e

echo "Testing JupyterLab installation..."

# Verify jupyter-lab is installed
if ! command -v jupyter-lab &> /dev/null; then
    echo "FAIL: jupyter-lab command not found"
    exit 1
fi

echo "jupyter-lab found at: $(which jupyter-lab)"
echo "Version: $(jupyter-lab --version)"

# Start JupyterLab in background and verify it starts
echo "Starting JupyterLab server..."
jupyter-lab --ip=0.0.0.0 --no-browser --allow-root --port=8888 &
JUPYTER_PID=$!

# Wait for server to start (check if port 8888 is listening)
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:8888 > /dev/null 2>&1; then
        echo "JupyterLab server is running on port 8888"
        break
    fi
    if ! kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "FAIL: JupyterLab process died unexpectedly"
        exit 1
    fi
    sleep 1
done

# Check if server is responding
if curl -s http://localhost:8888 > /dev/null 2>&1; then
    echo "SUCCESS: JupyterLab is responding"
else
    echo "FAIL: JupyterLab did not start within ${MAX_WAIT}s"
    kill $JUPYTER_PID 2>/dev/null || true
    exit 1
fi

# Clean up
echo "Stopping JupyterLab..."
kill $JUPYTER_PID 2>/dev/null || true
wait $JUPYTER_PID 2>/dev/null || true

echo "JupyterLab test completed successfully"
exit 0
