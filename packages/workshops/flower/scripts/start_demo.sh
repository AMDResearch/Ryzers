#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower Federated Learning Demo"
echo "========================================="
echo ""

# Check if setup was run
if [ ! -d "$FLOWER_PROJECT" ]; then
    echo "Error: Flower project not found. Run ./setup_env.sh first."
    exit 1
fi

# Export FLOWER_PROJECT for use in docker run flags (config.yaml uses $FLOWER_PROJECT)
export FLOWER_PROJECT

# Change to Ryzers root directory
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "Stopping all Flower containers..."
    docker stop "$SUPERLINK_NAME" 2>/dev/null || true
    docker stop "$SUPERNODE1_NAME" 2>/dev/null || true
    docker stop "$SUPERNODE2_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_SERVER_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_CLIENT1_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_CLIENT2_NAME" 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Step 1: Launch SuperLink in background
echo "[1/6] Launching SuperLink (background)..."
ryzers run "$SUPERLINK_NAME" "flower-superlink --insecure --isolation process" > /tmp/flower-superlink.log 2>&1 &
SUPERLINK_PID=$!
echo "  Started with PID $SUPERLINK_PID"

sleep 3

# Step 2: Launch SuperNode 1 in background
echo "[2/6] Launching SuperNode 1 (background)..."
ryzers run "$SUPERNODE1_NAME" "flower-supernode --insecure --superlink $SUPERLINK_NAME:9092 --node-config 'partition-id=0 num-partitions=2' --clientappio-api-address 0.0.0.0:9094 --isolation process" > /tmp/flower-supernode1.log 2>&1 &
SUPERNODE1_PID=$!
echo "  Started with PID $SUPERNODE1_PID"

# Step 3: Launch SuperNode 2 in background
echo "[3/6] Launching SuperNode 2 (background)..."
ryzers run "$SUPERNODE2_NAME" "flower-supernode --insecure --superlink $SUPERLINK_NAME:9092 --node-config 'partition-id=1 num-partitions=2' --clientappio-api-address 0.0.0.0:9095 --isolation process" > /tmp/flower-supernode2.log 2>&1 &
SUPERNODE2_PID=$!
echo "  Started with PID $SUPERNODE2_PID"

sleep 3

# Step 4: Launch ServerApp executor in background
echo "[4/6] Launching ServerApp executor (background)..."
ryzers run "$SUPEREXEC_SERVER_NAME" "flower-superexec --insecure --plugin-type serverapp --appio-api-address $SUPERLINK_NAME:9091" > /tmp/flower-superexec-server.log 2>&1 &
SUPEREXEC_SERVER_PID=$!
echo "  Started with PID $SUPEREXEC_SERVER_PID"

sleep 2

# Step 5: Launch ClientApp executors in background
echo "[5/6] Launching ClientApp executor 1 (background)..."
ryzers run "$SUPEREXEC_CLIENT1_NAME" "flower-superexec --insecure --plugin-type clientapp --appio-api-address $SUPERNODE1_NAME:9094" > /tmp/flower-superexec-client1.log 2>&1 &
SUPEREXEC_CLIENT1_PID=$!
echo "  Started with PID $SUPEREXEC_CLIENT1_PID"

echo "[5/6] Launching ClientApp executor 2 (background)..."
ryzers run "$SUPEREXEC_CLIENT2_NAME" "flower-superexec --insecure --plugin-type clientapp --appio-api-address $SUPERNODE2_NAME:9095" > /tmp/flower-superexec-client2.log 2>&1 &
SUPEREXEC_CLIENT2_PID=$!
echo "  Started with PID $SUPEREXEC_CLIENT2_PID"

sleep 3

# Show running containers
echo ""
echo "All components started. Running containers:"
docker ps --filter "name=flower-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 6: Run Flower training in this terminal (foreground/interactive)
echo "[6/6] Starting Federated Learning training..."
echo ""

# Check that all required containers are running
echo "Verifying all containers are running..."
RUNNING_COUNT=$(docker ps --filter "name=flower-" --format "{{.Names}}" | wc -l)
if [ "$RUNNING_COUNT" -lt 5 ]; then
    echo ""
    echo "ERROR: Expected 5+ containers, found $RUNNING_COUNT"
    echo ""
    echo "Running containers:"
    docker ps --filter "name=flower-" --format "  - {{.Names}}"
    echo ""
    echo "Background process logs:"
    echo "  SuperLink log: /tmp/flower-superlink.log"
    echo "  SuperNode1 log: /tmp/flower-supernode1.log"
    echo "  SuperNode2 log: /tmp/flower-supernode2.log"
    echo "  SuperExec Server log: /tmp/flower-superexec-server.log"
    echo "  SuperExec Client1 log: /tmp/flower-superexec-client1.log"
    echo "  SuperExec Client2 log: /tmp/flower-superexec-client2.log"
    echo ""
    echo "Check logs with: tail -20 /tmp/flower-*.log"
    echo ""
    exit 1
fi

# Create a simple training script in the workspace (will be mounted to /app)
cat > "$FLOWER_PROJECT/run_training.sh" << 'EOF'
#!/bin/bash
set -e
cd /app
echo "Installing project dependencies..."
pip install -q -e .
echo ""
echo "Starting federated learning training..."
echo ""
flwr run . local-deployment --stream
EOF
chmod +x "$FLOWER_PROJECT/run_training.sh"

# Run the script inside the container using ryzers run (interactive mode)
echo "Installing project and running federated learning..."
echo ""
ryzers run "$SUPEREXEC_SERVER_NAME" /app/run_training.sh

# Cleanup
rm -f "$FLOWER_PROJECT/run_training.sh"

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"

# Cleanup on exit
cleanup
