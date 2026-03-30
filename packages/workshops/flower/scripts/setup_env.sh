#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Setting up Flower demo environment..."

# Create workspace directory
mkdir -p "$FLOWER_WORKSPACE"

# Change to Ryzers root directory (ryzers run must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

# Initialize Flower project using the SuperExec ryzer (which has flwr CLI)
echo "Initializing Flower project..."

if [ ! -d "$FLOWER_PROJECT" ]; then
    # Run flwr new command inside SuperExec ryzer
    pushd "$RYZERS_ROOT"
    ryzers run --name "$SUPEREXEC_SERVER_NAME" \
        -v "$FLOWER_WORKSPACE:/workspace" \
        -- /bin/bash -c "cd /workspace && (flwr new @flwrlabs/quickstart-pytorch --non-interactive || flwr new quickstart-pytorch --framework pytorch)"
    popd

    echo "✓ Flower project created at: $FLOWER_PROJECT"
else
    echo "✓ Flower project already exists: $FLOWER_PROJECT"
fi

# Create Flower config file
mkdir -p ~/.flwr
cat > ~/.flwr/config.toml << EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

echo "✓ Setup complete!"
echo ""
echo "Next step: ./start_demo.sh"
