#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "Building Flower ryzer containers..."

# Change to Ryzers root directory (ryzers build must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
pushd "$RYZERS_ROOT"

# Build ryzer_env first (base environment)
echo "[0/3] Building ryzer_env (base environment)..."
ryzers build ryzer_env --name ryzer_env

# Build SuperLink ryzer
echo "[1/3] Building SuperLink..."
ryzers build flower-superlink --name "$SUPERLINK_NAME"

# Build SuperNode ryzer
echo "[2/3] Building SuperNode..."
ryzers build flower-supernode --name "$SUPERNODE1_NAME"
ryzers build flower-supernode --name "$SUPERNODE2_NAME"

# Build SuperExec ryzer
echo "[3/3] Building SuperExec..."
ryzers build flower-superexec --name "$SUPEREXEC_SERVER_NAME"
ryzers build flower-superexec --name "$SUPEREXEC_CLIENT1_NAME"
ryzers build flower-superexec --name "$SUPEREXEC_CLIENT2_NAME"

# Fix network and TTY in generated run scripts (BEFORE popd, while still in RYZERS_ROOT)
echo ""
echo "Fixing network and TTY configuration in run scripts..."
for script in ryzers.run.flower-*.sh; do
    if [ -f "$script" ]; then
        # Replace --network=host with --network flwr-network
        sed -i 's/--network=host/--network flwr-network/g' "$script"
        # Replace -it with -d for background daemon mode (multiple patterns to catch all cases)
        sed -i 's/ -it / -d /g; s/^docker run -it /docker run -d /g; s/ -it$/ -d/g' "$script"
        # Remove xhost line (X11 not needed for Flower)
        sed -i '/xhost +local:docker/d' "$script"
        echo "  ✓ Fixed $script"
        # Show the docker run line for verification
        echo "     Docker command: $(grep '^docker run' "$script" | head -1 | cut -c1-80)..."
    fi
done

popd

echo ""
echo "✓ All Flower ryzers built successfully!"
echo ""

# Now create the Flower project using the built SuperExec container
echo "Initializing Flower project..."

if [ ! -d "$FLOWER_PROJECT" ]; then
    # Create a temp ryzers run script that mounts workspace instead of project
    SUPEREXEC_SCRIPT="$RYZERS_ROOT/ryzers.run.${SUPEREXEC_SERVER_NAME}.sh"
    TEMP_INIT_SCRIPT="/tmp/ryzers.init.flower.sh.tmp"

    # Temporarily set FLOWER_PROJECT to workspace for the volume mount
    SAVED_FLOWER_PROJECT="$FLOWER_PROJECT"
    export FLOWER_PROJECT="$FLOWER_WORKSPACE"

    # Create temp script with --rm instead of -d for one-time command
    sed "s| -d | --rm |g" "$SUPEREXEC_SCRIPT" > "$TEMP_INIT_SCRIPT"
    chmod +x "$TEMP_INIT_SCRIPT"

    # Run flwr new command using ryzers script
    bash "$TEMP_INIT_SCRIPT" "/bin/bash -c 'cd /app && flwr new @flwrlabs/quickstart-pytorch'"

    # Restore FLOWER_PROJECT
    export FLOWER_PROJECT="$SAVED_FLOWER_PROJECT"

    # Cleanup temp script
    rm -f "$TEMP_INIT_SCRIPT"

    echo "✓ Flower project created at: $FLOWER_PROJECT"
else
    echo "✓ Flower project already exists: $FLOWER_PROJECT"
fi

# Create Flower config file in the project directory
mkdir -p "$FLOWER_PROJECT/.flwr"
chmod 755 "$FLOWER_PROJECT/.flwr" 2>/dev/null || true

if [ -f "$FLOWER_PROJECT/.flwr/config.toml" ]; then
    chmod 644 "$FLOWER_PROJECT/.flwr/config.toml" 2>/dev/null || rm -f "$FLOWER_PROJECT/.flwr/config.toml"
fi

cat > "$FLOWER_PROJECT/.flwr/config.toml" <<EOF
[superlink.local-deployment]
address = "$SUPERLINK_NAME:9093"
insecure = true
EOF

chmod 644 "$FLOWER_PROJECT/.flwr/config.toml"

# Also create in home directory for local use
mkdir -p ~/.flwr
if [ -f ~/.flwr/config.toml ]; then
    chmod 644 ~/.flwr/config.toml 2>/dev/null || rm -f ~/.flwr/config.toml
fi
cp "$FLOWER_PROJECT/.flwr/config.toml" ~/.flwr/config.toml
chmod 644 ~/.flwr/config.toml

# Install project dependencies locally (for verification)
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
    echo "Installing project dependencies..."
    cd "$FLOWER_PROJECT"
    pip install -q -e .
    cd - > /dev/null
fi

echo ""
echo "Next step: ./start_demo.sh"
