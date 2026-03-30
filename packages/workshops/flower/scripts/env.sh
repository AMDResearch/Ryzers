# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

# This script must be sourced "source env.sh" for paths to be set correctly
FLOWER_ENV_PATH="${BASH_SOURCE[0]}"
FLOWER_SCRIPTS="$(cd "$(dirname "$FLOWER_ENV_PATH")" && pwd)"
FLOWER_PATH=$(dirname "$FLOWER_SCRIPTS")
FLOWER_WORKSPACE="$FLOWER_PATH/workspace"
FLOWER_PROJECT="$FLOWER_WORKSPACE/quickstart-pytorch"

# Flower configuration
FLOWER_VERSION="1.27.0"

# Ryzer container names
SUPERLINK_NAME="flower-superlink"
SUPERNODE1_NAME="flower-supernode-1"
SUPERNODE2_NAME="flower-supernode-2"
SUPEREXEC_SERVER_NAME="flower-superexec-server"
SUPEREXEC_CLIENT1_NAME="flower-superexec-client-1"
SUPEREXEC_CLIENT2_NAME="flower-superexec-client-2"
