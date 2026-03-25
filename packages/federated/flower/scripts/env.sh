# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -ex

# This script must be sourced "source env.sh" for paths to be set correctly
FLOWER_ENV_PATH="${BASH_SOURCE[0]}"
FLOWER_SCRIPTS="$(cd "$(dirname "$FLOWER_ENV_PATH")" && pwd)"
FLOWER_PATH=`dirname $(realpath $FLOWER_SCRIPTS)`
REPO_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"

# Change to repo root where the ryzers.run.*.sh scripts are generated
cd "$REPO_ROOT"
