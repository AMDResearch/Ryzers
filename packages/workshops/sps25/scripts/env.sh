# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -ex

# This script must be sourced "source env.sh" for paths to be set correctly
SPSDEMO_ENV_PATH="${BASH_SOURCE[0]}"
SPSDEMO_SCRIPTS="$(cd "$(dirname "$SPSDEMO_ENV_PATH")" && pwd)"

SPSDEMO_PATH=`dirname $(realpath $SPSDEMO_SCRIPTS)`
SPSDEMO_VENV="$SPSDEMO_PATH/spsdemo_venv"
SPSDEMO_RYZERS="$SPSDEMO_PATH/spsdemo_ryzers"
