#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Notebook 2 without the notebook: bring up the headless O3DE display, the
# camera stream and the RAI agent chat as a Streamlit page.
#
#   ryzers run /ryzers/demo_roscon26_localinference.sh            # full demo
#   ryzers run /ryzers/demo_roscon26_localinference.sh --infra-only
#
# Open http://localhost:8501 once the scene has loaded. Ryzers runs with
# --network=host, so no port forwarding is involved.
#
# Expect four to five minutes from launch to the arm moving: Weston, O3DE,
# MoveIt and the perception services all start cold, and the first model turn
# is the slowest one.

set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source /ryzers/rai/install/setup.bash

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p -m 700 "${XDG_RUNTIME_DIR}"

# Point RAI's [openai] section at the local Lemonade server and load the model
# from the baked cache. Sourced, not run: it exports OPENAI_API_KEY and leaves
# lemond serving in the background.
# shellcheck disable=SC1091
source /ryzers/notebooks/scripts/lemonade_env.sh "${RAI_MODEL:-Gemma-4-E2B-it-GGUF}"

exec /ryzers/notebooks/scripts/manipulation_demo_headless.sh "$@"
