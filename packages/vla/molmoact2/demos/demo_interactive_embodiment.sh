#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Embodiment-swap interactive demo: compatibility entrypoint for the same browser
# UI as demo_interactive.sh, with LIBERO Panda replaced by another arm.
#
#   EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive_embodiment.sh
#   open http://localhost:8080   (remote box: ssh -L 8080:localhost:8080 <host>)
#
# Knobs: EMBODIMENT (ur5e or xarm6), EMBODIMENT_GRIPPER, plus the usual
# SUITE / TASK_ID / SEED / THINK / NUM_STEPS / PORT.
set -euo pipefail

export EMBODIMENT="${EMBODIMENT:-ur5e}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${TARGET:-/ryzers/demo_interactive.sh}"
[ -f "$TARGET" ] || TARGET="$HERE/demo_interactive.sh"
exec "$TARGET"
