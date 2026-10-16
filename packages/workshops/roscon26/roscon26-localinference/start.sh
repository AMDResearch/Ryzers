#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start JupyterLab for the ROSCon 2026 Local Inference workshop.
#
#   ryzers run /ryzers/start_roscon26_localinference.sh
#
# This is the replacement for the JupyterHub image's entrypoint. It exists
# because the notebook server itself needs ROS on its path: the `!ros2 topic
# list` cells in notebook 2 are shell escapes that inherit the *server's*
# environment, not the kernel's, and web_video_server is launched the same way.
# The kernels source ROS independently (see rai_kernel.sh), so a server started
# without this script still runs the Python, just not the shell cells.

set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ -f /ryzers/rai/install/setup.bash ]; then
    source /ryzers/rai/install/setup.bash
fi

# Weston and Xwayland both need these. XDG_RUNTIME_DIR is set by the composed
# lemonade-sdk ryzer; the mode matters, Weston refuses a world-readable one.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p -m 700 "${XDG_RUNTIME_DIR}"
mkdir -p -m 1777 /tmp/.X11-unix

cd /ryzers/notebooks

exec jupyter-lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.root_dir=/ryzers/notebooks \
    "$@"
