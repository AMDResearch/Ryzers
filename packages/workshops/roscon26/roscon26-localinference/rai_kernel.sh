#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Launcher for the "RAI (ROCm)" Jupyter kernel.
#
# Notebooks 0-2 import rclpy, rai_sim and rai_bench, which live in the ROS 2
# install tree and in the colcon overlay that `rai` builds at /ryzers/rai -
# neither is on the default sys.path. The JupyterHub image got away with a
# plain `python` argv because its entrypoint sourced both before starting the
# notebook server; under Ryzers the server is whatever the user typed, so the
# kernel sources them itself. That also means a kernel restart cannot lose the
# environment, and `ryzers run bash` behaves the same as the notebook.

set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ -f /ryzers/rai/install/setup.bash ]; then
    source /ryzers/rai/install/setup.bash
else
    echo "warning: /ryzers/rai/install/setup.bash missing -" \
         "build with 'ros rai' in the chain" >&2
fi

# Deliberately no `cd`. Jupyter starts a kernel in the notebook's own
# directory, and notebook 1 reads images/toucan.jpg from there; notebook 2
# chdir()s to /ryzers/rai itself for the cell that needs RAI's config.toml.
#
# /opt/capx-venv is the shim the Dockerfile points at the base interpreter, so
# in the composed image both kernels run the same Python and differ only in the
# environment they are given.
exec /opt/capx-venv/bin/python -m ipykernel_launcher "$@"
