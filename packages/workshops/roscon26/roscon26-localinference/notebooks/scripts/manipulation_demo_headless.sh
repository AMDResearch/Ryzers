#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Runs the RAI manipulation demo without a monitor.
#
# O3DE renders through Vulkan, which needs a presentable surface: Xvfb can't
# provide one (no DRI3, so only the llvmpipe software device is presentable),
# but Xwayland on a headless Weston compositor can - it hands the X server the
# real /dev/dri render node, so the iGPU does the rendering. The simulation
# camera is then republished as MJPEG by web_video_server and shown next to the
# agent chat in the Streamlit page, so the whole demo lives in the browser.

set -e

# --infra-only brings up the headless display and the camera stream, then exits
# and leaves them running. The notebook builds the UI itself in that mode, so
# there is no Streamlit page and no second browser tab.
INFRA_ONLY=0
[ "${1:-}" = "--infra-only" ] && INFRA_ONLY=1

RAI_DIR=/ryzers/rai
WESTON_LOG=/tmp/weston.log
VIDEO_PORT=8080

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p -m 700 "$XDG_RUNTIME_DIR"

# Xwayland needs this to exist. The container runs with --network=host, which
# shares the abstract socket namespace, so the abstract @X0 name is taken and
# Xwayland falls back to a socket file in here - Weston crashes if it's missing.
mkdir -p -m 1777 /tmp/.X11-unix

cleanup() {
    [ -n "$WESTON_PID" ] && kill "$WESTON_PID" 2>/dev/null
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    # $VIDEO_PID is the "ros2 run" wrapper, not the server binary it forks, and a
    # wedged server ignores SIGTERM (rclcpp's shutdown handler never gets to run).
    # Match on the name and SIGKILL, or the orphan keeps :$VIDEO_PORT and the next
    # launch dies with "Address already in use".
    pkill -9 -f web_video_server 2>/dev/null
    return 0
}
trap cleanup EXIT

# 1. Headless GPU display
if pgrep -f "weston --backend=headless" > /dev/null; then
    echo "Reusing the Weston compositor that is already running"
else
    echo "Starting headless Weston + Xwayland..."
    weston --backend=headless --renderer=gl \
           --width=1920 --height=1080 \
           --xwayland --socket=wayland-headless > "$WESTON_LOG" 2>&1 &
    WESTON_PID=$!
fi

for _ in $(seq 30); do
    DISPLAY=$(grep -oP 'xserver listening on display \K:[0-9]+' "$WESTON_LOG" | head -1)
    [ -n "$DISPLAY" ] && break
    sleep 1
done
if [ -z "$DISPLAY" ]; then
    echo "Xwayland did not come up - see $WESTON_LOG" >&2
    exit 1
fi
export DISPLAY
echo "Headless display ready on $DISPLAY"

# 2. ROS 2 + RAI environment (the demo resolves its assets relative to $RAI_DIR)
cd "$RAI_DIR"
source /opt/ros/${ROS_DISTRO}/setup.bash
source install/setup.bash

# 3. Republish the simulation camera as MJPEG for the browser
echo "Starting web_video_server on port $VIDEO_PORT..."
start_video_server() {
    ros2 run web_video_server web_video_server --ros-args -p port:=$VIDEO_PORT \
        >> /tmp/web_video_server.log 2>&1 &
    VIDEO_PID=$!
}
# Clear anything left over from an earlier run before claiming the port
if pkill -9 -f web_video_server 2>/dev/null; then sleep 2; fi
start_video_server

# web_video_server wedges when a stream client vanishes mid-frame - a crashed
# notebook kernel, a closed browser tab. It keeps the listen socket, spins at
# 100% CPU and stops answering, so the simulation panel and the notebook
# recorder both hang until someone restarts it by hand. Health-check the
# snapshot endpoint instead and respawn when it stops responding.
(
    sleep 30
    while true; do
        # Ask for the topic index, not the camera: the camera topic does not
        # exist until the simulation is up, but a wedged server stops answering
        # every request, this one included.
        if ! curl -fsS --max-time 10 -o /dev/null "http://localhost:$VIDEO_PORT/"; then
            echo "$(date -Is) web_video_server unresponsive - restarting" \
                >> /tmp/web_video_server.log
            pkill -9 -f web_video_server 2>/dev/null || true
            sleep 2
            start_video_server
            sleep 20  # give it time to come up before the next check
        fi
        sleep 15
    done
# Redirect the whole subshell, not just the echo: it outlives this script, and
# anything reading our stdout through a pipe would block until it exits.
) >> /tmp/web_video_server.log 2>&1 &
WATCHDOG_PID=$!

if [ "$INFRA_ONLY" = "1" ]; then
    # Hand weston, web_video_server and the watchdog over to the notebook by
    # dropping the trap - otherwise they die with this shell.
    trap - EXIT
    echo "DISPLAY=$DISPLAY"
    echo "Infrastructure ready - build the UI from the notebook"
    exit 0
fi

# 4. Streamlit page: simulation view + agent chat
echo "Starting the demo - open http://localhost:8501 once the scene has loaded"
streamlit run /ryzers/notebooks/scripts/manipulation_demo_streamlit.py \
    --server.headless true --server.address 0.0.0.0
