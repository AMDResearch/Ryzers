#!/usr/bin/env bash
# Run me before every benchmark/demo:   bash /ryzers/ros_preflight.sh
# Report without killing anything:      bash /ryzers/ros_preflight.sh --dry-run
#
# A benchmark run that fails or is interrupted leaves its nodes behind. Because
# the container uses --network=host --ipc=host with a shared ROS_DOMAIN_ID, the
# next run joins the same DDS graph as those orphans and collides with them,
# which surfaces as "ROS2 stack is not ready in time."
#
# This kills the leftovers, clears the daemon's cached graph, and verifies the
# preconditions the O3DE benchmark silently depends on.

set -uo pipefail

RAI_DIR="${RAI_DIR:-/ryzers/rai}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ] && DRY_RUN=1

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  FAIL: %s\n' "$*"; fail=1; }

# Leftovers from a previous run. lemond/lemonade are deliberately absent: the
# server is expensive to restart and holds the loaded model.
STALE_PATTERNS=(
    'RAIManipulationDemo.GameLauncher'
    'run_perception_services.py'
    'run_perception_agents.py'
    'panda_moveit_config_demo.launch.py'
    'manipulation-demo-streamlit.py'
    'manipulation_o3de.py'
    'robotic_manipulation'
    'move_group'
    'moveit'
    'static_transform_publisher'
    'robot_state_publisher'
    'rclcpp_components/component_container'
)

# Never signal ourselves or anything we're running under.
protected=" $$ $PPID "
pid=$$
while read -r ppid; do
    [ -z "$ppid" ] || [ "$ppid" = "0" ] && break
    protected+="$ppid "
    pid=$ppid
done < <(while :; do
    p=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) || break
    [ -z "$p" ] || [ "$p" = "0" ] && break
    echo "$p"
    pid=$p
done)

collect() {
    local out=()
    for pat in "${STALE_PATTERNS[@]}"; do
        while read -r p; do
            [ -z "$p" ] && continue
            [[ "$protected" == *" $p "* ]] && continue
            out+=("$p")
        done < <(pgrep -f -- "$pat" 2>/dev/null)
    done
    printf '%s\n' "${out[@]+"${out[@]}"}" | sort -u | sed '/^$/d'
}

echo "== 1. stale ROS2 / simulator processes =="
mapfile -t stale < <(collect)
if [ "${#stale[@]}" -eq 0 ]; then
    note "none found"
elif [ "$DRY_RUN" -eq 1 ]; then
    for p in "${stale[@]}"; do
        note "would kill $p  $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-80)"
    done
else
    for p in "${stale[@]}"; do
        note "TERM $p  $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-80)"
    done
    kill -TERM "${stale[@]}" 2>/dev/null

    # O3DE and move_group can take a few seconds to unwind.
    for _ in $(seq 1 10); do
        mapfile -t stale < <(collect)
        [ "${#stale[@]}" -eq 0 ] && break
        sleep 1
    done

    mapfile -t stale < <(collect)
    if [ "${#stale[@]}" -gt 0 ]; then
        note "still alive after 10s, sending KILL: ${stale[*]}"
        kill -KILL "${stale[@]}" 2>/dev/null
        sleep 2
    fi

    mapfile -t stale < <(collect)
    [ "${#stale[@]}" -eq 0 ] && note "all cleared" || bad "survived KILL: ${stale[*]}"
fi

echo "== 2. ROS2 environment =="
if [ ! -d "$RAI_DIR" ]; then
    bad "$RAI_DIR does not exist (set RAI_DIR)"
else
    cd "$RAI_DIR" || bad "cannot cd to $RAI_DIR"
fi

# The benchmark resolves the simulator binary, the MoveIt launch file and the
# perception script through paths relative to the workspace root, so an
# unsourced overlay or a stray cwd both break it in non-obvious ways.
# The ROS setup scripts read unset variables, so -u has to come off first.
set +u
[ -n "${ROS_DISTRO:-}" ] || . "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" 2>/dev/null
case " ${AMENT_PREFIX_PATH:-} " in
    *"$RAI_DIR/install"*) ;;
    *) . "$RAI_DIR/install/setup.bash" 2>/dev/null ;;
esac
set -u

case " ${AMENT_PREFIX_PATH:-} " in
    *"$RAI_DIR/install"*) note "overlay sourced (AMENT_PREFIX_PATH ok)" ;;
    *) bad "workspace overlay not on AMENT_PREFIX_PATH; openset.launch.py will not resolve" ;;
esac
note "cwd = $PWD"
note "ROS_DOMAIN_ID = ${ROS_DOMAIN_ID:-<unset, default 0>}"

echo "== 3. paths the benchmark resolves relatively =="
for p in \
    "demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher" \
    "src/examples/rai-manipulation-demo/Project/Examples/panda_moveit_config_demo.launch.py" \
    "src/rai_extensions/rai_perception/rai_perception/scripts/run_perception_services.py"
do
    [ -e "$p" ] && note "ok  $p" || bad "missing  $p"
done

for w in ~/.cache/rai/vision/weights/groundingdino_swint_ogc.pth \
         ~/.cache/rai/vision/weights/sam2_hiera_large.pt
do
    [ -s "$w" ] && note "ok  $(basename "$w")" || bad "missing perception weights: $w"
done

echo "== 4. ROS2 graph =="
# The daemon caches the graph and will happily report nodes that are already
# gone, so drop it and let the checks below repopulate from scratch.
if [ "$DRY_RUN" -eq 0 ]; then
    ros2 daemon stop >/dev/null 2>&1
    sleep 1
fi

nodes=$(timeout 25 ros2 node list 2>/dev/null | sed '/^$/d')
if [ -z "$nodes" ]; then
    note "graph is empty"
else
    bad "graph still advertises nodes:"
    printf '        %s\n' $nodes
    note "another container may share this domain; set a unique ROS_DOMAIN_ID"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "PREFLIGHT OK - safe to start a run."
else
    echo "PREFLIGHT FAILED - fix the items above; a run started now will likely"
    echo "die with 'ROS2 stack is not ready in time.'"
fi
exit "$fail"
