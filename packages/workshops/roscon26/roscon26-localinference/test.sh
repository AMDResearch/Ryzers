#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fast smoke test for the ROSCon26 Local Inference workshop layer.
#
# Scope is deliberately this ryzer's own contribution: the CaP-X workshop
# patches, the baked weights, HELIX/OpenCode, the two kernels, the headless
# display stack and the course content. It starts no services, touches no GPU
# and makes no network call, so it runs in seconds.
#
# Things a composed ryzer owns are *warnings*, not failures, so that a
# standalone `ryzers build roscon26-localinference` still reports something
# useful instead of dying on the first missing dependency.
#
# The deep per-stack checks are separate and much slower:
#   /ryzers/test_ros.sh                   (from the composed ros ryzer)
#   /ryzers/test_o3de.sh                  (from the composed o3de ryzer)
#   /ryzers/test_rai.sh                   (from the composed rai ryzer)
#   /ryzers/test_lemonade-sdk.sh          (from the composed lemonade ryzer)
#   /ryzers/test_capx.sh                  (from the composed capx ryzer)
#   /ryzers/test_roscon26_lemonade.sh     workshop: baked Gemma + a completion
#   /ryzers/test_roscon26_capx.sh         workshop: oracle eval on the patches
#   /ryzers/test_roscon26_rho.sh          workshop: HELIX harness
#   /ryzers/test_roscon26_rho_multitask.sh

set -u

NOTEBOOKS=/ryzers/notebooks
CAPX_ROOT="${CAPX_ROOT:-/ryzers/cap-x}"
CAPX_PY="${CAPX_VENV:-/opt/capx-venv}/bin/python"

FAILURES=0
WARNINGS=0

fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }
warn() { echo "WARN: $*"; WARNINGS=$((WARNINGS + 1)); }
ok()   { echo "  OK: $*"; }

echo "=========================================="
echo "ROSCon26 Local Inference workshop test"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. The interpreter shim. Every workshop script, and the recorded
#    opencode.json the notebook 4 replay reads, spells the CaP-X interpreter
#    /opt/capx-venv/bin/python. If this is broken nothing else matters.
# ---------------------------------------------------------------------------
echo ""
echo "--- interpreter ---"
if [[ -x "${CAPX_PY}" ]]; then
    ok "capx-venv shim: ${CAPX_PY} -> $(readlink -f "${CAPX_PY}")"
    echo "      $("${CAPX_PY}" --version)"
else
    fail "${CAPX_PY} is missing - the /opt/capx-venv shim did not resolve"
fi

# ---------------------------------------------------------------------------
# 2. Composed stacks. Warn-only: these belong to ros/rai/lemonade-sdk/capx.
# ---------------------------------------------------------------------------
echo ""
echo "--- composed stacks ---"
# ROS 2 installs to /opt/ros/<distro> and only reaches PATH once its setup
# script is sourced, which start_roscon26_localinference.sh does but a bare CMD
# does not. Source it here first, otherwise this warns on a perfectly good
# chain and teaches everyone to ignore the warning.
# ROS's own setup.bash reads AMENT_TRACE_SETUP_FILES without a default, which
# trips this script's `set -u`, so relax it just across the source.
if [ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]; then
    set +u
    # shellcheck disable=SC1090
    . "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
    set -u
fi
command -v ros2    >/dev/null 2>&1 && ok "ros2 (${ROS_DISTRO:-?})"  || warn "ros2 not found - add 'ros' to the chain"
command -v lemonade >/dev/null 2>&1 && ok "lemonade client"          || warn "lemonade not found - add 'lemonade-sdk' to the chain"
command -v lemond  >/dev/null 2>&1 && ok "lemond server"             || warn "lemond not found - add 'lemonade-sdk' to the chain"
command -v o3de    >/dev/null 2>&1 && ok "o3de"                      || warn "o3de not found - add 'o3de' to the chain"
[[ -d /ryzers/rai ]]    && ok "/ryzers/rai"    || warn "/ryzers/rai missing - add 'rai' to the chain"
[[ -d "${CAPX_ROOT}" ]] && ok "${CAPX_ROOT}"   || warn "${CAPX_ROOT} missing - add 'capx' to the chain"

if command -v jupyter-lab >/dev/null 2>&1; then
    ok "jupyter-lab $(jupyter-lab --version)"
    for kernel in capx rai; do
        if jupyter kernelspec list 2>/dev/null | grep -qE "^\s+${kernel}\s"; then
            ok "kernel registered: ${kernel}"
        else
            fail "kernelspec '${kernel}' is not visible to jupyter"
        fi
    done
else
    warn "jupyter-lab not found - add 'jupyterlab' to the chain for the notebooks"
fi

# ---------------------------------------------------------------------------
# 3. Headless display stack. This is what gives O3DE a presentable Vulkan
#    surface with no monitor attached; without it notebook 2 has no simulation.
# ---------------------------------------------------------------------------
echo ""
echo "--- headless display ---"
for binary in weston Xwayland; do
    command -v "${binary}" >/dev/null 2>&1 && ok "${binary}" || fail "${binary} not installed"
done
[[ -d /tmp/.X11-unix ]] && ok "/tmp/.X11-unix exists" || fail "/tmp/.X11-unix missing (Weston aborts without it)"

# ---------------------------------------------------------------------------
# 4. The workshop's CaP-X patches. Each assertion maps to one edit in
#    patch_capx_workshop.py, so a failure here names what drifted.
# ---------------------------------------------------------------------------
echo ""
echo "--- CaP-X workshop patches ---"
if [[ -d "${CAPX_ROOT}" ]]; then
    if "${CAPX_PY}" - "${CAPX_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
problems = []

# The two anchors patch_capx_workshop.py rewrites. R1ProControlApi and
# FrankaLiberoApi keep their SAM3 default on purpose and are not checked.
registry = (root / "capx/integrations/__init__.py").read_text()
anchors = (
    'register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3={}))',
    "lambda env: FrankaControlSpillWipeApi("
    "env, tcp_offset=[0.0, 0.0, -0.0158], use_sam3={}),",
)
for anchor in anchors:
    if anchor.format("True") in registry or anchor.format("False") not in registry:
        problems.append(f"registry: {anchor.format('False')[:60]}... not applied")

for name in ("launch_owlvit_server.py", "launch_sam2_server.py"):
    text = (root / "capx/serving" / name).read_text()
    if "dtype=load_dtype" not in text:
        problems.append(f"{name}: FP16 staging missing")
    if "_MODEL = _MODEL.float()" not in text:
        problems.append(f"{name}: on-device FP32 restore missing")

if "wait_timeout: float = 900.0" not in (root / "capx/envs/runner.py").read_text():
    problems.append("runner.py: service wait is not 900 s")

configs = [
    "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml",
    "env_configs/cube_restack/franka_robosuite_cube_restack.yaml",
    "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
]
for name in configs:
    config = root / name
    text = config.read_text()
    if "launch_sam3_server.main" in text:
        problems.append(f"{name}: still launches the SAM3 server")
    for expected in ("google/owlv2-large-patch14-ensemble", "facebook/sam2.1-hiera-large"):
        if expected not in text:
            problems.append(f"{name}: {expected} missing")
    fast = config.with_name(f"{config.stem}_fast{config.suffix}")
    if not fast.is_file():
        problems.append(f"{fast.name}: fast profile not generated")

for problem in problems:
    print(f"  {problem}")
raise SystemExit(1 if problems else 0)
PY
    then
        ok "registry, FP16 staging, 900 s wait and 4 scenario configs (+ _fast)"
    else
        fail "CaP-X workshop patches are not intact (see above)"
    fi
else
    warn "skipped - ${CAPX_ROOT} not present"
fi

# ---------------------------------------------------------------------------
# 5. Baked weights. Only enforced when the build actually asked for them, so
#    a deliberate BAKE_*=0 build reports a warning rather than a failure.
# ---------------------------------------------------------------------------
echo ""
echo "--- baked weights ---"
check_repo() {  # <hub-root> <models--dir> <what>
    if [[ -d "$1/hub/$2" ]]; then
        ok "$3 ($(du -sh "$1/hub/$2" 2>/dev/null | cut -f1))"
    elif [[ "$4" == "1" ]]; then
        fail "$3 was supposed to be baked but $1/hub/$2 is missing"
    else
        warn "$3 not baked - it will download on first use"
    fi
}
L="${LEMONADE_HF_HOME:-/opt/lemonade-cache/huggingface}"
C="${CAPX_CACHE:-/opt/capx-cache}"
check_repo "$L" models--unsloth--gemma-4-E2B-it-GGUF "Gemma-4 E2B GGUF"  "${ROSCON26_BAKED_LEMONADE:-1}"
check_repo "$L" models--unsloth--gemma-4-E4B-it-GGUF "Gemma-4 E4B GGUF"  "${ROSCON26_BAKED_LEMONADE:-1}"
check_repo "$C" models--facebook--sam2.1-hiera-large "SAM2.1 large"      "${ROSCON26_BAKED_PERCEPTION:-1}"
check_repo "$C" models--google--owlv2-large-patch14-ensemble "OWLv2 large" "${ROSCON26_BAKED_PERCEPTION:-1}"
check_repo "$C" models--facebook--sam2.1-hiera-small "SAM2.1 small (fast profile)" "${ROSCON26_BAKED_PERCEPTION:-1}"
check_repo "$C" models--google--owlv2-base-patch16-ensemble "OWLv2 base (fast profile)" "${ROSCON26_BAKED_PERCEPTION:-1}"

# ---------------------------------------------------------------------------
# 6. HELIX + OpenCode, the notebook 4 toolchain.
# ---------------------------------------------------------------------------
echo ""
echo "--- RHO toolchain ---"
if [[ -d /ryzers/helix ]]; then
    HEAD="$(git -C /ryzers/helix rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ "${HEAD}" == "${HELIX_REF:-}" ]]; then
        ok "helix at ${HEAD}"
    else
        fail "helix HEAD ${HEAD} != pinned ${HELIX_REF:-unset}"
    fi
else
    fail "/ryzers/helix missing"
fi
"${CAPX_VENV:-/opt/capx-venv}/bin/helix" --version >/dev/null 2>&1 \
    && ok "helix CLI $("${CAPX_VENV:-/opt/capx-venv}/bin/helix" --version 2>&1 | head -1)" \
    || fail "helix CLI not runnable at ${CAPX_VENV:-/opt/capx-venv}/bin/helix"
if command -v opencode >/dev/null 2>&1; then
    OC="$(opencode --version 2>/dev/null)"
    [[ "${OC}" == "${OPENCODE_VERSION:-}" ]] && ok "opencode ${OC}" \
        || fail "opencode ${OC} != pinned ${OPENCODE_VERSION:-unset}"
else
    fail "opencode not on PATH"
fi

# ---------------------------------------------------------------------------
# 7. Course content, including the recorded study notebook 4 reads. Missing
#    recorded assets are the single most common way this workshop breaks, so
#    rho_report's own preflight is what reports on them.
# ---------------------------------------------------------------------------
echo ""
echo "--- course content ---"
COUNT=$(find "${NOTEBOOKS}" -maxdepth 1 -name '*.ipynb' | wc -l)
[[ "${COUNT}" -eq 5 ]] && ok "5 notebooks" || fail "expected 5 notebooks, found ${COUNT}"
for asset in \
    scripts/notebook_demo.py scripts/capx_demo.py scripts/rho_demo.py \
    scripts/rho_report.py scripts/rho_multitask_demo.py \
    scripts/rho_multitask_study.py scripts/manipulation_demo_streamlit.py \
    images/toucan.jpg fixtures/rho_multitask/cube_stack.py
do
    [[ -f "${NOTEBOOKS}/${asset}" ]] && ok "${asset}" || fail "missing ${NOTEBOOKS}/${asset}"
done
for script in scripts/lemonade_env.sh scripts/manipulation_demo_headless.sh; do
    [[ -x "${NOTEBOOKS}/${script}" ]] && ok "${script} (executable)" \
        || fail "${NOTEBOOKS}/${script} is missing or not executable"
done

echo ""
echo "--- notebook 4 recorded study ---"
if PYTHONPATH="${NOTEBOOKS}/scripts" "${CAPX_PY}" - <<'PY'
import sys

import rho_report

status = rho_report.preflight("/ryzers/notebooks/recorded_results")
print(rho_report.format_preflight(status))
try:
    rho_report.require_ready(status)
except Exception as exc:  # noqa: BLE001 - the message is the point
    print(f"  {exc}")
    sys.exit(1)
PY
then
    ok "recorded_results is complete"
else
    fail "recorded_results is incomplete - notebook 4's recorded half will not render"
fi

# ---------------------------------------------------------------------------
# 8. RAI demo assets notebook 2 resolves by absolute path.
# ---------------------------------------------------------------------------
echo ""
echo "--- RAI demo assets ---"
if [[ -d /ryzers/rai ]]; then
    for asset in \
        examples/embodiments/manipulation_embodiment.json \
        examples/manipulation_common.py \
        examples/manipulation-demo-streamlit.py \
        config.toml
    do
        [[ -f "/ryzers/rai/${asset}" ]] && ok "rai/${asset}" || fail "missing /ryzers/rai/${asset}"
    done
    [[ -f /ryzers/rai/install/setup.bash ]] && ok "colcon overlay built" \
        || fail "/ryzers/rai/install/setup.bash missing - the manipulation demo will not launch"
else
    warn "skipped - /ryzers/rai not present"
fi

echo ""
echo "=========================================="
if [[ "${FAILURES}" -gt 0 ]]; then
    echo "FAILED: ${FAILURES} failure(s), ${WARNINGS} warning(s)"
    exit 1
fi
echo "SUCCESS: ROSCon26 Local Inference workshop test passed (${WARNINGS} warning(s))"
echo ""
echo "Start the workshop with:"
echo "  ryzers run /ryzers/start_roscon26_localinference.sh"
exit 0
