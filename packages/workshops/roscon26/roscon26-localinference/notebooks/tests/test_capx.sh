#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Test for the CaP-X stack used by 3_code_as_policy.ipynb.
# Three stages:
#   1. Sign-of-life: ROCm torch sees the GPU, the stack and Robosuite import,
#      MuJoCo renders headless through EGL, and the ungated OWLv2 + SAM2
#      weights are in the image cache.
#   2. Oracle eval (always runs, no LLM): one full CaP-Gym episode driven by the
#      environment's ground-truth code through the real sim + PyRoKi IK
#      pipeline, asserting task success (reward 1.0). This is the deterministic
#      correctness check for the eval pipeline on ROCm.
#   3. LLM eval (optional): a short real agentic eval - the model writes code,
#      the sim executes it, the run is scored. Runs against Lemonade if it is
#      already serving, or against any OpenAI-compatible endpoint given in
#      CAPX_LLM_SERVER_URL.
set -euo pipefail

CAPX_ROOT="${CAPX_ROOT:-/ryzers/cap-x}"
CAPX_PY="${CAPX_VENV:-/opt/capx-venv}/bin/python"
# The CaP-X kernel sets this too; a terminal run needs it to find the
# pre-fetched perception weights.
export HF_HOME="${CAPX_CACHE:-/opt/capx-cache}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${CAPX_ROOT}"

echo "================ [1/3] CaP-X / ROCm sign-of-life ================"
"${CAPX_PY}" -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'ROCm/HIP: {torch.version.hip}')
print(f'GPU ok  : {torch.cuda.is_available()}')
if torch.version.hip is None:
    raise SystemExit('This is not a ROCm/HIP PyTorch build')
if not torch.cuda.is_available():
    raise SystemExit('No ROCm GPU visible - check /dev/kfd and /dev/dri')
for i in range(torch.cuda.device_count()):
    print(f'  device {i}: {torch.cuda.get_device_name(i)}')
"

# jax is CPU-only by design (matches CaP-X's lock: jax 0.4.29, no GPU plugin).
"${CAPX_PY}" -c "
import capx, contact_graspnet_pytorch
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.vision.sam2 import init_sam2
import jax; print('jax', jax.__version__, 'devices (CPU expected):', jax.devices())
import pyroki, jaxls
import robosuite; print('robosuite', robosuite.__version__)
from capx.integrations import list_apis
assert 'FrankaControlApi' in list_apis(), 'perception API not registered'
print('capx + pyroki + API import OK')
"

"${CAPX_PY}" -c "
import sys
sys.path.insert(0, '/ryzers/notebooks/scripts')
from capx_demo import analyze_program
analysis = analyze_program('''
pose, _, extent = get_object_pose(\"red cube\", return_bbox_extent=True)
grasp, quat = sample_grasp_pose(\"red cube\")
goto_pose(grasp, quat, z_approach=0.1)
close_gripper()
''')
assert analysis['syntax_error'] is None
assert analysis['perception_calls'] == 2
assert analysis['planner_calls'] == 1
assert analysis['uses_bbox_extent']
assert analysis['uses_approach_offset']
print('generated-program introspection helpers OK')
"

"${CAPX_PY}" -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"box\" size=\".1 .1 .1\"/></worldbody></mujoco>')
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 64, 64)
try:
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    print('EGL render OK, frame shape', r.render().shape)
finally:
    r.close()
"

"${CAPX_PY}" -c "
from pathlib import Path
config = Path('${CAPX_ROOT}/env_configs/cube_stack/franka_robosuite_cube_stack.yaml')
assert config.is_file(), f'{config} missing'
fast_config = config.with_name('franka_robosuite_cube_stack_fast.yaml')
assert fast_config.is_file(), f'{fast_config} missing'
hub = Path('${HF_HOME}/hub')
repos = {
    'models--facebook--sam2.1-hiera-small',
    'models--google--owlv2-base-patch16-ensemble',
    'models--facebook--sam2.1-hiera-large',
    'models--google--owlv2-large-patch14-ensemble',
}
missing = sorted(repo for repo in repos if not (hub / repo).is_dir())
assert not missing, f'Perception weights not cached: {missing}'
text = config.read_text()
assert 'launch_owlvit_server.main' in text
assert 'launch_sam2_server.main' in text
assert 'google/owlv2-large-patch14-ensemble' in text
assert 'facebook/sam2.1-hiera-large' in text
assert 'launch_sam3_server.main' not in text
fast_text = fast_config.read_text()
assert 'google/owlv2-base-patch16-ensemble' in fast_text
assert 'facebook/sam2.1-hiera-small' in fast_text
for server_name in ('launch_owlvit_server.py', 'launch_sam2_server.py'):
    server_text = (Path('${CAPX_ROOT}/capx/serving') / server_name).read_text()
    assert 'dtype=load_dtype' in server_text, server_name
    assert '_MODEL = _MODEL.float()' in server_text, server_name
print('cube_stack large/default and optional fast perception profiles OK')
"

echo "================ [2/3] Oracle eval (no LLM) ================"
# The oracle path runs the full sim + control pipeline using ground-truth code.
# It only needs the PyRoKi IK server - no LLM, no segmentation, no GraspNet.
PYROKI_PID=""
cleanup_pyroki() {
    if [ -n "${PYROKI_PID:-}" ] && kill -0 "$PYROKI_PID" 2>/dev/null; then
        kill "$PYROKI_PID" 2>/dev/null || true
        wait "$PYROKI_PID" 2>/dev/null || true
    fi
}
trap cleanup_pyroki EXIT

"${CAPX_PY}" capx/serving/launch_pyroki_server.py \
    --port 8116 --robot panda_description --target-link panda_hand \
    > /tmp/pyroki.log 2>&1 &
PYROKI_PID=$!

echo "waiting for PyRoKi IK server..."
PYROKI_READY=0
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8116/ik -X POST -H "Content-Type: application/json" \
        -d '{"target_pose_wxyz_xyz":[1,0,0,0,0.4,0,0.3]}' >/dev/null 2>&1; then
        echo "PyRoKi ready (warmed JAX JIT) after ${i}s"
        PYROKI_READY=1
        break
    fi
    sleep 1
done
if [ "$PYROKI_READY" -ne 1 ]; then
    echo "PyRoKi failed to become ready"
    cat /tmp/pyroki.log
    exit 1
fi

if ! ORACLE_OUT=$(timeout 400 "${CAPX_PY}" tests/test_environments.py \
        --env-name franka_robosuite_pick_place_code_env 2>&1); then
    echo "ORACLE EVAL FAILED"
    tail -20 <<<"$ORACLE_OUT"
    exit 1
fi
grep -aE "Time taken:|Reward:|Success" <<<"$ORACLE_OUT" | tail -4 || true
cleanup_pyroki
trap - EXIT

if grep -aq '^Success$' <<<"$ORACLE_OUT"; then
    echo "ORACLE EVAL PASSED (reward 1.0)"
else
    echo "ORACLE EVAL FAILED"
    tail -20 <<<"$ORACLE_OUT"
    exit 1
fi

echo "================ [3/3] LLM eval (optional) ================"
# Default to the Lemonade server from notebook 1 when it is already serving.
if [ -z "${CAPX_LLM_SERVER_URL:-}" ] && lemonade status >/dev/null 2>&1; then
    CAPX_LLM_SERVER_URL="http://localhost:13305/api/v1/chat/completions"
    CAPX_LLM_MODEL="${CAPX_LLM_MODEL:-Gemma-4-E2B-it-GGUF}"
fi

if [ -z "${CAPX_LLM_SERVER_URL:-}" ]; then
    echo "SKIPPED - start Lemonade (lemond &) or set CAPX_LLM_SERVER_URL and"
    echo "CAPX_LLM_MODEL to run an agentic eval against an OpenAI-compatible"
    echo "endpoint."
else
    MODEL="${CAPX_LLM_MODEL:-Gemma-4-E2B-it-GGUF}"
    TRIALS="${CAPX_LLM_TRIALS:-2}"
    echo "Running ${TRIALS}-trial eval: model=${MODEL} server=${CAPX_LLM_SERVER_URL}"
    LLM_LOG=$(mktemp)
    if ! timeout 1800 "${CAPX_PY}" capx/envs/launch.py \
            --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
            --model "${MODEL}" \
            --server-url "${CAPX_LLM_SERVER_URL}" \
            --api-key "${CAPX_LLM_API_KEY:-}" \
            --total-trials "${TRIALS}" \
            --num-workers 1 \
            --max-tokens 4096 >"$LLM_LOG" 2>&1; then
        echo "LLM EVAL FAILED"
        tail -40 "$LLM_LOG"
        rm -f "$LLM_LOG"
        exit 1
    fi
    grep -aE "Trial [0-9]+ took|reward_|task_completed|Time taken to query" "$LLM_LOG" | tail -12 || true
    rm -f "$LLM_LOG"
    echo "LLM EVAL COMPLETE (see outputs/<model>/ for rewards and videos)"
fi

echo "================ CaP-X tests PASSED ================"
