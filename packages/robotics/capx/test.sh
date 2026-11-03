#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -euo pipefail

export CAPX_SIM="${CAPX_SIM:-robosuite}"
if [[ "$CAPX_SIM" == "libero" ]]; then
    ORACLE_ENV="franka_libero_pick_place_code_env_privileged"
else
    ORACLE_ENV="franka_robosuite_pick_place_code_env"
fi

PYROKI_PID=""
cleanup() {
    if [[ -n "$PYROKI_PID" ]] && kill -0 "$PYROKI_PID" 2>/dev/null; then
        kill "$PYROKI_PID" 2>/dev/null || true
        wait "$PYROKI_PID" 2>/dev/null || true
    fi
    PYROKI_PID=""
}
trap cleanup EXIT

cd /ryzers/cap-x

echo "================ [1/2] CaP-X / ROCm sign-of-life ================"
python3 - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import capx, contact_graspnet_pytorch, jax, jaxls, mujoco, pyroki, robosuite, sam3, torch

print(f"GPU ok  : {torch.cuda.is_available()}")
if torch.version.hip is None:
    raise SystemExit("This is not a ROCm/HIP PyTorch build")
if not torch.cuda.is_available():
    raise SystemExit("No ROCm GPU visible; check /dev/kfd and /dev/dri")
for index in range(torch.cuda.device_count()):
    print(f"  device {index}: {torch.cuda.get_device_name(index)}")

print("capx + pyroki import OK")
if os.environ["CAPX_SIM"] == "libero":
    import libero
    print("LIBERO + robosuite import OK")
else:
    print("robosuite", robosuite.__version__, "import OK")

m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 64, 64)
try:
    mujoco.mj_forward(m, d)
    renderer.update_scene(d)
    print("EGL render OK, frame shape", renderer.render().shape)
finally:
    renderer.close()
PY

echo "================ [2/2] Oracle eval: ${ORACLE_ENV} ================"
python3 capx/serving/launch_pyroki_server.py \
    --port 8116 --robot panda_description --target-link panda_hand \
    >/tmp/pyroki.log 2>&1 &
PYROKI_PID=$!

for second in {1..60}; do
    if curl -sf http://127.0.0.1:8116/ik -X POST \
        -H "Content-Type: application/json" \
        -d '{"target_pose_wxyz_xyz":[1,0,0,0,0.4,0,0.3]}' >/dev/null; then
        echo "PyRoKi ready (warmed JAX JIT) after ${second}s"
        break
    fi
    [[ "$second" -lt 60 ]] || {
        echo "PyRoKi failed to start"
        tail -40 /tmp/pyroki.log
        exit 1
    }
    sleep 1
done

if ! ORACLE_OUT="$(timeout 400 python3 tests/test_environments.py --env-name "$ORACLE_ENV" 2>&1)"; then
    echo "ORACLE EVAL FAILED"
    tail -20 <<<"$ORACLE_OUT"
    exit 1
fi
grep -aE "Time taken:|Reward:|Success" <<<"$ORACLE_OUT" | tail -4 || true
grep -aq '^Success$' <<<"$ORACLE_OUT" || { echo "ORACLE EVAL FAILED"; exit 1; }
echo "ORACLE EVAL PASSED (reward 1.0)"
cleanup

echo "================ CaP-X tests PASSED ================"
