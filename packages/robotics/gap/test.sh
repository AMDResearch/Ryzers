#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -euo pipefail

cd /ryzers/graph-as-policy

echo "================ GaP / ROCm smoke test ================"
python3 - <<'PY'
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gap
import jax
import mujoco
import numpy as np
import pyroki
import robosuite
import torch

if torch.version.hip is None:
    raise SystemExit("This is not a ROCm/HIP PyTorch build")
if not torch.cuda.is_available():
    raise SystemExit("No ROCm GPU visible; check /dev/kfd and /dev/dri")

print("GaP import OK")
print(f"PyTorch {torch.__version__}; HIP {torch.version.hip}")
for index in range(torch.cuda.device_count()):
    print(f"  device {index}: {torch.cuda.get_device_name(index)}")
print(f"JAX backend: {jax.default_backend()} (PyRoKi intentionally runs on CPU)")
print(f"Robosuite {robosuite.__version__} import OK")

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 64, 64)
try:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    print("MuJoCo EGL render OK:", renderer.render().shape)
finally:
    renderer.close()

from gap.envs.registry import resolve

for suite in ("libero_object_all_variance", "libero_spatial"):
    factory, key = resolve(suite)
    env, config = factory(key, 0, camera_names=None, enable_render=False)
    try:
        obs, info = env.reset(seed=1)
        rgb = obs["agentview"]["images"]["rgb"]
        assert rgb.shape == (512, 800, 3)
        assert info["task_prompt"]
        obs, _, terminated, truncated, _ = env.step(np.zeros(7))
        assert obs["robot_joint_pos_0"].shape == (8,)
        assert not terminated and not truncated
    finally:
        env.handle.env.close()
    print(f"{suite} reset and step OK")

# Verify measured-joint seeding without compiling the real solver.
from gap.connector import ik as ik_module


class _FakeJoints:
    num_actuated_joints = 8


class _FakeLinks:
    names = ["panda_hand"]


class _FakeRobot:
    joints = _FakeJoints()
    links = _FakeLinks()

    def forward_kinematics(self, cfg):
        assert np.asarray(cfg).shape == (8,)
        # PyRoKI link poses are [qw, qx, qy, qz, x, y, z].
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.11, 0.22, 0.33]])


original_basic = ik_module.solve_ik_basic
original_velocity = ik_module.solve_ik_vel_cost
seed = np.linspace(-0.3, 0.3, 7)
velocity_calls = 0

try:
    def unexpected_basic(**_kwargs):
        raise AssertionError("a seeded path must not re-solve waypoint zero")

    def continuous_ik(*, prev_cfg, initial_cfg, **_kwargs):
        global velocity_calls
        velocity_calls += 1
        assert np.allclose(initial_cfg, prev_cfg)
        result = np.asarray(prev_cfg).copy()
        result[0] += 0.01
        return result

    ik_module.solve_ik_basic = unexpected_basic
    ik_module.solve_ik_vel_cost = continuous_ik
    trajectory = ik_module.plan_trajectory_linear_ik(
        robot=_FakeRobot(),
        target_link_name="tool",
        start_pos=np.zeros(3),
        start_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        end_pos=np.array([0.03, 0.0, 0.0]),
        end_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        num_waypoints=4,
        ik_refinement_iters=1,
        seed_cfg=seed,
    )
    assert np.array_equal(trajectory[0, :7], seed)
    assert velocity_calls == 3
    assert np.max(np.abs(np.diff(trajectory[:, :7], axis=0))) < 0.5

finally:
    ik_module.solve_ik_basic = original_basic
    ik_module.solve_ik_vel_cost = original_velocity

print("PyRoKi measured-joint seeding OK")

# Verify the Cartesian start frame independently from the observed link pose.
captured = {}
original_plan = ik_module.plan_trajectory_linear_ik

try:
    def capture_plan(**kwargs):
        captured.update(kwargs)
        cfg = np.asarray(kwargs["seed_cfg"], dtype=np.float64)
        if cfg.size < 8:
            cfg = np.pad(cfg, (0, 8 - cfg.size))
        return np.stack([cfg, cfg])

    ik_module.plan_trajectory_linear_ik = capture_plan
    backend = ik_module.PyRokiBackend(
        tcp_offset=np.array([0.0, 0.0, -0.097]),
    )
    backend._robot = _FakeRobot()
    pose = lambda z: {
        "position": {"x": 0.0, "y": 0.0, "z": z},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    plan = backend.plan_linear(
        pose(9.0),
        pose(0.2),
        seed_joints=seed.tolist(),
        num_waypoints=2,
    )
    assert plan is not None
    assert np.allclose(captured["start_pos"], [0.11, 0.22, 0.33])
    assert np.allclose(captured["seed_cfg"], seed)
finally:
    ik_module.plan_trajectory_linear_ik = original_plan

print("PyRoKi Cartesian start-frame regression OK")

# Compile and execute the real PyRoKI path.
home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
backend = ik_module.PyRokiBackend()
robot = backend.robot
cfg = np.pad(home, (0, robot.joints.num_actuated_joints - home.size))
link_index = robot.links.names.index("panda_hand")
start_fk = np.asarray(robot.forward_kinematics(cfg))[link_index]
start = {
    "position": dict(zip(("x", "y", "z"), start_fk[4:7], strict=True)),
    "rotation": dict(zip(("w", "x", "y", "z"), start_fk[:4], strict=True)),
}
end = {
    "position": dict(start["position"]),
    "rotation": dict(start["rotation"]),
}
end["position"]["z"] += 0.03
plan = backend.plan_linear(
    start,
    end,
    seed_joints=home.tolist(),
    num_waypoints=5,
    ik_refinement_iters=1,
)
assert plan is not None
waypoints = np.stack([w["positions"] for w in plan["waypoints"]])
assert np.allclose(waypoints[0, :7], home)
assert np.max(np.abs(np.diff(waypoints[:, :7], axis=0))) < 0.5
end_fk = np.asarray(robot.forward_kinematics(waypoints[-1]))[link_index]
assert np.linalg.norm(end_fk[4:7] - np.array(list(end["position"].values()))) < 0.01

print("PyRoKi real Cartesian plan OK")

# Start each supported out-of-process bundle far enough to receive its catalog.
# Models are lazy, so this verifies code/dependency wiring without downloading
# gated weights or requiring an LLM key.
from gap_core.rpc.client import ToolClient

skills = Path("/ryzers/open-robot-skills/tools")
supported = [
    "curve",
    "gemini-er",
    "geometry",
    "grounding-dino",
    "molmo",
    "sam3",
    "vlm",
]
for name in supported:
    with ToolClient(name, skills / name) as client:
        if not client.catalog:
            raise RuntimeError(f"{name}: empty tool catalog")
        print(f"{name}: {len(client.catalog)} tool(s) registered")
PY

python3 -m pip check
pytest -q tests/runtime tests/connector/test_ik.py
gap --help >/dev/null
gap check >/dev/null
gap run examples/grocery_packing_pyroki/packing_graph \
    --sim libero_object_packing/0 --validate-only >/dev/null
echo "================ GaP / ROCm smoke test PASSED ================"
