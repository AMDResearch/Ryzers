# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboTwin episode loop (chunk-replay).

Mirrors RoboTwin's own eval loop but against the sim_robotwin.Policy seam: get an
observation, ask the policy for a [T, action_dim] joint-space chunk, execute the first
`replan_steps` rows via TASK_ENV.take_action, then replan. Each executed step yields a
live composed frame (head|observer over left|right wrist) to the on_frame callback. No
torch or model code here.
"""
from collections import deque


def run_episode(scene, policy, instruction, on_frame=None, should_stop=None, max_steps=None):
    """Drive one RoboTwin episode. Returns (success, steps_taken)."""
    replan_steps = int(getattr(policy, "replan_steps", 8))
    action_type = getattr(policy, "action_type", "qpos")
    limit = int(max_steps) if max_steps else scene.step_lim
    scene.set_instruction(instruction)
    policy.reset(instruction)

    queue = deque()
    steps = 0
    while scene.take_action_cnt < limit and not scene.success:
        if should_stop and should_stop():
            break
        if not queue:
            obs = scene.get_obs()
            chunk = policy.predict_action_chunk(obs, instruction)
            for row in chunk[:replan_steps]:
                queue.append(row)
        if not queue:
            break
        scene.take_action(queue.popleft(), action_type=action_type)
        steps += 1
        if on_frame is not None:
            on_frame(scene.eval_frame(), scene.take_action_cnt, len(queue) == 0)

    return scene.success, steps
