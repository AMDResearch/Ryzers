# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic LIBERO episode loop (chunk-replay).

Mirrors FastWAM's validated eval stepping exactly, but calls a generic Policy for the
action chunk instead of a hard-wired model: settle for `num_steps_wait` no-ops, then
repeatedly predict a chunk and execute its first `replan_steps` actions, rendering each
step through an `on_frame` callback. Used by both the sanity runner and the clean
interactive server.
"""
from sim_libero.libero_env import get_libero_dummy_action, get_libero_image, get_max_steps


def run_episode(scene, policy, instruction, on_frame=None, should_stop=None, max_steps=None):
    """Run one episode; returns (success, num_model_steps).

    on_frame(view_dict, step_idx, holding) is called every executed step.
    should_stop() -> True aborts early (interactive Stop button).
    """
    replan_steps = int(getattr(policy, "replan_steps", 5))
    num_steps_wait = int(getattr(policy, "num_steps_wait", 5))
    if max_steps is None:
        max_steps = get_max_steps(scene.suite)

    obs = scene.reset()
    policy.reset(instruction)

    pending = []
    success = False
    model_steps = 0
    t = 0
    while t < max_steps + num_steps_wait:
        if should_stop is not None and should_stop():
            break
        if t < num_steps_wait:
            obs, _, done, _ = scene.env.step(get_libero_dummy_action())
            t += 1
            continue

        if not pending:
            chunk = policy.predict_action_chunk(obs, instruction)
            pending = [list(a) for a in chunk[:replan_steps]]
            model_steps += 1

        imgs = get_libero_image(obs)
        if on_frame is not None:
            on_frame(imgs, t, False)

        obs, _, done, _ = scene.env.step(pending.pop(0))
        t += 1
        if done:
            success = True
            break

    return success, model_steps
