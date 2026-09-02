# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless sanity rollout: proves ROCm/Vulkan SAPIEN render + mplib execution + 4-view compose + encode with no model.

Env: TASK, TASK_CONFIG, SEED, MAX_STEPS, OUT_DIR, POLICY_FACTORY (module:function).
"""
import os
from datetime import datetime

from sim_robotwin.envutil import env_int, env_str
from sim_robotwin.policy import load_policy
from sim_robotwin.render import banner_frame, save_mp4
from sim_robotwin.rollout import run_episode
from sim_robotwin.taskenv import RoboTwinScene


def main():
    task = env_str("TASK", "click_bell")
    task_config = env_str("TASK_CONFIG", "demo_clean")
    seed = env_int("SEED", 100000)
    max_steps = env_int("MAX_STEPS", 60)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sanity] setting up {task} ({task_config}, seed={seed}) ...", flush=True)
    scene = RoboTwinScene.build_stable(task, task_config=task_config, seed=seed)
    instruction = scene.default_instruction()
    print(f"[sanity] scene ready: \"{instruction}\" | step_lim={scene.step_lim}", flush=True)

    policy = load_policy()
    print(f"[sanity] policy: {getattr(policy, 'name', type(policy).__name__)}", flush=True)

    frames = []

    def on_frame(rgb, step, replanning):
        frames.append(banner_frame(rgb, f"{policy.name}: {instruction}", 640))

    success, steps = run_episode(
        scene, policy, instruction, on_frame=on_frame, max_steps=max_steps
    )

    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"sanity_{task}_{ts}.mp4")
    if frames:
        save_mp4(frames, path, fps=15)
        print(f"[sanity] OK: {len(frames)} frames, {steps} steps, success={success}", flush=True)
        print(f"[sanity] saved {path}", flush=True)
    else:
        raise RuntimeError("no frames rendered")

    scene.close()


if __name__ == "__main__":
    main()
