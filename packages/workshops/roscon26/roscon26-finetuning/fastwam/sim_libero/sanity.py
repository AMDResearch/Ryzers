# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless sanity rollout for the LIBERO simulator base image.

Loads a policy (default: built-in RandomPolicy), rolls it through one scene for a
bounded number of steps, and saves an MP4 of the composed agentview|wrist view. Proves
the ROCm/EGL render + MuJoCo step + video encode path work end-to-end with no model.

Env: SUITE, TASK_ID, SEED, STEPS, OUT_DIR, POLICY_FACTORY (module:function).
"""
import os
from datetime import datetime

from sim_libero.envutil import env_int, env_str
from sim_libero.policy import load_policy
from sim_libero.render import banner_frame, compose_view, save_mp4
from sim_libero.rollout import run_episode
from sim_libero.scene import build_scene


def main():
    suite = env_str("SUITE", "libero_object")
    task_id = env_int("TASK_ID", 0)
    seed = env_int("SEED", 1000)
    steps = env_int("STEPS", 80)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sanity] building scene {suite}/{task_id} (seed={seed}) ...", flush=True)
    scene = build_scene(suite, task_id, seed=seed)
    print(f"[sanity] scene ready: \"{scene.description}\"", flush=True)

    policy = load_policy()
    print(f"[sanity] policy: {getattr(policy, 'name', type(policy).__name__)}", flush=True)

    frames = []

    def on_frame(imgs, step, holding):
        frames.append(banner_frame(compose_view(imgs), f"{policy.name}: {scene.description}", 640))

    success, model_steps = run_episode(
        scene, policy, scene.description, on_frame=on_frame, max_steps=steps
    )

    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"sanity_{suite}_{task_id}_{ts}.mp4")
    if frames:
        save_mp4(frames, path, fps=20)
        print(f"[sanity] OK: {len(frames)} frames, {model_steps} model calls, success={success}", flush=True)
        print(f"[sanity] saved {path}", flush=True)
    else:
        raise RuntimeError("no frames rendered")

    scene.close()


if __name__ == "__main__":
    main()
