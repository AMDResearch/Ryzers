#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Idempotently patch the allenai/lerobot fork's eval to dump per-episode executed
action trajectories (.npy) next to each rendered video, so we can make action
plots for every LIBERO rollout. Safe to re-run."""
import importlib.util
import sys

# Patch the INSTALLED module (non-editable pip install copies the source into
# site-packages), not the git checkout. Locate it via importlib so this works in
# both the dev container (/opt/venv) and the image's isolated venv (/opt/libero-venv).
spec = importlib.util.find_spec("lerobot.scripts.lerobot_eval")
PATH = spec.origin
print("patching:", PATH)
src = open(PATH).read()

if "AMD-PATCH: dump executed action" in src:
    print("already patched")
    sys.exit(0)

OLD = """            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))"""

NEW = """            for _b_ix, (stacked_frames, done_index) in enumerate(zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            )):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                # AMD-PATCH: dump executed action trajectory for action plots
                try:
                    import numpy as _np
                    _acts = rollout_data[ACTION][_b_ix, : done_index + 1].detach().cpu().numpy()
                    _np.save(videos_dir / f"eval_episode_{n_episodes_rendered}_actions.npy", _acts)
                except Exception as _e:
                    print("AMD-PATCH action dump failed:", _e)"""

if OLD not in src:
    print("ERROR: anchor not found; eval file layout changed", file=sys.stderr)
    sys.exit(2)

open(PATH, "w").write(src.replace(OLD, NEW, 1))
print("patched OK")
