#!/usr/bin/env python3
"""Render trajectory.npz to mp4 using headless MuJoCo + system Mesa/OSMesa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "assets"))

from headless_gl import reexec_with_gl_env  # noqa: E402

# Must happen before MuJoCo is imported: LD_LIBRARY_PATH only takes effect for a
# freshly started process.
reexec_with_gl_env(REPO_ROOT)

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco_playground import registry  # noqa: E402


def render_npz(npz_path: Path, output_path: Path, env_name: str) -> None:
    data = np.load(npz_path)
    qpos = data["qpos"]
    qvel = data["qvel"] if "qvel" in data else None
    mocap_pos = data["mocap_pos"] if "mocap_pos" in data else None
    mocap_quat = data["mocap_quat"] if "mocap_quat" in data else None
    dt = float(data["dt"])

    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg, config_overrides={"impl": "jax"})
    model = env.mj_model

    renderer = mujoco.Renderer(model, height=480, width=640)
    mj_data = mujoco.MjData(model)
    frames = []
    for i in range(len(qpos)):
        mj_data.qpos[:] = qpos[i]
        if qvel is not None:
            mj_data.qvel[:] = qvel[i]
        if mocap_pos is not None:
            mj_data.mocap_pos[:] = mocap_pos[i]
        if mocap_quat is not None:
            mj_data.mocap_quat[:] = mocap_quat[i]
        mujoco.mj_forward(model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())
    renderer.close()

    fps = 1.0 / dt
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"Wrote {len(frames)} frames to {output_path} at {fps:.1f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="trajectory.npz from 0_simulation_rl.ipynb")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "panda_pick_cube.mp4",
        help="output mp4 path",
    )
    parser.add_argument("--env", default="PandaPickCube", help="MuJoCo Playground env")
    args = parser.parse_args()
    render_npz(args.npz.resolve(), args.output.resolve(), args.env)


if __name__ == "__main__":
    main()
