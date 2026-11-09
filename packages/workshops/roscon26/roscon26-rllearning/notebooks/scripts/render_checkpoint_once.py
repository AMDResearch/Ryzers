#!/usr/bin/env python3
"""One-off: rollout + render a single Brax checkpoint (same seed as 0_simulation_rl.ipynb)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "assets"))

from headless_gl import (  # noqa: E402
    build_gl_env,
    enable_demo_quiet_mode,
    probe_render_subprocess,
    quiet_demo_output,
)

enable_demo_quiet_mode()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from brax.training import checkpoint as brax_checkpoint  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from ml_collections import config_dict  # noqa: E402
from mujoco_playground import registry  # noqa: E402


def load_ppo_policy(checkpoint_path: Path, env, deterministic=True):
    loaded_dict = json.loads((checkpoint_path / "ppo_network_config.json").read_text())
    factory_kwargs = loaded_dict["network_factory_kwargs"]
    if "activation" in factory_kwargs:
        factory_kwargs["activation"] = brax_checkpoint.networks.ACTIVATION[
            factory_kwargs["activation"]
        ]
    for init_fn_name in brax_checkpoint._KERNEL_INIT_FN_KEYWORDS:
        if init_fn_name not in factory_kwargs:
            continue
        init_fn_value = factory_kwargs[init_fn_name]
        if init_fn_value is None:
            del factory_kwargs[init_fn_name]
            continue
        factory_kwargs[init_fn_name] = brax_checkpoint.networks.KERNEL_INITIALIZER[
            init_fn_value
        ]
    loaded_dict["observation_size"] = env.observation_size
    loaded_dict["action_size"] = env.action_size
    config = config_dict.create(**loaded_dict)
    params = brax_checkpoint.load(checkpoint_path)
    ppo_network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn(params, deterministic=deterministic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint step directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    video_path = args.output.resolve()
    traj_path = video_path.with_suffix(".npz")

    can_render, probe_report, gl_env = probe_render_subprocess(REPO_ROOT)
    if not can_render:
        gl_env = build_gl_env(REPO_ROOT)
        raise SystemExit(f"No render backend worked:\n{probe_report}")

    with quiet_demo_output():
        env_cfg = registry.get_default_config("PandaPickCube")
        env = registry.load(
            "PandaPickCube",
            config=env_cfg,
            config_overrides={"impl": "jax"},
        )

    inference_fn = load_ppo_policy(checkpoint, env, deterministic=True)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    trajectory = [state]
    episode_reward = 0.0
    for _ in range(int(env_cfg.episode_length)):
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, action_rng)
        state = jit_step(state, action)
        trajectory.append(state)
        episode_reward += float(np.asarray(state.reward))
        if bool(np.asarray(state.done)):
            break

    qpos = np.stack([np.asarray(s.data.qpos) for s in trajectory])
    qvel = np.stack([np.asarray(s.data.qvel) for s in trajectory])
    mocap_pos = np.stack([np.asarray(s.data.mocap_pos) for s in trajectory])
    mocap_quat = np.stack([np.asarray(s.data.mocap_quat) for s in trajectory])
    rewards = np.array(
        [float(np.asarray(s.reward)) for s in trajectory[1:]], dtype=np.float32
    )
    np.savez(
        traj_path,
        qpos=qpos,
        qvel=qvel,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        rewards=rewards,
        episode_reward=episode_reward,
        dt=float(env.dt),
        env_name="PandaPickCube",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_trajectory.py"),
            str(traj_path),
            "-o",
            str(video_path),
            "--env",
            "PandaPickCube",
        ],
        cwd=REPO_ROOT,
        env=gl_env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)

    steps = len(trajectory) - 1
    print(f"checkpoint: {checkpoint}")
    print(f"reward={episode_reward:.3f}  steps={steps}")
    print(f"video: {video_path}")


if __name__ == "__main__":
    main()
