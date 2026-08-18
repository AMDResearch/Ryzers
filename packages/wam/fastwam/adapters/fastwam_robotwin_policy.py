# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FastWAM RoboTwin policy adapter for the simulation/robotwin interactive harness.

Implements the model-agnostic `sim_robotwin.Policy` seam by wrapping the FastWAM world-action
model. Reuses the *validated* RoboTwin deploy policy from
experiments/robotwin/fastwam_policy/deploy_policy.py verbatim (config compose, model
instantiate + checkpoint load, processor/normalizer, image/state builders and
`_infer_action_chunk`) so interactive rollouts match the closed-loop numbers.

Selected at runtime by the sim harness via
  POLICY_FACTORY=fastwam_robotwin_policy:build_policy

The parity-critical closed-loop path does NOT use this adapter: it runs RoboTwin's own
script/eval_policy.py against the same deploy_policy.py plugin. This wrapper only bridges
that plugin's inference into the interactive Policy seam.

Env: CKPT, DATASET_STATS, MIXED_PRECISION (bf16), REPLAN_STEPS, NUM_INFERENCE_STEPS,
ACTION_HORIZON, FASTWAM_REPO (/repos/fastwam). Requires /repos/fastwam,
/repos/fastwam/experiments/robotwin and /opt/sim on PYTHONPATH (the demo sets this).
"""
import os

import numpy as np

import experiments.robotwin.fastwam_policy.deploy_policy as D
from sim_robotwin.policy import Policy

DEFAULT_CKPT = "/models/fastwam_release/robotwin_uncond_3cam_384.pt"
DEFAULT_STATS = "/models/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"


class FastwamRoboTwinPolicy(Policy):
    name = "fastwam"

    def __init__(self, model):
        self.model = model  # deploy_policy.WorldActionRobotWinPolicy
        self.replan_steps = int(getattr(model, "replan_steps", 8))

    def reset(self, instruction):
        self.model.reset()

    def predict_action_chunk(self, obs, instruction):
        chunk = self.model._infer_action_chunk(observation=obs, instruction=instruction)
        return np.asarray(chunk, dtype=np.float32)


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    usr_args = {
        "ckpt_setting": os.environ.get("CKPT") or DEFAULT_CKPT,
        "dataset_stats_path": os.environ.get("DATASET_STATS") or DEFAULT_STATS,
        "device": "cuda",  # ROCm torch reports as cuda; deploy_policy falls back to cpu if absent
        "mixed_precision": os.environ.get("MIXED_PRECISION") or "bf16",
    }
    for key, env in (("replan_steps", "REPLAN_STEPS"),
                     ("num_inference_steps", "NUM_INFERENCE_STEPS"),
                     ("action_horizon", "ACTION_HORIZON")):
        val = os.environ.get(env)
        if val:
            usr_args[key] = val

    model = D.get_model(usr_args)
    print(f"[fastwam_robotwin_policy] model ready (ckpt={usr_args['ckpt_setting']}, "
          f"replan={getattr(model, 'replan_steps', '?')})", flush=True)
    return FastwamRoboTwinPolicy(model)
