# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Illustrative stub: wire a VLA-JEPA-style model into the LIBERO sim base.

NOT a runnable model - it shows the shape of a real adapter for a model developed as its
own ryzer (e.g. on another Strix Halo box) that chains on `simulation/libero`. Replace the
`import` and the two TODO blocks with your actual package's inference calls; the rest is the
LIBERO Policy contract and is identical for every model.

Build + run (from the VLA-JEPA package):
    ryzers build libero vla-jepa --name vla-jepa-libero
    POLICY_FACTORY=vjepa_libero_policy:build_policy \
      ryzers run --name vla-jepa-libero /ryzers/demos/demo_interactive.sh   # http://localhost:8080
"""
import os

import numpy as np

from sim_libero.policy import Policy

# from vjepa_vla import load_pretrained, VJepaActionHead   # <- your package's real API


class VJepaLiberoPolicy(Policy):
    name = "vla-jepa"
    replan_steps = 5
    num_steps_wait = 5

    def __init__(self, model, image_size=224):
        self.model = model
        self.image_size = image_size
        self._plan = None

    def reset(self, instruction):
        # V-JEPA-style planners typically cache a latent world-model plan per episode.
        self._plan = None

    def predict_action_chunk(self, obs, instruction):
        # 1) Build model inputs from the raw LIBERO obs (flip images back to upright, resize).
        third = np.ascontiguousarray(obs["agentview_image"][::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
        state = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                obs["robot0_gripper_qpos"]]).astype(np.float32)

        # 2) TODO: run your model. It must emit LIBERO OSC_POSE deltas [T, 7]:
        #    (dx, dy, dz, droll, dpitch, dyaw, gripper), gripper in {-1, +1}.
        #    e.g. actions = self.model.plan(third, wrist, state, instruction)
        raise NotImplementedError("replace with your VLA-JEPA inference")
        # return np.asarray(actions, dtype=np.float32)


def build_policy():
    ckpt = os.environ.get("CKPT") or "/models/vla_jepa/libero.pt"
    # model = load_pretrained(ckpt, device="cuda")   # ROCm torch reports as cuda
    model = None
    print(f"[vjepa_libero_policy] model ready (ckpt={ckpt})", flush=True)
    return VJepaLiberoPolicy(model)
