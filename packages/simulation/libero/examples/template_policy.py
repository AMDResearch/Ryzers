# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME template: plug your own model into the LIBERO sim base.

The sim base is model-agnostic; you build FROM it, ship an adapter like this, and select
it at runtime (do NOT edit this package).

Steps
-----
1. Chain on the sim base in your model's Dockerfile, installing your model UNDER the
   base's torch+numpy pins (needs numpy 1.26.4) so the LIBERO/robosuite/MuJoCo stack is
   not broken (FastWAM's scripts/strip_cuda_torch.py is a working reference):

       ARG BASE_IMAGE
       FROM ${BASE_IMAGE}
       COPY scripts/adapters/ /opt/model-adapters/

   Build:   ryzers build libero <yourmodel> --name <yourmodel>-libero

2. Implement the two methods below (load in build_policy, obs -> action chunk in
   predict_action_chunk).

3. Run with your factory on PYTHONPATH:

       PYTHONPATH=/opt/model-adapters:$PYTHONPATH \
       POLICY_FACTORY=template_policy:build_policy \
         ryzers run --name <yourmodel>-libero /ryzers/demos/demo_interactive.sh

Contract
--------
obs           : raw LIBERO/robosuite observation dict. Useful keys:
                  obs["agentview_image"]            HxWx3 uint8 (3rd-person; flipped)
                  obs["robot0_eye_in_hand_image"]   HxWx3 uint8 (wrist; flipped)
                  obs["robot0_eef_pos"]             (3,)  end-effector position
                  obs["robot0_eef_quat"]            (4,)  end-effector orientation (xyzw)
                  obs["robot0_gripper_qpos"]        (2,)  gripper joint positions
instruction   : str, the natural-language task.
return        : np.ndarray [T, 7] float32 = (dx,dy,dz,droll,dpitch,dyaw, gripper), LIBERO
                OSC_POSE delta control; gripper {-1 open, +1 close}. Harness executes the
                first `replan_steps` rows, then calls you again.
"""
import numpy as np

from sim_libero.policy import Policy


class TemplatePolicy(Policy):
    name = "template"
    replan_steps = 5      # env steps executed per predicted chunk before replanning
    num_steps_wait = 5    # no-op settle steps at episode start

    def __init__(self, model):
        self.model = model

    def reset(self, instruction):
        # Called once per episode; clear per-episode caches here.
        pass

    def predict_action_chunk(self, obs, instruction):
        # TODO: preprocess obs, run inference, return actions in the OSC_POSE delta space above.
        raise NotImplementedError("wire your model here")
        # return np.zeros((self.replan_steps, 7), dtype=np.float32)


def build_policy():
    # TODO: load your checkpoint / processor once here (env knobs: CKPT, DATASET_STATS, ...).
    model = None
    return TemplatePolicy(model)
