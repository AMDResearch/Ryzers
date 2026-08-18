# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME template: plug your own VLA/WAM into the RoboTwin simulator base (interactive).

The simulation/robotwin image is model-agnostic. To drive the *interactive* showcase with
your model you do NOT edit this package: you build YOUR ryzer FROM the sim base and ship an
adapter like this one, then select it at runtime.

Three steps
-----------
1. In your model package's Dockerfile, chain on the sim base:

       ARG BASE_IMAGE
       FROM ${BASE_IMAGE}
       # install your model UNDER the base's torch+numpy pins (RoboTwin/mplib need
       # numpy 1.26.4). See the FastWAM package's scripts/strip_cuda_torch.py for a
       # working PIP_CONSTRAINT reference.
       COPY adapters/ /opt/model-adapters/

   Build the chain:   ryzers build robotwin <yourmodel> --name <yourmodel>-robotwin

2. Implement the two methods below.

3. Run, pointing POLICY_FACTORY at your factory (adapter on PYTHONPATH):

       PYTHONPATH=/opt/model-adapters:$PYTHONPATH \
       POLICY_FACTORY=template_policy:build_policy \
         ryzers run --name <yourmodel>-robotwin /ryzers/demos/demo_interactive.sh

For the parity-critical CLOSED-LOOP eval, RoboTwin uses its own model-agnostic runner
(script/eval_policy.py + policy/<name>/deploy_policy.py) instead of this seam - drop your
`policy/<name>` plugin into /opt/RoboTwin/policy. This template is for the interactive
showcase (the LIBERO-style live Policy seam).

Contract
--------
obs           : RoboTwin observation dict. Useful keys:
                  obs["observation"]["head_camera"]["rgb"]   HxWx3 uint8
                  obs["observation"]["left_camera"]["rgb"]   HxWx3 uint8
                  obs["observation"]["right_camera"]["rgb"]  HxWx3 uint8
                  obs["joint_action"]["vector"]              (14,) current qpos:
                      [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]  (aloha-agilex)
instruction   : str, the natural-language task.
return        : np.ndarray [T, 14] float32, execute-ready joint-space qpos rows in the same
                layout as obs["joint_action"]["vector"]. The harness runs each row via
                TASK_ENV.take_action(row, action_type="qpos"), executing the first
                `replan_steps` rows before calling you again.
"""
import numpy as np

from sim_robotwin.policy import Policy


class TemplatePolicy(Policy):
    name = "template"
    replan_steps = 8      # env steps executed per predicted chunk before replanning

    def __init__(self, model):
        self.model = model

    def reset(self, instruction):
        # Called once per episode. Clear any per-episode caches.
        pass

    def predict_action_chunk(self, obs, instruction):
        head = obs["observation"]["head_camera"]["rgb"]
        left = obs["observation"]["left_camera"]["rgb"]
        right = obs["observation"]["right_camera"]["rgb"]
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)  # (14,)
        # TODO: run your model; return [T, 14] qpos rows (same layout as `state`).
        raise NotImplementedError("wire your model here")
        # return np.tile(state, (self.replan_steps, 1)).astype(np.float32)  # e.g. hold pose


def build_policy():
    # TODO: load your checkpoint / processor once here (env knobs: CKPT, DATASET_STATS, ...).
    model = None
    return TemplatePolicy(model)
