# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME template: plug your own VLA/WAM into the LIBERO simulator base.

The simulation/libero image is model-agnostic. To drive it with your model you do NOT
edit this package: you build YOUR ryzer FROM the sim base and ship an adapter like this
one, then select it at runtime.

Three steps
-----------
1. In your model package's Dockerfile, chain on the sim base:

       ARG BASE_IMAGE
       FROM ${BASE_IMAGE}
       # install your model UNDER the base's torch+numpy pins so you don't break the
       # LIBERO/robosuite/MuJoCo stack (it needs numpy 1.26.4). The FastWAM package's
       # scripts/strip_cuda_torch.py + `PIP_CONSTRAINT=$(pip freeze | grep numpy)` is a
       # working reference for this.
       COPY scripts/adapters/ /opt/model-adapters/

   Build the chain:   ryzers build libero <yourmodel> --name <yourmodel>-libero

2. Implement the two methods below (load the model in build_policy, convert obs -> action
   chunk in predict_action_chunk).

3. Run, pointing POLICY_FACTORY at your factory and putting the adapter on PYTHONPATH:

       PYTHONPATH=/opt/model-adapters:$PYTHONPATH \
       POLICY_FACTORY=template_policy:build_policy \
         ryzers run --name <yourmodel>-libero /ryzers/demos/demo_interactive.sh

The harness owns the env, rendering, MJPEG streaming and the episode loop. Your adapter
only turns one (obs, instruction) into an action chunk.

Contract
--------
obs           : raw LIBERO/robosuite observation dict. Useful keys:
                  obs["agentview_image"]            HxWx3 uint8 (3rd-person; vertically flipped)
                  obs["robot0_eye_in_hand_image"]   HxWx3 uint8 (wrist; vertically flipped)
                  obs["robot0_eef_pos"]             (3,)  end-effector position
                  obs["robot0_eef_quat"]            (4,)  end-effector orientation (xyzw)
                  obs["robot0_gripper_qpos"]        (2,)  gripper joint positions
instruction   : str, the natural-language task.
return        : np.ndarray [T, 7] float32 = (dx, dy, dz, droll, dpitch, dyaw, gripper),
                LIBERO OSC_POSE delta control; gripper in {-1 open, +1 close}. The harness
                executes the first `replan_steps` rows, then calls you again.
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
        # Called once per episode. Clear any per-episode caches / receding-horizon queue.
        pass

    def predict_action_chunk(self, obs, instruction):
        # TODO: preprocess obs into your model's inputs (resize/normalize images, build
        # the proprio/state vector), run inference, and return actions in the LIBERO
        # OSC_POSE delta space described above. Placeholder below just holds position.
        raise NotImplementedError("wire your model here")
        # return np.zeros((self.replan_steps, 7), dtype=np.float32)


def build_policy():
    # TODO: load your checkpoint / processor once here (env knobs: CKPT, DATASET_STATS, ...).
    model = None
    return TemplatePolicy(model)
