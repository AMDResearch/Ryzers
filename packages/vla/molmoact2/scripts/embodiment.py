# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Embodiment swap for the MolmoAct2 x LIBERO demos.

The MolmoAct2-LIBERO policy talks to the sim purely in end-effector space:
  observation.state = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]  (world frame)
  action            = [dx, dy, dz, droll, dpitch, dyaw, gripper]       (OSC_POSE delta)
Neither side carries joint angles, so the policy is embodiment-agnostic at the
interface. Swapping the Panda for another arm therefore only changes the
kinematics behind the same OSC controller; the action/observation contract is
unchanged as long as the gripper keeps a 2-dim qpos.

This module swaps the LIBERO arm by:
  1. registering a robosuite robot model under the name LIBERO expects
     ("Mounted<Arm>"), reusing robosuite's stock arm assets, and
  2. patching lerobot.envs.libero.OffScreenRenderEnv so every env is built with
     robots=[<Arm>] and a cross-mounted PandaGripper (so gripper_qpos stays
     2-dim and gripper state/action match the training distribution exactly).

Recorded init states are Panda-specific (different DoF), so callers must build
the env with init_states=False.

Enabled via env vars (see apply_from_env):
  EMBODIMENT=ur5e            arm to swap in (default: unset -> stock Panda)
  EMBODIMENT_GRIPPER=...     gripper to mount (default: PandaGripper)
  EMBODIMENT_ROT_DEG=ax,deg  optional constant tool-frame yaw/pitch/roll offset
                             on the orientation delta, e.g. "z,90" (default: none)
"""
import os

import numpy as np

_PANDA_TABLE_OFFSETS = {
    "bins": (-0.5, -0.1, 0),
    "empty": (-0.6, 0, 0),
    "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
    "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
    "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
    "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
    "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
}


def register_ur5e():
    """Define + register LIBERO-mountable UR5e variants (robosuite stock assets).

    LIBERO names its robot per scene type: tabletop/kitchen/study scenes ask for
    "Mounted<Arm>" while floor/coffee/living-room scenes ask for
    "OnTheGround<Arm>". We register both so any suite works.
    """
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.utils.mjcf_utils import xml_path_completion
    from robosuite.robots.single_arm import SingleArm
    from robosuite.robots import ROBOT_CLASS_MAPPING

    # IK-solved so the (cross-mounted PandaGripper) grip site starts at the Panda
    # LIBERO home eef pose pos=[-0.1485, 0, 0.2613], same orientation. This is the
    # key cross-embodiment fix: MolmoAct2 emits *delta* eef actions and localizes
    # the eef in image space, so the arm must start where the Panda would. Matching
    # the eef pose+orientation also realigns the wrist eye-in-hand camera for free,
    # since it rides rigidly on the gripper (identical cam->gripper transform on
    # both arms). Solved on libero_object/task3/seed1000 (pos err 0.4mm, rot 1e-4).
    _UR5E_INIT_QPOS = np.array([-0.2942, -1.6381, 1.9651, -1.8435, -1.5872, -1.8650])

    class MountedUR5e(ManipulatorModel):
        """UR5e (6-DoF) on the LIBERO Rethink mount."""

        def __init__(self, idn=0):
            super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

        @property
        def default_mount(self):
            return "RethinkMount"

        @property
        def default_gripper(self):
            return "Robotiq85Gripper"

        @property
        def default_controller_config(self):
            return "default_ur5e"

        @property
        def init_qpos(self):
            return _UR5E_INIT_QPOS

        @property
        def base_xpos_offset(self):
            return dict(_PANDA_TABLE_OFFSETS)

        @property
        def top_offset(self):
            return np.array((0, 0, 1.0))

        @property
        def _horizontal_radius(self):
            return 0.5

        @property
        def arm_type(self):
            return "single"

    class OnTheGroundUR5e(MountedUR5e):
        """UR5e (6-DoF) mounted directly on the ground (no Rethink pedestal)."""

        @property
        def default_mount(self):
            return None

    # Subclassing ManipulatorModel auto-registers the model in
    # robosuite.models.robots.REGISTERED_ROBOTS; we only declare each arm type.
    ROBOT_CLASS_MAPPING["MountedUR5e"] = SingleArm
    ROBOT_CLASS_MAPPING["OnTheGroundUR5e"] = SingleArm
    return MountedUR5e


# arm name -> (robosuite robot name passed to LIBERO, registrar)
_REGISTRARS = {
    "ur5e": ("UR5e", register_ur5e),
}


def _rot_offset_from_env():
    """Parse EMBODIMENT_ROT_DEG='axis,deg' into a 3x3 rotation, or None."""
    spec = os.environ.get("EMBODIMENT_ROT_DEG", "").strip()
    if not spec:
        return None
    axis, deg = spec.split(",")
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    if axis.lower() == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis.lower() == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # z


def make_action_transform():
    """Return f(action_7d)->action_7d. Identity unless EMBODIMENT_ROT_DEG is set.

    With a cross-mounted PandaGripper the gripper channel is already in the
    training distribution, so no gripper remap is needed. The only embodiment
    knob exposed here is a constant rotation offset on the orientation delta,
    for aligning a tool frame that differs from the Panda's.
    """
    R = _rot_offset_from_env()
    if R is None:
        return lambda a: a

    def _tf(action):
        a = np.asarray(action, dtype=np.float64).copy()
        a[3:6] = R @ a[3:6]
        return a.astype(np.float32)

    return _tf


_APPLIED = None


def apply(embodiment, gripper="PandaGripper"):
    """Patch lerobot's LIBERO env builder to use the given arm + gripper.

    Returns the robosuite robot name now in use (or None for stock Panda).
    Idempotent: repeated calls with the same arm are a no-op.
    """
    global _APPLIED
    emb = (embodiment or "").lower()
    if not emb or emb == "panda":
        return None
    if _APPLIED is not None:
        return _APPLIED
    if emb not in _REGISTRARS:
        raise ValueError(f"Unknown EMBODIMENT={embodiment!r}; known: {sorted(_REGISTRARS)} (xarm6 pending an MJCF)")

    robot_name, registrar = _REGISTRARS[emb]
    registrar()

    import lerobot.envs.libero as ll

    base_env_cls = ll.OffScreenRenderEnv

    class _EmbodiedEnv(base_env_cls):
        def __init__(self, **kwargs):
            kwargs.setdefault("robots", [robot_name])
            if gripper and gripper.lower() != "default":
                kwargs.setdefault("gripper_types", gripper)
            super().__init__(**kwargs)

    ll.OffScreenRenderEnv = _EmbodiedEnv

    # Recorded LIBERO init states are Panda-specific (different DoF), so loading
    # them into another arm corrupts/crashes the sim. Force init_states off for
    # any swapped arm regardless of how the env was configured, so the swap is
    # self-contained (no reliance on the caller passing --env.init_states=False).
    _orig_env_init = ll.LiberoEnv.__init__

    def _env_init(self, *a, **kw):
        kw["init_states"] = False
        _orig_env_init(self, *a, **kw)
        self.init_states = False
        self._init_states = None

    ll.LiberoEnv.__init__ = _env_init

    # The gym vector env batches the FULL observation against the Panda-shaped
    # observation_space (joint_pos/joint_vel = 7). A 6-DoF arm reports 6 joints,
    # which breaks the batched np.stack even though the policy only consumes the
    # 8-dim eef+gripper state. Pad joints to 7 so the obs matches the space; the
    # padded entries are unused by the policy.
    _orig_format = ll.LiberoEnv._format_raw_obs

    def _format(self, raw_obs):
        obs = _orig_format(self, raw_obs)
        rs = obs.get("robot_state") if isinstance(obs, dict) else None
        if rs and isinstance(rs.get("joints"), dict):
            for k in ("pos", "vel"):
                v = rs["joints"].get(k)
                if v is not None:
                    v = np.asarray(v)
                    if v.shape[-1] < 7:
                        pad = np.zeros(7 - v.shape[-1], dtype=v.dtype)
                        rs["joints"][k] = np.concatenate([v, pad])
        return obs

    ll.LiberoEnv._format_raw_obs = _format

    # Install the optional action transform on env.step so the policy's
    # Panda-frame action is remapped to the new arm before it hits OSC. Only
    # wired up when a rotation offset is actually configured (else it's identity).
    if os.environ.get("EMBODIMENT_ROT_DEG", "").strip():
        transform = make_action_transform()
        _orig_step = ll.LiberoEnv.step

        def _step(self, action):
            return _orig_step(self, transform(action))

        ll.LiberoEnv.step = _step

    _APPLIED = robot_name
    return robot_name


def apply_from_env():
    """Apply the embodiment swap configured via env vars. Returns robot name or None."""
    return apply(
        os.environ.get("EMBODIMENT", ""),
        gripper=os.environ.get("EMBODIMENT_GRIPPER", "PandaGripper"),
    )
