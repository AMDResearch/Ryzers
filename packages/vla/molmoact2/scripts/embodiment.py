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
     robots=[<Arm>] and the requested arm gripper. Native grippers are mapped
     back into Panda-style 2-dim qpos/qvel for policy compatibility.

Recorded init states are Panda-specific (different DoF), so callers must build
the env with init_states=False.

Enabled via env vars (see apply_from_env):
  EMBODIMENT=ur5e            arm to swap in (default: unset -> stock Panda)
  EMBODIMENT_GRIPPER=...     gripper to mount (default: the arm's native gripper:
                             UR5e -> Robotiq85, xArm6 -> XArmGripper). Set
                             explicitly to "PandaGripper" for cross-mount A/B.
  EMBODIMENT_EXECUTOR=absolute
                             xArm6-only: execute each policy delta as an absolute
                             eef target with feedback servo substeps, instead of
                             sending the one-shot delta directly to OSC_POSE.
                             Set to "joint_ik" for a MoveIt-style damped IK +
                             joint-trajectory executor inspired by xarm_ros2.
  EMBODIMENT_SERVO_STEPS=2    number of internal sim/control steps per policy action
                             when EMBODIMENT_EXECUTOR=absolute or joint_ik.
  EMBODIMENT_POS_SCALE=0.05   meters per unit policy position action.
  EMBODIMENT_ROT_SCALE=0.5    radians per unit policy orientation action.
  EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET=0
                             UR5e+Robotiq85 ablation only: offset reported eef
                             along the native tool axis, in meters.
  EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET=0
                             UR5e+Robotiq85 ablation only: offset wrist camera
                             along the native tool axis, in meters.
  EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND=1
                             UR5e+Robotiq85 ablation only: qpos interpolation
                             between PandaGripper home (0) and native home (1).
  EMBODIMENT_XARM6_CAMERA_MATCH=1
                             xArm6+XArmGripper only: place the wrist camera at
                             Panda's camera-in-grip-site transform.
  EMBODIMENT_IK_ORI_WEIGHT=0.35
                             orientation weight for joint_ik; lower values keep
                             elbow posture stable on 6-DoF xArm6.
  EMBODIMENT_IK_MAX_DQ=0.07   max joint update per internal joint_ik substep.
  EMBODIMENT_ROT_DEG=ax,deg  optional constant tool-frame yaw/pitch/roll offset
                             on the orientation delta, e.g. "z,90" (default: none)
"""
import os

import numpy as np

# UR5e init_qpos depends on the mounted gripper: the gripper's length sets where the
# grip site lands for a given arm pose, so each is IK-solved separately so the grip
# site starts at the Panda LIBERO home eef pose pos=[-0.1485, 0, 0.2613] (same
# orientation). PandaGripper is the cross-mount default; Robotiq85 is the UR5e's
# native gripper. Solved on libero_object/task3/seed1000 (pos err <= 0.5mm).
_UR5E_INIT_QPOS_PANDA = np.array([-0.2942, -1.6381, 1.9651, -1.8435, -1.5872, -1.8650])
_UR5E_INIT_QPOS_ROBOTIQ85 = np.array([-0.2922, -1.6619, 1.8606, -1.7151, -1.5871, -1.8630])
# xArm6 (URDF -> MJCF -> robosuite). IK-solved so each selected gripper's grip site
# starts at the Panda LIBERO home eef pose; see register_xarm6.
_XARM6_INIT_QPOS_PANDA = np.array([0.0, -0.02, -1.0592, 0.0, 1.1359, -0.0005])
# Native XArmGripper home: IK-solved against the xArm6+PandaGripper grip-site
# absolute pose (pos+orientation), because the native TCP is longer.
_XARM6_INIT_QPOS_NATIVE = np.array([0.0, -0.0473763, -1.244843, 0.0, 1.34892, -0.0005])
_PANDA_WRIST_CAM_IN_GRIPSITE = np.array([0.0, 0.05, -0.097])

# init_qpos[arm][gripper-lower] with a "_default" cross-mount (PandaGripper) entry.
# Solved per (arm, gripper) because the gripper's length sets where the grip site
# lands for a given arm pose.
_INIT_QPOS = {
    "ur5e": {"_default": _UR5E_INIT_QPOS_PANDA, "robotiq85gripper": _UR5E_INIT_QPOS_ROBOTIQ85},
    "xarm6": {"_default": _XARM6_INIT_QPOS_PANDA, "xarmgripper": _XARM6_INIT_QPOS_NATIVE},
}

_PANDA_OPEN = 0.04  # Panda per-finger fully-open joint value (closed = 0)

# Native grippers report a gripper_qpos that is not the Panda 2-dim [j1, -j1]. The
# MolmoAct2-LIBERO policy was trained on the Panda 2-dim gripper state, so we map a
# native gripper's driver joint into those units. No action remap is needed: the
# robosuite gripper controller is sign-based (current += speed * sign(action), same
# speed=0.01, "-1=open, +1=closed" for every gripper), so the same policy gripper
# command produces the same open/close behavior regardless of gripper.
#   driver_idx: index of the actuated joint within robot0_gripper_qpos
#   open:       driver joint value at fully open
#   closed:     driver joint value at fully closed
_NATIVE_GRIPPERS = {
    "robotiq85gripper": {"driver_idx": 0, "open": 0.0, "closed": 0.8},
    # Menagerie-style native model: two driver joints tied by a fixed tendon and
    # four-bar connect constraints. The first qpos remains the left driver.
    "xarmgripper": {"driver_idx": 0, "open": 0.0, "closed": 0.85},
}

# Per-arm default gripper. Native xArm6 is now the default route for
# cross-embodiment experiments; PandaGripper remains explicit fallback for A/B.
_FALLBACK_GRIPPER = "PandaGripper"
_DEFAULT_GRIPPER = {
    "ur5e": "Robotiq85Gripper",
    "xarm6": "XArmGripper",
}


def _resolve_gripper(emb, gripper):
    """Resolve which gripper to mount for an arm.

    Empty / 'default' / 'native' selects the arm's default native gripper. An
    explicit gripper name is honoured, so the Panda cross-mount stays available.
    """
    g = (gripper or "").strip()
    if not g or g.lower() in ("default", "native"):
        return _DEFAULT_GRIPPER.get(emb, _FALLBACK_GRIPPER)
    return g

# Set by apply() from the chosen arm+gripper; read by Mounted*.init_qpos and _format.
_ACTIVE_INIT_QPOS = _UR5E_INIT_QPOS_PANDA
_ACTIVE_NATIVE = None
_ACTIVE_EEF_OFFSET = 0.0
_ACTIVE_CAMERA_OFFSET = 0.0
_ACTIVE_XARM6_CAMERA_MATCH = False

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

    # init_qpos is IK-solved so the grip site starts at the Panda LIBERO home eef
    # pose pos=[-0.1485, 0, 0.2613], same orientation. This is the key
    # cross-embodiment fix: MolmoAct2 emits *delta* eef actions and localizes the
    # eef in image space, so the arm must start where the Panda would. The solved
    # pose is gripper-dependent (the gripper's length sets grip-site placement), so
    # the active value is chosen by apply() from the mounted gripper.
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
            return _ACTIVE_INIT_QPOS

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


def _xarm6_robot_xml():
    """Path to the bundled xArm6 robosuite robot.xml (URDF->MJCF->robosuite).

    Override with XARM6_ROBOT_XML; otherwise use the asset shipped next to this
    module (assets/xarm6/robot.xml), which is what gets baked into the image.
    """
    env = os.environ.get("XARM6_ROBOT_XML")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", "xarm6", "robot.xml")


def _xarm_gripper_xml():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", "xarm6", "xarm_gripper.xml")


def register_xarm_gripper():
    """Register the official UFACTORY xArm two-finger gripper with robosuite.

    robosuite's gripper registry is just a module-level dict consumed by
    gripper_factory(), so dynamic registration is sufficient and avoids patching the
    installed site-packages copy inside the container.
    """
    import robosuite.models.grippers as grippers
    from robosuite.models.grippers.gripper_model import GripperModel

    class XArmGripper(GripperModel):
        def __init__(self, idn=0):
            super().__init__(_xarm_gripper_xml(), idn=idn)

        def format_action(self, action):
            assert len(action) == 1
            self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
            return self.current_action

        @property
        def init_qpos(self):
            return np.zeros(6)

        @property
        def speed(self):
            return 0.01

        @property
        def dof(self):
            return 1

        @property
        def _important_geoms(self):
            return {
                "left_finger": ["left_finger_pad_1", "left_finger_pad_2"],
                "right_finger": ["right_finger_pad_1", "right_finger_pad_2"],
                "left_fingerpad": ["left_finger_pad_1", "left_finger_pad_2"],
                "right_fingerpad": ["right_finger_pad_1", "right_finger_pad_2"],
            }

    grippers.GRIPPER_MAPPING["XArmGripper"] = XArmGripper
    return XArmGripper


def register_xarm6():
    """Define + register LIBERO-mountable xArm6 variants.

    The xArm6 is not shipped by robosuite, so we bundle a robosuite-format
    robot.xml derived from the MIT-licensed UFACTORY xArm6 URDF (uf-gym). The
    gripper-mount body is `right_hand` at link6 (identity), matching the native
    xArm tool flange; the cross-mounted PandaGripper approaches along link6 +z.
    """
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.robots.single_arm import SingleArm
    from robosuite.robots import ROBOT_CLASS_MAPPING

    robot_xml = _xarm6_robot_xml()

    class MountedXArm6(ManipulatorModel):
        """xArm6 (6-DoF) on the LIBERO Rethink mount."""

        def __init__(self, idn=0):
            super().__init__(robot_xml, idn=idn)

        @property
        def default_mount(self):
            return "RethinkMount"

        @property
        def default_gripper(self):
            return _resolve_gripper("xarm6", None)

        @property
        def default_controller_config(self):
            return "default_ur5e"

        @property
        def init_qpos(self):
            return _ACTIVE_INIT_QPOS

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

    class OnTheGroundXArm6(MountedXArm6):
        """xArm6 (6-DoF) mounted directly on the ground (no Rethink pedestal)."""

        @property
        def default_mount(self):
            return None

    ROBOT_CLASS_MAPPING["MountedXArm6"] = SingleArm
    ROBOT_CLASS_MAPPING["OnTheGroundXArm6"] = SingleArm
    return MountedXArm6


# arm name -> (robosuite robot name passed to LIBERO, registrar)
_REGISTRARS = {
    "ur5e": ("UR5e", register_ur5e),
    "xarm6": ("XArm6", register_xarm6),
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


def _axisangle_to_mat(axisangle):
    """Rodrigues rotation matrix for a world-frame axis-angle vector."""
    v = np.asarray(axisangle, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    x, y, z = v / theta
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


_APPLIED = None


def apply(embodiment, gripper=None):
    """Patch lerobot's LIBERO env builder to use the given arm + gripper.

    gripper=None/'default'/'native' selects the arm's default native gripper; an
    explicit name is honoured.
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

    gripper = _resolve_gripper(emb, gripper)
    robot_name, registrar = _REGISTRARS[emb]
    if emb == "xarm6":
        register_xarm_gripper()

    # Pick the arm+gripper-specific IK home and (for non-Panda grippers) the gripper
    # observation remap. Read at env-build time via Mounted*.init_qpos / _format.
    global _ACTIVE_INIT_QPOS, _ACTIVE_NATIVE, _ACTIVE_EEF_OFFSET, _ACTIVE_CAMERA_OFFSET, _ACTIVE_XARM6_CAMERA_MATCH
    gkey = (gripper or "").lower()
    _ACTIVE_NATIVE = _NATIVE_GRIPPERS.get(gkey)
    arm_table = _INIT_QPOS[emb]
    _ACTIVE_INIT_QPOS = arm_table.get(gkey, arm_table["_default"])
    _ACTIVE_EEF_OFFSET = 0.0
    _ACTIVE_CAMERA_OFFSET = 0.0
    _ACTIVE_XARM6_CAMERA_MATCH = False
    if emb == "ur5e" and gkey == "robotiq85gripper":
        blend = float(os.environ.get("EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND") or "1.0")
        blend = float(np.clip(blend, 0.0, 1.0))
        _ACTIVE_INIT_QPOS = (1.0 - blend) * _UR5E_INIT_QPOS_PANDA + blend * _UR5E_INIT_QPOS_ROBOTIQ85
        _ACTIVE_EEF_OFFSET = float(os.environ.get("EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET") or "0.0")
        _ACTIVE_CAMERA_OFFSET = float(os.environ.get("EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET") or "0.0")
    if emb == "xarm6" and gkey == "xarmgripper":
        _ACTIVE_XARM6_CAMERA_MATCH = os.environ.get("EMBODIMENT_XARM6_CAMERA_MATCH", "1").strip() != "0"

    registrar()

    import lerobot.envs.libero as ll

    base_env_cls = ll.OffScreenRenderEnv

    def _osc_gain_override():
        """OSC (kp, damping_ratio) for this arm, or (None, None) to keep LIBERO's
        default (kp=150, dr=1). A non-redundant 6-DoF arm (xArm6) under-tracks the
        lateral (Y) eef goal at the default gain: robosuite's uncoupled pos/ori
        operational-inertia decoupling is a poorer approximation than for the 7-DoF
        Panda, so the effective Y bandwidth is too low and per-step OSC goals are not
        reached within control_freq's window (open-loop follower reproduced only
        ~27% of Panda's Y stroke). Raising kp restores the stroke; overdamping (dr>1)
        suppresses the resulting overshoot. The verified sweet spot kp=1000, dr=2.0
        halved follower eef error (0.13->0.09 m) and recovered ~80% of Panda's Y
        sweep. Tunable via EMBODIMENT_OSC_KP / EMBODIMENT_OSC_DR; this fits the arm
        to the LIBERO controller and intentionally deviates from real xArm dynamics.
        """
        kp = os.environ.get("EMBODIMENT_OSC_KP", "").strip()
        dr = os.environ.get("EMBODIMENT_OSC_DR", "").strip()
        if not kp and emb == "xarm6":
            kp = "1000"
            dr = dr or "2.0"
        if not kp:
            return None, None
        return float(kp), float(dr or "1.0")

    def _apply_osc_gains(rsenv):
        kp, dr = _osc_gain_override()
        if kp is None:
            return
        kd = 2.0 * np.sqrt(kp) * dr
        try:
            for robot in rsenv.robots:
                ctrl = robot.controller
                try:
                    ctrl.kp[:] = kp
                    ctrl.kd[:] = kd
                except (TypeError, AttributeError):
                    ctrl.kp = kp
                    ctrl.kd = kd
        except Exception:
            pass

    def _axis_error(target_mat, current_mat):
        import robosuite.utils.transform_utils as T

        rerr = np.asarray(target_mat, dtype=np.float64) @ np.asarray(current_mat, dtype=np.float64).T
        return T.quat2axisangle(T.mat2quat(rerr))

    def _eef_from_raw(raw_obs):
        import robosuite.utils.transform_utils as T

        pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
        if "robot0_eef_mat" in raw_obs:
            mat = np.asarray(raw_obs["robot0_eef_mat"], dtype=np.float64).reshape(3, 3)
        else:
            mat = T.quat2mat(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64).reshape(4))
        return pos, mat

    def _executor_mode():
        mode = os.environ.get("EMBODIMENT_EXECUTOR", "").strip().lower()
        if emb != "xarm6":
            return ""
        if mode in ("absolute", "abs", "servo"):
            return "absolute"
        if mode in ("joint_ik", "ik", "moveit", "trajectory", "joint_trajectory"):
            return "joint_ik"
        return ""

    def _robosuite_env(libero_env):
        # lerobot wraps the robosuite env in a GymWrapper-like object whose `.env`
        # holds raw `_get_observations()`. Some paths expose robosuite directly.
        return getattr(libero_env._env, "env", libero_env._env)

    def _absolute_servo_step(self, action):
        """Execute a MolmoAct2 delta as an absolute eef waypoint for xArm6.

        MolmoAct2 still predicts Panda-trained relative OSC actions. For xArm6,
        replaying those deltas one-shot accumulates under-tracking. This wrapper
        converts the delta into a world-frame target pose anchored at the current
        observed eef pose, then takes several feedback OSC substeps toward that
        target before the policy receives the next observation.
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.ndim != 1 or a.shape[0] < 7:
            raise ValueError(f"Expected 7-D action, got shape {np.asarray(action).shape}")

        pos_scale = float(os.environ.get("EMBODIMENT_POS_SCALE") or "0.05")
        rot_scale = float(os.environ.get("EMBODIMENT_ROT_SCALE") or "0.5")
        steps = max(1, int(os.environ.get("EMBODIMENT_SERVO_STEPS") or "2"))
        pos_tol = float(os.environ.get("EMBODIMENT_SERVO_POS_TOL") or "0.004")
        rot_tol = float(os.environ.get("EMBODIMENT_SERVO_ROT_TOL") or "0.08")

        rsenv = _robosuite_env(self)
        raw0 = rsenv._get_observations()
        start_pos, start_mat = _eef_from_raw(raw0)
        target_pos = start_pos + np.asarray(a[:3], dtype=np.float64) * pos_scale
        target_mat = _axisangle_to_mat(np.asarray(a[3:6], dtype=np.float64) * rot_scale) @ start_mat

        raw_obs = raw0
        info = {}
        reward_total = 0.0
        done = False
        is_success = False
        last_cmd = a.copy()

        for _ in range(steps):
            cur_pos, cur_mat = _eef_from_raw(raw_obs)
            pos_err = target_pos - cur_pos
            rot_err = _axis_error(target_mat, cur_mat)
            cmd = np.zeros(7, dtype=np.float32)
            cmd[:3] = np.clip(pos_err / pos_scale, -1.0, 1.0)
            cmd[3:6] = np.clip(rot_err / rot_scale, -1.0, 1.0)
            cmd[6] = a[6]
            last_cmd = cmd

            raw_obs, reward, done, info = self._env.step(cmd)
            reward_total += float(np.asarray(reward).reshape(-1)[0])
            is_success = bool(self._env.check_success())
            if done or is_success:
                break
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                break

        terminated = bool(done or is_success)
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
                "embodiment_executor": "absolute",
                "embodiment_servo_steps": steps,
                "embodiment_last_action": last_cmd,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()
        return observation, reward_total, terminated, False, info

    def _mj_model_data(sim):
        return (
            sim.model._model if hasattr(sim.model, "_model") else sim.model,
            sim.data._data if hasattr(sim.data, "_data") else sim.data,
        )

    def _tool_axis_offset_from_rsenv(rsenv, distance):
        """World-frame offset along the current native gripper tool axis."""
        if not distance:
            return None
        import mujoco

        model, data = _mj_model_data(rsenv.sim)
        grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_ft_frame")
        if grip_id < 0 or base_id < 0:
            return None
        direction = np.asarray(data.site_xpos[grip_id] - data.site_xpos[base_id], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return None
        return direction / norm * float(distance)

    def _tool_axis_offset(libero_env, distance):
        return _tool_axis_offset_from_rsenv(_robosuite_env(libero_env), distance)

    def _apply_camera_offset(rsenv):
        if not _ACTIVE_CAMERA_OFFSET:
            return
        import mujoco

        offset = _tool_axis_offset_from_rsenv(rsenv, _ACTIVE_CAMERA_OFFSET)
        if offset is None:
            return
        model, data = _mj_model_data(rsenv.sim)
        base_pos = getattr(rsenv, "_embodiment_camera_base_pos", None)
        if base_pos is None:
            base_pos = {}
            setattr(rsenv, "_embodiment_camera_base_pos", base_pos)
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if not name or "eye_in_hand" not in name:
                continue
            if cam_id not in base_pos:
                base_pos[cam_id] = np.asarray(model.cam_pos[cam_id], dtype=np.float64).copy()
            body_id = int(model.cam_bodyid[cam_id])
            body_mat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            model.cam_pos[cam_id] = base_pos[cam_id] + body_mat.T @ offset
        mujoco.mj_forward(model, data)

    def _apply_xarm6_camera_match(rsenv):
        if not _ACTIVE_XARM6_CAMERA_MATCH:
            return
        import mujoco

        model, data = _mj_model_data(rsenv.sim)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if site_id < 0:
            return
        site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        site_mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        target_world = site_pos + site_mat @ _PANDA_WRIST_CAM_IN_GRIPSITE
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if not name or "eye_in_hand" not in name:
                continue
            body_id = int(model.cam_bodyid[cam_id])
            body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
            body_mat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            model.cam_pos[cam_id] = body_mat.T @ (target_world - body_pos)
        mujoco.mj_forward(model, data)

    def _success(self, rsenv):
        for candidate in (getattr(self, "_env", None), rsenv):
            if candidate is None:
                continue
            fn = getattr(candidate, "check_success", None)
            if callable(fn):
                return bool(fn())
            fn = getattr(candidate, "_check_success", None)
            if callable(fn):
                return bool(fn())
        return False

    def _joint_ik_servo_step(self, action):
        """MoveIt/xarm_ros2-style joint trajectory executor for xArm6.

        xarm_ros2 does not drive Cartesian OSC directly. It solves IK / servo in
        joint space, applies singularity and joint-limit margins, then sends a
        FollowJointTrajectory position command. In robosuite we approximate that
        with damped least-squares IK on MuJoCo's site Jacobian, a small posture
        bias toward the xArm6 home, and per-substep joint clipping.
        """
        import mujoco

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.ndim != 1 or a.shape[0] < 7:
            raise ValueError(f"Expected 7-D action, got shape {np.asarray(action).shape}")

        rsenv = _robosuite_env(self)
        sim = rsenv.sim
        model, data = _mj_model_data(sim)

        pos_scale = float(os.environ.get("EMBODIMENT_POS_SCALE") or "0.05")
        rot_scale = float(os.environ.get("EMBODIMENT_ROT_SCALE") or "0.5")
        steps = max(1, int(os.environ.get("EMBODIMENT_SERVO_STEPS") or "2"))
        ori_w = float(os.environ.get("EMBODIMENT_IK_ORI_WEIGHT") or "0.35")
        damping = float(os.environ.get("EMBODIMENT_IK_DAMPING") or "0.08")
        max_dq = float(os.environ.get("EMBODIMENT_IK_MAX_DQ") or "0.07")
        posture_gain = float(os.environ.get("EMBODIMENT_IK_POSTURE_GAIN") or "0.15")
        limit_gain = float(os.environ.get("EMBODIMENT_IK_LIMIT_GAIN") or "0.25")
        limit_margin = float(os.environ.get("EMBODIMENT_IK_LIMIT_MARGIN") or "0.12")
        lower_sing = float(os.environ.get("EMBODIMENT_IK_SINGULAR_LOWER") or "17.0")
        hard_sing = float(os.environ.get("EMBODIMENT_IK_SINGULAR_HARD") or "30.0")

        raw0 = rsenv._get_observations()
        start_pos, start_mat = _eef_from_raw(raw0)
        target_pos = start_pos + np.asarray(a[:3], dtype=np.float64) * pos_scale
        target_mat = _axisangle_to_mat(np.asarray(a[3:6], dtype=np.float64) * rot_scale) @ start_mat

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if site_id < 0:
            raise RuntimeError("Could not find gripper0_grip_site for joint_ik executor")

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 7)]
        if any(j < 0 for j in jids):
            raise RuntimeError(f"Missing xArm6 joints for joint_ik executor: {jids}")
        qadr = np.asarray([int(model.jnt_qposadr[j]) for j in jids], dtype=np.int64)
        dadr = np.asarray([int(model.jnt_dofadr[j]) for j in jids], dtype=np.int64)
        jlo = np.asarray([model.jnt_range[j][0] for j in jids], dtype=np.float64)
        jhi = np.asarray([model.jnt_range[j][1] for j in jids], dtype=np.float64)
        q_home = np.asarray(_ACTIVE_INIT_QPOS, dtype=np.float64).reshape(-1)[:6]

        raw_obs = raw0
        info = {}
        reward_total = 0.0
        done = False
        is_success = False
        cond = 0.0

        for _ in range(steps):
            cur_pos, cur_mat = _eef_from_raw(raw_obs)
            pos_err = target_pos - cur_pos
            rot_err = _axis_error(target_mat, cur_mat)

            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, dadr], ori_w * jacr[:, dadr]])
            err = np.concatenate([pos_err, ori_w * rot_err])

            sv = np.linalg.svd(J, compute_uv=False)
            cond = float(sv[0] / max(sv[-1], 1e-9)) if len(sv) else 0.0
            if cond >= hard_sing:
                scale = 0.0
            elif cond > lower_sing:
                scale = (hard_sing - cond) / max(hard_sing - lower_sing, 1e-6)
            else:
                scale = 1.0
            err *= float(np.clip(scale, 0.0, 1.0))

            JJ = J @ J.T
            J_pinv = J.T @ np.linalg.solve(JJ + (damping ** 2) * np.eye(6), np.eye(6))
            dq = J_pinv @ err

            # Damped pseudoinverse leaves a small nullspace-like component; use it
            # to keep joint2/joint3 from flipping elbow branches when the Cartesian
            # target is near singularity.
            N = np.eye(6) - J_pinv @ J
            q = np.asarray(data.qpos[qadr], dtype=np.float64)
            dq += posture_gain * (N @ (q_home - q))

            near_lo = q < (jlo + limit_margin)
            near_hi = q > (jhi - limit_margin)
            dq[near_lo] += limit_gain * (jlo[near_lo] + limit_margin - q[near_lo]) / max(limit_margin, 1e-6)
            dq[near_hi] -= limit_gain * (q[near_hi] - (jhi[near_hi] - limit_margin)) / max(limit_margin, 1e-6)
            dq = np.clip(dq, -max_dq, max_dq)

            data.qpos[qadr] = np.clip(q + dq, jlo, jhi)
            try:
                data.qvel[dadr] = 0.0
            except Exception:
                pass
            mujoco.mj_forward(model, data)

            hold = np.zeros(7, dtype=np.float32)
            hold[6] = a[6]
            raw_obs, reward, done, info = self._env.step(hold)
            reward_total += float(np.asarray(reward).reshape(-1)[0])
            is_success = _success(self, rsenv)
            if done or is_success:
                break

        terminated = bool(done or is_success)
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
                "embodiment_executor": "joint_ik",
                "embodiment_servo_steps": steps,
                "embodiment_ik_cond": cond,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()
        return observation, reward_total, terminated, False, info

    class _EmbodiedEnv(base_env_cls):
        def __init__(self, **kwargs):
            kwargs.setdefault("robots", [robot_name])
            if gripper and gripper.lower() != "default":
                kwargs.setdefault("gripper_types", gripper)
            # Optional per-delta settling-time knob (EMBODIMENT_CONTROL_FREQ, Hz).
            # LIBERO runs OSC_POSE at control_freq=20 (0.05s/delta). A heavier/slower
            # arm (e.g. xArm6) can't fully reach each lateral OSC goal in that window,
            # so eef motion is attenuated (worst on the base-driven Y axis). Lowering
            # control_freq gives each delta more sim substeps to settle (the goal is
            # current_pose+delta, so this improves tracking, not stroke length) -- i.e.
            # the arm runs slower in sim-time but follows the policy's targets faithfully.
            cf = os.environ.get("EMBODIMENT_CONTROL_FREQ", "").strip()
            if cf:
                kwargs.setdefault("control_freq", float(cf))
            super().__init__(**kwargs)
            # The OSC controller is (re)built by robosuite at construction; patch its
            # gains in place. Re-applied after every reset (controller is rebuilt then).
            _apply_osc_gains(self)
            _apply_camera_offset(self)
            _apply_xarm6_camera_match(self)

        def reset(self, *args, **kwargs):
            out = super().reset(*args, **kwargs)
            _apply_osc_gains(self)
            _apply_camera_offset(self)
            _apply_xarm6_camera_match(self)
            return out

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
        if _ACTIVE_EEF_OFFSET and isinstance(raw_obs, dict):
            offset = _tool_axis_offset(self, _ACTIVE_EEF_OFFSET)
            if offset is not None and "robot0_eef_pos" in raw_obs:
                raw_obs = dict(raw_obs)
                raw_obs["robot0_eef_pos"] = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64) + offset
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
        # Native (non-Panda) grippers report a gripper_qpos/qvel with a different
        # joint count/units than the Panda 2-dim [j1, -j1] the policy expects. Map
        # the driver joint's closure fraction onto Panda finger units so both the
        # observation_space shape (2,) and the policy's state distribution match.
        if _ACTIVE_NATIVE and rs and isinstance(rs.get("gripper"), dict):
            di = _ACTIVE_NATIVE["driver_idx"]
            open_q = _ACTIVE_NATIVE.get("open", 0.0)
            closed = _ACTIVE_NATIVE["closed"]
            span = max(closed - open_q, 1e-6)
            g = rs["gripper"]
            q = g.get("qpos")
            if q is not None and np.asarray(q).shape[-1] != 2:
                frac = float(np.clip((np.asarray(q).reshape(-1)[di] - open_q) / span, 0.0, 1.0))
                g["qpos"] = np.array([(1.0 - frac) * _PANDA_OPEN, -(1.0 - frac) * _PANDA_OPEN])
            v = g.get("qvel")
            if v is not None and np.asarray(v).shape[-1] != 2:
                v0 = float(np.asarray(v).reshape(-1)[di])
                scale = _PANDA_OPEN / span
                g["qvel"] = np.array([-scale * v0, scale * v0])
        return obs

    ll.LiberoEnv._format_raw_obs = _format

    # Install optional execution wrappers on env.step. Rotation offsets transform
    # the policy action before execution; the absolute executor then turns the
    # transformed delta into an absolute xArm6 target and servoes internally.
    executor = _executor_mode()
    if os.environ.get("EMBODIMENT_ROT_DEG", "").strip() or executor:
        transform = make_action_transform()
        _orig_step = ll.LiberoEnv.step

        def _step(self, action):
            a = transform(action)
            if executor == "absolute":
                return _absolute_servo_step(self, a)
            if executor == "joint_ik":
                return _joint_ik_servo_step(self, a)
            return _orig_step(self, a)

        ll.LiberoEnv.step = _step

    _APPLIED = robot_name
    return robot_name


def apply_from_env():
    """Apply the embodiment swap configured via env vars. Returns robot name or None."""
    return apply(
        os.environ.get("EMBODIMENT", ""),
        gripper=os.environ.get("EMBODIMENT_GRIPPER", ""),
    )
