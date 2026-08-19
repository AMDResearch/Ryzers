# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""RoboTwin task-env wrapper: resolves task/embodiment/camera configs and drives a Base_Task.

RoboTwin uses relative paths (./task_config, ./assets, envs.<task>), so we chdir into
ROBOTWIN_ROOT.
"""
import importlib
import os

import numpy as np
import yaml

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/opt/RoboTwin")


def _ensure_cwd():
    """RoboTwin resolves ./task_config, ./assets, envs.<task> relative to its root."""
    if os.path.realpath(os.getcwd()) != os.path.realpath(ROBOTWIN_ROOT):
        os.chdir(ROBOTWIN_ROOT)


def _instantiate_task(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _unstable_error():
    """RoboTwin's UnStableError (unsettled objects at a seed); lazy import, sentinel fallback."""
    try:
        _ensure_cwd()
        from envs.utils.create_actor import UnStableError  # noqa: WPS433
        return UnStableError
    except Exception:  # noqa: BLE001
        class _NeverRaised(Exception):
            pass
        return _NeverRaised


def list_tasks():
    """Best-effort list of RoboTwin task names (envs/<task>.py) for the env-picker dropdown."""
    _ensure_cwd()
    envs_dir = os.path.join(ROBOTWIN_ROOT, "envs")
    skip = {"__init__", "utils", "camera"}
    names = []
    try:
        for fn in sorted(os.listdir(envs_dir)):
            if not fn.endswith(".py"):
                continue
            name = fn[:-3]
            if name.startswith("_") or name in skip:
                continue
            names.append(name)
    except OSError:
        pass
    return names


def load_task_args(task_name, task_config):
    """Build the RoboTwin `args` dict for a task (mirrors eval_policy.py / collect_data.py)."""
    _ensure_cwd()
    from envs import CONFIGS_PATH  # noqa: WPS433

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"no embodiment file for {name}")
        return robot_file

    def embodiment_cfg(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = embodiment_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_cfg(args["right_robot_file"])
    return args


class RoboTwinScene:
    """A single configured RoboTwin task episode (setup, obs, step, render, success)."""

    @classmethod
    def build_stable(cls, task_name, task_config="demo_clean", seed=0, video_dir=None,
                     max_attempts=15):
        """Build a scene, advancing the seed past ones that raise UnStableError (unsettled objects).

        Mirrors RoboTwin's eval_policy.py seed-skipping; .seed reflects the seed that succeeded.
        """
        unstable = _unstable_error()
        attempts = max(1, int(max_attempts))
        cur = int(seed)
        last_err = None
        for i in range(attempts):
            try:
                return cls(task_name, task_config=task_config, seed=cur, video_dir=video_dir)
            except unstable as e:
                last_err = e
                print(f"[scene] {task_name}: unstable seed {cur} "
                      f"({i + 1}/{attempts}), advancing seed", flush=True)
                cur += 1
        raise last_err if last_err is not None else RuntimeError(
            f"scene build failed for {task_name}")

    def __init__(self, task_name, task_config="demo_clean", seed=0, video_dir=None):
        self.task_name = task_name
        self.task_config = task_config
        self.seed = int(seed)
        self.args = load_task_args(task_name, task_config)
        self.args["policy_name"] = "sim_harness"
        self.env = _instantiate_task(task_name)
        self._video_dir = video_dir
        self.instruction = task_name.replace("_", " ")
        self._setup()

    def _setup(self):
        _ensure_cwd()
        setup_args = dict(self.args)
        setup_args["eval_mode"] = True
        setup_args["render_freq"] = 0
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
            setup_args["eval_video_save_dir"] = self._video_dir
        self.env.setup_demo(now_ep_num=0, seed=self.seed, is_test=True, **setup_args)
        self.env.set_instruction(instruction=self.default_instruction())

    def default_instruction(self):
        """Best-effort human-readable instruction for the task (falls back to the name)."""
        try:
            import json
            p = os.path.join("description", "task_instruction", f"{self.task_name}.json")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("full_description", self.task_name.replace("_", " "))
        except Exception:  # noqa: BLE001
            return self.task_name.replace("_", " ")

    def set_instruction(self, instruction):
        self.instruction = instruction
        self.env.set_instruction(instruction=instruction)

    def scene_objects(self):
        """Best-effort list of manipulable objects in the scene (cosmetic viewport panel)."""
        env = self.env
        names = []
        try:
            reg = getattr(env, "actor_name_dic", None)
            if isinstance(reg, dict) and reg:
                names = list(reg.keys())
        except Exception:  # noqa: BLE001
            names = []
        if not names:
            drop = {"robot", "left_robot", "right_robot", "table", "wall", "ground",
                    "plane", "arena"}
            try:
                for key, val in vars(env).items():
                    if key.startswith("_") or key in drop:
                        continue
                    if hasattr(val, "get_pose") and hasattr(val, "get_name"):
                        names.append(key)
            except Exception:  # noqa: BLE001
                pass
        seen, out = set(), []
        for n in names:
            label = str(n).replace("_", " ").strip()
            if label and label not in seen:
                seen.add(label)
                out.append(label)
        return out

    @property
    def step_lim(self):
        return int(getattr(self.env, "step_lim", 1000) or 1000)

    def get_obs(self):
        return self.env.get_obs()

    def state_vector(self, obs=None):
        obs = obs or self.env.now_obs
        return np.asarray(obs["joint_action"]["vector"], dtype=np.float32)

    def ee_state_vector(self, obs=None):
        """Current absolute EE pose for take_action(action_type="ee"):
        [left_xyz(3), left_quat_wxyz(4), left_grip(1), right_xyz(3), right_quat_wxyz(4), right_grip(1)] -> [16].
        EE counterpart of state_vector; RT demo re-issues it to HOLD while an EE policy plans.
        """
        obs = obs or self.env.now_obs
        ep = obs["endpose"]
        left = np.concatenate([np.asarray(ep["left_endpose"], dtype=np.float64),
                               [float(ep["left_gripper"])]])
        right = np.concatenate([np.asarray(ep["right_endpose"], dtype=np.float64),
                                [float(ep["right_gripper"])]])
        return np.concatenate([left, right])

    def take_action(self, action, action_type="qpos"):
        # qpos rows -> float32 (RoboTwin's data); ee rows -> float64 (planner IK precision).
        dtype = np.float64 if action_type == "ee" else np.float32
        self.env.take_action(np.asarray(action, dtype=dtype), action_type=action_type)

    def eval_frame(self):
        return np.asarray(self.env._get_eval_video_frame())[:, :, :3]

    @property
    def success(self):
        return bool(getattr(self.env, "eval_success", False))

    @property
    def take_action_cnt(self):
        return int(getattr(self.env, "take_action_cnt", 0))

    def close(self):
        try:
            self.env.close_env()
        except Exception:  # noqa: BLE001
            pass
