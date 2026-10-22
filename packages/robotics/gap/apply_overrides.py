#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Apply build-time overrides to the pinned GaP source."""

from pathlib import Path
import sys


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/ryzers/graph-as-policy")


def replace_once(relative_path: str, old: str, new: str) -> None:
    path = ROOT / relative_path
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{relative_path}: expected one match, found {count}")
    path.write_text(source.replace(old, new, 1))


def replace_between(
    relative_path: str, start: str, end: str, replacement: str
) -> None:
    path = ROOT / relative_path
    source = path.read_text()
    if source.count(start) != 1:
        raise RuntimeError(
            f"{relative_path}: expected one start marker, found {source.count(start)}"
        )
    if source.count(end) != 1:
        raise RuntimeError(
            f"{relative_path}: expected one end marker, found {source.count(end)}"
        )
    start_index = source.find(start)
    end_index = source.find(end, start_index)
    if end_index < start_index:
        raise RuntimeError(f"{relative_path}: end marker precedes start marker")
    path.write_text(source[:start_index] + replacement + source[end_index:])


replace_once(
    "pyproject.toml",
    '''# tool bundle); sim deps live in the [libero] extra. The connector's default
# motion-planning backend is cuRobo, shipped by the open-robot-skills curobo
# tool bundle; PyRoKi (CPU-JAX, in-process) is available as an opt-in
# fallback via the [pyroki] extra for no-GPU machines / 6-DOF arms.
''',
    '''# tool bundle); sim deps live in the [libero] extra. This image installs
# PyRoKI separately and uses it as the connector's motion backend.
''',
)
replace_between(
    "pyproject.toml",
    "  # Robotics: URDF + robot descriptions + cuRobo",
    '  "robot_descriptions>=1.16.0",\n',
    "  # Robotics models and geometry.\n",
)
replace_between(
    "pyproject.toml",
    '  "robot_descriptions>=1.16.0",\n',
    '  "trimesh",\n',
    '''  "robot_descriptions>=1.16.0",
  "yourdfpy>=0.0.56",
''',
)
replace_once(
    "pyproject.toml",
    '''# In-process PyRoKi IK fallback (CPU-JAX). Opt-in only — the default
# connector backend is cuRobo (open-robot-skills[curobo]). Pick this extra
# when running without a GPU, or when targeting 6-DOF YAM which cuRobo
# doesn't yet bundle. Wire the backend at construction time:
#   Connector(env, config, ik=PyRokiBackend(...))
''',
    '''# In-process PyRoKi IK (CPU JAX).
''',
)
replace_between(
    "pyproject.toml",
    "# CuRobo compiles CUDA extensions at install time against the environment's\n",
    "[build-system]\n",
    "",
)

replace_once(
    "gap/connector/core.py",
    '''        ik: Optional pre-built IK backend. Defaults to a
            :class:`gap.connector.ik.CuRoboBackend` built from *config* (GPU,
            v0.8 MotionPlanner — linear motion with IK fallback via
            ``plan_to_pose``). Pass an explicit
            :class:`gap.connector.ik.PyRokiBackend` to opt into the in-process
            CPU-JAX path (e.g. for YAM/6-DOF or no-GPU machines).
''',
    '''        ik: Optional pre-built IK backend. Defaults to
            :class:`gap.connector.ik.PyRokiBackend`, using CPU JAX.
''',
)
replace_once(
    "gap/connector/core.py",
    "        # Pluggable IK backend (cuRobo by default; lazy). Pass ``ik=`` to\n"
    "        # opt into PyRoKi or any other backend implementing the same surface.\n",
    "        # Pluggable IK backend, initialized lazily.\n",
)
replace_once(
    "gap/connector/core.py",
    "            from gap.connector.ik import CuRoboBackend\n",
    "            from gap.connector.ik import PyRokiBackend\n",
)
replace_once(
    "gap/connector/core.py",
    "                self._ik = CuRoboBackend(\n",
    "                self._ik = PyRokiBackend(\n",
)
replace_once(
    "gap/connector/core.py",
    '                    "Connector.ik: using cuRobo backend '
    '(default; arm_dof=%d)",\n',
    '                    "Connector.ik: using PyRoKi backend (arm_dof=%d)",\n',
)
replace_once(
    "gap/connector/core.py",
    '''            except NotImplementedError as e:
                # cuRobo doesn't bundle this arm (e.g. YAM 6-DOF). Surface a
                # clear pointer rather than silently auto-falling-back: per
                # the design decision, PyRoKi is opt-in only.
                raise RuntimeError(
                    f"Default IK backend (cuRobo) cannot serve this config: {e} "
                    f"Construct the connector with ik=PyRokiBackend(...) to "
                    f"opt into the in-process PyRoKi path."
                ) from e
''',
    '''            except NotImplementedError as e:
                raise RuntimeError(
                    f"Default IK backend cannot serve this config: {e}"
                ) from e
''',
)
replace_once(
    "gap/connector/core.py",
    '''        # Pass the current joint config as the linear seed — cuRobo's
        # MotionPlanner needs the actual start_config for FK; PyRoKi
        # accepts and ignores it.
''',
    '''        # Anchor the trajectory to the measured joint configuration.
''',
)
replace_once(
    "gap/connector/core.py",
    '''        """Move along a straight Cartesian line to ``pose``.

        The caller is responsible for providing target poses already in the
        IK link frame (panda_hand for Franka, link_6 for YAM) — matching the
        source's GoToPoseCartesian contract.
        """
''',
    '''        """Move along a straight Cartesian line to ``pose``.

        ``pose`` uses the connector's commanded tool frame.
        """
''',
)

replace_once(
    "gap/connector/ik.py",
    '''    jump_threshold: float = 0.5,
    ik_refinement_iters: int = 15,
) -> np.ndarray:
''',
    '''    jump_threshold: float = 0.5,
    ik_refinement_iters: int = 1,
    seed_cfg: np.ndarray | None = None,
) -> np.ndarray:
''',
)
replace_once(
    "gap/connector/ik.py",
    '''    # Linear interpolation for positions
''',
    '''    if ik_refinement_iters < 1:
        raise ValueError("ik_refinement_iters must be positive")

    # Linear interpolation for positions
''',
)
replace_once(
    "gap/connector/ik.py",
    '''    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations, strict=False)):
''',
    '''    trajectory = []
    prev_cfg = None
    jump_warnings = []
    start_index = 0

    if seed_cfg is not None:
        seed = np.asarray(seed_cfg, dtype=np.float64).reshape(-1)
        joint_count = int(robot.joints.num_actuated_joints)
        if seed.size < joint_count:
            seed = np.pad(seed, (0, joint_count - seed.size))
        elif seed.size > joint_count:
            seed = seed[:joint_count]
        trajectory.append(seed.copy())
        prev_cfg = seed
        start_index = 1

    waypoints = zip(
        positions[start_index:], orientations[start_index:], strict=False
    )
    for i, (pos, wxyz) in enumerate(waypoints, start=start_index):
''',
)
replace_once(
    "gap/connector/ik.py",
    '''        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(ik_refinement_iters):
                cfg = solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
''',
    '''        if use_prev_cfg and prev_cfg is not None:
            cfg = np.asarray(
                solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                    initial_cfg=prev_cfg,
                )
            )
''',
)
replace_once(
    "gap/connector/ik.py",
    "        del seed_joints  # unused; PyRoKi seeds per-waypoint internally\n",
    "",
)
replace_once(
    "gap/connector/ik.py",
    '''        Both ``start_world_pose`` and ``end_world_pose`` are TCP-frame poses:
        the backend converts them to the IK link frame using ``self._tcp_offset``
        before per-waypoint IK, matching ``solve_ik``'s contract. ``seed_joints``
        is accepted for interface parity with CuRoboBackend (PyRoKi's per-
        waypoint solver chains its own previous-config seed).
''',
    '''        ``end_world_pose`` is a commanded TCP pose. When measured
        ``seed_joints`` are available, their forward kinematics define the
        link-frame start and exact first waypoint.
''',
)
replace_once(
    "gap/connector/ik.py",
    '''        else:
            # Panda: convert both endpoints from TCP frame to panda_hand link
            # frame using the configured TCP offset / rotation. Matches
            # _solve_panda's per-call shift so solve_ik and plan_linear share
            # one user-facing convention (the caller passes TCP poses).
            start_pose = self._panda_tcp_to_link(start_world_pose)
            end_pose = self._panda_tcp_to_link(end_world_pose)
''',
    '''        else:
            if seed_joints is not None:
                seed = np.asarray(seed_joints, dtype=np.float64).reshape(-1)
                joint_count = int(self.robot.joints.num_actuated_joints)
                if seed.size < joint_count:
                    seed = np.pad(seed, (0, joint_count - seed.size))
                elif seed.size > joint_count:
                    seed = seed[:joint_count]
                fk = np.asarray(self.robot.forward_kinematics(seed))
                link_index = self.robot.links.names.index(self._target_link)
                link_pose = fk[link_index]
                start_pose = make_pose(link_pose[4:7], link_pose[:4])
            else:
                start_pose = self._panda_tcp_to_link(start_world_pose)
            end_pose = self._panda_tcp_to_link(end_world_pose)
''',
)
replace_once(
    "gap/connector/ik.py",
    '''                jump_threshold=jump_threshold,
                ik_refinement_iters=ik_refinement_iters,
            )
''',
    '''                jump_threshold=jump_threshold,
                ik_refinement_iters=ik_refinement_iters,
                seed_cfg=(
                    None
                    if seed_joints is None
                    else np.asarray(
                        list(reversed(seed_joints))
                        if self._is_yam()
                        else seed_joints,
                        dtype=np.float64,
                    )
                ),
            )
''',
)

replace_once(
    "gap/connector/ik.py",
    '''        num_waypoints: int = 40,
        ik_refinement_iters: int = 40,
        jump_threshold: float = 0.5,
    ) -> Trajectory | None:
        """Plan a Cartesian straight-line trajectory in TCP frame.
''',
    '''        num_waypoints: int = 40,
        ik_refinement_iters: int = 1,
        jump_threshold: float = 0.5,
    ) -> Trajectory | None:
        """Plan a Cartesian straight-line trajectory in TCP frame.
''',
)
replace_once(
    "gap/connector/ik.py",
    '''    num_waypoints: int = 40,
    ik_refinement_iters: int = 40,
    robot_urdf_path: str | None = None,
''',
    '''    num_waypoints: int = 40,
    ik_refinement_iters: int = 1,
    robot_urdf_path: str | None = None,
''',
)
replace_once(
    "gap/connector/ik.py",
    '''        Waypoints are in backend-native joint order; see
        ``trajectory_needs_joint_reverse``. Defaults match the source's
        ``PyRoKIPlanRequest(num_waypoints=40, ik_refinement_iters=40)``.
''',
    '''        Waypoints are in backend-native joint order; see
        ``trajectory_needs_joint_reverse``. One converged solve per waypoint
        avoids branch drift while retaining the source-compatible argument.
''',
)

replace_once(
    "gap-core/src/gap_core/rpc/client.py",
    '''        call_timeout_s: float | None = None,
        use_uv: bool = True,
    ) -> None:
        self.bundle_name = bundle_name
''',
    '''        call_timeout_s: float | None = None,
        use_uv: bool | None = None,
    ) -> None:
        if use_uv is None:
            use_uv = os.environ.get("GAP_TOOL_BUNDLE_USE_UV", "1") != "0"
        self.bundle_name = bundle_name
''',
)
replace_once(
    "gap/skills/capability.py",
    "import subprocess\n",
    "import subprocess\nimport sys\n",
)

new_probe = '''def probe_gpu(*, timeout: float = 5.0) -> ProbeResult:
    """Report GPUs visible to the installed PyTorch runtime."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "assert torch.cuda.is_available(), "
                    "'PyTorch reports no visible GPU'; "
                    "backend = ('ROCm ' + str(torch.version.hip) "
                    "if torch.version.hip else "
                    "'GPU runtime ' + str(torch.version.cuda)); "
                    "print(backend + ': ' + '; '.join("
                    "torch.cuda.get_device_name(i) "
                    "for i in range(torch.cuda.device_count())))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return ProbeResult(
            "missing",
            "Python executable not found",
            fix_hint="run GaP in its Python environment",
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            "error", f"PyTorch GPU probe timed out after {timeout:g}s"
        )
    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout).strip().splitlines()
        return ProbeResult(
            "missing",
            snippet[0]
            if snippet
            else f"PyTorch GPU probe exited {proc.returncode}",
        )
    gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not gpus:
        return ProbeResult("missing", "PyTorch reports no GPUs")
    return ProbeResult("ok", "; ".join(gpus))


'''

replace_between(
    "gap/skills/capability.py",
    "def probe_gpu(*, timeout: float = 5.0) -> ProbeResult:\n",
    "def _adc_path() -> Path:\n",
    new_probe,
)
replace_between(
    "gap/skills/capability.py",
    "Everything here is fast and offline:",
    "No model loads, no downloads, no network.\n",
    '''Everything here is fast and offline: static metadata, a PyTorch subprocess,
environment lookups, import probes, and optional cache checks.
''',
)

print(f"Applied GaP overrides to {ROOT}")
