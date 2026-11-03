#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Adapt the pinned open-robot-skills checkout to this image.

Without arguments, port the cuRobo-dependent skills to connector motion and
move the bundles this image cannot run beside the checkout. With ``--check``,
fail when any registered skill names a tool that nothing registers, or passes
a literal argument that tool's input schema does not accept.
"""

import argparse
import ast
from pathlib import Path
import re
import shutil


PORT_NOTE = "AMD/ROCm port: no collision checking."

#: Tool bundles and skills that need tools this image does not provide.
UNSUPPORTED = [
    "tools/curobo",
    "skills/anchoring-a-free-end",
    "skills/executing-feature-mating",
    "skills/executing-held-object-motion",
    "skills/planning-held-object-motion",
    "skills/plugging-a-cable-end",
    "skills/reconstructing-collision-worlds",
    "skills/selecting-reachable-grasp",
    "skills/tipping-over-a-surface-edge",
    # LIBERO has no scene that can exercise a held-object reobservation.
    "skills/registering-held-objects",
    # LIBERO has no cable-routing scene.
    "skills/seating-a-cable-crossing",
    "skills/transporting-objects/scripts/waypoint_move_carve.py",
    "skills/grasping-short-axis/scripts/plan_grasp_motion.py",
    "skills/grasping-with-planner/scripts/plan_grasp.py",
]


def replace_between(path: Path, start: str, end: str, replacement: str) -> None:
    source = path.read_text()
    if source.count(start) != 1:
        raise RuntimeError(
            f"{path}: expected one start marker, found {source.count(start)}"
        )
    if source.count(end) != 1:
        raise RuntimeError(
            f"{path}: expected one end marker, found {source.count(end)}"
        )
    start_index = source.find(start)
    end_index = source.find(end, start_index)
    if end_index < start_index:
        raise RuntimeError(f"{path}: end marker precedes start marker")
    path.write_text(source[:start_index] + replacement + source[end_index:])
    _note_python(path)


_patched_python: list[Path] = []


def _note_python(path: Path) -> None:
    if path.suffix == ".py" and path not in _patched_python:
        _patched_python.append(path)


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    path.write_text(source.replace(old, new, 1))
    _note_python(path)


def replace_to_end(path: Path, start: str, expected: str, replacement: str) -> None:
    """Replace from ``start`` through EOF when that span is exactly ``expected``."""
    source = path.read_text()
    if source.count(start) != 1:
        raise RuntimeError(
            f"{path}: expected one start marker, found {source.count(start)}"
        )
    index = source.find(start)
    actual = source[index:]
    if actual != expected:
        raise RuntimeError(
            f"{path}: section from {start!r} through end of file does not "
            f"match the expected original "
            f"({len(actual)} bytes, expected {len(expected)})"
        )
    path.write_text(source[:index] + replacement)
    _note_python(path)


def compile_patched_python() -> None:
    """Byte-compile every Python file this run rewrote."""
    import py_compile

    for path in _patched_python:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            raise RuntimeError(f"{path}: py_compile failed:\n{exc}") from exc


def add_port_note(skill: Path, detail: str) -> None:
    path = skill / "SKILL.md"
    heading = f"\n# {skill.name}\n"
    replace_once(path, heading, f"{heading}\n> {PORT_NOTE} {detail}\n")


def reject_curobo(path: Path, allowed: tuple[str, ...] = ()) -> None:
    leftover = [
        line for line in path.read_text().splitlines()
        if "curobo" in line.lower()
        and not line.startswith(f"> {PORT_NOTE}")
        and not any(a in line for a in allowed)
    ]
    if leftover:
        raise RuntimeError(f"{path} still mentions cuRobo:\n" + "\n".join(leftover))


# ---------------------------------------------------------------------------
# grasping-short-axis
# ---------------------------------------------------------------------------


def port_grasping_short_axis(root: Path) -> None:
    skill_dir = root / "skills/grasping-short-axis"
    plan = skill_dir / "scripts/plan_grasp.py"
    replace_between(
        plan,
        '"""Plan a collision-free CuRobo trajectory to one of several candidate '
        'grasp poses.\n',
        "class Output(TypedDict):\n",
        '''"""Descend in a straight line onto the first reachable candidate grasp pose.

``approach_above`` leaves the hand over the target at the grasp rotation.
This node tries each candidate in order with ``robot.go_to_pose_cartesian``
and stops at the first one that succeeds. It returns a trajectory holding
only the joint positions reached, so ``execute`` and ``finalize`` keep the
arm where it is. Raises :class:`RuntimeError` when no candidate is
reachable so the workflow routes to ``on_error``.

The descent is not collision-checked.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.types import Observation, Se3Pose, Trajectory, WorldConfig

logger = logging.getLogger(__name__)


''',
    )
    replace_between(
        plan,
        "    # NOTE: topk/retries/num_ik_seeds were inflated",
        "    topk: int = 8,\n",
        '''    # world_config, target_name, robot_file, the approach settings, retries
    # and num_ik_seeds are accepted so existing graphs keep their wiring; the
    # linear descent does not use them.
''',
    )
    replace_to_end(
        plan,
        '    start_joints = observation["arms"][0]["joint_state"]\n',
        '''    start_joints = observation["arms"][0]["joint_state"]

    # Auto-wrap a bare Se3Pose into a one-element list.
    if isinstance(grasp_poses, dict) and "position" in grasp_poses and "rotation" in grasp_poses:
        grasp_poses = [grasp_poses]
    poses = list(grasp_poses)[: int(topk)]
    if not poses:
        raise PlanningFailed("plan_grasp: empty input grasp_poses")

    # PER-POSE planning. A single goalset call over all candidates
    # (``plan_grasp_goalset``) jointly optimises pick + approach + grasp
    # for the whole set in one solver problem — an all-or-nothing solve
    # that fails fast (success=False, empty status) whenever the set is
    # hard, even when individual poses are independently plannable.
    # Looping ``curobo.plan_to_grasp_poses`` with a SINGLETON pose is far
    # more robust: N independent solves, return the first that plans.
    # General planning-robustness fix — no privileged info, no batch-sim
    # infra; first-feasible is deterministic and sufficient here.
    _ = grasp_approach_tstep_fraction
    _ = grasp_approach_offset
    n_attempt = len(poses)
    # cuRobo's per-pose IK uses random seeds (NN + random,
    # ``solve_batch(return_seeds=1)``), so a *feasible* pose can fail one
    # solve and succeed on a re-roll: the same workflow/snapshot gave
    # 1/25 plannable on one run and 0/25 on the next. Retry each
    # candidate up to ``retries`` times (fresh IK seeds each call) and
    # raise ``num_ik_seeds`` so a single solve is itself more thorough.
    # General planning-robustness fix for stochastically-marginal grasps.
    for i, pose in enumerate(poses):
        kwargs = dict(
            world_config=world_config,
            start_joint_position=start_joints,
            grasp_poses=[pose],
            grasp_pose_is_fingertip=True,
            use_world_collision=True,
            use_cuda_graph=False,
            robot_collision_sphere_buffer=-0.01,
            collision_activation_distance=0.005,
            ignore_obstacle_names=[target_name],
            use_grasp_approach=False,
            num_ik_seeds=int(num_ik_seeds),
        )
        if robot_file:
            kwargs["robot_file"] = robot_file
        for attempt in range(max(1, int(retries))):
            try:
                plan = ctx.tool("curobo.plan_to_grasp_poses", **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[plan_grasp] candidate %d/%d attempt %d errored "
                    "(%s); retrying/next",
                    i, n_attempt, attempt, exc,
                )
                continue
            if plan.get("success", False):
                logger.info(
                    "[plan_grasp] candidate %d/%d planned successfully "
                    "(attempt %d; per-pose over %d candidates, "
                    "retries=%d, num_ik_seeds=%d)",
                    i, n_attempt, attempt, n_attempt,
                    retries, num_ik_seeds,
                )
                return {"trajectory": plan["trajectory"]}

    raise PlanningFailed(
        f"plan_grasp: 0/{n_attempt} candidate grasps had a feasible CuRobo "
        f"plan over {retries} retries each (per-pose plan_to_grasp_poses, "
        f"num_ik_seeds={num_ik_seeds}, target_name={target_name!r}). "
        f"The candidate grasp poses are not reachable + collision-free "
        f"from the start configuration. This is a grasp-generation/"
        f"reachability problem, NOT perception (the target OBB is "
        f"correct). Levers: a different grasp generator or approach axis, "
        f"a pre-grasp repositioning move, or relaxed grasp orientation."
    )
''',
        '''    del world_config, observation, target_name, robot_file
    del grasp_approach_offset, grasp_approach_tstep_fraction, retries, num_ik_seeds

    # Auto-wrap a bare Se3Pose into a one-element list.
    if isinstance(grasp_poses, dict) and "position" in grasp_poses and "rotation" in grasp_poses:
        grasp_poses = [grasp_poses]
    poses = list(grasp_poses)[: int(topk)]
    if not poses:
        raise RuntimeError("plan_grasp: empty input grasp_poses")

    errors: list[str] = []
    for i, pose in enumerate(poses):
        try:
            ctx.tool("robot.go_to_pose_cartesian", pose=pose)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[plan_grasp] candidate %d/%d failed: %s", i + 1, len(poses), exc
            )
            errors.append(f"candidate {i + 1}: {exc}")
            continue
        logger.info("[plan_grasp] candidate %d/%d reached", i + 1, len(poses))
        joints = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
        positions = [float(v) for v in joints["positions"]]
        return {"trajectory": {"waypoints": [{"positions": positions}]}}

    raise RuntimeError(
        f"plan_grasp: none of the {len(poses)} candidate grasp poses was "
        "reachable by a straight-line descent with robot.go_to_pose_cartesian "
        "from the approach pose: " + "; ".join(errors)
    )
''',
    )

    skill = skill_dir / "SKILL.md"
    replace_once(
        skill,
        "description: Deterministic short-axis-aligned grasp with CuRobo. The grasp\n",
        "description: Deterministic short-axis-aligned grasp with a straight-line\n"
        "  descent. The grasp\n",
    )
    replace_once(
        skill,
        "tags: [grasping, manipulation, collision-aware, curobo, short-axis, handle]",
        "tags: [grasping, manipulation, cartesian, short-axis, handle]",
    )
    replace_once(
        skill,
        '''    - curobo.plan_to_grasp_poses
    - curobo.plan_grasp_motion
    - curobo.plan_directed_linear
''',
        "",
    )
    replace_once(
        skill,
        "    failed: Any failure during the grasp attempt — collision-aware planning "
        "failure or trajectory execution error",
        "    failed: Any failure during the grasp attempt — no candidate reachable "
        "by the straight-line descent, or a motion error",
    )
    replace_once(
        skill,
        "      `plan_grasp` auto-wraps the single pose into a one-element goalset.\n",
        "      `plan_grasp` auto-wraps the single pose into a one-element list.\n",
    )
    replace_between(
        skill,
        "      When building the collision world, ALWAYS pass\n",
        "    - >\n      ALWAYS insert a `finalize` state",
        '''      Wire `build_world` with `target_mask = Ref("in.target_mask")`,
      `target_obb = Ref("in.target_obb")` and a `target_name`, and pass
      `world_config` and the same `target_name` to `plan`. `plan_grasp.py`
      keeps these inputs so the graph wiring stays valid; its straight-line
      descent does not use them and is not collision-checked.
    - >
      ALWAYS include the `approach` state between `offset_from_base` and
      `observe`. `plan_grasp` descends in a straight line from wherever
      `approach` leaves the hand, so skipping `approach` starts the descent
      from a wrong wrist rotation and the grasp slips. `approach_above.py`
      lifts to a safe height, translates above the target, then rotates in
      place to the grasp orientation; all three legs route through
      `robot.go_to_pose_cartesian` and the connector's IK backend.
''',
    )
    replace_between(
        skill,
        "      is the #1 silent grasp-failure cause: `execute` runs\n",
        "    - >\n      Do NOT add a `verify_gripper_grasp` node",
        '''      keeps the gripper close from racing the arm: `plan` returns a
      one-waypoint trajectory at the joints it reached, `execute` and
      `finalize` hold that configuration, and `finalize` issues one
      `robot.move_to_joints` with full convergence so the arm is quiescent
      before `close`. If `robot.close_gripper` fires while the arm is still
      moving, the simulator can drop the close action and `target_held`
      fails with no error. Never wire `execute → close` directly.
''',
    )
    replace_between(
        skill,
        "    - >\n      OPTIONAL escalation lever — `plan_grasp_motion`\n",
        "  canonical_scripts:\n",
        "",
    )
    replace_once(skill, "    - plan_grasp_motion: scripts/plan_grasp_motion.py\n", "")
    replace_once(
        skill,
        '''    - title: Why disable use_grasp_approach
      path: references/design_grasp_curobo.md
''',
        "",
    )
    replace_between(
        skill,
        "Deterministic, geometry-locked grasping with CuRobo. The grasp pose is\n",
        "## When to use\n",
        '''Deterministic, geometry-locked grasping with a straight-line descent. The
grasp pose is computed directly from the target OBB — the gripper descends
along world −Z with its finger-opening axis snapped to the OBB's *shorter*
horizontal axis, so the jaws close across the narrow dimension of the bar.
An optional node then slides the grasp outward along the handle, clear of a
heavier attached body. The subgraph moves the hand above the target at the
grasp rotation, descends in a straight line with
`robot.go_to_pose_cartesian`, holds the reached joints, closes the gripper,
and outputs the EE pose at grasp.

## Install

This skill depends on the **geometry** tool bundle (for `build_world`):

```bash
uv sync --extra geometry   # (pip: pip install -e "open-robot-skills[geometry]")
```

''',
    )
    replace_once(
        skill,
        "- `curobo` not deployed.\n",
        "- The straight-line descent onto the target would pass through clutter;\n"
        "  it is not collision-checked.\n",
    )
    replace_once(
        skill,
        '''   `target_name = "target"`. `plan_grasp.py` auto-wraps the single bare
   Se3Pose into a one-element list. All four inputs are required.
''',
        '''   `target_name = "target"`. `plan_grasp.py` auto-wraps the single bare
   Se3Pose into a one-element list. All four inputs are required. It tries
   each candidate with `robot.go_to_pose_cartesian`, stops at the first
   that succeeds, and returns a one-waypoint trajectory at the reached
   joints; it raises `RuntimeError` when no candidate is reachable.
''',
    )
    replace_once(
        skill,
        "| `failed` | Any grasp-attempt failure: collision-aware planning failure or "
        "trajectory execution error",
        "| `failed` | Any grasp-attempt failure: no candidate reachable by the "
        "straight-line descent, or a motion error",
    )
    add_port_note(
        skill_dir,
        "`plan_grasp` replaces the cuRobo goalset plan with a straight-line "
        "descent through `robot.go_to_pose_cartesian`.",
    )
    reject_curobo(skill)
    reject_curobo(plan)


# ---------------------------------------------------------------------------
# grasping-with-planner
# ---------------------------------------------------------------------------


def port_grasping_with_planner(root: Path) -> None:
    skill_dir = root / "skills/grasping-with-planner"
    script = skill_dir / "scripts/grasp_descend_linear.py"
    replace_between(
        script,
        '"""Grasp via a fast axis-locked linear descend, with a collision-aware fallback.\n',
        "import logging\n",
        '''"""Grasp by rising, moving over the object, and descending vertically.

If the first top-down pose is unreachable, try each candidate pose in turn.
"""

''',
    )
    replace_once(
        script,
        '''    """Straight-line cartesian move. Raises on an infeasible solve — the
    caller's planner fallback handles that, so we do NOT silently retry here."""
''',
        '''    """Straight-line Cartesian move for one grasp segment."""
''',
    )
    replace_between(
        script,
        "def _descend_linear(",
        "def _wrist_world_cloud(",
        '''def _descend_linear(ctx: NodeContext, target_z: float, from_z: float,
                    rotation: dict | None = None) -> None:
    if float(from_z) - float(target_z) <= 0.002:
        return
    ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, ee["x"], ee["y"], target_z, rotation)


''',
    )
    replace_between(
        script,
        "def _planner_grasp(",
        "def run(",
        '''def _planner_grasp(
    ctx: NodeContext,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    hover_z: float,
    topk: int = 8,
) -> None:
    errors: list[str] = []
    candidates = list(candidate_poses)[: int(topk)]
    for index, pose in enumerate(candidates):
        try:
            _cartesian_grasp(ctx, pose, target_obb, hover_z)
            logger.info(
                "[grasp] candidate %d/%d succeeded",
                index + 1,
                len(candidates),
            )
            return
        except Exception as exc:
            errors.append(f"{index + 1}:{exc}")
    raise RuntimeError("grasp candidates failed: " + " | ".join(errors))


''',
    )
    replace_once(
        script,
        '''        logger.warning(
            "[grasp] cartesian path could not be found (%s); falling back to "
            "the collision-aware planner over the candidate fan.", exc,
        )
        _planner_grasp(ctx, candidate_poses, target_obb)
''',
        '''        logger.warning(
            "[grasp] Cartesian path failed (%s); trying the candidate fan.",
            exc,
        )
        _planner_grasp(ctx, candidate_poses, target_obb, hover_z)
''',
    )

    skill = skill_dir / "SKILL.md"
    replace_between(
        skill,
        "description: Top-down grasping via a fast axis-locked linear descend with a\n",
        "compatibility: requires gap>=0.1\n",
        '''description: Top-down grasping by rising, moving over the object, and
  descending vertically. A single `grasp_descend_linear` node tries the
  primary pose, then each candidate in the fan in order, and stops at the
  first straight-line descent that succeeds. Use when a perceived object
  (OBB) must be grasped.
''',
    )
    replace_once(
        skill,
        "metadata: {category: grasping, tags: [grasping, manipulation, collision-aware, curobo, linear]}",
        "metadata: {category: grasping, tags: [grasping, manipulation, linear]}",
    )
    replace_once(
        skill,
        '''    - curobo.plan_directed_linear
    - curobo.plan_to_grasp_poses
''',
        "",
    )
    replace_once(
        skill,
        "    failed: Any failure during the grasp attempt — the cartesian descend AND the collision-aware planner fallback both failed (a raise propagating to `on_error`). Coordinator routes to abort.",
        "    failed: Any failure during the grasp attempt — every candidate straight-line descent failed (a raise propagating to `on_error`). Coordinator routes to abort.",
    )
    replace_once(
        skill,
        '''      takes candidate 0 as the primary target AND the full fan as the
      planner-fallback goalset — both `Ref("compute_grasp.candidates.poses.0")`
''',
        '''      takes candidate 0 as the primary target AND the full fan as the
      fallback candidates — both `Ref("compute_grasp.candidates.poses.0")`
''',
    )
    replace_once(
        skill,
        '''      script already does rise → translate → Z-locked linear descend, and
      internally falls back to `curobo.plan_to_grasp_poses` (building its own
      per-observation collision world) when the straight-line solve fails.
''',
        '''      script rises, translates, and descends in a straight line, and
      retries the same descent on each candidate in the fan when the first
      pose fails.
''',
    )
    replace_between(
        skill,
        "Top-down grasping with a fast **axis-locked linear descend** and a\n",
        "## Install\n",
        '''Top-down grasping by rising, moving over the object, and descending
vertically. The node rises to a hover height, translates over the target
while rotating to the grasp yaw, and descends straight down onto the object.
If that pose is unreachable, it retries the same descent on each candidate
in the fan and stops at the first success. The descent is not
collision-checked. This is the `grocery_packing` grasp motion, distilled to
the general single-object case.

''',
    )
    replace_between(
        skill,
        "## Install\n",
        "## When to use\n",
        '''## Install

Motion uses the connector's `robot.go_to_pose_cartesian`. The descent is not
collision-checked.

''',
    )
    replace_once(
        skill,
        "- Default grasping skill whenever `curobo` is deployed — clean or cluttered.\n",
        "- A perceived object grasped by lowering straight onto it.\n",
    )
    replace_once(
        skill,
        "- `curobo` not deployed. Use `grasping-direct-ik` instead.\n",
        "- The straight-line descent onto the target would pass through clutter;\n"
        "  it is not collision-checked.\n",
    )
    replace_once(
        skill,
        '''   onto it (shallow grip near the perceived top, floored a hair above the
   object base so the fingers never ram the table). On an infeasible cartesian
   solve it falls back to `curobo.plan_to_grasp_poses` over `candidate_poses`.
''',
        '''   onto it (shallow grip near the perceived top, floored a hair above the
   object base so the fingers never ram the table). If that pose fails, it
   retries the same descent on each pose in `candidate_poses` and stops at
   the first success.
''',
    )
    replace_once(
        skill,
        "| `failed` | Grasp-attempt failure: the cartesian descend AND the cuRobo planner fallback both failed (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |",
        "| `failed` | Grasp-attempt failure: every candidate straight-line descent failed (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |",
    )
    add_port_note(
        skill_dir,
        "The Z-locked cuRobo descent is a straight-line "
        "`robot.go_to_pose_cartesian` descent, and the planner fallback "
        "retries that descent on each candidate pose.",
    )
    reject_curobo(script)
    reject_curobo(skill, allowed=("design_grasp_curobo.md",))


# ---------------------------------------------------------------------------
# transporting-objects
# ---------------------------------------------------------------------------


def port_transporting_objects(root: Path) -> None:
    skill_dir = root / "skills/transporting-objects"
    script = skill_dir / "scripts/transport_descend_linear.py"
    replace_once(
        script,
        '''an interior (basket, bin, box, tote). Lift/XY are straight-line cartesian
(``robot.go_to_pose_cartesian``); the descend uses cuRobo's AXIS-CONSTRAINED
linear move (``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``,
``orientation_mode="LOCK"``) — a guaranteed straight vertical drop. The descend
''',
        '''an interior (basket, bin, box, tote). Lift, XY and the descend are
straight-line cartesian (``robot.go_to_pose_cartesian``); the descend keeps the
hand's XY and rotation, so it is a straight vertical drop. The descend
''',
    )
    replace_once(
        script,
        '''        # A Cartesian servo can hit a joint-limit basin even when another arm
        # configuration reaches the same clear-space pose. Use the reusable
        # CuRobo single-pose planner for a genuine joint-space reconfiguration.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to curobo.plan_to_pose.",
            x,
            y,
            z,
            exc,
        )
        observation = ctx.tool("robot.get_observation")
        plan = ctx.tool(
            "curobo.plan_to_pose",
            target_pose=pose,
            start_joint_position=observation["arms"][0]["joint_state"],
        )
        if not plan.get("success") or not plan.get("trajectory"):
            raise RuntimeError("CuRobo could not plan the clear-space transport leg") from exc
        ctx.tool(
            "robot.execute_trajectory", trajectory=plan["trajectory"], max_steps_per_waypoint=60
        )
''',
        '''        # A Cartesian servo can hit a joint-limit basin even when another arm
        # configuration reaches the same clear-space pose. Use the connector's
        # single-pose IK move for a genuine joint-space reconfiguration.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to robot.go_to_pose.",
            x,
            y,
            z,
            exc,
        )
        ctx.tool("robot.go_to_pose", pose=pose)
''',
    )
    replace_once(
        script,
        '''def _cartesian(ctx: NodeContext, x: float, y: float, z: float, rotation: Quaternion) -> None:
''',
        '''def _wrist_to_tcp(ctx: NodeContext) -> float:
    """Distance from the ``robot.get_ee_pose`` wrist to the commanded TCP."""
    arm = ctx.tool("robot.describe_arm")
    gripper = ctx.tool("robot.describe_gripper")
    offset, approach = arm["tcp_offset"], gripper["approach_axis"]
    return abs(sum(float(offset[k]) * float(approach[k]) for k in ("x", "y", "z")))


def _cartesian(ctx: NodeContext, x: float, y: float, z: float, rotation: Quaternion) -> None:
''',
    )
    replace_once(
        script,
        '''    reached = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    error = math.sqrt(
        (float(reached["x"]) - float(x)) ** 2
        + (float(reached["y"]) - float(y)) ** 2
        + (float(reached["z"]) - float(z)) ** 2
    )
''',
        '''    reached = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    reached_z = float(reached["z"]) - _wrist_to_tcp(ctx)
    error = math.sqrt(
        (float(reached["x"]) - float(x)) ** 2
        + (float(reached["y"]) - float(y)) ** 2
        + (reached_z - float(z)) ** 2
    )
''',
    )
    replace_once(
        script,
        '''            f"reached=({reached['x']:.3f}, {reached['y']:.3f}, {reached['z']:.3f})"
''',
        '''            f"reached=({reached['x']:.3f}, {reached['y']:.3f}, {reached_z:.3f})"
''',
    )
    replace_once(
        script,
        '''            else max(place_z + 0.07, min(float(cur["z"]), place_z + 0.08))
''',
        '''            else max(place_z + 0.07, min(float(cur["z"]) - _wrist_to_tcp(ctx), place_z + 0.08))
''',
    )
    replace_once(
        script,
        '''    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool(
            "robot.execute_trajectory", trajectory=res["trajectory"], max_steps_per_waypoint=60
        )
    else:
        ee_pose = ctx.tool("robot.get_ee_pose")["pose"]
        ee = ee_pose["position"]
        _cartesian(ctx, ee["x"], ee["y"], target_z, ee_pose["rotation"])
''',
        '''    ee_pose = ctx.tool("robot.get_ee_pose")["pose"]
    ee = ee_pose["position"]
    _cartesian(ctx, ee["x"], ee["y"], target_z, ee_pose["rotation"])
''',
    )
    replace_once(
        script,
        '    raise RuntimeError("CuRobo found no reachable point inside the destination OBB")\n',
        '    raise RuntimeError("no reachable point inside the destination OBB")\n',
    )

    skill = skill_dir / "SKILL.md"
    replace_once(skill, "    - curobo.plan_to_pose\n", "")
    replace_once(
        skill,
        '''    - curobo.plan_with_grasped_object
    - curobo.plan_directed_linear
''',
        "",
    )
    replace_once(
        skill, "    - waypoint_move_carve: scripts/waypoint_move_carve.py\n", ""
    )
    replace_once(
        skill,
        '''- Cluttered transport paths where the lifted object risks colliding with
  other scene objects in transit — but see the `waypoint_move_carve`
  variant below, which routes the lift/translate through
  `curobo.plan_with_grasped_object` against a rebuilt collision world.
''',
        '''- Cluttered transport paths where the lifted object risks colliding with
  other scene objects in transit.
''',
    )
    replace_once(
        skill,
        '''  in). It lifts, moves over the container, and does an axis-locked straight-Z
  descend (`curobo.plan_directed_linear`, fingertip-frame). Lift and transport
''',
        '''  in). It lifts, moves over the container, and descends in a straight
  vertical line (`robot.go_to_pose_cartesian`, fingertip-frame). Lift and transport
''',
    )
    replace_between(
        skill,
        "   **Variant — collision-aware lift/translate (`waypoint_move_carve`)**.\n",
        "3. **`release`** — `type: script`, file `scripts/<sg>/descend_release.py`.\n",
        "",
    )
    replace_once(
        skill,
        '''   TCP-aware `robot.go_to_pose_cartesian` (cuRobo linear plan with a
   `plan_to_pose` fallback built into the backend), so the held object
''',
        '''   TCP-aware `robot.go_to_pose_cartesian` (the connector's linear IK
   plan), so the held object
''',
    )
    replace_once(
        skill,
        "waypoint_move,waypoint_move_carve,descend_release,",
        "waypoint_move,descend_release,",
    )
    release = skill_dir / "scripts/descend_release.py"
    replace_once(
        release,
        '''The default top-down rotation is asked of the live hand (``robot.grasp_frame``)
and the retract names the work arm (``robot.describe_arm``), so the node is
right on a hand whose approach axis is not tool-local +z and on a bimanual
robot whose other arm is holding something.
''',
        '''The default top-down rotation is asked of the live hand (``robot.grasp_frame``).
``robot.go_home`` takes no arm argument and returns every arm home.
''',
    )
    replace_once(
        release,
        '''    # Retract slowly enough that the freshly-placed object isn't disturbed
    # by the arm's lateral motion. This arm only: on a bimanual robot a
    # bare go_home un-parks whatever the support arm was holding, which is
    # why the arm is named rather than left to the runtime's default.
    ctx.tool("robot.go_home", arm_id=int(ctx.tool("robot.describe_arm")["arm_id"]))
''',
        '''    # Retract after the object has settled. robot.go_home takes no arm
    # argument and returns every arm home.
    ctx.tool("robot.go_home")
''',
    )
    replace_once(
        skill,
        '''   Descends, opens the gripper, retracts home (`robot.go_home` on the work
   arm named by `robot.describe_arm`; the default rotation is this hand's
   `robot.grasp_frame`). On success → exit `placed`; on failure → exit
''',
        '''   Descends, opens the gripper, retracts home (`robot.go_home`, which takes
   no arm argument and returns every arm home; the default rotation is this
   hand's `robot.grasp_frame`). On success → exit `placed`; on failure → exit
''',
    )
    add_port_note(
        skill_dir,
        "`transport_descend_linear` descends with `robot.go_to_pose_cartesian` "
        "instead of cuRobo's Z-locked linear plan and falls back to "
        "`robot.go_to_pose` instead of `curobo.plan_to_pose`. Its reach check "
        "and current height convert the `robot.get_ee_pose` wrist reading to "
        "the TCP. The `waypoint_move_carve` variant is not available. "
        "`descend_release` calls `robot.go_home` with no arguments.",
    )
    reject_curobo(skill)
    reject_curobo(script)


# ---------------------------------------------------------------------------
# grasping-direct-ik
# ---------------------------------------------------------------------------


def port_grasping_direct_ik(root: Path) -> None:
    skill_dir = root / "skills/grasping-direct-ik"
    script = skill_dir / "scripts/plan_to_pose.py"
    replace_between(
        script,
        '"""Plan a collision-free CuRobo trajectory to a single target pose.\n',
        "from typing import TypedDict\n",
        '''"""Move to a single target pose with the connector's IK.

Calls ``robot.go_to_pose`` with the target EE pose and returns a trajectory
holding only the joint positions reached, so a downstream execute keeps the
arm where it is. Raises :class:`gap.errors.PlanningFailed` when the move
fails so the workflow can route to a recovery path.

The move is not collision-checked; ``world_config``, ``observation`` and
``robot_file`` are accepted so existing graphs keep their wiring.
"""

''',
    )
    replace_to_end(
        script,
        '    start_joints = observation["arms"][0]["joint_state"]\n',
        '''    start_joints = observation["arms"][0]["joint_state"]

    kwargs = dict(
        world_config=world_config,
        start_joint_position=start_joints,
        target_pose=target_pose,
    )
    if robot_file:
        kwargs["robot_file"] = robot_file

    plan = ctx.tool("curobo.plan_to_pose", **kwargs)
    if not plan["success"]:
        raise PlanningFailed("CuRobo plan_to_pose failed")

    return {"trajectory": plan["trajectory"]}
''',
        '''    del world_config, observation, robot_file

    try:
        ctx.tool("robot.go_to_pose", pose=target_pose)
    except Exception as exc:  # noqa: BLE001
        raise PlanningFailed(f"robot.go_to_pose failed: {exc}") from exc

    joints = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    positions = [float(v) for v in joints["positions"]]
    return {"trajectory": {"waypoints": [{"positions": positions}]}}
''',
    )

    skill = skill_dir / "SKILL.md"
    replace_once(
        skill,
        "  (robot.describe_workspace) unless the workflow pins it. Use when no\n"
        "  trajectory planner (curobo) is deployed or the scene is uncluttered enough\n",
        "  (robot.describe_workspace) unless the workflow pins it. Use when no\n"
        "  collision-aware trajectory planner is deployed or the scene is uncluttered enough\n",
    )
    replace_once(
        skill,
        "    - curobo.plan_to_pose\n",
        "    - robot.get_observation\n",
    )
    replace_once(
        skill,
        "planner — works on platforms where CuRobo is not deployed, or in\n",
        "planner — works on platforms without a collision-aware planner, or in\n",
    )
    replace_once(
        skill,
        '''- The `curobo` tool bundle is not deployed (no collision-aware planner
  available).
''',
        "- No collision-aware planner is available.\n",
    )
    replace_once(
        skill,
        '''`scripts/<sg>/plan_to_pose.py` is the optional planned variant of a single
leg: it calls `curobo.plan_to_pose` from the observation's joint state to a
target pose and returns the `trajectory` (raising `PlanningFailed` when the
planner refuses) for a platform that deploys CuRobo as a fast single-pose IK
fallback without the goalset grasp planner.
''',
        '''`scripts/<sg>/plan_to_pose.py` is the optional single-leg variant: it moves
to the target pose with `robot.go_to_pose` and returns a one-waypoint
`trajectory` at the joints reached (raising `PlanningFailed` when the move
fails). `world_config` is accepted but not used; the move is not
collision-checked.
''',
    )
    replace_once(
        skill,
        "- `scripts/plan_to_pose.py` — the optional CuRobo single-pose leg.\n",
        "- `scripts/plan_to_pose.py` — the optional single-pose leg via `robot.go_to_pose`.\n",
    )
    add_port_note(
        skill_dir,
        "`plan_to_pose` moves with `robot.go_to_pose` instead of "
        "`curobo.plan_to_pose`.",
    )
    reject_curobo(skill)
    reject_curobo(script)


def move_unsupported(root: Path, parking: Path) -> None:
    for relative in UNSUPPORTED:
        source = root / relative
        target = parking / relative
        if not source.exists():
            raise RuntimeError(f"{source}: missing")
        if target.exists():
            raise RuntimeError(f"{target}: already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9-]*\.[a-z_][a-z0-9_]*$")
_FILE_SUFFIXES = frozenset({
    "py", "md", "json", "yaml", "yml", "txt", "png", "jpg", "npz", "npy",
    "urdf", "xml", "obj", "stl", "ply",
})


def _script_tool_names(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _TOOL_NAME.match(node.value)
            and node.value.rsplit(".", 1)[1] not in _FILE_SUFFIXES
        ):
            names.add(node.value)
    return names


def _bundles(root: Path, folder: str) -> list[Path]:
    return sorted(
        entry for entry in (root / folder).iterdir()
        if entry.is_dir()
        and not entry.name.startswith("_")
        and (entry / "SKILL.md").is_file()
    )


def _libero_connector_tools() -> dict:
    """Connector tool registry for this image's LIBERO env.

    Joint-control tools register only when the env can execute joint targets,
    so a connector around ``None`` omits them. The env is never initialized.
    """
    from types import SimpleNamespace

    from gap.connector.sim import SimConnector
    from gap.envs.libero_env import FrankaLiberoEnv

    env = FrankaLiberoEnv.__new__(FrankaLiberoEnv)
    return SimConnector(env, SimpleNamespace()).tool_registry._tools


def _input_schemas(root: Path) -> dict[str, set[str]]:
    """Accepted keyword names for each tool.

    Connector tools come from a throwaway connector's registry. Bundle tools
    come from the ``@tool`` declarations in each bundle directory. A connector
    name wins when both declare it.
    """
    from gap.agent._catalog import _ast_bundle_schemas

    schemas: dict[str, set[str]] = {}
    for name, descriptor in _libero_connector_tools().items():
        schemas[name] = set(descriptor.schema.inputs)
    for bundle in _bundles(root, "tools"):
        for name, schema in _ast_bundle_schemas(bundle).items():
            schemas.setdefault(name, set(schema.inputs))
    return schemas


def _tool_invocations(path: Path) -> list[tuple[int, str | None, list[str], str | None]]:
    """``ctx.tool`` calls as (line, tool name, literal keywords, unchecked reason)."""
    found = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_tool = (
            (isinstance(func, ast.Attribute) and func.attr == "tool")
            or (isinstance(func, ast.Name) and func.id == "tool")
        )
        if not is_tool:
            continue
        tool_name = None
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            tool_name = node.args[0].value
        literal = [kw.arg for kw in node.keywords if kw.arg is not None]
        reason = None
        if tool_name is None:
            reason = "tool name is not a literal"
        elif any(kw.arg is None for kw in node.keywords) or len(node.args) > 1:
            reason = "arguments passed dynamically"
        found.append((node.lineno, tool_name, literal, reason))
    return found


def check(root: Path) -> int:
    from gap.skills._meta_from_skill_md import parse_skill_md
    from gap.skills.validate import connector_tool_names

    available = set(connector_tool_names()) | set(_libero_connector_tools())
    for bundle in _bundles(root, "tools"):
        available.update(parse_skill_md(bundle / "SKILL.md").tools)
    schemas = _input_schemas(root)

    skills = _bundles(root, "skills")
    failures = []
    mismatches = []
    unchecked = []
    for bundle in skills:
        meta = parse_skill_md(bundle / "SKILL.md")
        referenced: dict[str, set[str]] = {}
        for name in meta.allowed_tools:
            referenced.setdefault(name, set()).add("allowed_tools")
        for name in meta.requires.connector if meta.requires is not None else []:
            referenced.setdefault(name, set()).add("requires.connector")
        scripts_dir = bundle / "scripts"
        if scripts_dir.is_dir():
            for script in sorted(scripts_dir.rglob("*.py")):
                relative = str(script.relative_to(bundle))
                for name in _script_tool_names(script):
                    referenced.setdefault(name, set()).add(relative)
                for line, tool_name, literal, reason in _tool_invocations(script):
                    where = f"{bundle.name}/{relative}:{line}"
                    if reason is not None:
                        unchecked.append(f"{where}: {tool_name or '?'} ({reason})")
                        if tool_name is None:
                            continue
                    accepted = schemas.get(tool_name)
                    if accepted is None:
                        unchecked.append(f"{where}: {tool_name} (no input schema)")
                        continue
                    extra = [name for name in literal if name not in accepted]
                    if extra:
                        mismatches.append(
                            f"{where}: {tool_name} does not accept {', '.join(extra)}"
                        )
        for name, where in sorted(referenced.items()):
            if name not in available:
                failures.append(f"{bundle.name}: {name} ({', '.join(sorted(where))})")

    failures.sort()
    mismatches.sort()
    unchecked.sort()
    print(f"{len(skills)} skill(s), {len(available)} available tool(s)")
    if failures:
        print("Skills naming tools nothing registers:")
        for line in failures:
            print(f"  {line}")
    else:
        print("Every registered skill names only available tools")
    if mismatches:
        print("Calls passing arguments the tool schema does not accept:")
        for line in mismatches:
            print(f"  {line}")
    else:
        print("Every literal tool argument is in that tool's input schema")
    print(f"Unchecked dynamic arguments: {len(unchecked)}")
    for line in unchecked:
        print(f"  {line}")
    return 1 if failures or mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="/ryzers/open-robot-skills",
                        type=Path)
    parser.add_argument("--check", action="store_true",
                        help="verify skill tool names and literal arguments")
    args = parser.parse_args()
    root = args.root
    if args.check:
        return check(root)

    port_grasping_short_axis(root)
    port_grasping_with_planner(root)
    port_transporting_objects(root)
    port_grasping_direct_ik(root)
    compile_patched_python()
    move_unsupported(root, root.parent / f"{root.name}-unsupported")
    print(f"Patched {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
