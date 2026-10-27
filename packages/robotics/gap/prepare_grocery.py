#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Create a PyRoKI variant of the grocery-packing workflow."""

from pathlib import Path
import shutil
import sys


SOURCE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/ryzers/graph-as-policy/examples/grocery_packing/packing_graph"
)
TARGET = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else (
        "/ryzers/graph-as-policy/examples/"
        "grocery_packing_pyroki/packing_graph"
    )
)


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


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    path.write_text(source.replace(old, new, 1))


shutil.copytree(SOURCE, TARGET)

grasp = TARGET / "scripts/grasp_move.py"
replace_between(
    grasp,
    '"""Grasp with a fast path and a robust fallback.\n',
    "import logging\n",
    '''"""Grasp by rising, moving over the object, and descending vertically.

If the first top-down pose is unreachable, try the remaining candidate poses.
"""

''',
)
replace_once(grasp, "_BASE_CLEARANCE = 0.012", "_BASE_CLEARANCE = 0.004")
replace_once(
    grasp,
    '''    """Straight-line cartesian move. Raises on an infeasible solve — the
    caller's planner fallback handles that, so we do NOT silently retry here."""
''',
    '''    """Straight-line Cartesian move for one grasp segment."""
''',
)
replace_between(
    grasp,
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
    grasp,
    "def _planner_grasp(",
    "def run(",
    '''def _planner_grasp(
    ctx: NodeContext,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    topk: int = 8,
    num_ik_seeds: int = 128,
) -> None:
    del num_ik_seeds
    errors: list[str] = []
    candidates = list(candidate_poses)[: int(topk)]
    for index, pose in enumerate(candidates):
        try:
            _cartesian_grasp(ctx, pose, target_obb, hover_z=0.2)
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
    grasp,
    '''        logger.warning(
            "[grasp] cartesian path could not be found (%s); falling back to "
            "the collision-aware planner over the candidate fan.", exc,
        )
''',
    '''        logger.warning(
            "[grasp] Cartesian path failed (%s); trying remaining candidates.",
            exc,
        )
''',
)

transport = TARGET / "scripts/transport_move.py"
replace_between(
    transport,
    '"""VAB-style transport:',
    "import logging\n",
    '''"""Lift, move over the basket, and descend vertically inside it."""

''',
)
replace_between(
    transport,
    "def _cartesian(",
    "def _descend_linear(",
    '''def _cartesian(ctx: NodeContext, x: float, y: float, z: float) -> None:
    pose = {
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "rotation": _DOWN,
    }
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:
        logger.warning(
            "[transport_move] Cartesian move to (%.3f, %.3f, %.3f) failed "
            "(%s); using an unconstrained IK move.",
            x,
            y,
            z,
            exc,
        )
        ctx.tool("robot.go_to_pose", pose=pose)


''',
)
replace_between(
    transport,
    "def _descend_linear(",
    "def run(",
    '''def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    if float(from_z) - float(target_z) <= 0.002:
        return
    ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, ee["x"], ee["y"], target_z)


''',
)

release = TARGET / "scripts/place_release.py"
replace_between(
    release,
    '"""VAB-style release:',
    "from typing import TypedDict\n",
    '''"""Open the gripper, let the object settle, and retract vertically."""

''',
)

print(TARGET)
