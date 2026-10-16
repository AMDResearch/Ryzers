# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded HELIX/CaP-X plumbing for the RHO workshop notebook."""

from __future__ import annotations

import atexit
import difflib
import io
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

CAPX_ROOT = Path(os.environ.get("CAPX_ROOT", "/ryzers/cap-x"))
CAPX_PYTHON = Path(os.environ.get("CAPX_PYTHON", "/opt/capx-venv/bin/python"))
HELIX = Path(os.environ.get("HELIX_BIN", "/opt/capx-venv/bin/helix"))
CAPX_MODEL = "Gemma-4-E4B-it-GGUF"
DEFAULT_RHO_MODEL = "Gemma-4-E2B-it-GGUF"
MODEL = os.environ.get("RHO_MODEL", DEFAULT_RHO_MODEL)
MODEL_API_ID = MODEL.removeprefix("user.")
CONFIG_PATH = os.environ.get(
    "RHO_CONFIG_PATH",
    "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
)
WORKSHOP_ROOT = Path(os.environ.get("RHO_WORKSHOP_ROOT", "/tmp/rho_workshop"))
CANDIDATE_ROOT = WORKSHOP_ROOT / "candidate"
VIDEO_ROOT = Path(os.environ.get("RHO_VIDEO_ROOT", str(WORKSHOP_ROOT / "videos")))
SERVICE_PORTS = (8113, 8115, 8116, 8117)
DEFAULT_TIMEOUT = 480
EVALUATION_TIMEOUT = 120

_OWNED_SERVERS: list[subprocess.Popen[Any]] = []
LAST_SERVICE_TIMING: dict[str, float] = {}


DEFAULT_PROGRAM = """\
# Code block 0
import numpy

# --- 1. Get object poses and extents ---

# Red cube data
red_pose, red_quat, red_extent = get_object_pose("red cube", return_bbox_extent=True)
# Green cube data
green_pose, _, green_extent = get_object_pose("green cube", return_bbox_extent=True)

# --- 2. Sample grasp pose for red cube ---
red_grasp_position, red_grasp_quat = sample_grasp_pose("red cube")

# --- 3. Approach and grasp the red cube ---
print("Approaching and grasping red cube...")
goto_pose(red_grasp_position, red_grasp_quat, z_approach=0.1)
close_gripper()

# --- 4. Lift the red cube to a safe height ---
# Calculate lift position: original position + 0.2m in Z
lift_position = red_grasp_position.copy()
lift_position[2] += 0.2
print("Lifting red cube to safe height...")
# Use z_approach=0.0 since we are actively moving the lifted object away from the initial grasp point
goto_pose(lift_position, red_grasp_quat, z_approach=0.0)

# --- 5. Calculate the target placement pose on the green cube ---

# Green cube center Z coordinate
green_center_z = green_pose[0][2]
# Half height of green cube
green_half_height = green_extent[2] / 2
# Half height of red cube
red_half_height = red_extent[2] / 2

# Calculate stacking height
place_z = green_center_z + green_half_height + red_half_height

# Target position (X, Y matches green cube center, Z is stacking height)
placement_position = numpy.array([green_pose[0][0], green_pose[0][1], place_z])

# --- 6. Approach and place the red cube ---
print("Moving to placement location on green cube...")
# Approach using z_approach=0.1 for controlled descent
goto_pose(placement_position, red_grasp_quat, z_approach=0.1)

# Release the cube
print("Releasing red cube.")
open_gripper()

# Optional: Move to a safe final pose if needed, but the task is complete.
# home_pose()
print("Task completed: Red cube stacked on green cube.")
"""

DEFAULT_PROVENANCE: dict[str, Any] = {
    "source": "recorded_capx_generation",
    "artifact": None,
    "model": CAPX_MODEL,
    "trial": 1,
    "recorded_reward": 0.7243017351331602,
    "recorded_task_completed": False,
    "source_run": "gemma-e4b-five-trials-20260821",
    "recorded_bug": (
        "green_pose is already the XYZ position, but the generated program "
        "indexes each scalar as though the position were nested."
    ),
}

POLICY_SOURCE = """\
from pathlib import Path


def build_program() -> str:
    return Path(__file__).with_name("program.py").read_text()
"""

API_REFERENCE = """\
# CaP-X cube-stack contract

Repair the recorded CaP-X generated program that should stack the red cube on
the green cube. Preserve its overall approach and correct the runtime failure
using the API contract.

- `sample_grasp_pose("red cube")` returns a grasp position and a reliable
  gripper quaternion.
- `get_object_pose(name, return_bbox_extent=True)` returns center position,
  quaternion, and full bounding-box side lengths.
- `goto_pose(position, quaternion, z_approach=0.1)` executes an approach and
  target motion; use a non-zero approach for grasping and placement.
- Lift the grasped cube far enough to clear the table and the target cube.
- The placement center height is both cubes' half-heights above the green
  center. Reuse the grasp quaternion for placement.
- `close_gripper()` and `open_gripper()` actuate the gripper.

The evaluator runs the artifact's recorded trial for training and a separately
configured held-out trial for validation.
Only edit files below `solver/`.
"""

OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "model": f"lemonade/{MODEL_API_ID}",
    "small_model": f"lemonade/{MODEL_API_ID}",
    "agent": {"build": {"temperature": 0.0, "steps": 8}},
    "experimental": {"primary_tools": ["read", "edit", "bash"]},
    "provider": {
        "lemonade": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local Lemonade",
            "options": {
                "baseURL": "http://127.0.0.1:13305/api/v1",
                "apiKey": "lemonade",
            },
            "models": {
                MODEL_API_ID: {
                    "name": MODEL_API_ID,
                    "tool_call": True,
                    "limit": {"context": 32768, "output": 4096},
                }
            },
        }
    },
    "permission": {
        "*": "allow",
        # OpenCode evaluates the last matching rule and tool paths are
        # workspace-relative, so keep the broad deny first and allow solver/.
        "edit": {
            "*": "deny",
            "solver/**": "allow",
            "**/solver/**": "allow",
        },
        "external_directory": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "task": "deny",
        "skill": "deny",
        "todowrite": "deny",
        "bash": {
            "*": "deny",
            "RHO_EVAL_ORIGIN=agent-self-check /opt/capx-venv/bin/python probe.py*": "allow",
            "/opt/capx-venv/bin/python -m py_compile solver/*.py": "allow",
            "git diff*": "allow",
            "git status*": "allow",
        },
    },
}

PROBE_SOURCE = """\
import os
import sys

sys.path.insert(
    0, os.environ.get("RHO_SUPPORT_ROOT", "/ryzers/notebooks/scripts")
)
from rho_demo import evaluate_cli

raise SystemExit(evaluate_cli())
"""

# The single-task workshop prompts below deliberately hand the mutator the
# diagnosis. Gemma E2B with eight turns does not reliably localize the defect
# on its own, and this path exists to fit one accepted mutation inside a live
# session. That makes it a demonstration of the HELIX loop's mechanics, not
# evidence that the agent can diagnose. The multi-task study in
# rho_multitask_demo.py takes the opposite stance and tells the mutator to
# diagnose rather than assume. Surface DEFAULT_GUIDANCE_DISCLOSURE wherever
# these results are shown.
DEFAULT_GUIDANCE_DISCLOSURE = (
    "This single-task configuration names the defect in helix.toml's objective "
    "and background. Gemma E2B is given the diagnosis so that one mutation can "
    "be accepted inside a live session, so treat an accepted result as evidence "
    "that the gate and frontier machinery work end to end, not as evidence that "
    "the agent diagnosed the failure. The recorded multi-task study instructs "
    "its mutator to diagnose rather than assume."
)

DEFAULT_OBJECTIVE = """\
Repair this authentic CaP-X generated program so it reliably stacks the red
cube on the green cube. Diagnose the recorded indexing failure, inspect
API_REFERENCE.md, and edit solver/program.py. get_object_pose returns a flat
XYZ vector: replace green_pose[0][2], green_pose[0][0], and green_pose[0][1]
with direct green_pose indexing. Keep the policy concise."""

DEFAULT_BACKGROUND = """\
This is a bounded workshop mutation. Read API_REFERENCE.md and the evaluator
diagnostics first. Your first mutation action MUST be an `edit` tool call on
solver/program.py: change green_pose[0][2] to green_pose[2],
green_pose[0][0] to green_pose[0], and green_pose[0][1] to green_pose[1].
Do not call skills, todo tools, or repeatedly re-read the same code. Inspect
`git diff` after the edit. Do not alter evaluation, configuration, permissions,
or files outside this repository. Run
`/opt/capx-venv/bin/python -m py_compile solver/*.py` and
`RHO_EVAL_ORIGIN=agent-self-check /opt/capx-venv/bin/python probe.py`
before finishing."""


def helix_config(
    generations: int = 1,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    background: str = DEFAULT_BACKGROUND,
    train_size: int = 1,
    val_size: int = 1,
    minibatch_size: int = 1,
    perfect_score_threshold: float = 1.0,
    max_evaluations: int | None = None,
    max_turns: int = 8,
    generations_limit: int = 4,
) -> str:
    if not 1 <= generations <= generations_limit:
        raise ValueError(
            f"workshop generations must be between 1 and {generations_limit}"
        )
    if '"""' in objective or '"""' in background:
        raise ValueError("HELIX prompts cannot contain TOML triple quotes")
    if minibatch_size > train_size:
        raise ValueError("minibatch cannot be larger than the training split")
    if max_evaluations is None:
        max_evaluations = max(8, 2 + 3 * generations)
    return f'''\
objective = """{objective}"""
seed = "."
rng_seed = 7
passthrough_env = [
  "CUDA_VISIBLE_DEVICES",
  "HIP_VISIBLE_DEVICES",
  "ROCR_VISIBLE_DEVICES",
  "HSA_OVERRIDE_GFX_VERSION",
  "LD_LIBRARY_PATH",
  "HF_HOME",
  "MUJOCO_GL",
  "PYOPENGL_PLATFORM",
  "CAPX_ROOT",
  "RHO_CONFIG_PATH",
  "RHO_PROGRESS_FILE",
  "RHO_SUPPORT_ROOT",
  "XDG_RUNTIME_DIR",
  "RHO_MOCK_EVAL",
]

[env]
CAPX_ROOT = "/ryzers/cap-x"
HF_HOME = "/opt/capx-cache"
MUJOCO_GL = "egl"
PYOPENGL_PLATFORM = "egl"
RHO_EVAL_TIMEOUT = "120"

[evaluator]
command = "/opt/capx-venv/bin/python probe.py"
protected_files = [
  "probe.py",
  "helix.toml",
  "opencode.json",
  "API_REFERENCE.md",
  "provenance.json",
]

[dataset]
train_size = {train_size}
val_size = {val_size}

[evolution]
max_generations = {generations}
perfect_score_threshold = {perfect_score_threshold}
max_evaluations = {max_evaluations}
merge_enabled = false
num_parallel_proposals = 1
mutations_per_parent = 1
minibatch_size = {minibatch_size}
max_workers = 1
cache_evaluation = true
acceptance_criterion = "strict_improvement"
frontier_type = "instance"

[agent]
backend = "opencode"
model = "lemonade/{MODEL_API_ID}"
max_turns = {max_turns}
background = """{background}"""

[sandbox]
enabled = false

[worktree]
base_dir = ".helix/worktrees"
'''


@dataclass
class BoundedRun:
    returncode: int
    timed_out: bool
    stdout: str
    elapsed_seconds: float


class _NotebookHelixProgress:
    """Collapse Rich frames and evaluator events into one notebook status."""

    _ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self, evaluation_log: Path) -> None:
        from IPython.display import HTML, display

        self._HTML = HTML
        self._handle = display(HTML("HELIX: starting…"), display_id=True)
        self._generation = "0/1"
        self._phase = "Initializing"
        self._phase_started = time.monotonic()
        self._evaluations = 0
        self._max_evaluations = 8
        self._evaluation_log = evaluation_log
        self._evaluation_log_offset = 0
        self._active_evaluation: dict[str, Any] | None = None
        self._evaluation_history: list[dict[str, Any]] = []
        self._helix_evaluations = 0
        self._agent_self_checks = 0
        self._last_tick_second = -1
        self._last_html = ""
        self._render()

    def _render(
        self,
        message: str | None = None,
        *,
        color: str = "#2563eb",
        complete: bool = False,
    ) -> None:
        detail = escape(message or self._phase)
        evaluation_details = ""
        if self._evaluation_history:
            rows = []
            for result in self._evaluation_history[-3:]:
                solved = bool(result.get("task_completed"))
                status = "solved" if solved else "not solved"
                status_color = "#15803d" if solved else "#b45309"
                task = (
                    f"{escape(str(result['task']))} · "
                    if result.get("task")
                    else ""
                )
                rows.append(
                    f'<span style="color:{status_color}">●</span> '
                    f"{escape(str(result['_display_label']))}: "
                    f"{task}"
                    f"{escape(str(result.get('split', 'train')))} "
                    f"trial {escape(str(result.get('trial', '?')))} · "
                    f"reward {float(result.get('reward', 0.0)):.3f} · {status} · "
                    f"{float(result.get('elapsed_seconds') or 0.0):.1f}s"
                )
            evaluation_details = (
                '<div style="margin-top:6px;color:#4b5563;font-size:0.92em">'
                + "<br>".join(rows)
                + "</div>"
            )
        if complete:
            progress = (
                '<progress value="1" max="1" style="width:320px"></progress> '
                f"{self._evaluations} HELIX evaluations used "
                f"({self._max_evaluations} maximum)"
            )
        else:
            progress = (
                f'<progress value="{self._evaluations}" max="{self._max_evaluations}" '
                'style="width:320px"></progress> '
                f"{self._evaluations} HELIX evaluations used "
                f"({self._max_evaluations} maximum)"
            )
        markup = (
            f"<b>HELIX generation {escape(self._generation)}</b> — {detail}<br>"
            f"{progress}{evaluation_details}"
        )
        if color != "#2563eb":
            markup = f'<span style="color:{color}">{markup}</span>'
        if markup != self._last_html:
            if self._handle is not None:
                self._handle.update(self._HTML(markup))
            self._last_html = markup

    def _poll_evaluations(self) -> str | None:
        try:
            with self._evaluation_log.open(encoding="utf-8") as stream:
                stream.seek(self._evaluation_log_offset)
                lines = stream.readlines()
                self._evaluation_log_offset = stream.tell()
        except (FileNotFoundError, OSError):
            return None

        message = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "started":
                self._active_evaluation = event
                self._last_tick_second = -1
            elif event.get("event") == "completed":
                self._active_evaluation = None
                if event.get("origin") == "agent-self-check":
                    self._agent_self_checks += 1
                    event["_display_label"] = (
                        f"Agent self-check {self._agent_self_checks}"
                    )
                else:
                    self._helix_evaluations += 1
                    event["_display_label"] = (
                        f"HELIX eval {self._helix_evaluations}"
                    )
                    self._evaluations = max(
                        self._evaluations, self._helix_evaluations
                    )
                self._evaluation_history.append(event)
                outcome = "solved" if event.get("task_completed") else "not solved"
                message = (
                    f"{event['_display_label']} complete · "
                    f"reward {float(event.get('reward', 0.0)):.3f} · {outcome}"
                )
        return message

    def tick(self) -> None:
        evaluation_message = self._poll_evaluations()
        if evaluation_message:
            self._render(evaluation_message)
            return

        if self._active_evaluation is not None:
            elapsed = max(
                0, int(time.time() - float(self._active_evaluation["started_at"]))
            )
            if elapsed != self._last_tick_second:
                self._last_tick_second = elapsed
                if self._active_evaluation.get("origin") == "agent-self-check":
                    label = f"Agent self-check {self._agent_self_checks + 1}"
                else:
                    label = f"HELIX eval {self._helix_evaluations + 1}"
                task = (
                    f"{self._active_evaluation['task']} · "
                    if self._active_evaluation.get("task")
                    else ""
                )
                self._render(
                    f"{label} running · "
                    f"{task}"
                    f"{self._active_evaluation.get('split', 'train')} "
                    f"trial {self._active_evaluation.get('trial', '?')} · "
                    f"{elapsed}s elapsed"
                )
            return

        elapsed = int(time.monotonic() - self._phase_started)
        if elapsed != self._last_tick_second:
            self._last_tick_second = elapsed
            self._render(f"{self._phase} · {elapsed}s elapsed")

    def __call__(self, raw_line: str) -> None:
        evaluation_message = self._poll_evaluations()
        line = self._ansi.sub("", raw_line).replace("\r", "\n").splitlines()[-1:]
        if not line:
            if evaluation_message:
                self._render(evaluation_message)
            return
        text = line[0].strip()

        generation = re.search(r"Generation\s+(\d+)\s*/\s*(\d+)", text)
        if generation:
            self._generation = f"{generation.group(1)}/{generation.group(2)}"

        phase = re.search(r"Status:\s*([^│]+)", text)
        if phase and self._active_evaluation is None:
            new_phase = phase.group(1).strip()
            if new_phase != self._phase:
                self._phase = new_phase
                self._phase_started = time.monotonic()
                self._last_tick_second = -1

        budget = re.search(r"(\d+)/(\d+)\s+evals", text)
        if budget:
            self._evaluations = int(budget.group(1))
            self._max_evaluations = int(budget.group(2))

        message = None
        for prefix in (
            "Creating seed worktree",
            "Evaluating seed",
            "Seed evaluated",
            "Minibatch gate",
            "Evolution complete",
        ):
            if prefix in text:
                message = text[text.index(prefix) :].strip()
                break
        if evaluation_message:
            self._render(evaluation_message)
        elif self._active_evaluation is not None:
            # Rich redraws its whole terminal frame while an evaluation runs.
            # Keep the side-channel evaluation status authoritative instead of
            # alternating it with stale "Applying mutation" frame lines.
            self.tick()
        elif message is not None:
            self._render(message)
        else:
            # Keep the elapsed-time rendering authoritative between meaningful
            # HELIX events. Rich redraws otherwise remove and restore the
            # seconds suffix several times per second.
            self.tick()

    def finish(self, result: BoundedRun) -> None:
        self._poll_evaluations()
        if result.returncode == 0:
            self._render(
                f"complete in {result.elapsed_seconds:.1f}s",
                color="#15803d",
                complete=True,
            )
        elif result.timed_out:
            self._render(
                f"stopped at the {result.elapsed_seconds:.0f}s deadline",
                color="#b45309",
            )
        else:
            self._render(f"failed (exit {result.returncode})", color="#b91c1c")


def _notebook_progress(evaluation_log: Path) -> _NotebookHelixProgress | None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            return _NotebookHelixProgress(evaluation_log)
    except Exception:
        pass
    return None


def _safe_reset(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        return
    temporary_root = Path("/tmp").resolve()
    if resolved == temporary_root or temporary_root not in resolved.parents:
        raise ValueError(f"refusing to remove non-workshop path: {resolved}")
    shutil.rmtree(resolved)


def _write_candidate(root: Path, program: str) -> None:
    (root / "solver").mkdir(parents=True, exist_ok=True)
    (root / "solver" / "__init__.py").write_text("")
    (root / "solver" / "geometry.py").write_text(
        "# Compatibility placeholder: the authentic policy lives in program.py.\n"
    )
    (root / "solver" / "program.py").write_text(program)
    (root / "solver" / "policy.py").write_text(POLICY_SOURCE)


def _provenance_metadata(
    provenance: Mapping[str, Any] | Path | str | None,
) -> dict[str, Any]:
    if provenance is None:
        return {}
    if isinstance(provenance, Mapping):
        return dict(provenance)
    loaded = json.loads(Path(provenance).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("provenance JSON must contain an object")
    return loaded


def _artifact_program(
    artifact: Path | str | None,
    provenance: Mapping[str, Any] | Path | str | None,
) -> tuple[str, dict[str, Any], Path | None]:
    metadata = _provenance_metadata(provenance)
    if artifact is None:
        recorded = dict(DEFAULT_PROVENANCE)
        recorded.update(metadata)
        return DEFAULT_PROGRAM, recorded, None

    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"CaP-X artifact does not exist: {artifact_path}")
    code_path = artifact_path
    if artifact_path.is_dir():
        embedded = artifact_path / "provenance.json"
        if embedded.exists():
            embedded_metadata = _provenance_metadata(embedded)
            embedded_metadata.update(metadata)
            metadata = embedded_metadata

        direct = [path for path in (artifact_path / "code.py", artifact_path / "program.py") if path.is_file()]
        candidates = direct or sorted(artifact_path.glob("trial_*/code.py"))
        if not candidates:
            candidates = sorted(artifact_path.rglob("code.py"))
        if len(candidates) > 1:
            selected_trial = int(
                next(
                    (metadata[key] for key in ("trial", "artifact_trial", "training_trial") if key in metadata),
                    1,
                )
            )
            matching = [
                path
                for path in candidates
                if any(re.search(rf"(?:^|_)trial_0*{selected_trial}(?:_|$)", part) for part in path.parts)
            ]
            if len(matching) == 1:
                candidates = matching
        if len(candidates) != 1:
            raise ValueError(
                f"artifact directory must identify exactly one CaP-X code.py or program.py; found {len(candidates)}"
            )
        code_path = candidates[0]
    if not code_path.is_file():
        raise ValueError(f"CaP-X code path is not a file: {code_path}")

    metadata.setdefault("source", "capx_artifact")
    metadata["artifact"] = str(artifact_path)
    metadata["code_path"] = str(code_path)
    return code_path.read_text(), metadata, code_path


def _artifact_trial(metadata: Mapping[str, Any], code_path: Path | None) -> int:
    evaluation = metadata.get("evaluation")
    nested_evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    source = metadata.get("provenance")
    nested_provenance = source if isinstance(source, Mapping) else {}
    declared = next(
        (
            value
            for value in (
                metadata.get("trial"),
                metadata.get("artifact_trial"),
                metadata.get("training_trial"),
                nested_evaluation.get("trial"),
                nested_provenance.get("source_trial"),
            )
            if value is not None
        ),
        None,
    )
    inferred: int | None = None
    if code_path is not None:
        for part in reversed(code_path.parts):
            match = re.search(r"(?:^|_)trial_(\d+)(?:_|$)", part)
            if match:
                inferred = int(match.group(1))
                break
    trial = int(declared if declared is not None else inferred or 1)
    if inferred is not None and declared is not None and trial != inferred:
        raise ValueError(f"provenance trial {trial} does not match artifact trial {inferred}")
    if trial < 0:
        raise ValueError("trial identifiers must be non-negative")
    return trial


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "user.name=RHO Workshop", "-c", "user.email=rho@localhost", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def prepare_workshop(
    root: Path | str = CANDIDATE_ROOT,
    *,
    artifact: Path | str | None = None,
    provenance: Mapping[str, Any] | Path | str | None = None,
    heldout_trial: int = 2,
    generations: int = 1,
    api_reference: str = API_REFERENCE,
    objective: str = DEFAULT_OBJECTIVE,
    background: str = DEFAULT_BACKGROUND,
    support_files: Mapping[str, str] | None = None,
    reset: bool = True,
    training_trials: Sequence[int] | None = None,
    heldout_trials: Sequence[int] | None = None,
    helix_overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Create a disposable repository around verbatim CaP-X generated code.

    ``training_trials``/``heldout_trials`` turn each split into several distinct
    layouts, which is what a pass-rate study needs; leaving them unset keeps the
    original one-trial-per-split behaviour.
    """
    overrides = dict(helix_overrides or {})
    generations_limit = int(overrides.get("generations_limit", 4))
    if not 1 <= generations <= generations_limit:
        raise ValueError(
            f"workshop generations must be between 1 and {generations_limit}"
        )
    if heldout_trial < 0:
        raise ValueError("held-out trial must be non-negative")
    if training_trials is not None and heldout_trials is not None:
        overlap = set(int(t) for t in training_trials) & set(int(t) for t in heldout_trials)
        if overlap:
            raise ValueError(f"held-out trials must be unseen during training: {sorted(overlap)}")
    root = Path(root).expanduser().resolve()
    if reset:
        _safe_reset(root)
    root.mkdir(parents=True, exist_ok=True)
    program, source_metadata, code_path = _artifact_program(artifact, provenance)
    training_trial = _artifact_trial(source_metadata, code_path)
    if "training_trial" in source_metadata:
        declared_training = int(source_metadata["training_trial"])
        if declared_training != training_trial:
            raise ValueError("training trial must match the CaP-X artifact's recorded trial")
    persisted_provenance = dict(source_metadata)
    persisted_provenance.update(
        {
            "artifact_trial": training_trial,
            "training_trial": training_trial,
            "heldout_trial": int(heldout_trial),
        }
    )
    if training_trials is not None:
        persisted_provenance["training_trials"] = [int(t) for t in training_trials]
    if heldout_trials is not None:
        persisted_provenance["heldout_trials"] = [int(t) for t in heldout_trials]

    _write_candidate(root, program)
    for relative, source in (support_files or {}).items():
        path = Path(relative)
        if path.is_absolute() or not path.parts or path.parts[0] != "solver":
            raise ValueError(f"support file must stay below solver/: {relative}")
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    (root / "API_REFERENCE.md").write_text(api_reference)
    (root / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG, indent=2) + "\n")
    (root / "helix.toml").write_text(
        helix_config(
            generations,
            objective=objective,
            background=background,
            **overrides,
        )
    )
    (root / "probe.py").write_text(PROBE_SOURCE)
    (root / "provenance.json").write_text(json.dumps(persisted_provenance, indent=2, sort_keys=True) + "\n")
    (root / ".gitignore").write_text(
        ".helix/\n.helix_artifacts/\n.helix_opencode_state/\n__pycache__/\n*.pyc\nhelix_batch.json\n"
    )

    _git(root, "init", "-b", "main")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Seed the recorded CaP-X program")
    return root


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def service_status() -> dict[int, bool]:
    return {port: _port_open(port) for port in SERVICE_PORTS}


def ensure_services(
    progress: Callable[[str], None] = print,
    *,
    model: str | None = MODEL,
) -> list[Any]:
    """Start/reuse Lemonade, OWLv2, SAM2, Contact-GraspNet, and PyRoKi.

    Pass ``model=None`` to start only the perception and control stack. A
    caller replaying frozen policies never reaches a language model, so
    loading one costs startup time and nothing else.
    """
    from capx_demo import SERVICE_LOG, ensure_lemonade, quiet_output

    lemonade_seconds = ensure_lemonade(model, progress=progress) if model else 0.0
    started = time.monotonic()
    old_cwd = Path.cwd()
    try:
        with quiet_output():
            os.chdir(CAPX_ROOT)
            from capx.envs.launch import LaunchArgs
            from capx.envs.runner import _start_api_servers
            from capx.utils.launch_utils import _load_config

            args = LaunchArgs(
                config_path=CONFIG_PATH,
                model=MODEL,
                server_url="http://127.0.0.1:13305/api/v1/chat/completions",
                temperature=0.2,
                max_tokens=4096,
            )
            _, _, api_servers = _load_config(args)
            servers = list(_start_api_servers(api_servers, 900.0))
            status = service_status()
            if not all(status.values()):
                missing = [port for port, ready in status.items() if not ready]
                servers.extend(_start_api_servers(api_servers, 900.0))
    finally:
        os.chdir(old_cwd)

    status = service_status()
    if not all(status.values()):
        missing = [port for port, ready in status.items() if not ready]
        raise RuntimeError(f"CaP-X services failed to start on ports: {missing}")

    for proc in servers:
        if hasattr(proc, "poll") and proc.poll() is None:
            _OWNED_SERVERS.append(proc)
    robotics_seconds = time.monotonic() - started
    LAST_SERVICE_TIMING.update(
        lemonade_seconds=lemonade_seconds,
        robotics_seconds=robotics_seconds,
        total_seconds=lemonade_seconds + robotics_seconds,
    )
    llm_note = f"LLM {lemonade_seconds:.1f}s" if model else "no LLM"
    progress(
        f"Services ready · {llm_note} · robotics {robotics_seconds:.1f}s"
        f" · details {SERVICE_LOG}"
    )
    return servers


def stop_owned_services() -> None:
    while _OWNED_SERVERS:
        proc = _OWNED_SERVERS.pop()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(stop_owned_services)


def _terminate_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_bounded(
    command: list[str],
    *,
    cwd: Path | str,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> BoundedRun:
    """Run a command in its own process group and enforce a wall-clock limit."""
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        for line in proc.stdout:
            output_queue.put(line)
        output_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    timed_out = False
    stream_done = False
    deadline = started + timeout_seconds
    process_exited_at: float | None = None

    while proc.poll() is None or not stream_done:
        if proc.poll() is not None:
            if process_exited_at is None:
                process_exited_at = time.monotonic()
            elif not stream_done and time.monotonic() - process_exited_at >= 1.0:
                # A detached descendant can inherit stdout after the bounded
                # process group exits. Do not let that orphaned pipe defeat the
                # wall-clock limit.
                break
        if proc.poll() is None and time.monotonic() >= deadline:
            timed_out = True
            _terminate_group(proc)
        try:
            line = output_queue.get(timeout=0.1)
        except queue.Empty:
            if heartbeat is not None:
                heartbeat()
            continue
        if line is None:
            stream_done = True
            continue
        lines.append(line)
        if progress is not None:
            progress(line.rstrip())

    proc.stdout.close()
    return BoundedRun(
        returncode=124 if timed_out else int(proc.returncode or 0),
        timed_out=timed_out,
        stdout="".join(lines),
        elapsed_seconds=time.monotonic() - started,
    )


def _trial_id(candidate_root: Path, split: str, example_id: str) -> int:
    if split not in {"train", "val"}:
        raise ValueError(f"unknown evaluation split: {split}")
    override = os.environ.get("RHO_TRIAL_ID")
    if override is not None:
        trial = int(override)
        if trial < 0:
            raise ValueError("trial override must be non-negative")
        return trial
    index = int(example_id)  # The legacy single-task evaluator uses positional IDs.
    path = candidate_root / "provenance.json"
    provenance = json.loads(path.read_text()) if path.exists() else {}
    # A pass-rate study scores one candidate across several distinct layouts, so
    # the positional id selects a trial rather than being discarded. Studies that
    # never record a trial list keep the original single-trial behaviour.
    trials = provenance.get("training_trials" if split == "train" else "heldout_trials")
    if trials:
        return int(trials[index % len(trials)])
    key = "training_trial" if split == "train" else "heldout_trial"
    return int(provenance.get(key, 1 if split == "train" else 2))


def _mock_evaluation(candidate_root: Path, split: str, example_id: str) -> dict[str, Any]:
    program = (candidate_root / "solver" / "program.py").read_text()
    compact = re.sub(r"\s+", "", program)
    invalid_accesses = (
        "green_pose[0][2]",
        "green_pose[0][0]",
        "green_pose[0][1]",
    )
    corrected_accesses = ("green_pose[2]", "green_pose[0]", "green_pose[1]")
    syntax_error = ""
    try:
        compile(program, "solver/program.py", "exec")
    except SyntaxError:
        syntax_error = traceback.format_exc()[-2400:]
    recorded_bug = any(access in compact for access in invalid_accesses)
    success = not syntax_error and not recorded_bug and all(access in compact for access in corrected_accesses)
    if syntax_error:
        feedback = "Mock execution rejected invalid Python; inspect traceback."
    elif recorded_bug:
        feedback = (
            "Recorded CaP-X failure reproduced: green_pose is an XYZ vector, "
            "so green_pose[0][…] indexes a scalar. Use green_pose[2], "
            "green_pose[0], and green_pose[1]."
        )
    elif success:
        feedback = "Mock execution accepted the corrected green_pose indexing."
    else:
        feedback = "Mock execution did not find the three direct green_pose accesses required by the recorded repair."
    return {
        "reward": float(success),
        "raw_reward": float(success),
        "task_completed": success,
        "split": split,
        "trial": _trial_id(candidate_root, split, example_id),
        "stdout": "",
        "stderr": "",
        "traceback": (
            syntax_error
            or ("IndexError: invalid index to scalar variable (green_pose[0][...])" if recorded_bug else "")
        ),
        "feedback": feedback,
        "video": None,
    }


def _info_value(info: Any, names: tuple[str, ...]) -> Any:
    if isinstance(info, Mapping):
        for name in names:
            if name in info and info[name] not in (None, ""):
                return info[name]
        for value in info.values():
            found = _info_value(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(info, (list, tuple)):
        for value in info:
            found = _info_value(value, names)
            if found not in (None, ""):
                return found
    return None


def _feedback_text(
    completed: bool,
    info: Any,
    stdout: str,
    stderr: str,
    sandbox_traceback: str,
) -> str:
    parts = ["Task completed." if completed else "Task not completed."]
    if stdout:
        parts.append(f"Sandbox stdout:\n{stdout[-1600:]}")
    if stderr:
        parts.append(f"Sandbox stderr:\n{stderr[-1600:]}")
    if sandbox_traceback:
        parts.append(f"Sandbox traceback:\n{sandbox_traceback[-2400:]}")
    if len(parts) == 1:
        parts.append(f"Simulator info: {str(info)[-1200:]}")
    return "\n\n".join(parts)


def seed_scene(env: Any, seed: int) -> int:
    """Make one trial id mean one scene, and report how many generators moved.

    Robosuite draws object placements from a Generator it builds when the
    environment is constructed, and CaP-X constructs it without a seed. The gym
    wrapper's ``reset(seed=...)`` only touches the legacy global ``np.random``,
    which the placement sampler never reads, so the same trial otherwise yields
    different cube positions on every evaluation. The sampler holds a reference
    to the Generator, so reseed it in place instead of replacing it.
    """
    import numpy as np

    low_level = getattr(env, "low_level_env", None)
    holders = [env, low_level, getattr(low_level, "robosuite_env", None)]
    reseeded = 0
    for holder in holders:
        for attribute in ("rng", "_rng"):
            generator = getattr(holder, attribute, None) if holder is not None else None
            if isinstance(generator, np.random.Generator):
                generator.bit_generator.state = np.random.default_rng(
                    seed
                ).bit_generator.state
                reseeded += 1
    np.random.seed(seed)
    return reseeded


NARRATION_PREFIX = "RHO_NARRATION "


class _NarrationTee(io.StringIO):
    """Buffer the policy's stdout while mirroring finished lines to ``sink``.

    CaP-X tees sandbox ``print()`` calls into whatever ``sys.stdout`` is bound to
    at the time, and ``_live_evaluation`` binds that to a buffer, so a replay is
    silent from outside the worker. Marking each line lets the parent process
    separate the policy's own narration from simulator setup noise and from the
    single JSON result line.
    """

    def __init__(self, sink: Any) -> None:
        super().__init__()
        self._sink = sink
        self._pending = ""

    def write(self, text: str) -> int:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if line.strip():
                self._sink.write(f"{NARRATION_PREFIX}{line}\n")
        self._sink.flush()
        return super().write(text)


def _live_evaluation(
    candidate_root: Path,
    split: str,
    example_id: str,
    *,
    capture: bool,
    config_path: str = CONFIG_PATH,
    policy_path: str = "solver/program.py",
) -> dict[str, Any]:
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    env = None
    try:
        # Candidate policies may import editable repository modules such as
        # solver.geometry and solver.runtime. Keep those imports available
        # after changing into the CaP-X source tree for simulator setup.
        sys.path.insert(0, str(candidate_root.resolve()))
        os.chdir(CAPX_ROOT)
        from capx.envs.configs.instantiate import instantiate
        from capx.envs.launch import LaunchArgs
        from capx.utils.launch_utils import _load_config

        args = LaunchArgs(
            config_path=config_path,
            model=MODEL,
            server_url="http://127.0.0.1:13305/api/v1/chat/completions",
            temperature=0.2,
            max_tokens=4096,
        )
        env_factory, _, _ = _load_config(args)
        env = instantiate(env_factory)
        trial = _trial_id(candidate_root, split, example_id)
        seed_scene(env, trial)
        env.reset(options={"trial": trial}, seed=trial)

        if capture:
            env.enable_video_capture()
        program = (candidate_root / policy_path).read_text()
        narrate = os.environ.get("RHO_LIVE_NARRATION") == "1"
        captured_stdout = _NarrationTee(sys.stdout) if narrate else io.StringIO()
        captured_stderr = io.StringIO()
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                _, reward, _, _, info = env.step(program)
        except Exception:
            sandbox_traceback = traceback.format_exc()[-2400:]
            stdout = captured_stdout.getvalue()
            stderr = captured_stderr.getvalue()
            return {
                "reward": 0.0,
                "raw_reward": 0.0,
                "task_completed": False,
                "split": split,
                "trial": trial,
                "stdout": stdout,
                "stderr": stderr,
                "traceback": sandbox_traceback,
                "feedback": _feedback_text(False, {}, stdout, stderr, sandbox_traceback),
                "video": None,
            }

        raw_reward = float(reward)
        sandbox_stdout = str(_info_value(info, ("sandbox_stdout", "stdout")) or "")
        sandbox_stderr = str(_info_value(info, ("sandbox_stderr", "stderr")) or "")
        sandbox_traceback = str(
            _info_value(
                info,
                ("sandbox_traceback", "traceback", "exception", "error"),
            )
            or ""
        )
        sandbox_rc_value = _info_value(info, ("sandbox_rc", "sandbox_returncode", "returncode"))
        stdout = captured_stdout.getvalue()
        stderr = captured_stderr.getvalue()
        if sandbox_stdout and sandbox_stdout not in stdout:
            stdout = (stdout + sandbox_stdout).strip()
        if sandbox_stderr and sandbox_stderr not in stderr:
            stderr = (stderr + sandbox_stderr).strip()
        execution_failed = bool(
            (sandbox_rc_value is not None and int(sandbox_rc_value) != 0) or sandbox_traceback or sandbox_stderr.strip()
        )
        # Partial robot motion can earn environment reward before generated
        # code crashes. RHO optimizes deployable policies, so an execution
        # failure receives zero evaluator score while retaining raw_reward for
        # diagnostics.
        score = 0.0 if execution_failed else raw_reward
        completed_value = _info_value(info, ("task_completed", "task_success", "success"))
        completed = not execution_failed and (
            bool(completed_value) if completed_value is not None else bool(score >= 1.0)
        )
        video: str | None = None
        if capture:
            from capx.utils.video_utils import _write_video

            frames = env.get_video_frames(clear=True)
            if frames:
                VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
                suffix = f"{split}_{example_id}_{int(time.time())}"
                _write_video(frames, str(VIDEO_ROOT), suffix=suffix)
                video = str(VIDEO_ROOT / f"video_{suffix}.mp4")
        return {
            "reward": score,
            "raw_reward": raw_reward,
            "task_completed": completed,
            "split": split,
            "trial": trial,
            "stdout": stdout,
            "stderr": stderr,
            "traceback": sandbox_traceback,
            "feedback": _feedback_text(completed, info, stdout, stderr, sandbox_traceback)
            + (f"\n\nRaw simulator reward before execution penalty: {raw_reward:.4f}." if execution_failed else ""),
            "video": video,
        }
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        sys.path[:] = old_sys_path
        os.chdir(old_cwd)


def _worker_result(
    candidate_root: Path,
    split: str,
    example_id: str,
    capture: bool,
    config_path: str = CONFIG_PATH,
    policy_path: str = "solver/program.py",
) -> dict[str, Any]:
    try:
        if os.environ.get("RHO_MOCK_EVAL") == "1":
            return _mock_evaluation(candidate_root, split, example_id)
        return _live_evaluation(
            candidate_root,
            split,
            example_id,
            capture=capture,
            config_path=config_path,
            policy_path=policy_path,
        )
    except Exception:
        return {
            "reward": 0.0,
            "raw_reward": 0.0,
            "task_completed": False,
            "split": split,
            "trial": _trial_id(candidate_root, split, example_id),
            "stdout": "",
            "stderr": "",
            "traceback": traceback.format_exc()[-2400:],
            "feedback": "Candidate raised during execution; inspect traceback.",
            "video": None,
        }


def score_candidate(
    candidate_root: Path | str,
    split: str = "train",
    example_id: str = "0",
    *,
    trial: int | None = None,
    capture: bool = False,
    timeout_seconds: float = EVALUATION_TIMEOUT,
    config_path: str = CONFIG_PATH,
    policy_path: str = "solver/program.py",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one layout in a killable child without modifying the candidate.

    ``progress`` receives each line the policy writes as it is written, which is
    how a caller watches a frozen repository work rather than waiting on a
    silent subprocess.
    """
    candidate_root = Path(candidate_root).resolve()
    if trial is not None and trial < 0:
        raise ValueError("trial must be non-negative")
    selected_trial = int(trial) if trial is not None else _trial_id(candidate_root, split, example_id)
    worker_env = os.environ.copy()
    worker_env["RHO_VIDEO_ROOT"] = str(VIDEO_ROOT)
    if trial is not None:
        worker_env["RHO_TRIAL_ID"] = str(trial)
    if progress is not None:
        # Only a caller that is watching pays for the extra stdout traffic; the
        # HELIX evaluation path reads execution_tail and stays untouched.
        worker_env["RHO_LIVE_NARRATION"] = "1"
    worker = run_bounded(
        [
            str(CAPX_PYTHON if CAPX_PYTHON.exists() else Path(sys.executable)),
            str(Path(__file__).resolve()),
            "_evaluate_worker",
            str(candidate_root),
            split,
            str(example_id),
            "1" if capture else "0",
            config_path,
            policy_path,
        ],
        cwd=candidate_root,
        timeout_seconds=timeout_seconds,
        env=worker_env,
        progress=progress,
    )
    if worker.timed_out:
        return {
            "reward": 0.0,
            "raw_reward": 0.0,
            "task_completed": False,
            "split": split,
            "trial": selected_trial,
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "feedback": f"Evaluation timed out after {timeout_seconds:.0f}s.",
            "video": None,
            "timed_out": True,
            "elapsed_seconds": worker.elapsed_seconds,
        }
    for line in reversed(worker.stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "reward" in result:
            result["execution_tail"] = "\n".join(worker.stdout.splitlines()[-12:-1])[-1200:]
            result["timed_out"] = False
            result["elapsed_seconds"] = worker.elapsed_seconds
            return result
    return {
        "reward": 0.0,
        "raw_reward": 0.0,
        "task_completed": False,
        "split": split,
        "trial": selected_trial,
        "stdout": "",
        "stderr": "",
        "traceback": worker.stdout[-2400:],
        "feedback": "Evaluator worker returned no JSON result.",
        "video": None,
        "timed_out": False,
        "elapsed_seconds": worker.elapsed_seconds,
    }


def _append_progress_event(event: Mapping[str, Any]) -> None:
    path_value = os.environ.get("RHO_PROGRESS_FILE")
    if not path_value:
        return
    try:
        with Path(path_value).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event), separators=(",", ":")) + "\n")
    except OSError:
        # Notebook feedback must never make an evaluation fail.
        pass


def evaluate_cli() -> int:
    """Emit HELIX's exact positional per-example result protocol."""
    root = Path.cwd()
    batch_path = root / "helix_batch.json"
    ids = json.loads(batch_path.read_text()) if batch_path.exists() else ["0"]
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must be a JSON list of strings")
    split = os.environ.get("HELIX_SPLIT", "train")
    timeout = float(os.environ.get("RHO_EVAL_TIMEOUT", EVALUATION_TIMEOUT))
    origin = os.environ.get("RHO_EVAL_ORIGIN", "helix")
    payload: list[list[Any]] = []
    for example_id in ids:
        trial = _trial_id(root, split, example_id)
        evaluation_id = f"{os.getpid()}-{time.time_ns()}-{example_id}"
        _append_progress_event(
            {
                "event": "started",
                "evaluation_id": evaluation_id,
                "started_at": time.time(),
                "origin": origin,
                "split": split,
                "trial": trial,
            }
        )
        result = score_candidate(root, split, example_id, timeout_seconds=timeout)
        _append_progress_event(
            {
                "event": "completed",
                "evaluation_id": evaluation_id,
                "origin": origin,
                "split": result["split"],
                "trial": result["trial"],
                "reward": result["reward"],
                "raw_reward": result.get("raw_reward", result["reward"]),
                "task_completed": result["task_completed"],
                "timed_out": result.get("timed_out", False),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
        side_info = {
            "reward": result["reward"],
            "raw_reward": result.get("raw_reward", result["reward"]),
            "task_completed": result["task_completed"],
            "split": result["split"],
            "trial": result["trial"],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "traceback": result.get("traceback", ""),
            "feedback": result.get("feedback", ""),
            "video": result.get("video"),
            "execution_tail": result.get("execution_tail", ""),
            "timed_out": result.get("timed_out", False),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "scores": {"completion": result["reward"]},
        }
        payload.append([float(result["reward"]), side_info])
    print("HELIX_RESULT=" + json.dumps(payload, separators=(",", ":")))
    return 0


def run_helix(
    root: Path | str = CANDIDATE_ROOT,
    *,
    generations: int = 1,
    timeout_seconds: float = DEFAULT_TIMEOUT,
    progress: Callable[[str], None] = print,
    merge: bool = False,
) -> BoundedRun:
    """Stream a bounded HELIX evolution in the disposable candidate repo."""
    root = Path(root).resolve()
    # The candidate's own helix.toml decides how long a search may run, so the
    # workshop repository stays bounded at 4 while a study can configure more
    # without this guard needing to know which is which.
    config_path = root / "helix.toml"
    permitted = 4
    if config_path.is_file():
        declared = re.search(
            r"^max_generations\s*=\s*(\d+)", config_path.read_text(), re.M
        )
        if declared:
            permitted = int(declared.group(1))
    if not 1 <= generations <= permitted:
        raise ValueError(
            f"generations must be between 1 and {permitted} for this repository"
        )
    evaluation_log = Path(
        f"/tmp/rho-evaluations-{os.getpid()}-{time.time_ns()}.jsonl"
    )
    evaluation_log.write_text("", encoding="utf-8")
    run_env = os.environ.copy()
    run_env.pop("RHO_EVAL_ORIGIN", None)
    run_env["RHO_PROGRESS_FILE"] = str(evaluation_log)
    command = [
        str(HELIX if HELIX.exists() else Path("helix")),
        "evolve",
        "--dir",
        str(root),
        "--config",
        "helix.toml",
        "--generations",
        str(generations),
    ]
    if not merge:
        command.append("--no-merge")
    notebook_display = (
        _notebook_progress(evaluation_log) if progress is print else None
    )
    result = run_bounded(
        command,
        cwd=root,
        timeout_seconds=timeout_seconds,
        env=run_env,
        progress=notebook_display or progress,
        heartbeat=notebook_display.tick if notebook_display is not None else None,
    )
    if notebook_display is not None:
        notebook_display.finish(result)
    evaluation_log.unlink(missing_ok=True)
    if result.timed_out:
        progress(f"HELIX stopped at the {timeout_seconds:.0f}s workshop deadline.")
    return result


def source_diff(before: Path | str, after: Path | str) -> str:
    before, after = Path(before), Path(after)
    chunks: list[str] = []
    relatives = {
        path.relative_to(before)
        for path in (before / "solver").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    relatives.update(
        path.relative_to(after)
        for path in (after / "solver").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for relative in sorted(relatives, key=lambda path: path.as_posix()):
        old_path = before / relative
        new_path = after / relative
        old = old_path.read_text().splitlines(keepends=True) if old_path.exists() else []
        new = new_path.read_text().splitlines(keepends=True) if new_path.exists() else []
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=(f"seed/{relative.as_posix()}" if old_path.exists() else "/dev/null"),
                tofile=(f"best/{relative.as_posix()}" if new_path.exists() else "/dev/null"),
            )
        )
    return "".join(chunks)


def semantic_mutation(diff: str) -> list[str]:
    removed: dict[str, str] = {}
    changes: list[str] = []
    assignment = re.compile(r"^[+-]([A-Z][A-Z0-9_]*)\s*=\s*(.+)$")
    for line in diff.splitlines():
        match = assignment.match(line)
        if not match:
            continue
        name, value = match.groups()
        if line.startswith("-"):
            removed[name] = value
        elif name in removed:
            changes.append(f"{name}: {removed[name]} -> {value}")
    if not changes and diff:
        files = sorted({line.removeprefix("+++ best/") for line in diff.splitlines() if line.startswith("+++ best/")})
        changes = [f"Changed {name}" for name in files]
    return changes


def export_best(root: Path | str = CANDIDATE_ROOT) -> Path:
    root = Path(root)
    destination = root.parent / "live_best"
    _safe_reset(destination)
    try:
        done = subprocess.run(
            [
                str(HELIX if HELIX.exists() else Path("helix")),
                "best",
                "--dir",
                str(root),
                "--export",
                str(destination),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except OSError:
        return root
    return destination if done.returncode == 0 and destination.exists() else root


def summarize_run(root: Path | str = CANDIDATE_ROOT) -> dict[str, Any]:
    root = Path(root)
    best = export_best(root)
    best_diff = source_diff(root, best)
    state_path = root / ".helix" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    child_ids = [candidate_id for candidate_id in state.get("frontier", []) if candidate_id != "g0-s0"]
    candidates = [root / ".helix" / "worktrees" / candidate_id for candidate_id in child_ids]
    candidate_diffs = {
        candidate.name: source_diff(root, candidate) for candidate in candidates if (candidate / "solver").is_dir()
    }
    child = candidates[-1] if candidates else None
    child_diff = candidate_diffs.get(child.name, "") if child is not None else ""
    return {
        "accepted": bool(best_diff.strip()),
        "improved_best": bool(best_diff.strip()),
        "live_best": str(best),
        "best_diff": best_diff,
        "child_candidate": str(child) if child is not None else None,
        "semantic_mutation": semantic_mutation(child_diff or best_diff),
        "child_diff": child_diff,
        "candidate_diffs": candidate_diffs,
        "fallback": {
            "candidate": str(root),
            "trace": {"mutation": []},
            "label": (
                "NO LIVE CHILD ACCEPTED — the authentic failed seed is retained; no prerecorded success is substituted."
            ),
        },
    }


def live_smoke_cli() -> int:
    """Run the file-backed live path used by the optional image smoke test."""
    ensure_services()
    # Cold-loading four perception/control services is a one-time environment
    # setup cost. The workshop's ten-minute bound applies to the RHO mutation
    # and paired rollouts that follow, matching the notebook's cell structure.
    started = time.monotonic()
    root = prepare_workshop()
    seed = score_candidate(root, "train")
    run = run_helix(root, generations=1, timeout_seconds=DEFAULT_TIMEOUT)
    summary = summarize_run(root)
    heldout = score_candidate(summary["live_best"], "val", capture=True)
    elapsed = time.monotonic() - started
    if run.timed_out:
        raise RuntimeError("HELIX exceeded its 480-second hard deadline")
    if run.returncode != 0:
        raise RuntimeError(run.stdout[-4000:])
    if elapsed >= 600:
        raise RuntimeError(f"live workshop path took {elapsed:.1f}s")
    if heldout.get("timed_out"):
        raise RuntimeError(f"held-out evaluation timed out: {heldout}")
    result = {
        "seed_reward": seed["reward"],
        "accepted": summary["accepted"],
        "live_best": summary["live_best"],
        "best_diff": summary["best_diff"],
        "heldout_reward": heldout["reward"],
        "heldout_completed": heldout["task_completed"],
        "heldout_video": heldout.get("video"),
        "heldout_feedback": heldout.get("feedback", ""),
        "helix_seconds": round(run.elapsed_seconds, 1),
        "total_seconds": round(elapsed, 1),
    }
    print("RHO_LIVE_RESULT=" + json.dumps(result, separators=(",", ":")))
    return 0


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: rho_demo.py {prepare|evaluate|run|summary}")
        return 2
    if argv[0] == "prepare":
        artifact = Path(argv[1]) if len(argv) > 1 else None
        provenance = Path(argv[2]) if len(argv) > 2 else None
        print(prepare_workshop(artifact=artifact, provenance=provenance))
        return 0
    if argv[0] == "evaluate":
        return evaluate_cli()
    if argv[0] == "run":
        result = run_helix()
        print(json.dumps(asdict(result), indent=2))
        return result.returncode
    if argv[0] == "summary":
        print(json.dumps(summarize_run(), indent=2))
        return 0
    if argv[0] == "live-smoke":
        return live_smoke_cli()
    if argv[0] == "_evaluate_worker":
        result = _worker_result(
            Path(argv[1]),
            argv[2],
            argv[3],
            bool(int(argv[4])),
            argv[5] if len(argv) > 5 else CONFIG_PATH,
            argv[6] if len(argv) > 6 else "solver/program.py",
        )
        print(json.dumps(result, separators=(",", ":")))
        return 0
    if argv[0] == "_sleep":
        time.sleep(float(argv[1]))
        return 0
    raise ValueError(f"unknown command: {argv[0]}")


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
