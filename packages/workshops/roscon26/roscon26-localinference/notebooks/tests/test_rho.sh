#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Static RHO harness checks always run. Set RHO_RUN_LIVE=1 for one real local
# Gemma/OpenCode/HELIX mutation against the CaP-X simulator.
set -euo pipefail

CAPX_PY="${CAPX_VENV:-/opt/capx-venv}/bin/python"
HELIX_REF="29cfa6e5eae902f6bc6d2113e51499e92c6109ee"
OPENCODE_VERSION="1.18.18"

export PYTHONPATH="/ryzers/notebooks/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CAPX_ROOT="${CAPX_ROOT:-/ryzers/cap-x}"
export HF_HOME="${CAPX_CACHE:-/opt/capx-cache}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd /ryzers

echo "================ RHO toolchain ================"
test "$(git -C /ryzers/helix rev-parse HEAD)" = "${HELIX_REF}"
"${CAPX_PY}" -c "import capx, helix, torch; assert torch.version.hip"
"${CAPX_PY}" -c "from helix import __version__; assert __version__ == '0.2.1', __version__"
"${CAPX_VENV:-/opt/capx-venv}/bin/helix" --version
test "$(opencode --version)" = "${OPENCODE_VERSION}"
node --version

echo "================ RHO static harness ================"
RHO_MOCK_EVAL=1 "${CAPX_PY}" - <<'PY'
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import rho_demo
from helix.config import load_config

root = rho_demo.prepare_workshop(
    support_files={"solver/strategy.py": "CLEARANCE = 0.1\n"}
)
required = {
    "solver/__init__.py",
    "solver/geometry.py",
    "solver/program.py",
    "solver/policy.py",
    "solver/strategy.py",
    "API_REFERENCE.md",
    "probe.py",
    "opencode.json",
    "helix.toml",
    "provenance.json",
}
assert required <= {
    str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
}
assert (root / ".git").is_dir()
assert (root / "solver" / "program.py").read_text() == rho_demo.DEFAULT_PROGRAM
assert "program.py" in (root / "solver" / "policy.py").read_text()
assert (root / "solver" / "strategy.py").read_text() == "CLEARANCE = 0.1\n"

with tempfile.TemporaryDirectory(prefix="rho-custom-root-", dir="/tmp") as temporary:
    custom_root = Path(temporary) / "candidate"
    custom_root.mkdir()
    stale = custom_root / "stale.txt"
    stale.write_text("remove me")
    prepared = rho_demo.prepare_workshop(custom_root)
    assert prepared == custom_root.resolve()
    assert not stale.exists()

provenance = json.loads((root / "provenance.json").read_text())
assert provenance["source"] == "recorded_capx_generation"
assert provenance["artifact_trial"] == provenance["training_trial"] == 1
assert provenance["heldout_trial"] == 2
assert provenance["recorded_task_completed"] is False
assert not hasattr(rho_demo, "SEED_GEOMETRY")
assert not hasattr(rho_demo, "FROZEN_GEOMETRY")
assert not hasattr(rho_demo, "FALLBACK_ROOT")

config = load_config(root / "helix.toml")
assert config.agent.backend == "opencode"
assert config.agent.model == f"lemonade/{rho_demo.MODEL}"
assert config.dataset.train_size == config.dataset.val_size == 1
assert config.evolution.max_generations == 1
assert config.evolution.minibatch_size == 1
assert config.evolution.max_workers == 1
assert config.evolution.merge_enabled is False
assert config.evolution.acceptance_criterion == "strict_improvement"
assert set(config.evaluator.protected_files) >= {
    "probe.py",
    "helix.toml",
    "opencode.json",
    "provenance.json",
}
assert "max_generations = 2" in rho_demo.helix_config(2)
four_generation_config = rho_demo.helix_config(4)
assert "max_generations = 4" in four_generation_config
assert "max_evaluations = 14" in four_generation_config
custom_config = rho_demo.helix_config(
    objective="Repair another task.",
    background="Edit solver/program.py first.",
)
assert 'objective = """Repair another task."""' in custom_config
assert 'background = """Edit solver/program.py first."""' in custom_config
assert '"RHO_CONFIG_PATH"' in custom_config
assert '"RHO_PROGRESS_FILE"' in custom_config
try:
    rho_demo.helix_config(5)
except ValueError:
    pass
else:
    raise AssertionError("more than four generations must be rejected")

opencode = json.loads((root / "opencode.json").read_text())
permissions = opencode["permission"]
assert permissions["external_directory"] == "deny"
assert permissions["edit"]["*"] == "deny"
assert permissions["edit"]["solver/**"] == "allow"
assert permissions["edit"]["**/solver/**"] == "allow"
assert permissions["bash"]["*"] == "deny"
assert (
    permissions["bash"][
        "RHO_EVAL_ORIGIN=agent-self-check /opt/capx-venv/bin/python probe.py*"
    ]
    == "allow"
)
assert "/opt/capx-venv/bin/python probe.py*" not in permissions["bash"]
assert list(permissions["edit"])[0] == "*"
assert list(permissions["bash"])[0] == "*"
assert "RHO_EVAL_ORIGIN=agent-self-check" in rho_demo.DEFAULT_BACKGROUND

# The default authentic recording deterministically reproduces its scalar-index
# error without importing CaP-X.
before = rho_demo.score_candidate(root, "train", timeout_seconds=5)
assert before["reward"] == 0.0 and before["task_completed"] is False
assert before["trial"] == 1
assert before["elapsed_seconds"] >= 0.0
assert "green_pose[0]" in before["traceback"]

(root / "helix_batch.json").write_text('["0"]\n')
env = os.environ.copy()
with tempfile.NamedTemporaryFile(prefix="rho-evaluation-progress-", delete=False) as stream:
    progress_path = Path(stream.name)
env.update(
    RHO_MOCK_EVAL="1",
    HELIX_SPLIT="train",
    RHO_PROGRESS_FILE=str(progress_path),
)
done = subprocess.run(
    [sys.executable, str(Path(rho_demo.__file__)), "evaluate"],
    cwd=root,
    env=env,
    check=True,
    capture_output=True,
    text=True,
)
lines = done.stdout.splitlines()
assert len(lines) == 1 and lines[0].startswith("HELIX_RESULT="), done.stdout
payload = json.loads(lines[0].split("=", 1)[1])
assert payload[0][0] == 0.0
assert payload[0][1]["task_completed"] is False
assert payload[0][1]["trial"] == 1
progress_events = [
    json.loads(line) for line in progress_path.read_text().splitlines()
]
progress_path.unlink()
assert [event["event"] for event in progress_events] == ["started", "completed"]
assert progress_events[0]["trial"] == progress_events[1]["trial"] == 1
assert progress_events[1]["reward"] == 0.0
assert progress_events[0]["origin"] == progress_events[1]["origin"] == "helix"

class DisplayHandle:
    def __init__(self):
        self.updates = []

    def update(self, value):
        self.updates.append(value.data)


with tempfile.NamedTemporaryFile(
    prefix="rho-widget-progress-", mode="w", delete=False
) as stream:
    widget_progress_path = Path(stream.name)
handle = DisplayHandle()
with patch("IPython.display.display", return_value=handle):
    widget = rho_demo._NotebookHelixProgress(widget_progress_path)
with widget_progress_path.open("a") as stream:
    stream.write(json.dumps(progress_events[0]) + "\n")
widget("Status: Applying mutation")
assert "HELIX eval 1 running" in handle.updates[-1]
widget("Status: Evaluating")
assert "Applying mutation" not in handle.updates[-1]
assert "Status: Evaluating" not in handle.updates[-1]
with widget_progress_path.open("a") as stream:
    stream.write(json.dumps(progress_events[1]) + "\n")
widget.tick()
assert "reward 0.000" in handle.updates[-1]
assert "HELIX eval 1" in handle.updates[-1]
assert widget._evaluations == 1
widget("Status: Applying mutation")
assert "Applying mutation · 0s elapsed" in handle.updates[-1]
stable_update = handle.updates[-1]
widget("unrelated Rich redraw")
assert handle.updates[-1] == stable_update

self_check_started = {
    **progress_events[0],
    "evaluation_id": "self-check",
    "origin": "agent-self-check",
}
self_check_completed = {
    **progress_events[1],
    "evaluation_id": "self-check",
    "origin": "agent-self-check",
    "reward": 1.0,
    "task_completed": True,
}
with widget_progress_path.open("a") as stream:
    stream.write(json.dumps(self_check_started) + "\n")
    stream.write(json.dumps(self_check_completed) + "\n")
widget.tick()
assert "Agent self-check 1" in handle.updates[-1]
assert widget._evaluations == 1
widget_progress_path.unlink()

assert {
    "reward",
    "stdout",
    "stderr",
    "traceback",
    "feedback",
    "video",
    "scores",
} <= payload[0][1].keys()

program_path = root / "solver" / "program.py"
corrected = program_path.read_text()
corrected = corrected.replace("green_pose[0][2]", "green_pose[2]")
corrected = corrected.replace("green_pose[0][0]", "green_pose[0]")
corrected = corrected.replace("green_pose[0][1]", "green_pose[1]")
program_path.write_text(corrected)
after = rho_demo.score_candidate(root, "val", timeout_seconds=5)
assert after["reward"] == 1.0 and after["task_completed"]
assert after["trial"] == 2
provenance_before_override = (root / "provenance.json").read_text()
random_trial = rho_demo.score_candidate(
    root, "val", trial=7331, timeout_seconds=5
)
assert random_trial["reward"] == 1.0 and random_trial["trial"] == 7331
assert (root / "provenance.json").read_text() == provenance_before_override

# A real CaP-X trial directory is accepted as input, copied byte-for-byte, and
# fixes the training trial to the artifact while preserving caller metadata.
with tempfile.TemporaryDirectory(prefix="rho-artifact-") as temporary:
    temporary = Path(temporary)
    artifact = temporary / "trial_07_sandboxrc_1_reward_0.000_taskcompleted_0"
    artifact.mkdir()
    artifact_code = "# authentic generated bytes\nprint('artifact')\n"
    (artifact / "code.py").write_text(artifact_code)
    artifact_root = temporary / "candidate"
    rho_demo.prepare_workshop(
        artifact_root,
        artifact=artifact,
        provenance={"run": "workshop-recording"},
        heldout_trial=9,
    )
    assert (artifact_root / "solver" / "program.py").read_text() == artifact_code
    artifact_provenance = json.loads(
        (artifact_root / "provenance.json").read_text()
    )
    assert artifact_provenance["training_trial"] == 7
    assert artifact_provenance["heldout_trial"] == 9
    assert artifact_provenance["run"] == "workshop-recording"

bounded = rho_demo.run_bounded(
    [sys.executable, str(Path(rho_demo.__file__)), "_sleep", "2"],
    cwd=root,
    timeout_seconds=0.2,
)
assert bounded.timed_out and bounded.returncode == 124
print("mock repair, HELIX_RESULT, permissions, and timeout OK")
PY

if [[ "${RHO_RUN_LIVE:-0}" != "1" ]]; then
    echo "Live mutation skipped; run with RHO_RUN_LIVE=1 on a GPU workshop pod."
    echo "================ RHO tests PASSED ================"
    exit 0
fi

echo "================ RHO live one-generation smoke ================"
"${CAPX_PY}" /ryzers/notebooks/scripts/rho_demo.py live-smoke

echo "================ RHO tests PASSED ================"
