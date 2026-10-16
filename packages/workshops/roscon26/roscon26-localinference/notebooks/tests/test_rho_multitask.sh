#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -euo pipefail

CAPX_PY="${CAPX_VENV:-/opt/capx-venv}/bin/python"
export PYTHONPATH="/ryzers/notebooks/scripts${PYTHONPATH:+:${PYTHONPATH}}"

echo "================ Multi-task RHO static harness ================"
RHO_MULTITASK_MOCK=1 "${CAPX_PY}" - <<'PY'
import json
import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from helix.config import load_config
import rho_demo
import rho_multitask_demo as demo

override_env = os.environ.copy()
override_env.pop("RHO_MULTITASK_MODEL", None)
override_env["RHO_MODEL"] = "rho-model-override"
override = subprocess.run(
    [
        sys.executable,
        "-c",
        "import rho_multitask_demo as demo; print(demo.DEFAULT_MODEL)",
    ],
    env=override_env,
    check=True,
    capture_output=True,
    text=True,
)
assert override.stdout.strip() == "rho-model-override"


with tempfile.TemporaryDirectory(prefix="rho-multitask-", dir="/tmp") as temporary:
    root = demo.prepare_workshop(Path(temporary) / "candidate")
    manifest = json.loads((root / "scenarios.json").read_text())
    assert manifest["schema_version"] == "rho-multitask-scenarios/v2"
    assert manifest["splits"] == {
        "train": ["stack_train", "lift_train"],
        "val": [
            "stack_val1",
            "stack_val2",
            "stack_val3",
            "lift_val1",
            "lift_val2",
            "lift_val3",
        ],
    }
    assert set(manifest["scenarios"]) == {
        "stack_train",
        "lift_train",
        "stack_val1",
        "stack_val2",
        "stack_val3",
        "lift_val1",
        "lift_val2",
        "lift_val3",
    }
    assert {scenario["task"] for scenario in manifest["scenarios"].values()} == {
        "cube_stack",
        "cube_lift",
    }
    assert {
        path.name for path in (root / "solver" / "tasks").glob("*.py")
    } == {"__init__.py", "cube_stack.py", "cube_lift.py"}
    # Each task contributes three distinct validation layouts.
    val_trials = {
        scenario_id: manifest["scenarios"][scenario_id]["trial"]
        for scenario_id in manifest["splits"]["val"]
    }
    assert sorted(val_trials.values()) == [2, 2, 3, 3, 4, 4]
    assert 'os.environ.get("RHO_SUPPORT_ROOT", "/ryzers/notebooks/scripts")' in (
        root / "probe.py"
    ).read_text()
    assert '"RHO_SUPPORT_ROOT"' in (root / "helix.toml").read_text()
    os.environ["RHO_TRIAL_ID"] = "2"
    try:
        assert rho_demo._trial_id(root, "val", "stack_val") == 2
    finally:
        os.environ.pop("RHO_TRIAL_ID")
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["schema_version"] == "rho-multitask-fixtures/v2"
    assert provenance["seed_model"] == "Gemma-4-E4B-it-GGUF"
    assert {"cube_stack", "cube_lift"} <= set(provenance["policies"])
    for policy in provenance["policies"].values():
        assert len(policy["source_policy_sha256"]) == 64
        assert policy["source_prompt"]
        assert policy["source_trial"] >= 1
        # A sweep run outside a git checkout records no revision, so the key
        # must be present but may legitimately be null.
        assert "source_git_commit" in policy
    assert demo.resolve_scenarios(root, "train", ["0", "1"])[0][0] == "stack_train"
    assert demo.resolve_scenarios(root, "val", ["1"])[0][0] == "stack_val2"
    assert demo.resolve_scenarios(root, "val", ["3"])[0][0] == "lift_val1"
    for invalid_id in ("stack_val1", "-1", "2"):
        try:
            demo.resolve_scenarios(root, "train", [invalid_id])
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid train scenario was accepted: {invalid_id}")
    shuffled = list(manifest["splits"]["train"])
    random.Random(29).shuffle(shuffled)
    assert shuffled == ["lift_train", "stack_train"]

    config = load_config(root / "helix.toml")
    assert config.dataset.train_size == 2
    assert config.dataset.val_size == 6
    assert config.evolution.max_generations == 2
    assert config.evolution.perfect_score_threshold == 1.1
    assert config.evolution.minibatch_size == 1
    assert config.evolution.num_parallel_proposals == 2
    assert config.evolution.max_workers == 1
    assert config.evolution.merge_enabled is True
    assert config.evolution.max_merge_invocations == 2
    assert config.evolution.merge_val_overlap_floor == 1
    assert config.evolution.merge_subsample_size == 6
    assert config.evolution.frontier_type == "instance"
    assert config.evolution.acceptance_criterion == "strict_improvement"

    opencode = json.loads((root / "opencode.json").read_text())
    assert opencode["model"] == f"lemonade/{demo.opencode_model_id()}"
    assert opencode["provider"]["lemonade"]["models"][
        demo.opencode_model_id()
    ]["tool_call"] is True
    assert opencode["agent"]["build"]["steps"] == 24
    assert config.agent.model == f"lemonade/{demo.opencode_model_id()}"
    assert config.agent.max_turns == 24
    assert opencode["permission"]["edit"]["solver/**"] == "allow"
    assert opencode["permission"]["edit"]["*"] == "deny"
    assert "change green_pose" not in demo.BACKGROUND
    assert "except ValueError" not in demo.BACKGROUND

    protected = [
        "probe.py",
        "helix.toml",
        "opencode.json",
        "CONTRACT.md",
        "scenarios.json",
        "provenance.json",
    ]
    protected_before = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in protected
    }
    (root / "helix_batch.json").write_text('["0","1"]\n')
    env = os.environ.copy()
    env.update(RHO_MULTITASK_MOCK="1", HELIX_SPLIT="train")
    done = subprocess.run(
        [sys.executable, str(Path(demo.__file__)), "evaluate"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(done.stdout.split("HELIX_RESULT=", 1)[1])
    assert [entry[1]["task"] for entry in payload] == ["cube_stack", "cube_lift"]
    assert [entry[0] for entry in payload] == [0.0, 0.0]
    assert set(payload[0][1]["scores"]) == {"completion", "raw_reward", "deployable"}
    assert demo.MOCK_LABEL in payload[0][1]["feedback"]
    assert protected_before == {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in protected
    }

    stack = root / "solver" / "tasks" / "cube_stack.py"
    stack.write_text(
        stack.read_text()
        .replace("green_pose[0][2]", "green_pose[2]")
        .replace("green_pose[0][0]", "green_pose[0]")
        .replace("green_pose[0][1]", "green_pose[1]")
    )
    lift = root / "solver" / "tasks" / "cube_lift.py"
    lift.write_text("import numpy\n" + lift.read_text())
    train_results = [
        demo.score_scenario(root, "train", scenario_id, scenario)
        for scenario_id, scenario in demo.resolve_scenarios(root, "train", ["0", "1"])
    ]
    assert [result["reward"] for result in train_results] == [1.0, 1.0]
    try:
        demo.score_scenario(
            root,
            "val",
            "unknown_val",
            {
                "task": "unknown_task",
                "trial": 2,
                "policy_path": "solver/tasks/cube_stack.py",
                "config_path": "unused.yaml",
            },
        )
    except ValueError as exc:
        assert "unsupported mock task" in str(exc)
    else:
        raise AssertionError("unsupported mock task was accepted")

    root = demo.prepare_workshop(root)
    mocked = demo.materialize_mock_evolution(root)
    assert demo.MOCK_LABEL in mocked.stdout
    summary = demo.frontier_summary(root)
    assert summary["generation"] == 2
    assert summary["candidates"]["g1-s1"]["wins"] == [
        "stack_val1",
        "stack_val2",
        "stack_val3",
    ]
    assert summary["candidates"]["g1-s2"]["wins"] == [
        "lift_val1",
        "lift_val2",
        "lift_val3",
    ]
    assert summary["candidates"]["g0-s0"]["frontier"] is False
    assert summary["candidates"]["g2-m1"]["frontier"] is True
    assert all(
        set(candidate["scores"]) == set(manifest["splits"]["val"])
        for candidate in summary["candidates"].values()
    )
    assert summary["merge_counter"] == 1
    lesson = demo.evolution_lesson(summary)
    assert lesson["multi_key_frontier"] is True
    assert lesson["covered_difficult_keys"] == sorted(manifest["splits"]["val"])
    assert lesson["specialist_pair"] is True
    assert lesson["specialists"]["cube_stack"] == ["g1-s1"]
    assert lesson["specialists"]["cube_lift"] == ["g1-s2"]
    assert lesson["broad_candidates"] == ["g2-m1"]
    assert lesson["merge_attempted"] is True
    assert summary["merge_ancestry"] == [
        {"candidate": "g2-m1", "parents": ["g1-s1", "g1-s2"]}
    ]
    assert summary["merge_output_dedup_triplets"] == [
        ["g1-s1", "g1-s2", "4d6f636b4d65726765436f6d6d69745368613031"]
    ]
    lineage = {item["id"]: item for item in summary["lineage"]}
    assert lineage["g1-s1"]["changed_files"] == ["solver/tasks/cube_stack.py"]
    assert lineage["g1-s2"]["changed_files"] == ["solver/tasks/cube_lift.py"]
    assert lineage["g2-m1"]["parents"] == ["g1-s1", "g1-s2"]
    assert all(
        lineage[candidate_id]["gate_result"] == "passed_strict_train_gate"
        for candidate_id in ("g1-s1", "g1-s2")
    )
    assert lineage["g2-m1"]["gate_result"] == "passed_merge_validation_gate"
    for candidate_id, changed in (
        ("g1-s1", "cube_stack.py"),
        ("g1-s2", "cube_lift.py"),
    ):
        assert changed in rho_demo.source_diff(
            root / ".helix" / "worktrees" / "g0-s0",
            root / ".helix" / "worktrees" / candidate_id,
        )

    captured = {}
    original_run_bounded = rho_demo.run_bounded

    def fake_run_bounded(command, **kwargs):
        captured["command"] = command
        return rho_demo.BoundedRun(0, False, "", 0.1)

    rho_demo.run_bounded = fake_run_bounded
    try:
        run = demo.run_helix(root, progress=lambda _: None)
    finally:
        rho_demo.run_bounded = original_run_bounded
    assert run.returncode == 0
    assert "--no-merge" not in captured["command"]
    assert captured["command"][-2:] == ["--generations", "2"]

    hidden_calls = []
    original_score_scenario = demo.score_scenario

    def fake_score_scenario(candidate_root, split, scenario_id, scenario, **kwargs):
        hidden_calls.append((scenario_id, scenario["task"], scenario["trial"], kwargs["capture"]))
        return {
            "scenario_id": scenario_id,
            "task": scenario["task"],
            "trial": scenario["trial"],
            "reward": 1.0,
            "raw_reward": 1.0,
            "task_completed": True,
            "stderr": "",
            "traceback": "",
            "timed_out": False,
        }

    demo.score_scenario = fake_score_scenario
    try:
        hidden = demo.hidden_rollouts(
            root,
            trials={
                "cube_stack": [100, 101, 102, 103, 104],
                "cube_lift": [200, 201, 202, 203, 204],
            },
            capture=True,
        )
    finally:
        demo.score_scenario = original_score_scenario
    assert len(hidden) == 10
    assert len(hidden_calls) == 10
    assert all(call[3] is True for call in hidden_calls)
    assert hidden_calls[0][:3] == ("hidden_cube_stack_100", "cube_stack", 100)
    assert hidden_calls[-1][:3] == ("hidden_cube_lift_204", "cube_lift", 204)

    def rollout(task, trial, reward, completed):
        return {
            "task": task,
            "trial": trial,
            "reward": reward,
            "raw_reward": reward,
            "task_completed": completed,
            "stderr": "",
            "traceback": "",
            "timed_out": False,
        }

    before_rollouts = [
        *[rollout("cube_stack", trial, 0.0, False) for trial in range(100, 105)],
        *[rollout("cube_lift", trial, 1.0, True) for trial in range(200, 205)],
    ]
    after_rollouts = [
        rollout("cube_stack", 100, 1.0, True),
        *[rollout("cube_stack", trial, 0.0, False) for trial in range(101, 105)],
        *[rollout("cube_lift", trial, 1.0, True) for trial in range(200, 205)],
    ]
    before_summary = demo.summarize_rollouts(before_rollouts)
    after_summary = demo.summarize_rollouts(after_rollouts)
    assert set(before_summary) == {"cube_stack", "cube_lift"}
    assert not hasattr(demo, "hidden_success_criterion")
    criterion = demo.deployment_success_criterion(before_summary, after_summary)
    assert criterion["required_tasks"] == ["cube_stack", "cube_lift"]
    assert criterion["rollouts"] == 10
    assert criterion["completed_before"] == 5
    assert criterion["completed_after"] == 6
    assert criterion["completion_improved"] is True
    assert criterion["reward_improved"] is False
    assert criterion["deployment_improved"] is True
    assert criterion["met"] is True

    noisy_rollouts = [
        *[rollout("cube_stack", trial, 0.001, False) for trial in range(100, 105)],
        *[rollout("cube_lift", trial, 1.0, True) for trial in range(200, 205)],
    ]
    noisy_criterion = demo.deployment_success_criterion(
        before_summary,
        demo.summarize_rollouts(noisy_rollouts),
    )
    assert noisy_criterion["completion_improved"] is False
    assert noisy_criterion["reward_improved"] is False
    assert noisy_criterion["met"] is False

    reward_gain_rollouts = [
        *[rollout("cube_stack", trial, 0.2, False) for trial in range(100, 105)],
        *[rollout("cube_lift", trial, 1.0, True) for trial in range(200, 205)],
    ]
    reward_criterion = demo.deployment_success_criterion(
        before_summary,
        demo.summarize_rollouts(reward_gain_rollouts),
    )
    assert reward_criterion["completed_before"] == reward_criterion["completed_after"]
    assert abs(reward_criterion["mean_reward_after"] - 0.6) < 1e-9
    assert reward_criterion["completion_improved"] is False
    assert reward_criterion["reward_improved"] is True
    assert reward_criterion["met"] is True

    neutral_before = {
        "pick": {"rollouts": 2, "completed": 1, "mean_reward": 0.25},
        "place": {"rollouts": 2, "completed": 1, "mean_reward": 0.25},
    }
    neutral_after = {
        "pick": {"rollouts": 2, "completed": 1, "mean_reward": 0.35},
        "place": {"rollouts": 2, "completed": 1, "mean_reward": 0.35},
    }
    neutral_criterion = demo.deployment_success_criterion(
        neutral_before,
        neutral_after,
        required_tasks=("pick", "place"),
    )
    assert neutral_criterion["required_tasks"] == ["pick", "place"]
    assert neutral_criterion["reward_improved"] is True

    invalid_criteria = [
        (
            before_summary,
            {"cube_stack": after_summary["cube_stack"]},
            {},
        ),
        (
            before_summary,
            after_summary,
            {"required_tasks": ("cube_stack", "cube_stack")},
        ),
        (
            before_summary,
            after_summary,
            {"minimum_hard_reward_gain": -0.01},
        ),
        (
            before_summary,
            {
                **after_summary,
                "cube_stack": {**after_summary["cube_stack"], "rollouts": 4},
            },
            {},
        ),
    ]
    for invalid_before, invalid_after, kwargs in invalid_criteria:
        try:
            demo.deployment_success_criterion(
                invalid_before,
                invalid_after,
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid deployment summaries were accepted")

print("multi-task manifest, evaluator, frontier, and merge plumbing OK")
PY

echo "================ Multi-task RHO tests PASSED ================"
