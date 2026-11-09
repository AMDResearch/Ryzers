#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run and record the two-task RHO repository-evolution study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from time import monotonic


HIDDEN_TRIALS = {
    "cube_stack": [8740, 9351, 6027, 7419, 4883],
    "cube_lift": [2596, 3841, 5107, 6773, 8215],
}

RESULT_KEYS = (
    "scenario_id",
    "task",
    "trial",
    "reward",
    "raw_reward",
    "task_completed",
    "timed_out",
    "stderr",
    "traceback",
    "feedback",
    "video",
    "elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="user.Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    # Recording runs search wider and longer than the bounded workshop demo.
    parser.add_argument("--proposals", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=40)
    parser.add_argument(
        "--hidden-repeats",
        type=int,
        default=1,
        help=(
            "replays of each held-out trial; one rollout per trial is "
            "noise-dominated, so a stable pass rate needs several"
        ),
    )
    return parser.parse_args()


def compact(result: dict) -> dict:
    return {key: result.get(key) for key in RESULT_KEYS}


def repeated_hidden_rollouts(experiment, root: Path, *, repeats: int) -> list[dict]:
    """Replay every held-out trial ``repeats`` times.

    A single rollout per trial is noise-dominated: the same frozen policy has
    been measured at 2/5 and 4/5 on the same cube_stack trials. Only the first
    pass captures video, since the notebook shows one clip per trial and the
    later passes exist to stabilise the pass rate.
    """
    results: list[dict] = []
    for index in range(repeats):
        results.extend(
            compact(result)
            for result in experiment.hidden_rollouts(
                root,
                trials=HIDDEN_TRIALS,
                capture=(index == 0),
            )
        )
    return results


def evaluate_split(experiment, root: Path, split: str) -> list[dict]:
    names = json.loads((root / "scenarios.json").read_text())["splits"][split]
    resolved = experiment.resolve_scenarios(
        root,
        split,
        [str(index) for index in range(len(names))],
    )
    return [
        compact(
            experiment.score_scenario(
                root,
                split,
                scenario_id,
                scenario,
                timeout_seconds=180.0,
            )
        )
        for scenario_id, scenario in resolved
    ]


def selected_candidate(root: Path, frontier: dict) -> tuple[str, Path, bool]:
    """Pick the deployed repository the way RHO defines it.

    RHO deploys the highest-mean-validation-reward member of the *terminal
    frontier*. Coverage dominance can prune a candidate whose score sum beats
    every retained one -- (0.9, 0.9) is best on neither instance and loses to
    (1.0, 0.0) and (0.0, 1.0) -- so the retained set has to be filtered before
    the maximum is taken.
    """
    candidates = frontier["candidates"]
    retained = {
        candidate_id: candidate
        for candidate_id, candidate in candidates.items()
        if candidate.get("frontier")
    }
    from_frontier = bool(retained)
    pool = retained or candidates
    candidate_id = max(
        pool,
        key=lambda current: (
            sum(pool[current]["scores"].values()),
            current,
        ),
    )
    worktree = root / ".helix" / "worktrees" / candidate_id
    return candidate_id, worktree if worktree.is_dir() else root, from_frontier


def export_repository(source: Path, destination: Path) -> list[str]:
    """Copy an evolvable surface so the notebook can replay rollouts live."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source / "solver",
        destination / "solver",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Every protected file travels too: the notebook prints the evaluator and
    # the mutator permission set to show what the search space excludes.
    for name in (
        "scenarios.json",
        "provenance.json",
        "CONTRACT.md",
        "helix.toml",
        "probe.py",
        "opencode.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    return sorted(
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file()
    )


def relativize_media(results: list[dict], base: Path) -> None:
    """Store video paths relative to the recorded root.

    The study host and the course image mount this tree at different absolute
    paths, so the report must not bake in the producing host's layout.
    """
    for result in results:
        video = result.get("video")
        if not video:
            continue
        try:
            result["video"] = str(Path(video).resolve().relative_to(base))
        except ValueError:
            continue


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["RHO_MODEL"] = args.model
    os.environ["RHO_MULTITASK_MODEL"] = args.model
    os.environ["RHO_WORKSHOP_ROOT"] = str(output_dir)
    os.environ["RHO_VIDEO_ROOT"] = str(output_dir / "videos")
    os.environ.setdefault("RHO_SUPPORT_ROOT", str(Path(__file__).resolve().parent))

    import rho_demo
    import rho_multitask_demo as experiment

    root = output_dir / "candidate"
    report_path = output_dir / "rho_multitask_report.json"
    try:
        setup_started = monotonic()
        experiment.ensure_services()
        setup_seconds = monotonic() - setup_started

        experiment.prepare_workshop(
            root,
            model=args.model,
            generations=args.generations,
            helix_overrides={
                "generations_limit": args.generations,
                "num_parallel_proposals": args.proposals,
                "minibatch_size": args.minibatch_size,
                "max_evaluations": args.max_evaluations,
            },
        )
        manifest = json.loads((root / "scenarios.json").read_text())
        provenance = json.loads((root / "provenance.json").read_text())

        baseline_started = monotonic()
        baseline_validation = evaluate_split(experiment, root, "val")
        hidden_before = repeated_hidden_rollouts(
            experiment, root, repeats=args.hidden_repeats
        )
        baseline_seconds = monotonic() - baseline_started

        evolution_started = monotonic()
        run = experiment.run_helix(
            root,
            generations=args.generations,
            timeout_seconds=args.timeout,
        )
        evolution_seconds = monotonic() - evolution_started
        frontier = experiment.frontier_summary(root)
        lesson = experiment.evolution_lesson(frontier)
        best_id, best_root, selected_from_frontier = selected_candidate(root, frontier)

        hidden_after_started = monotonic()
        hidden_after = repeated_hidden_rollouts(
            experiment, best_root, repeats=args.hidden_repeats
        )
        hidden_after_seconds = monotonic() - hidden_after_started

        before_summary = experiment.summarize_rollouts(hidden_before)
        after_summary = experiment.summarize_rollouts(hidden_after)
        criterion = experiment.deployment_success_criterion(
            before_summary,
            after_summary,
        )
        lineage = frontier["lineage"]

        # The notebook replays one held-out rollout live against each of these,
        # so they travel with the report rather than living only in /tmp.
        exported = {
            "seed": export_repository(root, output_dir / "repos" / "seed"),
            "selected": export_repository(best_root, output_dir / "repos" / "selected"),
        }
        for results in (baseline_validation, hidden_before, hidden_after):
            relativize_media(results, output_dir)
        report = {
            "schema_version": "rho-multitask-helix-report/v2",
            "mode": "live_capx",
            "recorded_fallback": False,
            "study_provenance": {
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "hostname": socket.gethostname(),
                "image_id": os.environ.get("EXPERIMENT_IMAGE_ID"),
                "source_revision": os.environ.get("EXPERIMENT_SOURCE_REVISION"),
            },
            "mutation_model_loader_alias": args.model,
            "mutation_model_api_id": experiment.opencode_model_id(args.model),
            "seed_model": provenance["seed_model"],
            "generations": args.generations,
            "proposal_slots_per_generation": args.proposals,
            "manifest": manifest,
            "provenance": provenance,
            "baseline_validation": baseline_validation,
            "helix": asdict(run),
            "frontier": frontier,
            "lesson": lesson,
            "selected_candidate": best_id,
            "selected_from_frontier": selected_from_frontier,
            "selected_diff": rho_demo.source_diff(root, best_root),
            "exported_repositories": exported,
            "hidden_trials": {
                task: list(trials) for task, trials in HIDDEN_TRIALS.items()
            },
            "hidden_trials_used_by_evolution": False,
            "hidden_rollouts_per_task": len(next(iter(HIDDEN_TRIALS.values())))
            * args.hidden_repeats,
            "hidden_repeats": args.hidden_repeats,
            "hidden_before": hidden_before,
            "hidden_after": hidden_after,
            "hidden_before_summary": before_summary,
            "hidden_after_summary": after_summary,
            "success_criterion": criterion,
            "timing": {
                "setup_seconds": setup_seconds,
                "baseline_seconds": baseline_seconds,
                "evolution_seconds": evolution_seconds,
                "hidden_after_seconds": hidden_after_seconds,
                "agent_event_span_seconds": sum(
                    float(item.get("agent_metrics", {}).get("total_seconds", 0.0))
                    for item in lineage
                ),
                "retained_validation_simulator_seconds": sum(
                    float(item.get("validation_simulator_seconds", 0.0))
                    for item in lineage
                ),
            },
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "RHO_MULTITASK_RESULT="
            + json.dumps(
                {
                    "selected_candidate": best_id,
                    "selected_from_frontier": selected_from_frontier,
                    "criterion_met": criterion["met"],
                    "completed_before": criterion["completed_before"],
                    "completed_after": criterion["completed_after"],
                    "mean_reward_before": criterion["mean_reward_before"],
                    "mean_reward_after": criterion["mean_reward_after"],
                    "timed_out": run.timed_out,
                    "returncode": run.returncode,
                    "report": str(report_path),
                },
                separators=(",", ":"),
            )
        )
        return 124 if run.timed_out else run.returncode
    finally:
        rho_demo.stop_owned_services()


if __name__ == "__main__":
    sys.exit(main())
