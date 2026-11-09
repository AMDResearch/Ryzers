# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Presentation helpers for the recorded multi-task RHO study.

Kept separate from ``rho_multitask_demo`` because that module is imported by
``probe.py`` inside every candidate worktree. Nothing here is allowed to run on
the evaluator path, so pandas and matplotlib are imported lazily.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rho-multitask-helix-report/v2"
DEFAULT_RECORDED_ROOT = Path("/ryzers/notebooks/recorded_results")
REPORT_NAME = "rho_multitask_report.json"

TASK_LABELS = {
    "cube_stack": "cube stack",
    "spill_wipe": "spill wipe",
    "cube_lift": "cube lift",
}
SCENARIO_LABELS = {
    "stack_val": "cube stack (validation)",
    "wipe_val": "spill wipe (validation)",
    "lift_val": "cube lift (validation)",
}

_SHORT_TASK_NAMES = {"stack": "cube stack", "wipe": "spill wipe", "lift": "cube lift"}


def scenario_label(key: str) -> str:
    """Human label for a scenario id, including numbered multi-trial splits."""
    if key in SCENARIO_LABELS:
        return SCENARIO_LABELS[key]
    match = re.fullmatch(r"([a-z]+)_(train|val)(\d*)", key)
    if not match:
        return key
    short, split, index = match.groups()
    name = _SHORT_TASK_NAMES.get(short, short.replace("_", " "))
    split_name = "validation" if split == "val" else "train"
    return f"{name} ({split_name} {index})" if index else f"{name} ({split_name})"

MUTATION_MODEL = "user.Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"

# The 30B mutation model is not baked into the course image. It is 18 GB and
# only the search needs it, so anyone reproducing the study fetches it first.
# Lemonade's built-in Qwen3-Coder alias resolves to a different quantization,
# hence the explicit checkpoint and file.
PULL_MODEL_COMMAND = (
    f"lemonade pull {MUTATION_MODEL} \\\n"
    "  --checkpoint main \\\n"
    "    unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:"
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \\\n"
    "  --recipe llamacpp \\\n"
    "  --label coding"
)

# rho_multitask_demo._safe_reset() refuses to clear a candidate tree anywhere
# outside /tmp, so the study has to be produced there and moved afterwards.
# Media paths in the report are stored relative to its own directory, so the
# finished study survives relocation.
REGENERATE_COMMAND = (
    "python scripts/rho_multitask_study.py \\\n"
    "  --output-dir /tmp/recorded_results \\\n"
    f"  --model {MUTATION_MODEL} \\\n"
    "  --generations 2 \\\n"
    "  --proposals 4 \\\n"
    "  --minibatch-size 2 \\\n"
    "  --hidden-repeats 3 \\\n"
    "  --max-evaluations 900 \\\n"
    "  --timeout 7200\n"
    "rm -rf /tmp/recorded_results/candidate\n"
    "cp -a /tmp/recorded_results /ryzers/notebooks/recorded_results"
)


# ---------------------------------------------------------------------------
# Loading and asset resolution
# ---------------------------------------------------------------------------


def recorded_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_RECORDED_ROOT


def resolve_media(value: Any, root: Path | str | None = None) -> Path | None:
    """Resolve a recorded video path against the recorded root.

    The study writes absolute paths from whichever host produced the run, so a
    report replayed inside the course image has to fall back to the recorded
    root and finally to a basename match under ``videos/``.
    """
    if not value:
        return None
    base = recorded_root(root)
    candidate = Path(str(value))
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        relative = base / candidate
        if relative.is_file():
            return relative
    fallback = base / "videos" / candidate.name
    if fallback.is_file():
        return fallback
    return None


def preflight(root: Path | str | None = None) -> dict[str, Any]:
    """Report which recorded assets are present without raising.

    Run this before the session. A missing asset here is a rehearsal problem;
    the same asset missing mid-talk is a broken demo.
    """
    base = recorded_root(root)
    report_path = base / REPORT_NAME
    checks: list[dict[str, Any]] = []

    def record(label: str, path: Path, required: bool) -> bool:
        present = path.exists()
        checks.append(
            {
                "asset": label,
                "path": str(path),
                "present": present,
                "required": required,
            }
        )
        return present

    has_report = record("study report", report_path, True)
    record("seed repository", base / "repos" / "seed" / "solver", True)
    record("selected repository", base / "repos" / "selected" / "solver", True)
    # The notebook prints these by name, so directory presence is not enough.
    for name in ("helix.toml", "probe.py", "opencode.json"):
        record(f"seed {name}", base / "repos" / "seed" / name, True)

    schema_ok = False
    if has_report:
        try:
            report = json.loads(report_path.read_text())
            schema_ok = report.get("schema_version") == SCHEMA_VERSION
        except (OSError, json.JSONDecodeError):
            schema_ok = False

    required_ok = all(check["present"] for check in checks if check["required"])
    return {
        "recorded_root": str(base),
        "checks": checks,
        "schema_ok": schema_ok,
        "ready": bool(required_ok and schema_ok),
        "regenerate_command": REGENERATE_COMMAND,
    }


def format_preflight(status: dict[str, Any]) -> str:
    lines = [f"Recorded root: {status['recorded_root']}"]
    for check in status["checks"]:
        mark = "OK  " if check["present"] else ("MISS" if check["required"] else "----")
        suffix = "" if check["required"] else " (optional)"
        lines.append(f"  [{mark}] {check['asset']}{suffix}: {check['path']}")
    lines.append(
        f"  [{'OK  ' if status['schema_ok'] else 'MISS'}] schema {SCHEMA_VERSION}"
    )
    if not status["ready"]:
        lines.append("")
        lines.append("Recorded assets are incomplete. Regenerate them with:")
        lines.append("")
        lines.append(status["regenerate_command"])
    return "\n".join(lines)


def require_ready(status: dict[str, Any]) -> None:
    """Fail with the regeneration command instead of a bare missing-file error."""
    if status["ready"]:
        return
    missing = [
        check["asset"]
        for check in status["checks"]
        if check["required"] and not check["present"]
    ]
    detail = ", ".join(missing) if missing else f"schema is not {SCHEMA_VERSION}"
    raise FileNotFoundError(
        f"Recorded study assets are incomplete ({detail}) under "
        f"{status['recorded_root']}.\n\nRegenerate them with:\n\n"
        f"{status['regenerate_command']}"
    )


def load_report(root: Path | str | None = None) -> dict[str, Any]:
    base = recorded_root(root)
    path = base / REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing recorded report: {path}\n\nRegenerate it with:\n\n"
            f"{REGENERATE_COMMAND}"
        )
    report = json.loads(path.read_text())
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Recorded report schema is {report.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}. Regenerate it with:\n\n"
            f"{REGENERATE_COMMAND}"
        )
    return report


# ---------------------------------------------------------------------------
# Frontier / Pareto
# ---------------------------------------------------------------------------


def validation_keys(report: dict[str, Any]) -> list[str]:
    return list(report["manifest"]["splits"]["val"])


def instance_owners(report: dict[str, Any]) -> dict[str, list[str]]:
    """Map each validation instance to the candidates that currently own it.

    HELIX retains on coverage dominance rather than pairwise comparison: a
    candidate survives while it is best on at least one instance.
    """
    keys = validation_keys(report)
    active = report["frontier"].get("active_frontier", {})
    owners: dict[str, list[str]] = {}
    for raw_key, winners in active.items():
        label = (
            keys[int(raw_key)]
            if str(raw_key).isdigit() and int(raw_key) < len(keys)
            else str(raw_key)
        )
        owners[label] = list(winners)
    return owners


def frontier_frame(report: dict[str, Any]):
    """One row per candidate: per-instance scores, ownership, retention."""
    import pandas as pd

    keys = validation_keys(report)
    candidates = report["frontier"]["candidates"]
    lineage = {str(item["id"]): item for item in report["frontier"].get("lineage", [])}
    selected = report.get("selected_candidate")

    rows = []
    for candidate_id, candidate in candidates.items():
        scores = candidate.get("scores", {})
        record = lineage.get(str(candidate_id), {})
        row: dict[str, Any] = {"candidate": candidate_id}
        for key in keys:
            row[scenario_label(key)] = float(scores.get(key, 0.0))
        row["mean"] = (
            sum(float(scores.get(key, 0.0)) for key in keys) / len(keys) if keys else 0.0
        )
        row["owns"] = ", ".join(candidate.get("wins", [])) or "-"
        row["retained"] = bool(candidate.get("frontier"))
        row["operation"] = record.get("operation", "-")
        row["deployed"] = candidate_id == selected
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("candidate")
    return frame.sort_values(["retained", "mean"], ascending=[False, False])


def progress_frame(report: dict[str, Any]):
    """One row per proposal the search made, in the order it made them.

    Candidates that clear the training gate get validation scores and appear in
    the frontier; proposals the gate rejected exist only in the lineage. Both
    matter -- the rejections are what the search cost.
    """
    import pandas as pd

    keys = validation_keys(report)
    candidates = report["frontier"]["candidates"]
    selected = report.get("selected_candidate")

    rows = []
    for record in report["frontier"].get("lineage", []):
        candidate_id = str(record["id"])
        candidate = candidates.get(candidate_id)
        validated = candidate is not None
        scores = (candidate or {}).get("scores", {})
        rows.append(
            {
                "candidate": candidate_id,
                "generation": int(record.get("generation", 0)),
                "operation": record.get("operation", "-"),
                "gate": record.get("gate_result", "-"),
                "validated": validated,
                "mean validation": (
                    sum(float(scores.get(key, 0.0)) for key in keys) / len(keys)
                    if validated and keys
                    else None
                ),
                "retained": bool((candidate or {}).get("frontier")),
                "deployed": candidate_id == selected,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["generation", "candidate"]).set_index("candidate")


def plot_progress(report: dict[str, Any], *, figsize: tuple[float, float] = (8.5, 5.0)):
    """Mean validation reward per generation, one marker per proposal.

    This is the plot that answers whether the search worked. Selection deploys
    the highest mean-validation candidate, so the running best is the number
    the deployed policy inherits.
    """
    import matplotlib.pyplot as plt

    frame = progress_frame(report)
    if frame.empty:
        raise ValueError("no lineage recorded, so there is nothing to plot")

    figure, axes = plt.subplots(figsize=figsize)
    scored = frame[frame["validated"]]
    rejected = frame[~frame["validated"]]

    # The running best is what selection would deploy if the search stopped here.
    best_so_far = []
    running = float("-inf")
    for generation in sorted(frame["generation"].unique()):
        upto = scored[scored["generation"] <= generation]["mean validation"]
        if len(upto):
            running = max(running, float(upto.max()))
        best_so_far.append((generation, running if running > float("-inf") else 0.0))
    axes.step(
        [g for g, _ in best_so_far],
        [v for _, v in best_so_far],
        where="post",
        color="#1d4ed8",
        linewidth=1.8,
        alpha=0.85,
        zorder=2,
        label="best so far",
    )

    jitter = {}
    for candidate_id, row in scored.iterrows():
        generation = int(row["generation"])
        seen = jitter.get(generation, 0)
        jitter[generation] = seen + 1
        axes.scatter(
            generation + 0.06 * seen,
            row["mean validation"],
            s=240 if row["deployed"] else 130,
            marker="*" if row["deployed"] else "o",
            facecolor="#2563eb" if row["retained"] else "none",
            edgecolor="#1d4ed8" if row["retained"] else "#94a3b8",
            linewidths=1.5,
            zorder=3,
        )

    if len(rejected):
        counts = rejected.groupby("generation").size()
        axes.scatter(
            counts.index,
            [-0.045] * len(counts),
            marker="x",
            s=60,
            color="#cbd5e1",
            zorder=3,
        )
        for generation, count in counts.items():
            axes.annotate(
                str(count),
                (generation, -0.045),
                textcoords="offset points",
                xytext=(7, -4),
                fontsize=8,
                color="#94a3b8",
            )

    axes.set_xlabel("generation")
    axes.set_ylabel("mean validation reward")
    axes.set_title("Search progress: every proposal, and the best score so far")
    axes.set_ylim(-0.09, 1.06)
    axes.set_xticks(sorted(frame["generation"].unique()))
    axes.grid(alpha=0.25, zorder=0)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=9, label="retained on frontier",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#94a3b8", markersize=9, label="covered / pruned",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=14, label="deployed",
        ),
        plt.Line2D(
            [], [], marker="x", linestyle="", color="#cbd5e1",
            markersize=8, label="rejected by the training gate",
        ),
        plt.Line2D([], [], color="#1d4ed8", linewidth=1.8, label="best so far"),
    ]
    axes.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout()
    return figure


def plot_frontier(report: dict[str, Any], *, figsize: tuple[float, float] = (7.0, 6.0)):
    """Scatter the candidates over two validation instances.

    With two instances, coverage dominance is exactly readable off the plot:
    the dashed lines mark the per-instance best scores, every candidate that
    touches a dashed line owns that instance and is retained, and everything
    strictly inside the box is covered by someone else and gets pruned.
    """
    import matplotlib.pyplot as plt

    keys = validation_keys(report)
    if len(keys) != 2:
        return plot_frontier_grid(report)
    x_key, y_key = keys
    candidates = report["frontier"]["candidates"]
    selected = report.get("selected_candidate")

    figure, axes = plt.subplots(figsize=figsize)
    best_x = max(
        (float(c.get("scores", {}).get(x_key, 0.0)) for c in candidates.values()),
        default=0.0,
    )
    best_y = max(
        (float(c.get("scores", {}).get(y_key, 0.0)) for c in candidates.values()),
        default=0.0,
    )

    # Everything strictly inside this box is beaten on both instances at once,
    # which is exactly the condition for coverage pruning.
    axes.add_patch(
        plt.Rectangle(
            (-0.05, -0.05),
            best_x + 0.05,
            best_y + 0.05,
            facecolor="#fee2e2",
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    axes.axvline(best_x, linestyle="--", color="#94a3b8", linewidth=1.2, zorder=1)
    axes.axhline(best_y, linestyle="--", color="#94a3b8", linewidth=1.2, zorder=1)

    seen: dict[tuple[float, float], int] = {}
    for candidate_id, candidate in sorted(candidates.items()):
        scores = candidate.get("scores", {})
        x = float(scores.get(x_key, 0.0))
        y = float(scores.get(y_key, 0.0))
        retained = bool(candidate.get("frontier"))
        deployed = candidate_id == selected

        # Identical score vectors are common and would otherwise overplot.
        collisions = seen.get((x, y), 0)
        seen[(x, y)] = collisions + 1
        offset = 0.018 * collisions

        axes.scatter(
            x + offset,
            y + offset,
            s=230 if deployed else 150,
            marker="*" if deployed else "o",
            facecolor=("#2563eb" if retained else "none"),
            edgecolor=("#1d4ed8" if retained else "#94a3b8"),
            linewidths=1.6,
            zorder=3,
        )
        axes.annotate(
            candidate_id,
            (x + offset, y + offset),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=9,
            color=("#1e3a8a" if retained else "#64748b"),
            zorder=4,
        )

    axes.set_xlabel(f"{scenario_label(x_key)} reward")
    axes.set_ylabel(f"{scenario_label(y_key)} reward")
    axes.set_title("Instance frontier: retained candidates own at least one task")
    axes.set_xlim(-0.05, max(1.0, best_x) + 0.18)
    axes.set_ylim(-0.05, max(1.0, best_y) + 0.18)
    axes.grid(alpha=0.25, zorder=0)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=10, label="retained on frontier",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#94a3b8", markersize=10, label="covered / pruned",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=15, label="deployed candidate",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#fee2e2", edgecolor="none",
            label="covered on both instances",
        ),
    ]
    # Candidates cluster in the corners, so an in-axes legend reliably hides one.
    axes.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout()
    return figure


def task_groups(report: dict[str, Any]) -> dict[str, list[str]]:
    """Group validation instance ids by the task they belong to."""
    groups: dict[str, list[str]] = {}
    for key in validation_keys(report):
        match = re.fullmatch(r"([a-z]+)_(?:train|val)\d*", key)
        short = match.group(1) if match else key
        groups.setdefault(_SHORT_TASK_NAMES.get(short, short), []).append(key)
    return groups


def task_frontier_frame(report: dict[str, Any]):
    """One row per candidate, scored as a mean over each task's instances."""
    import pandas as pd

    groups = task_groups(report)
    candidates = report["frontier"]["candidates"]
    selected = report.get("selected_candidate")

    rows = []
    for candidate_id, candidate in candidates.items():
        scores = candidate.get("scores", {})
        row: dict[str, Any] = {"candidate": candidate_id}
        for task, keys in groups.items():
            row[task] = sum(float(scores.get(key, 0.0)) for key in keys) / len(keys)
        row["retained"] = bool(candidate.get("frontier"))
        row["deployed"] = candidate_id == selected
        rows.append(row)
    return pd.DataFrame(rows).set_index("candidate")


def plot_task_frontier(
    report: dict[str, Any], *, figsize: tuple[float, float] = (7.0, 6.0)
):
    """Scatter candidates by their per-task mean score.

    With several layouts per task a full instance-pair grid is unreadable, so
    this collapses each task to the fraction of its layouts a candidate solved.
    Coverage is still decided per instance, so a retained candidate may sit
    inside the cloud here; ``coverage_frame`` remains the authority.
    """
    import matplotlib.pyplot as plt

    groups = task_groups(report)
    if len(groups) != 2:
        return plot_frontier_grid(report)

    frame = task_frontier_frame(report)
    (x_task, x_keys), (y_task, y_keys) = list(groups.items())

    figure, axes = plt.subplots(figsize=figsize)
    seen: dict[tuple[float, float], int] = {}
    for candidate_id, row in frame.iterrows():
        x, y = float(row[x_task]), float(row[y_task])
        collisions = seen.get((x, y), 0)
        seen[(x, y)] = collisions + 1
        offset = 0.015 * collisions
        axes.scatter(
            x + offset,
            y + offset,
            s=250 if row["deployed"] else 140,
            marker="*" if row["deployed"] else "o",
            facecolor="#2563eb" if row["retained"] else "none",
            edgecolor="#1d4ed8" if row["retained"] else "#94a3b8",
            linewidths=1.5,
            zorder=3,
        )
        axes.annotate(
            candidate_id,
            (x + offset, y + offset),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=8,
            color="#1e3a8a" if row["retained"] else "#64748b",
            zorder=4,
        )

    ticks = [index / len(x_keys) for index in range(len(x_keys) + 1)]
    axes.set_xticks(ticks)
    axes.set_yticks([index / len(y_keys) for index in range(len(y_keys) + 1)])
    axes.set_xlabel(f"{x_task}: fraction of {len(x_keys)} layouts solved")
    axes.set_ylabel(f"{y_task}: fraction of {len(y_keys)} layouts solved")
    axes.set_title("Per-task frontier: how much of each task a candidate solved")
    axes.set_xlim(-0.08, 1.14)
    axes.set_ylim(-0.08, 1.14)
    axes.grid(alpha=0.25, zorder=0)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=10, label="retained on frontier",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#94a3b8", markersize=10, label="covered / pruned",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=15, label="deployed",
        ),
    ]
    axes.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout()
    return figure


def plot_frontier_grid(report: dict[str, Any], *, panel_size: float = 3.4):
    """One frontier panel per pair of validation instances.

    Coverage dominance is defined across all instances at once and cannot be
    read off any single panel, so a candidate may sit inside the shaded region
    here and still be retained for winning a third instance. The panels show
    where candidates specialise; ``coverage_frame`` is the authority on why
    each one was kept.
    """
    import itertools

    import matplotlib.pyplot as plt

    keys = validation_keys(report)
    if len(keys) < 2:
        raise ValueError(f"a frontier needs at least two validation instances, got {keys}")
    candidates = report["frontier"]["candidates"]
    selected = report.get("selected_candidate")

    pairs = list(itertools.combinations(keys, 2))
    columns = min(3, len(pairs))
    rows = (len(pairs) + columns - 1) // columns
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(panel_size * columns + 1.0, panel_size * rows + 1.2),
        squeeze=False,
    )

    for index, (x_key, y_key) in enumerate(pairs):
        axes = axes_grid[index // columns][index % columns]
        seen: dict[tuple[float, float], int] = {}
        for candidate_id, candidate in sorted(candidates.items()):
            scores = candidate.get("scores", {})
            x = float(scores.get(x_key, 0.0))
            y = float(scores.get(y_key, 0.0))
            collisions = seen.get((x, y), 0)
            seen[(x, y)] = collisions + 1
            offset = 0.02 * collisions
            retained = bool(candidate.get("frontier"))
            deployed = candidate_id == selected
            axes.scatter(
                x + offset,
                y + offset,
                s=200 if deployed else 90,
                marker="*" if deployed else "o",
                facecolor=("#2563eb" if retained else "none"),
                edgecolor=("#1d4ed8" if retained else "#94a3b8"),
                linewidths=1.4,
                zorder=3,
            )
        axes.set_xlabel(scenario_label(x_key), fontsize=9)
        axes.set_ylabel(scenario_label(y_key), fontsize=9)
        axes.set_xlim(-0.08, 1.14)
        axes.set_ylim(-0.08, 1.14)
        axes.tick_params(labelsize=8)
        axes.grid(alpha=0.25, zorder=0)

    for index in range(len(pairs), rows * columns):
        axes_grid[index // columns][index % columns].axis("off")

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=9, label="retained on frontier",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#94a3b8", markersize=9, label="covered / pruned",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="#2563eb",
            markeredgecolor="#1d4ed8", markersize=14, label="deployed",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.suptitle("Where candidates specialise, one panel per task pair", fontsize=11)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure


def coverage_frame(report: dict[str, Any]):
    """Which candidate owns each validation instance, and at what score."""
    import pandas as pd

    candidates = report["frontier"]["candidates"]
    rows = []
    for instance, owners in instance_owners(report).items():
        rows.append(
            {
                "validation instance": scenario_label(instance),
                "owned by": ", ".join(owners) or "-",
                "best score": max(
                    (
                        float(
                            candidates.get(owner, {}).get("scores", {}).get(instance, 0.0)
                        )
                        for owner in owners
                    ),
                    default=0.0,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("validation instance")


def selection_rule(report: dict[str, Any]) -> str:
    """Explain the deployment choice the way the RHO paper states it."""
    selected = report.get("selected_candidate")
    frame = frontier_frame(report)
    retained = frame[frame["retained"]]
    if selected in retained.index:
        mean = float(retained.loc[selected, "mean"])
        return (
            f"Deployed {selected}: the highest-mean-validation-reward member of the "
            f"terminal frontier (mean {mean:.3f} over {len(validation_keys(report))} "
            f"instances, chosen from {len(retained)} retained candidates)."
        )
    return (
        f"Deployed {selected}, which is NOT a retained frontier member. "
        "RHO deploys the highest-mean member of the terminal frontier, so this "
        "run needs investigating before the numbers are quoted."
    )


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def lineage_frame(report: dict[str, Any]):
    """Per-candidate ancestry, gate outcome, validation score, files."""
    import pandas as pd

    keys = validation_keys(report)
    candidates = report["frontier"]["candidates"]
    rows = []
    for record in report["frontier"].get("lineage", []):
        candidate_id = str(record["id"])
        changed = record.get("changed_files") or record.get("files_changed") or []
        candidate = candidates.get(candidate_id)
        scores = (candidate or {}).get("scores", {})
        rows.append(
            {
                "candidate": candidate_id,
                "generation": record.get("generation"),
                "operation": record.get("operation"),
                "parents": ", ".join(record.get("parents") or []) or "-",
                "sampled train task": record.get("sampled_train_task") or "-",
                "gate": record.get("gate_result", "-"),
                # Rejected proposals never reach validation, so this is None
                # for them rather than zero.
                "mean validation": (
                    sum(float(scores.get(key, 0.0)) for key in keys) / len(keys)
                    if candidate is not None and keys
                    else None
                ),
                "files changed": ", ".join(sorted(changed)) or "-",
                "retained": bool(
                    candidates.get(candidate_id, {}).get("frontier", False)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def lineage_tree(report: dict[str, Any]) -> str:
    """Render ancestry as an indented tree, marking merges and retention."""
    candidates = report["frontier"]["candidates"]
    records = {str(item["id"]): item for item in report["frontier"].get("lineage", [])}
    selected = report.get("selected_candidate")

    children: dict[str | None, list[str]] = {}
    for candidate_id, record in records.items():
        children.setdefault(record.get("parent"), []).append(candidate_id)

    lines: list[str] = []

    def walk(candidate_id: str, prefix: str, connector: str) -> None:
        record = records[candidate_id]
        retained = bool(candidates.get(candidate_id, {}).get("frontier", False))
        scores = candidates.get(candidate_id, {}).get("scores", {})
        score_text = " · ".join(
            f"{scenario_label(key)} {float(scores.get(key, 0.0)):.2f}"
            for key in validation_keys(report)
        )
        markers = []
        if record.get("operation") == "merge":
            parents = ", ".join(record.get("parents") or [])
            markers.append(f"merge of {parents}")
        markers.append("retained" if retained else "pruned")
        if candidate_id == selected:
            markers.append("DEPLOYED")
        lines.append(
            f"{prefix}{connector}{candidate_id}"
            f"  [{' · '.join(markers)}]  {score_text}"
        )
        child_prefix = prefix + ("" if not connector else
                                 ("    " if connector == "└── " else "│   "))
        siblings = sorted(children.get(candidate_id, []))
        for index, child in enumerate(siblings):
            last = index == len(siblings) - 1
            walk(child, child_prefix, "└── " if last else "├── ")

    for root_id in sorted(children.get(None, [])):
        walk(root_id, "", "")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diffs over the multi-file surface
# ---------------------------------------------------------------------------


def diff_sections(diff: str) -> dict[str, str]:
    """Split a unified diff produced by ``rho_demo.source_diff`` per file."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in (diff or "").splitlines(keepends=True):
        header = re.match(r"^\+\+\+ (?:best/)?(.+?)\s*$", line)
        if header:
            name = header.group(1).strip()
            current = "(deleted)" if name == "/dev/null" else name
            sections.setdefault(current, [])
        if current is not None:
            sections[current].append(line)
    return {name: "".join(lines) for name, lines in sections.items()}


def diff_summary(diff: str):
    """Added/removed line counts per file, so the surface is visible at a glance."""
    import pandas as pd

    rows = []
    for name, section in diff_sections(diff).items():
        added = sum(
            1
            for line in section.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1
            for line in section.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )
        rows.append({"file": name, "added": added, "removed": removed})
    if not rows:
        return pd.DataFrame(columns=["file", "added", "removed"]).set_index("file")
    return pd.DataFrame(rows).set_index("file").sort_index()


# ---------------------------------------------------------------------------
# Held-out evidence
# ---------------------------------------------------------------------------


def held_out_frame(report: dict[str, Any]):
    """Seed vs deployed on trials evolution never saw."""
    import pandas as pd

    before = report.get("hidden_before_summary", {})
    after = report.get("hidden_after_summary", {})
    rows = []
    for task in sorted(set(before) | set(after)):
        seed = before.get(task, {})
        best = after.get(task, {})
        rows.append(
            {
                "task": TASK_LABELS.get(task, task),
                "rollouts": seed.get("rollouts", 0),
                "seed solved": seed.get("completed", 0),
                "deployed solved": best.get("completed", 0),
                "seed mean reward": seed.get("mean_reward", 0.0),
                "deployed mean reward": best.get("mean_reward", 0.0),
                "seed exec failures": seed.get("execution_failures", 0),
                "deployed exec failures": best.get("execution_failures", 0),
            }
        )
    return pd.DataFrame(rows).set_index("task")


def held_out_trial(report: dict[str, Any], task: str, index: int = 0) -> int:
    """One of the trials evolution never sampled, for a live replay."""
    trials = report.get("hidden_trials", {}).get(task)
    if trials:
        return int(trials[index])
    seen = [
        int(result["trial"])
        for result in report.get("hidden_before", [])
        if result.get("task") == task
    ]
    if seen:
        return seen[index]
    raise KeyError(f"no recorded held-out trial for task {task!r}")


# Deployed-policy pass rate over five replays of each validation trial, measured
# with scripts/rho_replay_scan.py once rho_demo.seed_scene made a trial mean a
# fixed scene. Object placement is now reproducible; grasp sampling still runs
# in the perception service and is not, which is where the misses come from.
REPLAY_PASS_RATES: dict[tuple[str, int], float] = {
    ("cube_stack", 2): 0.6,
    ("cube_stack", 3): 0.8,
    ("cube_stack", 4): 0.4,
    ("cube_lift", 2): 0.8,
    ("cube_lift", 3): 0.4,
    ("cube_lift", 4): 0.8,
}


def validation_trial(report: dict[str, Any], task: str, index: int | None = None) -> int:
    """A task's validation trial, best-measured first unless ``index`` is given."""
    trials = [
        int(entry["trial"])
        for entry in report.get("baseline_validation", [])
        if entry.get("task") == task
    ]
    if not trials:
        raise KeyError(f"no recorded validation scenario for task {task!r}")
    if index is not None:
        return trials[index]
    return max(trials, key=lambda trial: REPLAY_PASS_RATES.get((task, trial), 0.0))


def rollout_narrator(
    indent: str = "   ",
    width: int = 96,
    writer: Any = print,
) -> Any:
    """Build a ``progress`` callback that prints a policy's narration as it runs.

    Pass the result to ``replay_trials(progress=...)``. Simulator setup noise and
    the evaluator's JSON result line are dropped, leaving the ``print()`` calls
    the policy itself makes, which arrive as the robot reaches each stage.
    """
    import rho_demo

    def narrate(line: str) -> None:
        if not line.startswith(rho_demo.NARRATION_PREFIX):
            return
        text = line[len(rho_demo.NARRATION_PREFIX) :].rstrip()
        if len(text) > width:
            text = text[: width - 1] + "…"
        writer(f"{indent}{text}")

    return narrate


def validation_tasks(report: dict[str, Any]) -> list[str]:
    """Every task with a validation trial, in report order."""
    ordered: list[str] = []
    for entry in report.get("baseline_validation", []):
        task = str(entry.get("task"))
        if task not in ordered:
            ordered.append(task)
    return ordered


def held_out_videos(
    report: dict[str, Any],
    task: str,
    trial: int,
    root: Path | str | None = None,
) -> dict[str, Path | None]:
    """Locate the recorded before/after clips for one held-out trial."""
    found: dict[str, Path | None] = {"seed": None, "deployed": None}
    for key, label in (("hidden_before", "seed"), ("hidden_after", "deployed")):
        for result in report.get(key, []):
            if result.get("task") == task and int(result.get("trial", -1)) == int(trial):
                found[label] = resolve_media(result.get("video"), root)
                break
    return found


def study_cost(report: dict[str, Any]) -> dict[str, Any]:
    """The wall-clock ledger, so the live/recorded split is honest on stage."""
    timing = report.get("timing", {})
    return {
        "generations": report.get("generations"),
        "proposal_slots_per_generation": report.get("proposal_slots_per_generation"),
        "mutation_model": report.get("mutation_model_loader_alias"),
        "setup_seconds": timing.get("setup_seconds", 0.0),
        "baseline_seconds": timing.get("baseline_seconds", 0.0),
        "evolution_seconds": timing.get("evolution_seconds", 0.0),
        "held_out_seconds": timing.get("hidden_after_seconds", 0.0),
        "total_seconds": sum(
            float(timing.get(key, 0.0))
            for key in (
                "setup_seconds",
                "baseline_seconds",
                "evolution_seconds",
                "hidden_after_seconds",
            )
        ),
    }


# ---------------------------------------------------------------------------
# Why the mutator is a large model
# ---------------------------------------------------------------------------

LEMONADE_CHAT_URL = "http://127.0.0.1:13305/api/v1/chat/completions"

MUTATOR_PROBE_MODELS = (
    "Gemma-4-E4B-it-GGUF",
    "Qwen3-Coder-30B-A3B-Instruct-GGUF",
)

# The exact repair the cube-stack seed needs, stripped of everything agentic:
# no files to open, no evaluator feedback to parse, no tool calls to sequence.
MUTATOR_PROBE_PROMPT = """\
You are editing a Python robot policy.

`get_object_pose(name, return_bbox_extent=True)` returns three values:
  position   - a flat numpy XYZ array, e.g. array([0.51, 0.02, -0.09])
  quaternion - a WXYZ array
  extent     - full XYZ side lengths, e.g. array([0.06, 0.05, 0.05])

A helper exists in solver/geometry.py:

    def stack_center(target_position, target_extent, object_extent):
        \"\"\"Return the XYZ center for stacking one object on a target object.\"\"\"

The program is executed with exec() and has NO package context, so relative
imports fail.

Given these variables already exist:
    green_pose, _, green_extent = get_object_pose("green cube", return_bbox_extent=True)
    red_pose, _, red_extent = get_object_pose("red cube", return_bbox_extent=True)

Write ONLY the import line and the single line that assigns
`placement_position` using stack_center. No explanation, no code fence.
"""


def _probe_sample(model: str, temperature: float = 0.4, timeout: float = 300.0) -> str:
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": MUTATOR_PROBE_PROMPT}],
            "temperature": temperature,
            "max_tokens": 160,
        }
    ).encode()
    request = urllib.request.Request(
        LEMONADE_CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.load(handle)["choices"][0]["message"]["content"].strip()


def grade_probe_answer(text: str) -> dict[str, bool]:
    """Score one answer against the two mistakes seen in the agentic runs.

    Deliberately does not require the import line to be present. Both models
    sometimes reply with the assignment alone, and that formatting slip is not
    the failure being measured here.
    """
    flat = text.replace("```python", "").replace("```", "")
    relative_import = bool(re.search(r"from\s+\.geometry\s+import", flat))
    call = re.search(r"stack_center\(([^)]*)\)", flat)
    arguments = call.group(1) if call else ""
    scalar_argument = bool(re.search(r"green_pose\s*\[", arguments))
    passes_full_pose = bool(re.search(r"\bgreen_pose\b", arguments))
    return {
        # Gemma intermittently returns nothing at all, which is a reliability
        # story rather than a wrong answer, so it is counted separately.
        "empty": not text.strip(),
        "relative_import": relative_import,
        "scalar_argument": scalar_argument,
        "usable": passes_full_pose and not scalar_argument and not relative_import,
    }


def mutator_probe_frame(
    models: tuple[str, ...] = MUTATOR_PROBE_MODELS,
    samples: int = 3,
    temperature: float = 0.4,
):
    """Ask each model for the repair directly, with no agentic loop involved.

    Separates "does the model know the fix" from "can the model land the fix
    while driving tools", which is the distinction the RHO mutator depends on.
    """
    import pandas as pd

    rows = []
    for model in models:
        usable = 0
        relative_imports = 0
        scalar_arguments = 0
        empty = 0
        errors: list[str] = []
        for _ in range(samples):
            try:
                grade = grade_probe_answer(_probe_sample(model, temperature))
            except Exception as exc:  # noqa: BLE001 - surface, do not crash the talk
                errors.append(type(exc).__name__)
                continue
            usable += grade["usable"]
            relative_imports += grade["relative_import"]
            scalar_arguments += grade["scalar_argument"]
            empty += grade["empty"]
        rows.append(
            {
                "model": model,
                "usable repair": f"{usable}/{samples}",
                "relative import": relative_imports,
                "scalar argument": scalar_arguments,
                "empty reply": empty,
                "request errors": ", ".join(sorted(set(errors))) or "none",
            }
        )
    return pd.DataFrame(rows).set_index("model")
