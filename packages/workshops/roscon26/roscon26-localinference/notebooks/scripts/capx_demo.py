# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Setup, profiling, and video helpers for the CaP-X notebook."""

import os

# MuJoCo selects its headless renderer at import time.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import ast  # noqa: E402
import base64  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager, redirect_stderr, redirect_stdout  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402

CAPX_ROOT = Path(os.environ.get("CAPX_ROOT", "/ryzers/cap-x"))

# CaP-X resolves configs and assets relative to its repository.
if CAPX_ROOT.is_dir():
    os.chdir(CAPX_ROOT)

try:
    import capx  # noqa: F401,E402
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The CaP-X stack is not on this interpreter's path. In Jupyter, pick the "
        "'CaP-X (ROCm)' kernel from the menu in the top right."
    ) from exc

LEMONADE_PORT = 13305
DEFAULT_MODEL = "Gemma-4-E2B-it-GGUF"

LEMONADE_ENV = Path(
    os.environ.get(
        "LEMONADE_ENV",
        "/ryzers/notebooks/scripts/lemonade_env.sh",
    )
)

LEMONADE_CACHE = os.environ.get("LEMONADE_CACHE", "/opt/lemonade-cache/lemonade")
LEMONADE_HF_HOME = os.environ.get("LEMONADE_HF_HOME", "/opt/lemonade-cache/huggingface")
LLAMA_METRICS_URL = os.environ.get("LLAMA_METRICS_URL", "http://127.0.0.1:8001/metrics")
SERVICE_LOG = Path(os.environ.get("CAPX_SERVICE_LOG", "/tmp/capx-services.log"))

WORK = Path(os.environ.get("CAPX_WORK", "/tmp/capx_notebook"))
WORK.mkdir(parents=True, exist_ok=True)

TRIAL_DIR = re.compile(r"trial_(\d+)_sandboxrc_(\d+)_reward_([\d.]+)_taskcompleted_(\d)")

PRIMITIVES = (
    "get_object_pose",
    "sample_grasp_pose",
    "goto_pose",
    "open_gripper",
    "close_gripper",
    "home_pose",
)

SCENARIOS = {
    "cube stack": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    "cube restack": "env_configs/cube_restack/franka_robosuite_cube_restack.yaml",
    "cube lift": "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml",
    "nut assembly": "env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml",
    "spill wipe": "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
    "two arm handover": "env_configs/two_arm_handover/two_arm_handover.yaml",
}


def lemonade_alive(timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"http://localhost:{LEMONADE_PORT}/api/v1/health", timeout=timeout).ok
    except requests.RequestException:
        return False


def ensure_lemonade(model: str = DEFAULT_MODEL, progress=print) -> float:
    """Start Lemonade if needed and load one model."""
    started = time.monotonic()
    action = "Loading" if lemonade_alive() else "Starting Lemonade and loading"
    progress(f"{action} {model} from the image cache...")

    # setsid keeps the daemon alive across kernel restarts.
    proc = subprocess.Popen(
        ["setsid", "bash", str(LEMONADE_ENV), "--serve-only", model],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "HF_HOME": LEMONADE_HF_HOME,
            "LEMONADE_CACHE": LEMONADE_CACHE,
            "LEMONADE_HF_HOME": LEMONADE_HF_HOME,
        },
    )
    load_error = None
    for line in proc.stdout:
        rendered = line.rstrip()
        if "Error loading model:" in rendered:
            load_error = rendered
        elif "Downloading" in rendered or "Fetching" in rendered:
            progress(f"  {rendered}")
    proc.wait()

    if load_error is not None:
        raise RuntimeError(load_error)
    if proc.returncode != 0 or not lemonade_alive():
        raise RuntimeError(f"{LEMONADE_ENV} --serve-only {model} failed (exit {proc.returncode}) - see /tmp/lemond.log")
    elapsed = time.monotonic() - started
    progress(f"Lemonade ready in {elapsed:.1f}s")
    return elapsed


@contextmanager
def quiet_output(log_path: Path = SERVICE_LOG) -> Iterator[Path]:
    """Send noisy native and child-process output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as stream:
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield log_path
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def llama_metrics(timeout: float = 2.0) -> dict[str, float]:
    """Read cumulative prompt and generation counters from llama.cpp."""
    names = {
        "prompt_tokens_total",
        "prompt_seconds_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
    }
    try:
        text = requests.get(LLAMA_METRICS_URL, timeout=timeout).text
    except requests.RequestException:
        return {}
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        key, _, raw_value = line.partition(" ")
        name = key.removeprefix("llamacpp:")
        if name in names:
            metrics[name] = float(raw_value)
    return metrics


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: round(value - before.get(name, value), 3) for name, value in after.items()}


def show_video(env, name: str = "notebook_run", width: int = 640) -> str | None:
    """Encode and display the frames captured during the last step."""
    from capx.utils.video_utils import _write_video
    from IPython.display import Video, display

    frames = env.get_video_frames(clear=True)
    if not frames:
        print("no frames captured - was enable_video_capture called before the step?")
        return None

    WORK.mkdir(parents=True, exist_ok=True)
    _write_video(frames, str(WORK), suffix=name)
    path = str(WORK / f"video_{name}.mp4")
    display(Video(path, embed=True, width=width))
    return path


def _ffmpeg() -> str:
    """The ffmpeg binary, preferring the one imageio ships with."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _scaled(video: Path, dest: Path, width: int) -> Path:
    """Scale a clip for embedding, falling back to the original."""
    try:
        done = subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-vf", f"scale={width}:-2", "-an", str(dest)],
            capture_output=True,
            text=True,
        )
    except OSError:
        return video
    return dest if done.returncode == 0 and dest.exists() else video


def _show_video_grid(
    entries: list[tuple[str | int, float, bool, Path | None]],
    width: int,
    progress,
) -> None:
    from IPython.display import HTML, display

    thumbs = WORK / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)

    figures = []
    embedded = 0
    for trial, reward, solved, video in entries:
        caption = f"{trial} · reward {reward:.3f}"
        caption += " · solved" if solved else ""
        if video is None:
            figures.append(
                f'<figure style="margin:0;text-align:center;font:12px/1.4 sans-serif">'
                f'<div style="width:{width}px;height:{width * 3 // 4}px;display:flex;'
                f"align-items:center;justify-content:center;border:1px dashed currentColor;"
                f'opacity:.5">no video</div><figcaption>{caption}</figcaption></figure>'
            )
            continue

        safe_trial = re.sub(r"[^0-9A-Za-z_-]+", "_", str(trial))
        source = _scaled(video, thumbs / f"trial_{safe_trial}.mp4", width)
        if source is video:
            progress(f"  could not scale {video.name}, embedding it as it is")

        data = base64.b64encode(source.read_bytes()).decode()
        embedded += len(data)
        figures.append(
            f'<figure style="margin:0;text-align:center;font:12px/1.4 sans-serif">'
            f'<video src="data:video/mp4;base64,{data}" width="{width}" '
            f"controls loop muted playsinline></video>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    def row(items: list[str]) -> str:
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'
            'align-items:flex-start;margin-bottom:12px">' + "".join(items) + "</div>"
        )

    top = (len(figures) + 1) // 2
    display(HTML(row(figures[:top]) + (row(figures[top:]) if len(figures) > top else "")))
    progress(f"{len(figures)} rollout videos · {embedded / 1e6:.1f} MB embedded")


def show_trial_grid(trials: list[dict], width: int = 240, progress=print) -> None:
    """Display CaP-X trial videos with rewards."""
    entries = []
    for trial in trials:
        trial_dir = trial.get("dir")
        video = (
            next(iter(sorted(trial_dir.glob("video_combined*.mp4"))), None)
            if trial_dir is not None
            else None
        )
        entries.append((f"seed {trial['trial']}", trial["reward"], trial["solved"], video))
    _show_video_grid(entries, width, progress)


def show_comparison_grid(
    labeled_rollouts: list[tuple[str, dict]],
    width: int = 320,
    progress=print,
) -> None:
    """Display labeled rollouts on one row for a before/after comparison.

    ``_show_video_grid`` balances entries over two rows, which stacks a two-clip
    comparison vertically. Side by side is the whole point here.
    """
    from IPython.display import HTML, display

    thumbs = WORK / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)

    figures = []
    for label, rollout in labeled_rollouts:
        raw_video = rollout.get("video")
        video = Path(raw_video) if raw_video else None
        if video is not None and not video.is_file():
            video = None
        reward = float(rollout.get("reward") or 0.0)
        solved = bool(rollout.get("task_completed"))
        caption = (
            f"<b>{label}</b><br>reward {reward:.3f}"
            f" · {'solved' if solved else 'not solved'}"
        )
        if video is None:
            figures.append(
                f'<figure style="margin:0;text-align:center;font:12px/1.4 sans-serif">'
                f'<div style="width:{width}px;height:{width * 3 // 4}px;display:flex;'
                f"align-items:center;justify-content:center;border:1px dashed currentColor;"
                f'opacity:.5">no video</div><figcaption>{caption}</figcaption></figure>'
            )
            continue
        safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(label))
        source = _scaled(video, thumbs / f"compare_{safe}.mp4", width)
        data = base64.b64encode(source.read_bytes()).decode()
        figures.append(
            f'<figure style="margin:0;text-align:center;font:12px/1.4 sans-serif">'
            f'<video src="data:video/mp4;base64,{data}" width="{width}" '
            f"controls loop muted playsinline></video>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    display(
        HTML(
            '<div style="display:flex;flex-wrap:wrap;gap:16px;'
            'justify-content:center;align-items:flex-start">'
            + "".join(figures)
            + "</div>"
        )
    )


def show_rollout_grid(rollouts: list[dict], width: int = 240, progress=print) -> None:
    """Display RHO rollout videos across seeds."""
    entries = []
    for rollout in rollouts:
        raw_video = rollout.get("video")
        video = Path(raw_video) if raw_video else None
        if video is not None and not video.is_file():
            video = None
        entries.append(
            (
                f"{rollout.get('task', 'rollout')} seed {int(rollout['trial'])}",
                float(rollout.get("reward") or 0.0),
                bool(rollout.get("task_completed")),
                video,
            )
        )
    _show_video_grid(entries, width, progress)


def show_paired_rollout_grid(
    before: list[dict],
    after: list[dict],
    width: int = 220,
    progress=print,
) -> None:
    """Display matched before/after rollout videos in adjacent pairs."""
    indexed_after = {
        (str(item.get("task", "")), int(item["trial"])): item for item in after
    }
    entries = []
    for baseline in before:
        key = (str(baseline.get("task", "")), int(baseline["trial"]))
        evolved = indexed_after.get(key)
        for phase, rollout in (("before", baseline), ("after", evolved)):
            raw_video = rollout.get("video") if rollout else None
            video = Path(raw_video) if raw_video else None
            if video is not None and not video.is_file():
                video = None
            entries.append(
                (
                    f"{key[0]} seed {key[1]} {phase}",
                    float(rollout.get("reward") or 0.0) if rollout else 0.0,
                    bool(rollout.get("task_completed")) if rollout else False,
                    video,
                )
            )
    _show_video_grid(entries, width, progress)


# Gemma sometimes emits a control token straight into the fenced code block,
# most often appended to the final line as `open_gripper()<unused56>`. It is a
# decoding artifact rather than anything the model "wrote", but it reaches the
# sandbox as a SyntaxError and costs the whole rollout. CaP-X's _extract_code()
# only strips Markdown fences, so the cleanup happens here.
SPECIAL_TOKEN = re.compile(
    r"<unused\d+>|</?s>|<pad>|<eos>|<bos>|<end_of_turn>|<start_of_turn>"
)


def clean_program(program: str) -> tuple[str, list[str]]:
    """Strip stray model control tokens from generated code.

    Returns the cleaned program and the tokens removed, so a caller can show
    that the repair happened instead of hiding it.
    """
    found = SPECIAL_TOKEN.findall(program)
    if not found:
        return program, []
    return SPECIAL_TOKEN.sub("", program).rstrip(), found


def analyze_program(program: str) -> dict:
    """Summarize how generated code uses CaP-X's grounded robot primitives."""
    calls = {name: 0 for name in PRIMITIVES}
    syntax_error = None
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        tree = None
        syntax_error = f"{exc.msg} (line {exc.lineno})"
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in calls:
                    calls[node.func.id] += 1

    compact = "".join(program.split())
    return {
        "syntax_error": syntax_error,
        "primitive_calls": calls,
        "perception_calls": calls["get_object_pose"] + calls["sample_grasp_pose"],
        "planner_calls": calls["goto_pose"] + calls["home_pose"],
        "uses_bbox_extent": "return_bbox_extent=True" in compact,
        "uses_approach_offset": "z_approach=" in compact,
        "nested_pose_indexing": bool(
            re.search(r"[A-Za-z_][A-Za-z0-9_]*_pose\[0\]\[[012]\]", compact)
        ),
        "line_count": len(program.splitlines()),
    }


def trial_program(trial: dict) -> str:
    """Read the generated program retained in a CaP-X trial artifact."""
    trial_dir = trial.get("dir")
    if trial_dir is None:
        return ""
    code_path = Path(trial_dir) / "code.py"
    return code_path.read_text() if code_path.is_file() else ""


def trial_introspection(trials: list[dict]) -> list[dict]:
    """Return compact code and failure evidence for a set of CaP-X trials."""
    rows = []
    for trial in trials:
        program = trial_program(trial)
        analysis = analyze_program(program)
        reward = float(trial.get("reward") or 0.0)
        if trial.get("solved"):
            outcome = "completed"
        elif trial.get("error"):
            outcome = "execution failure"
        elif reward > 0:
            outcome = "partial reward"
        else:
            outcome = "no progress"
        rows.append(
            {
                "trial": int(trial.get("trial", 0)),
                "outcome": outcome,
                "reward": reward,
                "primitive_calls": sum(analysis["primitive_calls"].values()),
                "perception_calls": analysis["perception_calls"],
                "planner_calls": analysis["planner_calls"],
                "bbox_extent": analysis["uses_bbox_extent"],
                "approach_offset": analysis["uses_approach_offset"],
                "nested_pose_indexing": analysis["nested_pose_indexing"],
                "syntax_error": analysis["syntax_error"],
                "program": program,
            }
        )
    return rows


def benchmark(
    model: str,
    server_url: str,
    config_path: str,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    trials: int = 5,
    oracle: bool = False,
    verbose: bool = False,
    progress=print,
) -> list[dict]:
    """Run the CaP-X CLI serially over several layouts."""
    out_dir = WORK / "eval"
    cmd = [
        sys.executable,
        "capx/envs/launch.py",
        "--config-path",
        config_path,
        "--model",
        model,
        "--server-url",
        server_url,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--total-trials",
        str(trials),
        "--num-workers",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    if oracle:
        cmd.extend(["--use-oracle-code", "True"])
    progress(f"Running {trials} CaP-X trial{'s' if trials != 1 else ''}...")

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(CAPX_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    for line in proc.stdout:
        output.append(line)
        if verbose:
            progress(line.rstrip())
    proc.wait()
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = "".join(output[-40:]).strip()
        raise RuntimeError(f"CaP-X failed (exit {proc.returncode}):\n{tail}")
    progress(f"CaP-X finished in {elapsed:.1f}s")

    return read_results(out_dir, model, verbose=verbose, progress=progress)


def read_results(out_dir: Path, model: str, *, verbose: bool = False, progress=print) -> list[dict]:
    """Collect CaP-X trial artifacts."""
    root = out_dir.parent / model.replace("/", "_") / out_dir.name
    if not root.is_dir():
        found = sorted(out_dir.parent.glob(f"*/{out_dir.name}"), key=lambda p: p.stat().st_mtime)
        if not found:
            progress(f"no results under {out_dir.parent}")
            return []
        root = found[-1]

    summary = root / "summaries.txt"
    if verbose and summary.exists():
        progress("\n" + summary.read_text())

    by_trial = {}
    for d in sorted(root.glob("trial_*")):
        m = TRIAL_DIR.match(d.name)
        if m:
            entry = {
                "trial": int(m[1]),
                "error": m[2] != "0",
                "reward": float(m[3]),
                "solved": m[4] == "1",
                "dir": d,
            }
            current = by_trial.get(entry["trial"])
            if current is None or d.stat().st_mtime > current["dir"].stat().st_mtime:
                by_trial[entry["trial"]] = entry
    trials = [by_trial[trial] for trial in sorted(by_trial)]

    progress(f"{'trial':>5} {'sandbox':>8} {'reward':>7} {'solved':>7}")
    for t in trials:
        progress(f"{t['trial']:>5} {'error' if t['error'] else 'ok':>8} {t['reward']:>7.3f} {str(t['solved']):>7}")
    if trials:
        solved = sum(t["solved"] for t in trials)
        mean = np.mean([t["reward"] for t in trials])
        progress(f"\nsuccess rate: {solved}/{len(trials)}   mean reward: {mean:.3f}")
    return trials


def benchmark_scenarios(
    model: str,
    server_url: str,
    scenarios: dict | None = None,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    trials: int = 1,
    oracle: bool = False,
    verbose: bool = False,
    progress=print,
) -> list[dict]:
    """Run one or more episodes of several different tasks, side by side."""
    if trials < 1:
        raise ValueError("trials must be at least 1")

    scenarios = scenarios or SCENARIOS
    results = []
    for label, config_path in scenarios.items():
        progress(f"\n===== {label}: {config_path} =====")
        out_dir = WORK / "scenarios" / re.sub(r"[^0-9A-Za-z]+", "_", label)
        cmd = [
            sys.executable,
            "capx/envs/launch.py",
            "--config-path",
            config_path,
            "--model",
            model,
            "--server-url",
            server_url,
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
            "--total-trials",
            str(trials),
            "--num-workers",
            "1",
            "--output-dir",
            str(out_dir),
        ]
        if oracle:
            cmd.extend(["--use-oracle-code", "True"])

        started = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(CAPX_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output: list[str] = []
        for line in proc.stdout:
            output.append(line)
            if verbose or ("Trial" in line and "took" in line):
                progress(line.rstrip())
        proc.wait()
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            tail = "".join(output[-40:]).strip()
            progress(f"{label} failed (exit {proc.returncode}):\n{tail}")

        got = read_results(
            out_dir,
            model,
            verbose=False,
            progress=lambda *args, **kwargs: None,
        )
        if not got:
            got = [
                {
                    "trial": 0,
                    "reward": 0.0,
                    "solved": False,
                    "error": True,
                    "dir": None,
                }
            ]
        per_trial_elapsed = elapsed / len(got)
        for entry in got:
            entry["label"] = label
            entry["elapsed_seconds"] = per_trial_elapsed
            results.append(entry)

    progress(f"\n{'scenario':<14}{'trial':>7}{'sandbox':>9}{'reward':>8}{'solved':>8}")
    for result in results:
        progress(
            f"{result['label']:<14}{result.get('trial', 0):>7}"
            f"{'error' if result.get('error') else 'ok':>9}"
            f"{result['reward']:>8.3f}{str(result['solved']):>8}"
        )
    solved = sum(result["solved"] for result in results)
    progress(f"\nsolved {solved}/{len(results)} scenarios")
    return results
