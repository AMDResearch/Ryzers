# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Infrastructure helpers for the humanoid locomotion workshop notebooks.

The notebook narrative intentionally stays focused on locomotion, terrain, rewards,
and training. Runner construction, batched evaluation, table rendering, and
inline playback live here instead.
"""

from __future__ import annotations

import os

# quadrants (Genesis' compiler) logs "Graphical python shell detected, using wrapped
# sys.stdout" at info level while it is imported inside a Jupyter kernel. Its C++
# logger takes the level from QD_LOG_LEVEL when the library loads, but gs.init()
# later passes its own log_level and warns if the variable is still set. So raise
# the level only for the import; the variable is gone again before gs.init() runs.
if "QD_LOG_LEVEL" not in os.environ:
    os.environ["QD_LOG_LEVEL"] = "warn"
    try:
        import quadrants  # noqa: F401  (first import; sets the logger level)
    finally:
        del os.environ["QD_LOG_LEVEL"]

import gc
import logging
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from collections.abc import Sequence
import time
from pathlib import Path

import torch

_GENESIS_NOISE = (
    "Mesh is not watertight",
    "SDF pre-processing",
    "not watertight; skipping wall-thickness",
    "torsional friction",
    "rolling friction",
    "Adding <gs.engine.entities",
    "Added contact sensor",
    "Genesis constraint jacobian",
    "Building scene",
    "Compiling simulation kernels",
    "Building visualizer",
    "Generated terrain",
    "Scene ",
)


class _GenesisNoiseFilter(logging.Filter):
    """Drop repetitive Genesis INFO lines and known harmless warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.levelno == logging.INFO:
            return False
        if record.levelno == logging.WARNING:
            return not any(fragment in message for fragment in _GENESIS_NOISE)
        return True

from gslab.envs import GenesisManagerBasedRlEnv
from gslab.rl import RslRlVecEnvWrapper
from gslab.rl.rsl_rl.runner import GslabOnPolicyRunner
from gslab.tasks.registry import load_env_cfg, load_rl_cfg
from gslab.vis import VideoConfig, VideoRecorder
from gslab.vis.notebook import DEFAULT_CAMERA_LOOKAT, DEFAULT_CAMERA_POS


# Rollout length for the metrics table. Robots start at the patch centre (the same
# spawn as the playback video) and must cross the whole staircase for a fall to be
# possible: on the box stairs the first riser is 1.25 m out, the top landing spans
# 3.2-4.2 m and the descent ends at 5.8 m. A competent policy at ~0.6 m/s covers
# that in 10 s. Shorter windows only measure the first steps and let a slow or
# creeping policy score zero falls; a 2.5 s window did exactly that.
# 500 control steps at 0.10-0.13 s each is about a minute per checkpoint.
EVAL_DURATION_S = 10.0
# Forward spawn shift for the metrics rollout, in meters. Kept at 0 so the metrics
# and the video start from the same place; raise it to spend more of a short
# rollout on the steps.
EVAL_SPAWN_OFFSET_X = 0.0
# Robots per terrain cell in the metrics scene. A Genesis step costs about the same
# for 24 robots as for 96, so four per cell (96 robots) buys 6% granularity per
# level and 1% overall for free; one per cell made 25% jumps out of single falls.
EVAL_ROBOTS_PER_CELL = 4
# Playback clips: 720p at 25 fps. Genesis renders and encodes a frame every
# round(1 / (fps * physics_dt)) substeps, so 25 fps captures every second control
# step (50 Hz) and halves the render cost, which dominates the watch() cells.
VIDEO_RES = (1280, 720)
VIDEO_FPS = 25
# Robots in the playback clip. They are spread over the difficulty levels of the box
# stairs, hardest level nearest the camera; with more robots than levels, levels repeat.
VIDEO_ROBOTS = 5
# Clip length. The box staircase ends 5.8 m from the spawn point; at 0.45-0.65 m/s the
# robots climb, cross the landing and descend within 15 s and stay inside the static
# camera's frame. 30 s had them walk 20 m, out of the cell and out of view.
VIDEO_DURATION_S = 15.0
# Playback scenarios: terrain column and where in the cell the robots start.
# "upstairs": box stairs from the run-up (first riser 1.25 m ahead), climb, landing, descend.
# "downstairs": same staircases, spawned on the top landing so the clip is the descent.
# "terrain": the rough-ground column, from the cell centre.
VIDEO_SCENARIOS = ("upstairs", "downstairs", "terrain")


def clear_stale_kernel_cache_lock(max_age_s: float = 600.0) -> Path | None:
    """Remove a leftover quadrants kernel-cache lock so the offline cache keeps working.

    quadrants (Genesis' compiler) guards its on-disk kernel cache with a lock file
    that it deletes on release; the file only outlives a process that was killed
    mid-operation (a stopped container, a kernel shut down during a scene build).
    Every later process then logs ``Lock ... qdcache.lock failed`` and gives up on
    the cache, recompiling kernels each time. The lock is held for seconds, so a
    file older than ``max_age_s`` is stale and safe to delete. Returns the removed
    path, or None.
    """
    cache_root = Path(
        os.environ.get("QD_OFFLINE_CACHE_FILE_PATH") or Path.home() / ".cache/quadrants/qdcache"
    )
    lock = cache_root / "kernel_compilation_manager" / "qdcache.lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return None
    if age < max_age_s:
        return None
    try:
        lock.unlink()
    except OSError:
        return None
    return lock


def genesis_backend():
    """The Genesis backend for this workshop: AMDGPU, on a ROCm build of PyTorch."""
    import genesis as gs

    if torch.version.hip is None:
        raise RuntimeError(
            "PyTorch is not a ROCm build. Install it from the ROCm 7.13 wheel index "
            "(https://repo.amd.com/rocm/whl/gfx1151/), see INSTALL.md."
        )
    return gs.amdgpu


def ensure_helper_path() -> Path:
    """Add the notebooks helper directory to ``sys.path`` when needed."""
    module_dir = Path(__file__).resolve().parent
    seen: set[Path] = set()
    for helper_dir in (
        module_dir,
        Path.cwd().resolve(),
        Path.cwd().resolve() / "notebooks",
    ):
        if helper_dir in seen:
            continue
        seen.add(helper_dir)
        if (helper_dir / "workshop_helpers.py").is_file():
            path = str(helper_dir)
            if path not in sys.path:
                sys.path.insert(0, path)
            return helper_dir
    raise RuntimeError("Start JupyterLab from the workshop folder.")


def setup_notebook_path() -> Path:
    """Backward-compatible alias for :func:`ensure_helper_path`."""
    return ensure_helper_path()


def find_workshop_root(start: Path | None = None) -> Path:
    """Locate the workshop root from the current notebook working directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "INSTALL.md").is_file() and (candidate / "checkpoints").is_dir():
            return candidate
    raise RuntimeError("Could not find the workshop folder. Start Jupyter from that folder.")


def require_genesis_ready() -> None:
    """Ensure the infrastructure smoke test has initialized Genesis on the GPU."""
    import genesis as gs

    if not getattr(gs, "_initialized", False) or gs.backend != genesis_backend():
        raise RuntimeError("Run the first-cell infrastructure smoke test before continuing.")


def bootstrap_workshop(start: Path | None = None) -> Path:
    """Prepare helper imports and return the workshop root."""
    ensure_helper_path()
    root = find_workshop_root(start)
    require_genesis_ready()
    return root


@dataclass(frozen=True)
class WorkshopPaths:
    root: Path
    baseline: Path
    trained: Path
    supplied_trained: Path
    reference: Path
    run_dir: Path
    videos: Path


def workshop_paths(root: Path) -> WorkshopPaths:
    """Return the stable checkpoint and log paths used across notebooks."""
    root = root.resolve()
    return WorkshopPaths(
        root=root,
        baseline=root / "checkpoints/g1_flat_baseline.pt",
        trained=root / "logs/roscon_stairs/model_stairs.pt",
        # A finished run of the training notebook (25-min budget), for attendees whose
        # own run is not there yet.
        supplied_trained=root / "checkpoints/g1_stairs_easy_trained.pt",
        # The reference: 2,048 robots x 24 steps x 200 PPO updates, over an hour here.
        reference=root / "checkpoints/g1_ref.pt",
        run_dir=root / "logs/roscon_stairs",
        videos=root / "logs/roscon_videos",
    )


def print_device_summary(workshop_root: Path, extra: dict[str, object] | None = None) -> None:
    """Print accelerator and workshop path information."""
    import genesis as gs

    from gslab.tasks.registry import list_tasks

    gpu = torch.cuda.get_device_properties(0)
    print(f"workshop  : {workshop_root}")
    print(f"accelerator: {gpu.name} ({gpu.total_memory / 2**30:.1f} GiB)")
    print(f"PyTorch    : {torch.__version__}")
    print(f"Genesis    : {gs.__version__}")
    if extra:
        for key, value in extra.items():
            rel = value.relative_to(workshop_root) if isinstance(value, Path) else value
            print(f"{key:<11}: {rel}")


def show_video(path: Path, title: str | None = None) -> None:
    """Display a recorded MP4 inline in the notebook."""
    from IPython.display import HTML, Video, display

    path = path.resolve()
    if title:
        display(HTML(f"<p><b>{title}</b> — <code>{path.name}</code></p>"))
    display(Video(str(path), embed=True, width=900))


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "playback"


def quiet_workshop_logs() -> None:
    """Silence noisy third-party warnings for notebook workshop runs."""
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"quadrants\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*PURE\.VIOLATION.*",
        category=UserWarning,
    )

    try:
        import genesis as gs
    except ImportError:
        return

    if not getattr(gs, "_initialized", False):
        return

    noise_filter = _GenesisNoiseFilter()
    gs.logger._logger.addFilter(noise_filter)
    gs.logger.handler.addFilter(noise_filter)
    gs.logger._logger.setLevel(logging.ERROR)
    gs.logger.handler.setLevel(logging.ERROR)


@contextmanager
def genesis_build_quiet():
    """Hide Genesis scene-build chatter while preserving real errors."""
    try:
        import genesis as gs
    except ImportError:
        yield
        return

    if not getattr(gs, "_initialized", False):
        yield
        return

    previous_level = gs.logger.level
    gs.logger._logger.setLevel(logging.ERROR)
    gs.logger.handler.setLevel(logging.ERROR)
    try:
        yield
    finally:
        gs.logger._logger.setLevel(previous_level)
        gs.logger.handler.setLevel(previous_level)


def run_environment_check(performance_mode: bool = True) -> None:
    """Verify the workshop wheel and execute one Genesis step on the GPU backend."""
    ensure_helper_path()
    clear_stale_kernel_cache_lock()
    from importlib.metadata import version

    import genesis as gs
    import gslab
    import gslab.tasks  # verify task registration before changing the default device

    print(f"gslab  : {version('gslab')} ({gslab.SRC_PATH})")
    print(f"Genesis: {gs.__version__}")
    print(f"PyTorch: {torch.__version__}")

    backend = genesis_backend()  # raises for non-ROCm builds
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the AMD GPU.")

    if not getattr(gs, "_initialized", False):
        gs.init(backend=backend, seed=0, performance_mode=performance_mode)
    if gs.backend != backend:
        raise RuntimeError(f"Genesis selected gs.{gs.backend.name}, expected gs.{backend.name}.")

    quiet_workshop_logs()

    probe = torch.arange(8, device="cuda", dtype=torch.float32)
    if probe.square().sum().item() != 140.0:
        raise RuntimeError("PyTorch ROCm compute probe returned an unexpected result.")

    smoke_scene = None
    smoke_box = None
    try:
        with genesis_build_quiet():
            smoke_scene = gs.Scene(
                show_viewer=False,
                profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            )
            smoke_scene.add_entity(gs.morphs.Plane())
            smoke_box = smoke_scene.add_entity(
                gs.morphs.Box(size=(0.10, 0.10, 0.10), pos=(0.0, 0.0, 0.20))
            )
            smoke_scene.build(n_envs=1)
            smoke_scene.step()
        if not torch.isfinite(smoke_box.get_pos()).all().item():
            raise RuntimeError("Genesis produced a non-finite physics state.")
    finally:
        if smoke_scene is not None:
            smoke_scene.destroy()
        smoke_scene = smoke_box = None
        clear_device_cache()

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"ROCm    : {torch.version.hip}")
    print(f"PASS: PyTorch sees the GPU and Genesis completed a step on gs.{backend.name}.")


def clear_device_cache() -> None:
    """Collect released scenes before the next large GPU allocation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_checkpoint(path: Path) -> Path:
    """Return a usable checkpoint or raise an actionable workshop error."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not ready: {path}. Restore the checkpoint or wait for training."
        )
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"{path.name} is unexpectedly small and is not a usable model checkpoint."
        )
    return path


def _quiet_call(function, *args, **kwargs):
    """Suppress verbose model architecture output while preserving exceptions."""
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return function(*args, **kwargs)


def make_runner(
    env,
    task: str,
    checkpoint: Path | None = None,
    log_dir: str | Path | None = None,
    agent_cfg=None,
    device: str = "cuda",
):
    """Construct an rsl_rl runner quietly and optionally load actor weights."""
    if agent_cfg is None:
        agent_cfg = load_rl_cfg(task)
        agent_cfg.logger = "tensorboard"

    def build():
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = GslabOnPolicyRunner(
            wrapped,
            asdict(agent_cfg),
            str(log_dir) if log_dir is not None else None,
            device,
        )
        if checkpoint is not None:
            runner.load(
                str(require_checkpoint(Path(checkpoint))),
                load_cfg={"actor": True},
                strict=True,
            )
        return runner

    return _quiet_call(build), agent_cfg


def load_policy(runner, checkpoint: Path, device: str = "cuda"):
    """Load actor weights without printing the full model architecture."""
    checkpoint = require_checkpoint(Path(checkpoint))
    _quiet_call(
        runner.load,
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
    )
    return runner.get_inference_policy(device=device)


def assign_environments(
    env,
    levels: int | torch.Tensor,
    terrain_types: int | torch.Tensor,
    spawn_offset_x: float = 0.0,
) -> None:
    """Assign each robot to an explicit terrain cell.

    ``spawn_offset_x`` shifts the spawn point forward (world +x, the direction
    the robots are commanded to walk) from the patch origin; the standing height
    is re-queried from the heightfield at the shifted position.
    """
    levels = torch.as_tensor(levels, device=env.device, dtype=torch.long)
    terrain_types = torch.as_tensor(
        terrain_types, device=env.device, dtype=torch.long
    )
    if levels.ndim == 0:
        levels = levels.expand(env.num_envs)
    if terrain_types.ndim == 0:
        terrain_types = terrain_types.expand(env.num_envs)
    if levels.numel() != env.num_envs or terrain_types.numel() != env.num_envs:
        raise ValueError("Terrain assignments must contain one value per robot.")

    terrain = env.terrain
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = terrain_types
    env_ids = torch.arange(env.num_envs, device=env.device)
    origins = terrain._origins_for(env_ids)
    if spawn_offset_x:
        origins[:, 0] += spawn_offset_x
        origins[:, 2] = terrain.height_at(origins)
    terrain.env_origins[:] = origins
    env.refresh_terrain_spawn()


def hold_command(env, command: torch.Tensor) -> None:
    """Prevent velocity commands from resampling during a comparison."""
    env.commands[:] = command
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        if term is not None:
            term.hold()
            term.command[:] = command


def _reset_assignment(env, levels, terrain_types, spawn_offset_x: float = 0.0) -> None:
    assign_environments(env, levels, terrain_types, spawn_offset_x)
    env.seed(0)
    env.reset()


def _rollout(
    env,
    policy,
    duration_s: float,
    command: tuple[float, float, float],
):
    command_tensor = torch.tensor(
        command, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1)
    start_x = env.robot.get_pos()[:, 0].clone()
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    travelled = torch.zeros(env.num_envs, device=env.device)

    for _ in range(round(duration_s / env.step_dt)):
        hold_command(env, command_tensor)
        env.obs_buf = env._compute_observations()
        with torch.inference_mode():
            actions = policy(env.get_observations())
        _, _, terminated, _, _ = env.step(actions)

        newly_fallen = terminated & ~fell
        if newly_fallen.any():
            travelled[newly_fallen] = (
                env.robot.get_pos()[newly_fallen, 0] - start_x[newly_fallen]
            )
        fell |= terminated

    survived = ~fell
    travelled[survived] = env.robot.get_pos()[survived, 0] - start_x[survived]
    return fell, travelled


def _evaluate_batched_env(
    env,
    runner,
    checkpoint: Path,
    num_levels: int,
    num_types: int,
    duration_s: float,
    command: tuple[float, float, float],
    device: str,
    spawn_offset_x: float = EVAL_SPAWN_OFFSET_X,
):
    """Cover every difficulty level and terrain type with one batch of robots.

    A Genesis step costs roughly the same for 8 robots as for 24 (the solver is
    launch-bound at this scale, ~0.1 s per control step), so the cheapest way to
    cover the 6 x 4 terrain grid is one rollout with a robot on every cell. That
    is the path taken when ``env.num_envs >= num_levels * num_types``. Smaller
    batches fall back to cycling through the levels one rollout per level.
    """
    if env.num_envs < num_types:
        raise ValueError(
            f"Use at least {num_types} robots so every terrain type is represented."
        )

    policy = load_policy(runner, checkpoint, device=device)
    env_ids = torch.arange(env.num_envs, device=env.device)
    cells = num_levels * num_types

    if env.num_envs >= cells:
        cell = env_ids % cells  # extra robots double up on cells round-robin
        levels = cell // num_types
        terrain_types = cell % num_types
        _reset_assignment(env, levels, terrain_types, spawn_offset_x)
        fell, travelled = _rollout(env, policy, duration_s, command)
        rows = []
        for level in range(num_levels):
            mask = levels == level
            rows.append(
                (
                    level,
                    100.0 * fell[mask].float().mean().item(),
                    travelled[mask].mean().item(),
                )
            )
        overall = (100.0 * fell.float().mean().item(), travelled.mean().item())
        return rows, overall

    terrain_types = env_ids % num_types
    rows = []
    all_falls = []
    all_travel = []

    for level in range(num_levels):
        levels = torch.full(
            (env.num_envs,), level, device=env.device, dtype=torch.long
        )
        _reset_assignment(env, levels, terrain_types, spawn_offset_x)
        fell, travelled = _rollout(env, policy, duration_s, command)
        rows.append(
            (
                level,
                100.0 * fell.float().mean().item(),
                travelled.mean().item(),
            )
        )
        all_falls.append(fell)
        all_travel.append(travelled)

    falls = torch.cat(all_falls)
    travel = torch.cat(all_travel)
    overall = (100.0 * falls.float().mean().item(), travel.mean().item())
    return rows, overall


def show_results(rows, overall, title: str) -> None:
    """Render metrics as rows and difficulty levels as columns."""
    from IPython.display import Markdown, display

    headings = [f"level {level}" for level, _, _ in rows] + ["all"]
    falls = [f"{value:.0f}%" for _, value, _ in rows] + [f"{overall[0]:.0f}%"]
    travel = [f"{value:.2f} m" for _, _, value in rows] + [f"{overall[1]:.2f} m"]
    table = [
        f"**{title} ({len(rows)} difficulty levels)**",
        "",
        "| metric | " + " | ".join(headings) + " |",
        "|---|" + "---:|" * len(headings),
        "| falls | " + " | ".join(falls) + " |",
        "| travel | " + " | ".join(travel) + " |",
    ]
    display(Markdown("\n".join(table)))


def evaluate_task(
    task: str,
    checkpoint: Path,
    num_envs: int | None = None,
    duration_s: float = EVAL_DURATION_S,
    command: tuple[float, float, float] = (0.8, 0.0, 0.0),
    device: str = "cuda",
):
    """Evaluate every difficulty in one rollout (one robot per terrain cell by default)."""
    env = runner = None
    try:
        cfg = load_env_cfg(task, play=True)
        cfg.curriculum = {}
        cfg.scene.terrain.spawn_jitter = 0.0
        cfg.episode_length_s = duration_s
        cfg.seed = 0
        generator = cfg.scene.terrain.terrain_generator
        num_levels = generator.num_rows
        num_types = len(generator.sub_terrains)
        if num_envs is None:
            num_envs = EVAL_ROBOTS_PER_CELL * num_levels * num_types
        cfg.scene.num_envs = num_envs

        env = GenesisManagerBasedRlEnv(cfg=cfg)
        runner, _ = make_runner(env, task, device=device)
        return _evaluate_batched_env(
            env,
            runner,
            checkpoint,
            num_levels,
            num_types,
            duration_s,
            command,
            device,
        )
    finally:
        if env is not None:
            env.close()
        runner = env = None
        clear_device_cache()


class EvaluationSession:
    """One evaluation scene reused for metrics and video playback.

    By default it holds ``EVAL_ROBOTS_PER_CELL`` robots on every terrain cell
    (6 levels x 4 types = 24 cells, 96 robots), so a single rollout measures the
    whole difficulty ladder in about a minute.
    """

    def __init__(
        self,
        task: str,
        num_envs: int | None = None,
        device: str = "cuda",
        workshop_root: Path | None = None,
    ) -> None:
        self.task = task
        self.num_envs = num_envs
        self.device = device
        self.workshop_root = (
            workshop_root.resolve() if workshop_root is not None else find_workshop_root()
        )

        cfg = load_env_cfg(task, play=True)
        cfg.curriculum = {}
        cfg.scene.terrain.spawn_jitter = 0.0
        cfg.episode_length_s = 30.0
        cfg.seed = 0
        cfg.scene.terrain.debug_vis = False

        generator = cfg.scene.terrain.terrain_generator
        self.terrain_names = list(generator.sub_terrains)
        self.num_levels = generator.num_rows
        self.num_types = len(self.terrain_names)
        if num_envs is None:
            num_envs = EVAL_ROBOTS_PER_CELL * self.num_levels * self.num_types
        self.num_envs = num_envs
        if num_envs < self.num_types:
            raise ValueError(
                f"num_envs must be at least {self.num_types} to cover every terrain type."
            )
        cfg.scene.num_envs = num_envs

        self.view_level = self.num_levels - 1
        self.view_type = self.terrain_names.index("real_stairs")

        grid_x = generator.size[0] * self.num_levels
        grid_y = generator.size[1] * self.num_types
        terrain_span = max(grid_x, grid_y)
        self.camera_far = max(150.0, 4.0 * terrain_span)

        with genesis_build_quiet():
            self.env = GenesisManagerBasedRlEnv(cfg=cfg)
        self.runner, _ = make_runner(self.env, task, device=device)
        self._warm_up()

    def _warm_up(self, steps: int = 10) -> None:
        """Trigger Genesis kernel compilation here, not in the first evaluate() call.

        The first control steps of a fresh scene JIT-compile the physics kernels
        (several seconds). Paying that while the scene is built keeps every
        evaluate() call at its steady-state cost.
        """
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        cells = self.num_levels * self.num_types
        _reset_assignment(
            self.env,
            (env_ids % cells) // self.num_types,
            env_ids % self.num_types,
            EVAL_SPAWN_OFFSET_X,
        )
        policy = self.runner.get_inference_policy(device=self.device)
        _rollout(self.env, policy, steps * self.env.step_dt, (0.8, 0.0, 0.0))
        clear_device_cache()

    def summary(self) -> str:
        cells = self.num_levels * self.num_types
        mode = (
            f"one rollout, {self.num_envs // cells} robot(s) per terrain cell"
            if self.num_envs >= cells
            else "batch cycled through the levels, one rollout per level"
        )
        return (
            f"evaluation scene: {self.num_envs} robots; "
            f"{self.num_levels} levels × {self.num_types} types ({mode})\n"
            f"playback: {VIDEO_ROBOTS} robots on {self.terrain_names[self.view_type]}, "
            f"levels {self.playback_levels(VIDEO_ROBOTS)} (recorded to MP4)"
        )

    def evaluate(
        self,
        checkpoint: Path,
        duration_s: float = EVAL_DURATION_S,
        command: tuple[float, float, float] = (0.8, 0.0, 0.0),
        spawn_offset_x: float = EVAL_SPAWN_OFFSET_X,
    ):
        """Falls and forward travel per difficulty level for one checkpoint.

        One rollout of ``duration_s`` with a robot on every terrain cell, spawned
        ``spawn_offset_x`` m forward so the steps are reached immediately.
        """
        return _evaluate_batched_env(
            self.env,
            self.runner,
            checkpoint,
            self.num_levels,
            self.num_types,
            duration_s,
            command,
            self.device,
            spawn_offset_x,
        )

    def show_results(self, rows, overall, title: str) -> None:
        show_results(rows, overall, title)

    def playback_levels(self, num_robots: int) -> list[int]:
        """Difficulty level for each playback robot: hardest first, wrapping if needed."""
        if num_robots < 1:
            raise ValueError("num_robots must be at least 1")
        top = self.num_levels - 1
        return [(top - i) % self.num_levels for i in range(num_robots)]

    def playback_scenario(self, scenario: str) -> tuple[int, float]:
        """Terrain type index and forward spawn offset (m) for a playback scenario."""
        if scenario not in VIDEO_SCENARIOS:
            raise ValueError(f"scenario must be one of {VIDEO_SCENARIOS}, got {scenario!r}")
        generator = self.env.terrain.cfg.terrain_generator
        if scenario == "terrain":
            rough = [name for name in self.terrain_names if "rough" in name]
            if not rough:
                raise ValueError(f"no rough terrain column in {self.terrain_names}")
            return self.terrain_names.index(rough[0]), 0.0
        stairs = generator.sub_terrains[self.terrain_names[self.view_type]]
        if scenario == "upstairs":
            return self.view_type, 0.0
        # Top landing of the box staircase: run-up + ascending flight, measured from
        # the spawn origin (middle of the run-up). Start 0.3 m onto the landing.
        landing_start = stairs.approach_length + stairs.num_steps * stairs.step_width
        return self.view_type, landing_start + 0.3 - stairs.approach_length / 2

    def _playback_camera(self, levels: Sequence[int], view_type: int, spawn_offset_x: float) -> dict:
        """A static camera that frames every playback robot's cell.

        Cells of one terrain type are stacked along +x, 8 m apart, and the robots
        walk in +x. The camera sits behind the lowest-level cell, off to the side and
        elevated, looking down the row: the robots walk away from it, the risers face
        the camera on their shaded side (the light comes from +x, +y, +z) and so stay
        visible against the white ground, which they do not from the front. Offsets
        scale with the row length so the farthest robot stays in view.
        """
        origins = self.env.terrain.terrain_origins  # (levels, types, 3), world coords
        xs = [float(origins[level, view_type, 0]) + spawn_offset_x for level in levels]
        y = float(origins[levels[0], view_type, 1])
        cell = float(self.env.terrain.cfg.terrain_generator.size[0])
        x_near, x_far = min(xs), max(xs)
        span = x_far - x_near + cell  # first spawn to last cell's end
        pos = (x_near - 5.0 - 0.1 * span, y + 4.0 + 0.2 * span, 3.0 + 0.1 * span)
        lookat = (x_near + 0.45 * span, y, 0.5)
        return dict(pos=pos, lookat=lookat, fov=45, follow=False)

    def watch(
        self,
        checkpoint: Path,
        label: str,
        duration_s: float = VIDEO_DURATION_S,
        command: tuple[float, float, float] = (0.8, 0.0, 0.0),
        output_path: Path | None = None,
        num_robots: int = VIDEO_ROBOTS,
        scenario: str = "upstairs",
    ) -> Path:
        """Record ``num_robots`` robots, one per difficulty level, and display the MP4.

        ``scenario`` picks the terrain column and the start position: ``"upstairs"``
        (box stairs from the run-up), ``"downstairs"`` (same staircases, starting on
        the top landing) or ``"terrain"`` (rough ground). One robot uses the close
        follow camera; several robots use a static camera pulled back far enough to
        frame every cell in the row.
        """
        from tqdm.auto import tqdm

        checkpoint = require_checkpoint(Path(checkpoint))
        levels = self.playback_levels(num_robots)
        view_type, spawn_offset_x = self.playback_scenario(scenario)
        video_dir = workshop_paths(self.workshop_root).videos
        video_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = video_dir / f"{_slugify(label)}_{scenario}.mp4"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cfg = load_env_cfg(self.task, play=True)
        cfg.curriculum = {}
        cfg.scene.terrain.spawn_jitter = 0.0
        cfg.episode_length_s = duration_s
        cfg.seed = 0
        cfg.scene.terrain.debug_vis = False
        cfg.scene.num_envs = num_robots

        if num_robots == 1:
            camera = dict(pos=DEFAULT_CAMERA_POS, lookat=DEFAULT_CAMERA_LOOKAT, fov=40, follow=True)
        else:
            camera = self._playback_camera(levels, view_type, spawn_offset_x)
        recorder = VideoRecorder(
            VideoConfig(
                width=VIDEO_RES[0],
                height=VIDEO_RES[1],
                far=self.camera_far,
                output=str(output_path),
                **camera,
            )
        )
        terrain_name = self.terrain_names[view_type]

        playback_env = playback_runner = None
        try:
            quiet_workshop_logs()
            with genesis_build_quiet():
                playback_env = GenesisManagerBasedRlEnv(
                    cfg=cfg,
                    pre_build_hooks=[recorder],
                )
            playback_runner, _ = make_runner(
                playback_env, self.task, device=self.device
            )
            policy = load_policy(playback_runner, checkpoint, device=self.device)
            assign_environments(
                playback_env,
                torch.tensor(levels, device=playback_env.device, dtype=torch.long),
                view_type,
                spawn_offset_x,
            )
            playback_env.seed(0)
            playback_env.reset()

            steps = round(duration_s / playback_env.step_dt)
            command_tensor = torch.tensor(
                command, device=playback_env.device, dtype=torch.float32
            ).expand(playback_env.num_envs, -1)

            print(
                f"{label} — {scenario}: {duration_s:.0f} s on {terrain_name}, {num_robots} robot(s) "
                f"at level(s) {levels}, {VIDEO_RES[0]}x{VIDEO_RES[1]} at {VIDEO_FPS} fps"
            )
            # Native Genesis recording: frames are rendered and encoded inside
            # scene.step() via camera.update_recording(), paced to VIDEO_FPS by Genesis.
            recorder.cam.start_recording(
                save_to_filename=str(output_path),
                fps=VIDEO_FPS,
            )
            obs = playback_env.get_observations()
            for _step in tqdm(
                range(steps),
                desc=f"Simulating {label}",
                unit="step",
            ):
                hold_command(playback_env, command_tensor)
                playback_env.obs_buf = playback_env._compute_observations()
                with torch.inference_mode():
                    actions = policy(obs)
                playback_env.step(actions)
                hold_command(playback_env, command_tensor)
                obs = playback_env.get_observations()

            recorder.cam.stop_recording()
            print(f"Saved video to {output_path}")
        finally:
            if playback_env is not None:
                playback_env.close()
            playback_runner = playback_env = None
            clear_device_cache()

        show_video(output_path, title=label)
        return output_path

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.runner = self.env = None
        clear_device_cache()
