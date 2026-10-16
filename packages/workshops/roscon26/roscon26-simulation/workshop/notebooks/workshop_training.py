# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Notebook training helpers: quiet logs and live PPO progress."""

from __future__ import annotations

import datetime
import statistics
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from workshop_helpers import clear_device_cache, genesis_build_quiet, make_runner, quiet_workshop_logs

if TYPE_CHECKING:
    from gslab.rl.rsl_rl.runner import GslabOnPolicyRunner


class TrainingProgressDisplay:
    """Live progress bar and reward curves for PPO training in Jupyter."""

    def __init__(self, total_iterations: int, time_budget_s: float | None = None) -> None:
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        self._total = max(total_iterations, 1)
        self._time_budget_s = time_budget_s
        self._started = time.monotonic()
        self._iterations: list[int] = []
        self._rewards: list[float] = []
        self._episode_lengths: list[float] = []
        self._closed = False
        self._setup_stop: threading.Event | None = None
        self._setup_thread: threading.Thread | None = None

        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=self._total,
            description="PPO:",
            bar_style="info",
            layout=widgets.Layout(width="100%"),
        )
        self.status = widgets.HTML(value="Preparing training environment…")

        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Mean reward", "Mean episode length"),
        )
        figure.add_trace(
            go.Scatter(x=[], y=[], mode="lines", line=dict(color="#2563eb", width=2)),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=[], y=[], mode="lines", line=dict(color="#16a34a", width=2)),
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="iteration", row=1, col=1)
        figure.update_xaxes(title_text="iteration", row=1, col=2)
        figure.update_yaxes(title_text="reward", row=1, col=1)
        figure.update_yaxes(title_text="steps", row=1, col=2)
        figure.update_layout(
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
        )
        self.plot = go.FigureWidget(figure)

        self.container = widgets.VBox([self.progress, self.status, self.plot])
        self._displayed = False

    def show(self) -> None:
        if self._displayed:
            return
        from IPython.display import display

        display(self.container)
        self._displayed = True

    def begin_setup(self, message: str) -> None:
        """Show a ticking elapsed timer while Genesis compiles or the runner loads."""
        self.end_setup()
        self.progress.description = "Setup:"
        self.progress.max = 1
        self.progress.value = 0
        self.progress.bar_style = "warning"
        self._setup_message = message
        self._setup_stop = threading.Event()
        self._setup_thread = threading.Thread(target=self._setup_tick, daemon=True)
        self._setup_thread.start()

    def _setup_tick(self) -> None:
        start = time.monotonic()
        while self._setup_stop is not None and not self._setup_stop.wait(1.0):
            elapsed = int(time.monotonic() - start)
            self.status.value = (
                f"<b>{self._setup_message}</b> — {elapsed}s elapsed "
                f"(Genesis compiles kernels on first run; this is normal)"
            )

    def end_setup(self) -> None:
        if self._setup_stop is not None:
            self._setup_stop.set()
            if self._setup_thread is not None:
                self._setup_thread.join(timeout=0.2)
            self._setup_stop = None
            self._setup_thread = None

        self.progress.description = "PPO:"
        self.progress.max = self._total
        self.progress.value = 0
        self.progress.bar_style = "info"

    def update(
        self,
        *,
        it: int,
        start_it: int,
        total_it: int,
        logger,
        loss_dict: dict,
        collect_time: float,
        learn_time: float,
        action_std,
        **_,
    ) -> None:
        done = it + 1 - start_it
        self.progress.value = min(done, self._total)
        self.progress.max = max(total_it - start_it, 1)

        mean_reward = (
            statistics.mean(logger.rewbuffer) if len(logger.rewbuffer) > 0 else float("nan")
        )
        mean_length = (
            statistics.mean(logger.lenbuffer) if len(logger.lenbuffer) > 0 else float("nan")
        )

        self._iterations.append(it)
        self._rewards.append(mean_reward)
        self._episode_lengths.append(mean_length)

        iteration_time = collect_time + learn_time
        collection_size = (
            logger.cfg["num_steps_per_env"] * logger.num_envs * logger.gpu_world_size
        )
        fps = int(collection_size / iteration_time) if iteration_time > 0 else 0
        planned_total = total_it
        if self._time_budget_s is not None and done > 0:
            # Iterations that fit in the wall-clock budget, given the average
            # iteration time so far (setup time already counted in elapsed).
            elapsed = time.monotonic() - self._started
            per_iteration = logger.tot_time / done
            fits = done + int((self._time_budget_s - elapsed) // per_iteration)
            planned_total = start_it + max(min(fits, total_it - start_it), done)
            self.progress.max = max(planned_total - start_it, 1)
        remaining = max(planned_total - it - 1, 0)
        eta_seconds = int(logger.tot_time / done * remaining) if done > 0 else 0
        value_loss = loss_dict.get("value", float("nan"))
        budget_note = (
            f" &nbsp;|&nbsp; budget <b>{int(self._time_budget_s // 60)} min</b>"
            if self._time_budget_s is not None
            else ""
        )

        self.status.value = (
            f"<b>Iteration {it + 1}/{planned_total}</b> &nbsp;|&nbsp; "
            f"reward <b>{mean_reward:.2f}</b> &nbsp;|&nbsp; "
            f"ep len <b>{mean_length:.1f}</b> &nbsp;|&nbsp; "
            f"value loss <b>{value_loss:.3f}</b> &nbsp;|&nbsp; "
            f"fps <b>{fps}</b> &nbsp;|&nbsp; "
            f"ETA <b>{datetime.timedelta(seconds=eta_seconds)}</b>"
            f"{budget_note}"
        )

        with self.plot.batch_update():
            self.plot.data[0].x = tuple(self._iterations)
            self.plot.data[0].y = tuple(self._rewards)
            self.plot.data[1].x = tuple(self._iterations)
            self.plot.data[1].y = tuple(self._episode_lengths)

    def finish(self, message: str) -> None:
        self.end_setup()
        self.progress.bar_style = "success"
        self.progress.value = self.progress.max
        self.status.value = f"<b>{message}</b>"

    def close(self) -> None:
        if self._closed:
            return
        self.end_setup()
        self._closed = True


class _TimeBudgetReached(Exception):
    """Raised from the per-iteration log hook to stop PPO before the budget runs out."""


def _patch_runner_logger(
    runner: GslabOnPolicyRunner,
    display: TrainingProgressDisplay,
    deadline: float | None = None,
) -> None:
    logger = runner.logger
    logger._workshop_original_log = logger.log
    logger._workshop_original_init = logger.init_logging_writer

    def quiet_init_logging_writer() -> None:
        sink = StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            logger._workshop_original_init()

    def widget_log(**kwargs) -> None:
        sink = StringIO()
        with redirect_stdout(sink):
            logger._workshop_original_log(**kwargs, print_minimal=True)
        display.update(logger=logger, **kwargs)
        if deadline is not None:
            # Stop when the next iteration would overshoot the deadline. Use the
            # slower of the last and the mean iteration time so a slow machine
            # is not caught out by a fast first iteration.
            done = kwargs["it"] + 1 - kwargs["start_it"]
            last = kwargs["collect_time"] + kwargs["learn_time"]
            per_iteration = max(last, logger.tot_time / max(done, 1))
            if time.monotonic() + per_iteration > deadline:
                raise _TimeBudgetReached(done)

    logger.init_logging_writer = quiet_init_logging_writer
    logger.log = widget_log


def _restore_runner_logger(runner: GslabOnPolicyRunner | None) -> None:
    if runner is None:
        return
    logger = runner.logger
    if hasattr(logger, "_workshop_original_log"):
        logger.log = logger._workshop_original_log
        del logger._workshop_original_log
    if hasattr(logger, "_workshop_original_init"):
        logger.init_logging_writer = logger._workshop_original_init
        del logger._workshop_original_init


def run_training(
    *,
    env_cfg,
    task: str,
    baseline: Path,
    run_dir: Path,
    agent_cfg,
    output_checkpoint: Path,
    iterations: int,
    workshop_root: Path | None = None,
    device: str = "cuda",
    time_budget_s: float | None = None,
) -> Path:
    """Build the scene, fine-tune with PPO, and show live notebook progress.

    ``iterations`` is the maximum number of PPO updates. With ``time_budget_s``
    set, training also stops once the next update would not finish inside the
    budget, measured from the start of this call (so scene build and kernel
    compilation count), and the checkpoint is saved as usual. Genesis step time
    varies a lot between machines and even between boots of the same machine,
    so the budget is what keeps the workshop schedule; the iteration cap is what
    keeps a fast machine from over-training.
    """
    from gslab.envs import GenesisManagerBasedRlEnv

    quiet_workshop_logs()
    started = time.monotonic()
    deadline = started + time_budget_s if time_budget_s is not None else None
    train_env = None
    train_runner = None
    display = TrainingProgressDisplay(iterations, time_budget_s)
    display.show()

    try:
        display.begin_setup("Building simulation scene")
        try:
            with genesis_build_quiet():
                train_env = GenesisManagerBasedRlEnv(cfg=env_cfg)
        finally:
            display.end_setup()

        display.begin_setup("Loading policy checkpoint")
        try:
            train_runner, _ = make_runner(
                train_env,
                task,
                baseline,
                run_dir,
                agent_cfg,
                device,
            )
        finally:
            display.end_setup()

        display.status.value = "<b>Training started</b>"
        _patch_runner_logger(train_runner, display, deadline)
        completed = iterations
        try:
            train_runner.learn(
                num_learning_iterations=iterations,
                init_at_random_ep_len=True,
            )
        except _TimeBudgetReached as stop:
            completed = int(stop.args[0])
            # learn() normally closes the TensorBoard writer on its way out.
            if train_runner.logger.writer is not None:
                train_runner.logger.stop_logging_writer()
        train_runner.save(str(output_checkpoint))
        elapsed = time.monotonic() - started

        if workshop_root is not None:
            saved_path = output_checkpoint.relative_to(workshop_root)
        else:
            saved_path = output_checkpoint
        summary = (
            f"{completed} PPO updates in {datetime.timedelta(seconds=int(elapsed))}"
            + (" (time budget reached)" if completed < iterations else "")
        )
        display.finish(f"Saved checkpoint to {saved_path} — {summary}")
        print(f"\n{summary}")
        print(f"saved: {saved_path}")
        return output_checkpoint
    finally:
        _restore_runner_logger(train_runner)
        if train_env is not None:
            train_env.close()
        display.close()
        clear_device_cache()
