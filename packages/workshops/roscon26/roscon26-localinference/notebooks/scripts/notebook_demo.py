# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# In-notebook front end for RAI's manipulation demo.
#
# The Streamlit version of this demo runs as a separate server the browser
# reaches over jupyter-server-proxy, so the demo lives in a second tab. Here the
# agent, the O3DE bridge and the chat all run in the notebook kernel and the UI
# is ipywidgets, so the whole demo is the output of one cell. Camera frames are
# fetched kernel-side from web_video_server and pushed into an Image widget over
# the kernel's existing websocket - the browser never opens the stream itself.

# rclpy has to be initialized before torch is loaded: torch's bundled C++
# runtime corrupts the heap for rcl's init, aborting the process with
# "free(): invalid size". Everything below pulls in torch through rai_perception,
# so this import must stay first.
import rclpy

if not rclpy.ok():
    rclpy.init()

import contextlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

# matplotlib and timm both warn while the imports below load them, and neither
# is actionable here - they would otherwise be the first thing in the cell,
# before any progress message. Registered before the imports that trigger them.
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings("ignore", message=".*timm\\.models\\.layers is deprecated")

import ipywidgets as widgets  # noqa: E402
import requests  # noqa: E402
from IPython.display import display  # noqa: E402
from langchain_core.callbacks.base import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from launch import LaunchDescription  # noqa: E402
from launch.actions import IncludeLaunchDescription  # noqa: E402
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: E402
from launch_ros.actions import Node  # noqa: E402
from launch_ros.substitutions import FindPackageShare  # noqa: E402
from rai.communication.ros2.connectors.ros2_connector import ROS2Connector  # noqa: E402
from rai.messages import HumanMultimodalMessage  # noqa: E402
from rai_bench.manipulation_o3de import get_scenarios  # noqa: E402
from rai_bench.manipulation_o3de.benchmark import Scenario  # noqa: E402
from rai_sim.o3de.o3de_bridge import (  # noqa: E402
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
)
from rai_sim.simulation_bridge import SceneConfig  # noqa: E402

RAI_DIR = Path("/ryzers/rai")
DEMO_SCRIPT = "/ryzers/notebooks/scripts/manipulation_demo_headless.sh"
O3DE_CONFIG = "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml"
CAMERA_TOPIC = "/color_image5"
VIDEO_PORT = 8080
LEMONADE_URL = "http://localhost:13305"
DEFAULT_MODEL = "Gemma-4-E2B-it-GGUF"
LEMONADE_CACHE = os.environ.get(
    "LEMONADE_CACHE", "/opt/lemonade-cache/lemonade"
)
LEMONADE_HF_HOME = os.environ.get(
    "LEMONADE_HF_HOME", "/opt/lemonade-cache/huggingface"
)

# Same layouts the Streamlit sidebar offers, keyed by scene config file stem
SCENARIO_NAMES = {
    "3rc": "3 Red Cubes",
    "4carrots": "4 Carrots",
    "2rc_2a": "2 Red Cubes, 2 Apples",
    "3rc_2a_1carrot": "3 Red Cubes, 2 Apples, 1 Carrot",
    "3carrots_3a_2rc": "3 Carrots, 3 Apples, 2 Red Cubes",
}
GREETING = "Hi! I am a robotic arm. What can I do for you?"


LAUNCH_LOG = "/tmp/demo_launch.log"
_launch_log_handle = None


def _log_file(path: str = LAUNCH_LOG):
    """One append handle for the whole session.

    Kept open deliberately: the logging handler installed below holds a
    reference to it and keeps writing long after the launch call returns.
    """
    global _launch_log_handle
    if _launch_log_handle is None or _launch_log_handle.closed:
        _launch_log_handle = open(path, "a", buffering=1)  # noqa: SIM115
    return _launch_log_handle


def quiet_ros_logging(path: str = LAUNCH_LOG) -> None:
    """Stop ROS 2 from writing to the notebook cell.

    Two separate sources, neither of which a file-descriptor swap can catch:

    The "[move_group-4] ..." lines are child process output that the launch
    service reads off a pipe and re-emits through Python logging. launch builds
    exactly one screen handler, lazily, bound to whatever sys.stdout was at the
    time - in a kernel that is ipykernel's ZMQ stream. Replacing the handler
    points all of it at a file instead, permanently, which matters because the
    launch service keeps running in the background after build_demo returns.

    The unprefixed lines - tf2's TF_OLD_DATA, the connector's INFO - come from
    rcl inside this process, so they take a severity bump instead.
    """
    import launch.logging

    # launch's own StreamHandler subclass, not logging.StreamHandler: it carries
    # setFormatterFor(), which get_output_loggers() calls on whatever it finds.
    handler = launch.logging.handlers.StreamHandler(_log_file(path))
    formatter = getattr(launch.logging.launch_config, "screen_formatter", None)
    if formatter is not None:
        handler.setFormatter(formatter)
    launch.logging.launch_config.screen_handler = handler

    rclpy.logging.set_logger_level("", rclpy.logging.LoggingSeverity.ERROR)


@contextlib.contextmanager
def quiet(log_path: str = LAUNCH_LOG):
    """Divert this process's stdout/stderr to a log file.

    Has to be done at the file-descriptor level. The ROS 2 launch service, the
    O3DE process and rcl's C++ logger all write to fd 1/2 directly, so
    contextlib.redirect_stdout never sees them - and ipykernel captures those
    fds, which is why their output lands in the cell at all.

    print() is unaffected: in a kernel sys.stdout is a ZMQ stream that does not
    go through fd 1, so progress messages still reach the notebook.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    log = open(log_path, "a")  # noqa: SIM115
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        yield log_path
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        log.close()


# ---------------------------------------------------------------------------
# Environment: Lemonade, headless display, camera stream
# ---------------------------------------------------------------------------


def ensure_lemonade(model: str = DEFAULT_MODEL) -> None:
    """Python equivalent of ``source lemonade_env.sh``.

    Starts the Lemonade server if it is down, loads the model, and points RAI's
    [openai] section at it. Safe to call again - every step is a no-op once done.
    """

    def running() -> bool:
        return subprocess.run(["lemonade", "status"], capture_output=True).returncode == 0

    if not running():
        # The handle has to outlive this call - lemond keeps writing to it long
        # after we return, so a context manager would close it out from under.
        log = open("/tmp/lemond.log", "a")  # noqa: SIM115
        subprocess.Popen(
            ["lemond", LEMONADE_CACHE],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "HF_HOME": LEMONADE_HF_HOME,
                "LEMONADE_CACHE": LEMONADE_CACHE,
                "LEMONADE_HF_HOME": LEMONADE_HF_HOME,
            },
        )
        for _ in range(120):
            if running():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Lemonade did not come up - see /tmp/lemond.log")

    subprocess.run(["lemonade", "load", model], check=True, capture_output=True)

    # RAI reads the backend out of config.toml, not the environment
    config = RAI_DIR / "config.toml"
    config.write_text(_point_openai_at(config.read_text(), model))
    os.environ.setdefault("OPENAI_API_KEY", "lemonade")


def _point_openai_at(text: str, model: str) -> str:
    """Rewrite the [openai] section of RAI's config.toml to use Lemonade.

    Scoped to that one section on purpose. Every vendor section has
    simple_model/complex_model keys, and in [vendor] they name the *vendor*
    ("openai"), not a model - rewriting those globally makes RAI look up a
    vendor named after the model and fail with AttributeError.
    """
    out, in_openai = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_openai = stripped == "[openai]"
        elif in_openai:
            body, newline = line.rstrip("\n"), line[len(line.rstrip("\n")) :]
            body = re.sub(
                r"^(simple_model|complex_model)\s*=.*$",
                lambda m: f'{m.group(1)} = "{model}"',
                body,
            )
            body = re.sub(r"^base_url\s*=.*$", f'base_url = "{LEMONADE_URL}/api/v0"', body)
            line = body + newline
        out.append(line)
    return "".join(out)


def camera_url(kind: str = "stream", **params) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"http://localhost:{VIDEO_PORT}/{kind}?topic={CAMERA_TOPIC}&{query}"


def _camera_alive(timeout: int = 10) -> bool:
    try:
        r = requests.get(camera_url("snapshot", quality=20), timeout=timeout)
        return r.ok and len(r.content) > 0
    except requests.RequestException:
        return False


def _server_alive(timeout: int = 8) -> bool:
    """Is web_video_server answering at all?

    Deliberately asks for the topic index rather than the camera: the camera
    topic only exists once O3DE is up, which happens after this, while a wedged
    server stops answering every request including this one.
    """
    try:
        return requests.get(f"http://localhost:{VIDEO_PORT}/", timeout=timeout).ok
    except requests.RequestException:
        return False


def _wait_for_server(deadline: float) -> bool:
    """web_video_server takes a few seconds to bind after a restart, and
    answers nothing at all until then."""
    while time.time() < deadline:
        if _server_alive():
            return True
        time.sleep(3)
    return False


def start_infrastructure(timeout: int = 180) -> str:
    """Bring up the headless display and the camera stream, and adopt $DISPLAY.

    O3DE is launched from this kernel later on, so the kernel itself needs the
    DISPLAY that Xwayland picked - that is what the script reports back.
    """
    proc = subprocess.run(
        ["setsid", "bash", DEMO_SCRIPT, "--infra-only"],
        cwd=RAI_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{DEMO_SCRIPT} --infra-only failed:\n{proc.stdout}\n{proc.stderr}")

    match = re.search(r"^DISPLAY=(\S+)$", proc.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"No display reported by the demo script:\n{proc.stdout}")
    os.environ["DISPLAY"] = match.group(1)

    # Only that the server is up - the camera topic appears later, when the
    # simulation starts publishing. The view waits for frames on its own.
    if not _wait_for_server(time.time() + 60):
        raise RuntimeError(f"web_video_server is not answering on :{VIDEO_PORT} - see /tmp/web_video_server.log")
    return os.environ["DISPLAY"]


# ---------------------------------------------------------------------------
# Simulation + agent (ported from RAI's manipulation-demo-streamlit.py)
# ---------------------------------------------------------------------------


def launch_description() -> LaunchDescription:
    launch_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                "src/examples/rai-manipulation-demo/Project/Examples/panda_moveit_config_demo.launch.py",
            ]
        )
    )
    launch_robotic_manipulation = Node(
        package="robotic_manipulation",
        executable="robotic_manipulation",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    launch_openset = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("rai_bringup"), "/launch/openset.launch.py"]),
        # v1 tool names alongside v2's, so either agent version can run
        launch_arguments={"enable_legacy_service_names": "true"}.items(),
    )
    return LaunchDescription([launch_openset, launch_moveit, launch_robotic_manipulation])


def _scenario_for(scenario_path: str) -> Scenario:
    return Scenario(
        task=None,
        scene_config=SceneConfig.load_base_config(Path(scenario_path)),
        scene_config_path=scenario_path,
    )


def initialize_o3de(scenario_path: str, agent_version: str = "v2"):
    """Launch the simulation, the ROS 2 stack and the starting scene."""
    simulation_config = O3DExROS2SimulationConfig.load_config(config_path=Path(O3DE_CONFIG))
    if agent_version == "v2":
        # v1 asks for the legacy perception service names, v2 for the new ones
        renamed = {
            "/grounding_dino_classify": "/detection",
            "/grounded_sam_segment": "/segmentation",
        }
        services = simulation_config.required_robotic_ros2_interfaces["services"]
        simulation_config.required_robotic_ros2_interfaces["services"] = [renamed.get(s, s) for s in services]

    scenario = _scenario_for(scenario_path)
    o3de = O3DEngineArmManipulationBridge(ROS2Connector(executor_type="multi_threaded"))
    o3de.init_simulation(simulation_config=simulation_config)
    o3de.launch_robotic_stack(
        required_robotic_ros2_interfaces=simulation_config.required_robotic_ros2_interfaces,
        launch_description=launch_description(),
    )
    o3de.setup_scene(scenario.scene_config)
    return o3de, scenario


def setup_new_scene(o3de, scenario_path: str) -> Scenario:
    scenario = _scenario_for(scenario_path)
    o3de.setup_scene(scenario.scene_config)
    return scenario


def available_layouts() -> dict:
    """Layout label -> scene config path, for the layouts the demo ships with."""
    layouts = {}
    for scenario in get_scenarios(levels=["medium", "hard", "very_hard"]):
        stem = Path(scenario.scene_config_path).stem
        if stem in SCENARIO_NAMES:
            layouts[SCENARIO_NAMES[stem]] = scenario.scene_config_path
    return layouts


# ---------------------------------------------------------------------------
# Widget UI
# ---------------------------------------------------------------------------


class _ToolCallLog(BaseCallbackHandler):
    """Streams the agent's tool calls into an Output widget as they happen.

    Callbacks fire on the worker thread, so this writes with append_stdout -
    "with output:" only captures on the thread that entered it.
    """

    def __init__(self, output: widgets.Output):
        self.output = output
        self._started = {}

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        name = (serialized or {}).get("name", "tool")
        self._started[run_id] = (name, time.time())
        self.output.append_stdout(f"  -> {name}({input_str})\n")

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        name, started = self._started.pop(run_id, ("tool", time.time()))
        text = str(output).replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "..."
        self.output.append_stdout(f"  <- {name} [{time.time() - started:.1f}s] {text}\n")

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        name, _ = self._started.pop(run_id, ("tool", 0))
        self.output.append_stdout(f"  !! {name} failed: {error}\n")


class ManipulationDemo:
    """The Streamlit page's behaviour, as a widget tree in a notebook cell.

    Build it once, then call ``display()``. The camera view runs on its own
    thread and each instruction runs on a worker thread, so the kernel stays
    responsive while the arm moves.
    """

    def __init__(self, o3de, scenario, agent, camera_tool, fps: int = 6):
        self.o3de = o3de
        self.scenario = scenario
        self.agent = agent
        self.camera_tool = camera_tool
        self.fps = fps
        self.messages = [AIMessage(content=GREETING)]
        self.layouts = available_layouts()

        self._stop = threading.Event()
        self._camera_thread = None
        self._worker = None

        self._build_widgets()

    # -- widgets ----------------------------------------------------------

    def _build_widgets(self):
        self.view = widgets.Image(format="jpeg", width=640, height=360)
        self.status = widgets.HTML("<i>Connecting to the simulation camera...</i>")

        current = Path(self.scenario.scene_config_path).stem
        self.layout_picker = widgets.Dropdown(
            options=list(self.layouts),
            value=SCENARIO_NAMES.get(current, next(iter(self.layouts), None)),
            description="Layout:",
        )
        self.layout_picker.observe(self._on_layout_change, names="value")

        self.reload_button = widgets.Button(description="Reload layout", tooltip="Rebuild the current scene")
        self.reload_button.on_click(lambda _: self._change_layout(self.layout_picker.value))

        self.clear_button = widgets.Button(
            description="Clear history",
            tooltip="Forget previous tool results so the agent re-checks the scene",
        )
        self.clear_button.on_click(self._on_clear)

        # "overflow", not "overflow_y": ipywidgets 8 removed the per-axis traits,
        # so overflow_y was silently dropped and the log never clipped. It kept a
        # 360px box for layout while its content ran past it, painting over the
        # input row underneath. With overflow set the log scrolls inside its box
        # and the row below stays put.
        self.log = widgets.Output(
            layout=widgets.Layout(
                height="360px",
                overflow="auto",
                border="1px solid #ddd",
                padding="6px",
                flex="0 0 auto",
            )
        )
        self.entry = widgets.Text(
            placeholder="Tell the arm what to do, then press Enter",
            # flex rather than width=100%: at 100% the entry and the button add up
            # to more than the row, which pushes Send out and scrolls the panel
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
        )
        # on_submit is the only hook for the Enter key; ipywidgets 8 deprecates
        # it without offering a replacement, so just keep the notice out of the
        # cell output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.entry.on_submit(self._on_submit)

        self.send_button = widgets.Button(
            description="Send",
            button_style="primary",
            layout=widgets.Layout(width="90px", flex="0 0 auto", margin="0 0 0 6px"),
        )
        self.send_button.on_click(lambda _: self._on_submit(self.entry))

        self.log.append_stdout(f"assistant: {GREETING}\n\n")

        left = widgets.VBox(
            [
                self.view,
                self.status,
                self.layout_picker,
                widgets.HBox([self.reload_button, self.clear_button]),
            ]
        )
        # The margin is the gap the log must never close on: the log is a fixed
        # 360px block and this row a fixed-height one after it, so the entry sits
        # at the bottom of the panel no matter how much the agent logs.
        self.input_row = widgets.HBox(
            [self.entry, self.send_button],
            layout=widgets.Layout(
                width="100%",
                margin="10px 0 0 0",
                flex="0 0 auto",
                align_items="center",
            ),
        )
        right = widgets.VBox(
            [self.log, self.input_row],
            layout=widgets.Layout(width="100%", overflow="hidden"),
        )
        self.ui = widgets.HBox([left, right])

    def display(self):
        """Start the camera and show the UI.

        Returns nothing on purpose - returning self would make Jupyter echo the
        repr under the widget.
        """
        self._start_camera()
        display(self.ui)

    # -- camera -----------------------------------------------------------

    def _start_camera(self):
        if self._camera_thread and self._camera_thread.is_alive():
            return
        self._stop.clear()
        self._camera_thread = threading.Thread(target=self._pump_frames, daemon=True)
        self._camera_thread.start()

    def _pump_frames(self):
        """One long-lived MJPEG connection, split on the JPEG markers.

        Polling /snapshot per frame instead would churn through subscribers and
        eventually hang web_video_server.
        """
        url = camera_url("stream", quality=70, width=640, height=360)
        while not self._stop.is_set():
            try:
                with requests.get(url, stream=True, timeout=30) as r:
                    buf, last = b"", 0.0
                    for chunk in r.iter_content(8192):
                        if self._stop.is_set():
                            return
                        buf += chunk
                        while True:
                            start, end = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                            if start == -1 or end == -1 or end < start:
                                break
                            frame, buf = buf[start : end + 2], buf[end + 2 :]
                            now = time.time()
                            if now - last >= 1 / self.fps:
                                self.view.value = frame
                                if self.status.value:
                                    self.status.value = ""
                                last = now
            except requests.RequestException:
                # The stream is empty until O3DE renders its first frame, and
                # drops if web_video_server is restarted - reconnect either way.
                self.status.value = "<i>Waiting for the simulation camera...</i>"
                time.sleep(5)

    def stop(self):
        """Stop the camera thread. The simulation keeps running."""
        self._stop.set()

    # -- chat -------------------------------------------------------------

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _set_enabled(self, enabled: bool):
        for w in (self.entry, self.send_button, self.layout_picker, self.reload_button, self.clear_button):
            w.disabled = not enabled

    def _on_submit(self, sender):
        instruction = sender.value.strip()
        if not instruction or self._busy():
            return
        sender.value = ""
        self._set_enabled(False)
        self.log.append_stdout(f"you: {instruction}\n")
        self._worker = threading.Thread(target=self._run_agent, args=(instruction,), daemon=True)
        self._worker.start()

    def _run_agent(self, instruction: str):
        started = time.time()
        try:
            # The agent starts from what the camera sees, like the chat panel does
            _, artifact = self.camera_tool._run()
            message = HumanMultimodalMessage(content=instruction, images=artifact.get("images", []))
            self.messages.append(message)
            result = self.agent.invoke(
                {"messages": self.messages},
                config=RunnableConfig(
                    {
                        "callbacks": [_ToolCallLog(self.log)],
                        "recursion_limit": 100,
                    }
                ),
            )
            self.messages = result["messages"]
            reply = result["messages"][-1].content
            self.log.append_stdout(f"assistant: {reply}\n({time.time() - started:.0f}s)\n\n")
        except Exception:
            self.log.append_stdout(f"error:\n{traceback.format_exc()}\n")
        finally:
            self._set_enabled(True)

    def _reset_history(self):
        # Drops cached tool results, so the agent looks at the scene again
        # instead of trusting an observation from a previous task
        self.messages = [AIMessage(content=GREETING)]
        self.log.outputs = ()
        self.log.append_stdout(f"assistant: {GREETING}\n\n")

    def _on_clear(self, _):
        # Guard only the button: the scene worker calls _reset_history directly,
        # since it is itself the thread _busy() would report.
        if self._busy():
            return
        self._reset_history()

    # -- scene ------------------------------------------------------------

    def _on_layout_change(self, change):
        self._change_layout(change["new"])

    def _change_layout(self, label: str):
        if self._busy():
            return
        path = self.layouts.get(label)
        if path is None:
            return
        self._set_enabled(False)
        self.log.append_stdout(f"[scene] loading {label}...\n")

        def work():
            try:
                self.o3de.clear_scene()
            except Exception as exc:
                self.log.append_stdout(f"[scene] could not clear: {exc}\n")
            try:
                self.scenario = setup_new_scene(self.o3de, path)
                self._reset_history()
                self.log.append_stdout(f"[scene] {label} ready\n\n")
            except Exception:
                self.log.append_stdout(f"[scene] failed:\n{traceback.format_exc()}\n")
            finally:
                self._set_enabled(True)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()


def build_demo(
    layout: str = "3 Red Cubes",
    agent_version: str = "v2",
    model: str = DEFAULT_MODEL,
    progress=print,
    verbose: bool = False,
) -> ManipulationDemo:
    """Everything the Streamlit page did at startup, in one call.

    Takes a few minutes: the scene has to render and the ROS 2 stack has to come
    up before the agent can see anything.

    The launch is loud - MoveIt, O3DE and the perception services between them
    print hundreds of lines. That goes to LAUNCH_LOG unless verbose is set, so
    the cell shows the progress messages and then the demo.
    """
    os.chdir(RAI_DIR)
    hush = contextlib.nullcontext if verbose else quiet

    progress("Starting Lemonade and pointing RAI at it...")
    ensure_lemonade(model)

    progress("Starting the headless display and camera stream...")
    progress(f"  display {start_infrastructure()}")

    layouts = available_layouts()
    if layout not in layouts:
        raise ValueError(f"Unknown layout {layout!r} - pick one of {list(layouts)}")

    progress("Launching the simulation and the ROS 2 stack (a few minutes)...")
    if not verbose:
        # Must happen before the launch: launch caches its screen handler the
        # first time a process logs, and O3DE inherits fd 1/2 when it is spawned
        quiet_ros_logging()
    with hush():
        o3de, scenario = initialize_o3de(layouts[layout], agent_version=agent_version)

    progress("Building the agent...")
    with hush():
        # Imported here rather than at module scope: manipulation_common reads
        # RAI's config.toml, which ensure_lemonade has only just pointed at
        # Lemonade.
        sys.path.insert(0, str(RAI_DIR / "examples"))
        from manipulation_common import create_agent

        agent, camera_tool = create_agent(version=agent_version)

    if not verbose:
        progress(f"  (launch output in {LAUNCH_LOG})")
    progress("Ready.")
    return ManipulationDemo(o3de, scenario, agent, camera_tool)
