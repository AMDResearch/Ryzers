"""Headless MuJoCo GL setup using system Mesa/OSMesa libraries.

Expects `libosmesa6`, Mesa EGL, and DRI drivers from the course Docker image.
Rendering must run in a subprocess started with GL variables already set, because
the dynamic loader captures its search path at process start.

Backends are probed in order because which one works depends on the Mesa build:

- `osmesa` needs `libOSMesa.so.8`.
- the EGL backends need software rendering (`LIBGL_ALWAYS_SOFTWARE=1`).
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Set in child environments so a re-executed process does not loop.
CONFIGURED_FLAG = "HEADLESS_GL_CONFIGURED"


@contextlib.contextmanager
def quiet_demo_output():
    """Hide optional-backend chatter and benign JAX cast warnings during demos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            yield


def enable_demo_quiet_mode() -> None:
    """Apply process-wide filters for a clean notebook demo."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="overflow encountered in cast")

# Backend name -> variables that select it. Every GL variable listed in any
# candidate is cleared first, so candidates never inherit each other's state.
BACKEND_CANDIDATES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "osmesa",
        {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
    ),
    (
        "egl-surfaceless",
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "surfaceless",
        },
    ),
    (
        "egl-device",
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "device",
        },
    ),
    (
        "egl-default",
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LIBGL_ALWAYS_SOFTWARE": "1",
        },
    ),
)

DEFAULT_BACKEND = BACKEND_CANDIDATES[0][0]

_BACKEND_KEYS = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE",
    "EGL_PLATFORM",
)

_GL_ENV_KEYS = (
    *_BACKEND_KEYS,
    "LD_LIBRARY_PATH",
    "MESA_LOADER_DRIVER_OVERRIDE",
    "GALLIUM_DRIVER",
    "LIBGL_DRIVERS_PATH",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    CONFIGURED_FLAG,
)

_PROBE_CODE = (
    "import mujoco\n"
    "model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')\n"
    "renderer = mujoco.Renderer(model, height=64, width=64)\n"
    "renderer.render()\n"
    "renderer.close()\n"
    "print('probe ok')\n"
)

_SYSTEM_DRI = Path("/usr/lib/x86_64-linux-gnu/dri")
_EGL_VENDOR_FILES = (
    Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json"),
    Path("/usr/share/glvnd/egl_vendor.d/10_mesa.json"),
)


def build_gl_env(repo_root: Path, backend: str = DEFAULT_BACKEND) -> dict[str, str]:
    """Environment for rendering with the named backend."""
    del repo_root  # kept for call-site compatibility
    overrides = dict(BACKEND_CANDIDATES).get(backend)
    if overrides is None:
        raise ValueError(f"Unknown backend {backend!r}")

    env = os.environ.copy()
    for key in _BACKEND_KEYS:
        env.pop(key, None)

    env["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    env.update(overrides)

    if _SYSTEM_DRI.is_dir():
        env["LIBGL_DRIVERS_PATH"] = str(_SYSTEM_DRI)

    for vendor in _EGL_VENDOR_FILES:
        if vendor.is_file():
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
            break

    env[CONFIGURED_FLAG] = "1"
    return env


def apply_gl_env(env: dict[str, str]) -> None:
    """Copy GL variables into this process, for reporting and child processes.

    This does not make rendering work in the current interpreter; `dlopen`
    ignores `LD_LIBRARY_PATH` changes made after process start.
    """
    for key in _GL_ENV_KEYS:
        if key in env:
            os.environ[key] = env[key]
        else:
            os.environ.pop(key, None)


def probe_render_subprocess(
    repo_root: Path,
) -> tuple[bool, str, dict[str, str] | None]:
    """Find a backend that can render, by trying each one in a fresh process.

    Returns whether rendering worked, a description of the outcome, and the
    environment that succeeded.
    """
    failures = []
    for backend, _ in BACKEND_CANDIDATES:
        env = build_gl_env(repo_root, backend)
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            env=env,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode == 0:
            return True, backend, env
        error = (result.stderr or result.stdout or "no output").strip()
        failures.append(f"--- {backend} ---\n{error[-600:]}")

    return False, "\n\n".join(failures), None


def resolve_gl_env(repo_root: Path) -> dict[str, str]:
    """Environment that renders successfully, or the first candidate."""
    ok, _, env = probe_render_subprocess(repo_root)
    if ok and env is not None:
        return env
    return build_gl_env(repo_root)


def reexec_with_gl_env(repo_root: Path) -> None:
    """Restart this process so the dynamic loader sees the GL libraries."""
    if os.environ.get(CONFIGURED_FLAG) == "1":
        return
    os.execve(sys.executable, [sys.executable, *sys.argv], resolve_gl_env(repo_root))
