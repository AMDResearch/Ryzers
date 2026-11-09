# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------------------------------------
# This ryzer is a THIN workshop layer. Everything heavy is composed:
#
#   ryzers build jupyterlab ros o3de rai lemonade-sdk capx roscon26-localinference
#
#   jupyterlab    notebook server + the 8888 port mapping
#   ros           ROS 2 Jazzy desktop, cv_bridge, web_video_server (notebook 2)
#   o3de          Vulkan SDK + the O3DE engine behind the RAI manipulation scene
#   rai           /ryzers/rai, the manipulation demo, GroundingDINO/SAM2 weights
#   lemonade-sdk  lemond, the local OpenAI-compatible server on :13305
#   capx          /ryzers/cap-x, Robosuite, PyRoKi, Contact-GraspNet
#
# What is left for this layer is only what the workshop adds on top:
#   * headless Weston + Xwayland, so O3DE gets a presentable Vulkan surface
#   * the workshop's CaP-X patches (OWLv2+SAM2 instead of gated SAM3, FP16
#     checkpoint staging, a 900 s service wait)
#   * the baked Gemma GGUFs and perception weights, so the room runs offline
#   * HELIX + OpenCode, the repo-evolution toolchain notebook 4 documents
#   * the two Jupyter kernels the notebooks select, and the course content
# ---------------------------------------------------------------------------

# HELIX drives repository-level evolution in notebook 4; OpenCode is its local
# coding-agent backend. Both pins are what tests/test_rho.sh asserts.
ARG HELIX_REF=29cfa6e5eae902f6bc6d2113e51499e92c6109ee
ARG HELIX_VERSION=0.2.1
ARG OPENCODE_VERSION=1.18.18

# Baked weights, in GB:
#   BAKE_LEMONADE_MODELS=1     Gemma-4 E2B + E4B Q4_K_M + mmproj-F16   ~10.1
#   BAKE_PERCEPTION_WEIGHTS=1  SAM2.1 large/small, OWLv2 large/base     ~3.5
# Set either to 0 to fetch at runtime instead (needs network in the room).
ARG BAKE_LEMONADE_MODELS=1
ARG BAKE_PERCEPTION_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HELIX_REF=${HELIX_REF} \
    OPENCODE_VERSION=${OPENCODE_VERSION} \
    ROSCON26_BAKED_LEMONADE=${BAKE_LEMONADE_MODELS} \
    ROSCON26_BAKED_PERCEPTION=${BAKE_PERCEPTION_WEIGHTS} \
    CAPX_ROOT=/ryzers/cap-x \
    CAPX_VENV=/opt/capx-venv \
    CAPX_CACHE=/opt/capx-cache \
    CAPX_PYTHON=/opt/capx-venv/bin/python \
    HELIX_BIN=/opt/capx-venv/bin/helix \
    LEMONADE_CACHE=/opt/lemonade-cache/lemonade \
    LEMONADE_HF_HOME=/opt/lemonade-cache/huggingface \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

WORKDIR /ryzers

# ---------------------------------------------------------------------------
# Headless display + Node.
#
# O3DE renders through Vulkan, which needs a *presentable* surface. Xvfb cannot
# give it one (no DRI3, so only the llvmpipe software device is presentable),
# but Xwayland on a headless Weston compositor can: it hands the X server the
# real /dev/dri render node, so the iGPU does the rendering. This is the single
# reason weston/xwayland are here and not in the `o3de` ryzer - the ryzer
# assumes a desktop, the workshop assumes a browser.
#
# nodejs/npm exist only to carry OpenCode. ffmpeg, unzip and the Vulkan/Mesa
# drivers already arrive with `ros`, `rai` and `capx`; they are listed anyway so
# a standalone `ryzers build roscon26-localinference` still produces something
# coherent, and apt is a no-op when they are present.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip \
        ffmpeg \
        nodejs npm \
        weston xwayland \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# /opt/capx-venv compatibility shim.
#
# The workshop scripts, the recorded opencode.json permission rules and the
# prompts HELIX hands the mutation model all spell the CaP-X interpreter
# `/opt/capx-venv/bin/python`, because the JupyterHub image kept CaP-X in its
# own venv. The composed `capx` ryzer installs into the base image's
# interpreter instead, so point the old name at the real prefix rather than
# rewriting the literal in a dozen places - including inside recorded JSON that
# must keep matching what the search actually ran.
#
# sys.prefix is read from the interpreter rather than hardcoded, so this works
# whether the base image uses /opt/venv (rocm/pytorch does) or a bare /usr.
# ---------------------------------------------------------------------------
RUN PREFIX="$(python3 -c 'import sys; print(sys.prefix)')" \
    && test -x "${PREFIX}/bin/python" \
    && ln -sfn "${PREFIX}" /opt/capx-venv \
    && echo "capx-venv shim -> ${PREFIX}" \
    && : "Soft, because a standalone build has no capx to find. In a chain" \
         "build this line is the first thing to report a broken shim, and" \
         "the composed /ryzers/test_capx.sh exercises the stack properly." \
    && /opt/capx-venv/bin/python -c \
      "import capx, robosuite, torch; assert torch.version.hip, 'not a ROCm torch'; \
print('composed CaP-X visible through the shim:', torch.__version__)" \
       || echo "WARNING: CaP-X not importable - add 'capx' to the build chain"

# ---------------------------------------------------------------------------
# Notebook-only Python additions.
#
# `-c /ryzers/constraints.txt` is the constraints file the composed `capx`
# layer wrote (base torch + its uv override-dependencies, numpy==1.26.4 among
# them). Reusing it is what stops a convenience package dragging numpy to 2.x
# and taking rai_perception's numpy-1-ABI scikit-learn with it. It is absent in
# a standalone build, hence the guard.
#
# streamlit is deliberately NOT listed: rai_core already depends on
# "streamlit>=1.44,<2" and owns that pin. pandas is capped below 3.0 because
# nothing in this image has been exercised against the pandas 3 API changes.
# ---------------------------------------------------------------------------
RUN CONSTRAINTS="$([ -f /ryzers/constraints.txt ] && echo '-c /ryzers/constraints.txt' || true)" \
    && pip3 install --no-cache-dir --break-system-packages ${CONSTRAINTS} \
        "ipywidgets==8.1.9" \
        "pandas>=2.2,<3" \
        "ipykernel>=6.29"

# ---------------------------------------------------------------------------
# Workshop CaP-X patches, applied on top of the composed `capx` layer.
#
# These are the four modifications the JupyterHub image carried that the
# standalone ryzer does not. They are a patch layer rather than a capx fork so
# that CAPX_REF, the dependency resolution and the submodule selection stay
# owned by packages/robotics/capx.
#
#   1. registry      FrankaControlApi / FrankaControlSpillWipeApi default to
#                    use_sam3=False, so the four live scenarios ground text with
#                    OWLv2 and segment with box-prompted SAM2. The capx ryzer
#                    only *adds* a FrankaControlApiSam2 alias; the env_configs
#                    the notebooks load name the plain API, so the default has
#                    to move. This also drops the HF_TOKEN requirement.
#   2. FP16 staging  ROCm spends minutes transferring float32 CPU tensors to an
#                    APU. Load as float16 and restore float32 on-device.
#   3. 900 s wait    already applied by the capx ryzer; the script accepts that
#                    and re-applies only if it finds the upstream 120 s.
#   4. env_configs   swap the gated SAM3 server for OWLv2-large + SAM2.1-large
#                    and emit matching *_fast.yaml profiles (base/small).
#
# Every replacement asserts its match count, so an upstream change breaks the
# build here instead of silently producing an image that fails in the room.
# ---------------------------------------------------------------------------
COPY patch_capx_workshop.py /ryzers/patch_capx_workshop.py
RUN python3 -I /ryzers/patch_capx_workshop.py /ryzers/cap-x

# ---------------------------------------------------------------------------
# Perception weights (~3.5 GB), staged into /opt/capx-cache.
#
# Baked rather than fetched: the room has no reliable network, and both OWLv2
# and SAM2.1 are ungated, so there is no token to configure. The *_fast.yaml
# profile's smaller pair costs 0.8 GB of that and is what makes an A/B against
# the large models possible without a download mid-session.
#
# Duplicate .pt/.bin copies of the same checkpoints are excluded - allow_patterns
# names the safetensors only, which roughly halves what lands in the image.
# ---------------------------------------------------------------------------
COPY fetch_workshop_models.py /ryzers/fetch_workshop_models.py
RUN mkdir -p "${CAPX_CACHE}" \
    && if [ "${BAKE_PERCEPTION_WEIGHTS}" = "0" ]; then \
         echo "BAKE_PERCEPTION_WEIGHTS=0 - SAM2/OWLv2 will download on first use"; \
       elif ! python3 -c "import huggingface_hub" 2>/dev/null; then \
         echo "WARNING: huggingface_hub missing - add 'capx' to the build chain"; \
       else \
         HF_HOME="${CAPX_CACHE}" python3 -I /ryzers/fetch_workshop_models.py perception; \
       fi

# ---------------------------------------------------------------------------
# Lemonade models (~10.1 GB), staged into /opt/lemonade-cache.
#
# Notebooks 1-3 are built around these two checkpoints, so they are baked.
# Only the quantizations Lemonade's built-in aliases resolve to are fetched -
# the full GGUF repositories carry twenty-odd redundant quantizations each.
#
# The Qwen3-Coder-30B mutation model is deliberately NOT baked. Notebook 4
# replays a recorded evolution and never calls a language model, so caching
# another 18 GB would serve one opt-in path; the notebook prints the
# `lemonade pull` command that fetches and registers it.
#
# LEMONADE_CACHE / LEMONADE_HF_HOME keep this out of $HOME on purpose. That
# started as a JupyterHub workaround, but it still earns its place here: both
# $HOME/.cache/huggingface and $HOME/.cache/lemonade are bind-mounted from the
# host by the composed lemonade-sdk ryzer, which would hide image-baked weights.
# ---------------------------------------------------------------------------
RUN mkdir -p "${LEMONADE_CACHE}" "${LEMONADE_HF_HOME}" \
    && if [ "${BAKE_LEMONADE_MODELS}" = "0" ]; then \
         echo "BAKE_LEMONADE_MODELS=0 - run 'lemonade pull' before the session"; \
       elif ! python3 -c "import huggingface_hub" 2>/dev/null; then \
         echo "WARNING: huggingface_hub missing - add 'capx' to the build chain"; \
       else \
         HF_HOME="${LEMONADE_HF_HOME}" python3 -I /ryzers/fetch_workshop_models.py lemonade; \
       fi

# ---------------------------------------------------------------------------
# HELIX + OpenCode.
#
# Last of the install layers on purpose: a harness bump here must not rebuild
# the simulator, the perception stack or ten gigabytes of weights.
# ---------------------------------------------------------------------------
RUN CONSTRAINTS="$([ -f /ryzers/constraints.txt ] && echo '-c /ryzers/constraints.txt' || true)" \
    && git clone https://github.com/KE7/helix.git /ryzers/helix \
    && git -C /ryzers/helix checkout "${HELIX_REF}" \
    && pip3 install --no-cache-dir --break-system-packages ${CONSTRAINTS} -e /ryzers/helix \
    && npm install --global "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force \
    && /opt/capx-venv/bin/helix --version \
    && python3 -c "from helix import __version__ as v; \
assert v == '${HELIX_VERSION}', f'helix {v} != ${HELIX_VERSION}'; print('helix', v)" \
    && test "$(opencode --version)" = "${OPENCODE_VERSION}" \
    && { python3 -c "import capx, helix, torch; assert torch.version.hip; \
print('HELIX/OpenCode workshop stack OK')" \
         || echo "WARNING: CaP-X/ROCm torch not importable - standalone build?"; }

# ---------------------------------------------------------------------------
# Jupyter kernels.
#
# Two kernelspecs, matching the `name` recorded in the notebooks' metadata
# ("rai" for 0-2, "capx" for 3-4), so they select themselves on open.
#
# In the composed image both resolve to the same interpreter - the CaP-X venv
# of the JupyterHub build is now the base environment. They are still two
# kernels because their *environments* differ: CaP-X needs the baked perception
# cache and the EGL backend chosen before MuJoCo imports, while RAI needs the
# ROS 2 overlay and must not inherit CAPX_ROOT's chdir.
# ---------------------------------------------------------------------------
COPY rai_kernel.sh /ryzers/rai_kernel.sh
RUN chmod +x /ryzers/rai_kernel.sh \
    && mkdir -p /usr/local/share/jupyter/kernels/capx \
               /usr/local/share/jupyter/kernels/rai \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/capx-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "CaP-X (ROCm)",' \
      '  "language": "python",' \
      '  "env": {' \
      '    "HF_HOME": "/opt/capx-cache",' \
      '    "MUJOCO_GL": "egl",' \
      '    "PYOPENGL_PLATFORM": "egl",' \
      '    "CAPX_ROOT": "/ryzers/cap-x"' \
      '  }' \
      '}' \
      > /usr/local/share/jupyter/kernels/capx/kernel.json \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/ryzers/rai_kernel.sh", "-f", "{connection_file}"],' \
      '  "display_name": "RAI (ROCm)",' \
      '  "language": "python"' \
      '}' \
      > /usr/local/share/jupyter/kernels/rai/kernel.json \
    && : "Fail the build now rather than in the room if jupyter cannot see" \
         "them. Skipped when jupyterlab is not in the chain." \
    && if command -v jupyter >/dev/null 2>&1; then \
         jupyter kernelspec list; \
         jupyter kernelspec list 2>&1 | grep -q 'capx' || exit 1; \
         jupyter kernelspec list 2>&1 | grep -q 'rai' || exit 1; \
       else \
         echo "jupyter not present - build with 'jupyterlab' in the chain"; \
       fi

# ---------------------------------------------------------------------------
# Course content and entry points.
# ---------------------------------------------------------------------------
COPY notebooks /ryzers/notebooks/
COPY start.sh /ryzers/start_roscon26_localinference.sh
COPY test.sh /ryzers/test_roscon26_localinference.sh
COPY demo.sh /ryzers/demo_roscon26_localinference.sh

# The deep per-stack checks keep the names they have in the course repository
# under notebooks/tests, and are exposed under roscon26-prefixed names at
# /ryzers. The prefix matters: /ryzers/test_capx.sh, /ryzers/test_rai.sh,
# /ryzers/test_ros.sh and /ryzers/test_lemonade-sdk.sh already belong to the
# composed ryzers, and those still work - overwriting them would throw away
# the upstream checks in exchange for the workshop ones.
RUN chmod +x \
        /ryzers/start_roscon26_localinference.sh \
        /ryzers/test_roscon26_localinference.sh \
        /ryzers/demo_roscon26_localinference.sh \
        /ryzers/notebooks/scripts/*.sh \
        /ryzers/notebooks/tests/*.sh \
    && ln -sf /ryzers/notebooks/tests/test_capx.sh          /ryzers/test_roscon26_capx.sh \
    && ln -sf /ryzers/notebooks/tests/test_lemonade-sdk.sh  /ryzers/test_roscon26_lemonade.sh \
    && ln -sf /ryzers/notebooks/tests/test_rho.sh           /ryzers/test_roscon26_rho.sh \
    && ln -sf /ryzers/notebooks/tests/test_rho_multitask.sh /ryzers/test_roscon26_rho_multitask.sh \
    && mkdir -p /ryzers/notebooks/outputs \
    && : "Xwayland falls back to a socket file in /tmp/.X11-unix because" \
         "--network=host shares the abstract socket namespace and @X0 is" \
         "already taken on the host. Weston crashes if the directory is gone." \
    && mkdir -p -m 1777 /tmp/.X11-unix

WORKDIR /ryzers/notebooks

# No EXPOSE here. Every port this workshop serves on is already declared by a
# composed ryzer (8888 jupyterlab, 13305 lemonade-sdk, 8501 rai, 11311 ros) or
# bound to loopback inside the container (8080 web_video_server, 8113/8115/8116/
# 8117 the CaP-X services). Ryzers runs with --network=host regardless.

CMD ["/ryzers/test_roscon26_localinference.sh"]
