# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

# MuJoCo Playground pinned to the commit the workshop checkpoints were trained
# against. The PandaPickCube observation/action contract must match the Orbax
# checkpoints in notebooks/assets or the Brax policy fails to load.
ARG PLAYGROUND_COMMIT=cf88ae6e9c38654199d85c5c976795a7dc350571

# The only gfx1151-verified ROCm JAX set. The AMD gfx1151 wheel index itself
# stops at 0.9.x, and jax 0.9.x hits HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
# in a fused scatter on this chip (fixed in XLA for jax >= 0.10). So jax and its
# ROCm plugin come from PyPI, and the AMD index supplies the gfx1151 rocm-sdk
# library wheels the plugin loads.
ARG JAX_VERSION=0.11.1
ARG ROCM_WHEEL_VERSION=7.13.0
ARG GPU_TARGET=gfx1151

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    RL_VENV=/opt/rl-venv

WORKDIR /ryzers

# Headless GL for MuJoCo offscreen rendering. assets/headless_gl.py probes
# osmesa first and then EGL, all through llvmpipe software rendering, so Mesa
# and the DRI drivers are what matter here rather than the ROCm stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git unzip \
        libegl1 libegl-dev libgl1 libgles-dev \
        libgl1-mesa-dri libglew-dev \
        libosmesa6 libosmesa6-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Isolated RL environment.
#
# ROCm JAX resolves its HIP libraries from the `rocm` *wheel* SDK (7.13), while
# the base image's torch uses the system ROCm at /opt/rocm (7.14). Installing
# both into the same site-packages puts two ROCm runtimes on one loader path.
# A standalone venv (deliberately NOT --system-site-packages) keeps the JAX
# stack off the base interpreter, so composing this ryzer with a torch package
# leaves that torch untouched. Same reason capx/rai/gslab each get their own.
# ---------------------------------------------------------------------------
RUN python3 -m venv ${RL_VENV} \
    && ${RL_VENV}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# pip searches every index and takes the best match across them, which is the
# behaviour uv spells --index-strategy unsafe-best-match. jax/jaxlib/plugin/pjrt
# resolve from PyPI; `rocm[libraries]` only exists on the gfx1151 index.
RUN ${RL_VENV}/bin/pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://repo.amd.com/rocm/whl/${GPU_TARGET}/ \
        "jax==${JAX_VERSION}" \
        "jaxlib==${JAX_VERSION}" \
        "jax-rocm7-plugin==${JAX_VERSION}" \
        "jax-rocm7-pjrt==${JAX_VERSION}" \
        "rocm[libraries]==${ROCM_WHEEL_VERSION}" \
    && ${RL_VENV}/bin/pip freeze \
        | grep -iE '^(jax|jaxlib|jax-rocm7-plugin|jax-rocm7-pjrt|rocm|rocm-sdk-core|rocm-sdk-libraries)[-a-z0-9_]*==' \
        > /opt/rl-constraints.txt \
    && cat /opt/rl-constraints.txt

# Simulation + RL stack. Versions carried over from the validated workshop
# image. brax declares jax>=0.4.6 with no upper bound, which pip is free to
# "satisfy" by pulling a newer jax/jaxlib from PyPI -- leaving jax-rocm7-plugin
# pinned at an older version it no longer matches, and silently losing the GPU.
# The constraints file frozen above holds the whole ROCm set in lockstep, so a
# conflict becomes a build error rather than a CPU fallback at workshop time.
RUN PIP_CONSTRAINT=/opt/rl-constraints.txt ${RL_VENV}/bin/pip install --no-cache-dir \
        "mujoco==3.10.0" \
        "mujoco-mjx==3.10.0" \
        "brax==0.14.2" \
        "flax" \
        "orbax-checkpoint>=0.11.22" \
        "etils" \
        "ml-collections" \
        "mediapy" \
        "imageio" \
        "imageio-ffmpeg" \
        "absl-py" \
        "lxml" \
        "tqdm" \
        "warp-lang" \
        "ipykernel" \
    && ${RL_VENV}/bin/pip install --no-cache-dir --no-deps \
        "git+https://github.com/google-deepmind/mujoco_playground.git@${PLAYGROUND_COMMIT}" \
    && ${RL_VENV}/bin/python -c \
      "from importlib.metadata import version; \
versions = {p: version(p) for p in ('jax', 'jaxlib', 'jax-rocm7-plugin', 'jax-rocm7-pjrt')}; \
assert len(set(versions.values())) == 1, f'ROCm JAX set out of lockstep: {versions}'; \
print('ROCm JAX set in lockstep:', versions)"

COPY notebooks /ryzers/notebooks/
WORKDIR /ryzers/notebooks

# Unpack the Brax PPO checkpoints the notebook rolls out. Shipped zipped
# because each one is a tree of ~60 small Orbax shard files.
RUN unzip -q assets/PandaPickCube-20260807-131132.zip -d assets \
    && unzip -q assets/PandaPickCube-20260817-150103.zip -d assets \
    && test -f assets/PandaPickCube-20260807-131132/checkpoints/000008192000/ppo_network_config.json \
    && test -f assets/PandaPickCube-20260817-150103/checkpoints/000045875200/ppo_network_config.json

# Fetch the MuJoCo Menagerie Franka assets at build time. registry.load() caches
# them under site-packages, so doing it here keeps the first notebook cell from
# reaching the network.
RUN MUJOCO_GL=disable ${RL_VENV}/bin/python -c \
      "from mujoco_playground import registry; \
env = registry.load('PandaPickCube', config_overrides={'impl': 'jax'}); \
print('PandaPickCube env OK, obs=', env.observation_size, 'act=', env.action_size)"

# The RL kernel. The notebook shells out to scripts/render_trajectory.py via
# sys.executable, so selecting this kernel is also what puts the render
# subprocess on the right interpreter. XLA_PYTHON_CLIENT_PREALLOCATE=false stops
# JAX claiming 75% of VRAM up front, which matters on an APU where the GPU
# shares system memory with everything else in the container.
RUN mkdir -p /usr/local/share/jupyter/kernels/roscon26-rl \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/rl-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "ROSCon26 RL (ROCm)",' \
      '  "language": "python",' \
      '  "env": {' \
      '    "XLA_PYTHON_CLIENT_PREALLOCATE": "false"' \
      '  }' \
      '}' \
      > /usr/local/share/jupyter/kernels/roscon26-rl/kernel.json

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Copy test + demo scripts
COPY test.sh /ryzers/test_roscon26_rllearning.sh
COPY demo.sh /ryzers/demo_roscon26_rllearning.sh
RUN chmod +x /ryzers/test_roscon26_rllearning.sh /ryzers/demo_roscon26_rllearning.sh

# Default command runs the test (for interactive use: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root)
CMD ["/ryzers/test_roscon26_rllearning.sh"]
