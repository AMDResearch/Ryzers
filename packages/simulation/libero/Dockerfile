# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic LIBERO simulator base image for AMD Strix Halo (gfx1151).
# Ships the LIBERO/MuJoCo/robosuite closed-loop stack (headless EGL) plus the
# sim_libero harness (Policy seam + closed-loop/interactive/sanity runners and a
# built-in RandomPolicy). No policy/model code lives here: any policy chains on top
# of this image (ryzers build simulation/libero <policy>) and is selected at runtime
# via POLICY_FACTORY. LIBERO assets are cloned into the image; datasets/weights are
# supplied by the policy layer or mounted at runtime, never baked here.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential \
      libegl-dev libgles-dev libosmesa6-dev libglew-dev libglu1-mesa libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# LIBERO closed-loop sim. numpy is pinned to 1.26.4 (robosuite's numba has no py3.12
# wheel for numpy 2.x); the constraint then protects the base ROCm torch + numpy so
# they are never replaced by transitive deps. LIBERO installs with --no-deps for the
# same reason. `future`, `cloudpickle`, `gym==0.25.2` and `termcolor` are LIBERO/robosuite
# runtime imports not declared as install deps (previously pulled in transitively by the
# fastwam install), so they are added explicitly. The `libero` package ships no top-level
# __init__.py, so it resolves via PYTHONPATH.
ENV PYTHONPATH=/opt/LIBERO:/opt/sim
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.2" easydict thop \
      "future==1.0.0" "cloudpickle==3.1.2" "gym==0.25.2" termcolor \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO /opt/LIBERO && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps -e /opt/LIBERO

# The sim harness (Policy seam, closed-loop/interactive/sanity runners, RandomPolicy).
COPY lib/ /opt/sim/

# Install the torch.load compat shim as a startup .pth in site-packages so it applies to
# every Python process in the image (including any policy's own eval script, which need
# not import sim_libero) without shadowing the distro's sitecustomize. See _torch_compat.
RUN cp /opt/sim/sim_libero_compat.pth "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"

# First `import libero.libero` writes ~/.libero/config.yaml interactively; feed defaults
# so it is baked into the image and never prompts at runtime. Also verify the deeper
# env import (robosuite controllers) and the sim_libero harness resolve. No GPU render
# here (build has no device); the sanity demo exercises rendering at run time.
RUN printf 'N\nN\nN\nN\nN\n' | python -c "import mujoco, robosuite; from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; import sim_libero; from sim_libero.policy import load_policy; print('OK mujoco', mujoco.__version__, '| sim_libero + LIBERO envs import cleanly')"

COPY test.py /ryzers/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
