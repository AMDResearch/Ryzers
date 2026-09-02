# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic LIBERO sim base for Strix Halo (gfx1151): LIBERO/MuJoCo/robosuite
# closed-loop stack (headless EGL) + sim_libero harness. Policies chain on top and
# are selected at runtime via POLICY_FACTORY; weights/datasets are never baked here.

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

# numpy pinned 1.26.4 (robosuite numba has no py3.12/numpy-2 wheel); constraint keeps the
# base ROCm torch+numpy. LIBERO installs --no-deps; missing runtime imports added explicitly.
# libero ships no top-level __init__.py, so PYTHONPATH resolves it.
ENV PYTHONPATH=/opt/LIBERO:/opt/sim
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.2" easydict thop \
      "future==1.0.0" "cloudpickle==3.1.2" "gym==0.25.2" termcolor \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO /opt/LIBERO && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps -e /opt/LIBERO

# The sim harness (Policy seam + runners + RandomPolicy).
COPY lib/ /opt/sim/

# torch.load compat shim as a startup .pth so it applies to every Python process
# (incl. a policy's own eval script). See _torch_compat.
RUN cp /opt/sim/sim_libero_compat.pth "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"

# First `import libero.libero` prompts to write ~/.libero/config.yaml; feed defaults so it
# bakes in and never prompts at runtime. Also verifies the stack imports (no GPU render at
# build time; the sanity demo exercises rendering at run time).
RUN printf 'N\nN\nN\nN\nN\n' | python -c "import mujoco, robosuite; from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; import sim_libero; from sim_libero.policy import load_policy; print('OK mujoco', mujoco.__version__, '| sim_libero + LIBERO envs import cleanly')"

COPY test.py /ryzers/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
