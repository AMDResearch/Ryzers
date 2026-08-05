# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# CaP-X (https://github.com/capgym/cap-x) — Code-as-Policy agents for robot
# manipulation — on AMD Ryzen AI via ROCm.
#
# This image runs the CaP-Gym EVAL path for two simulator families:
#   - Robosuite (1.5.x)
#   - LIBERO-PRO (robosuite 1.4.x)
# Because the two pin conflicting robosuite versions, pick ONE per image with:
#   --build-arg CAPX_SIM=robosuite   (default)   or   CAPX_SIM=libero

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

# robosuite | libero
ARG CAPX_SIM=robosuite
# Pinned to an audited CaP-X commit so submodule gitlinks resolve to the exact
# versions in CaP-X's uv.lock. Override to track upstream.
ARG CAPX_REF=53e9966d7a8e2fa7494676772bccc35280f5c0ed

# MuJoCo headless rendering uses EGL/OpenGL rather than CUDA.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CAPX_SIM=${CAPX_SIM} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

WORKDIR /ryzers

# System deps: build toolchain + EGL/GL stack for MuJoCo, pyrender, open3d.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates cmake build-essential pkg-config \
        python3-dev \
        libegl1 libgl1 libglib2.0-0 libgles2 \
        libosmesa6 libosmesa6-dev \
        libglew-dev libglfw3 libglfw3-dev \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --global credential.helper store

# Pin-constraints that protect the base ROCm torch and match CaP-X's
# uv override-dependencies. torch/torchvision are pinned to the base image's
# installed versions so pip will NEVER pull a different (CUDA) build.
RUN python3 - <<'PY' > /ryzers/constraints.txt
import torch, torchvision
print(f"torch=={torch.__version__.split('+')[0]}")
print(f"torchvision=={torchvision.__version__.split('+')[0]}")
PY

# Match CaP-X's uv override-dependencies and the Python 3.12 Robosuite
# resolution in its lockfile. In particular, Robosuite 1.5.1 still calls the
# pre-3.10 mj_fullM API, so allowing the current MuJoCo release breaks startup.
# FIX: CaP-X also pins PyOpenGL-accelerate==3.1.6, but that version has no cp312
# wheel and fails to compile on the base image's Python 3.12. -accelerate is an
# optional perf add-on (not required), so we omit the pin and only constrain
# PyOpenGL itself (pure-python, installs fine).
RUN printf '%s\n' \
        'numpy==1.26.4' \
        'PyOpenGL==3.1.6' \
        'matplotlib>=3.10,<3.11' \
        'opencv-python-headless<4.13' \
        'jax==0.4.29' \
        'jaxlib==0.4.29' \
        'mujoco==3.5.0' \
        'mink==1.1.0' \
        'networkx==3.6.1' \
        'pyglet==2.1.13' \
    >> /ryzers/constraints.txt && cat /ryzers/constraints.txt

# Clone CaP-X. Clone WITHOUT --recurse-submodules, then init only the submodules
# the eval images actually use. This skips curobo (huge + nested CUDA submodules),
# b1k (BEHAVIOR/Isaac Sim), and verl (RL training) — none needed for the
# Robosuite/LIBERO eval path, and a large time/space saving.
RUN git clone https://github.com/capgym/cap-x /ryzers/cap-x \
    && cd /ryzers/cap-x && git checkout ${CAPX_REF} \
    && git submodule update --init \
        capx/third_party/sam3 \
        capx/third_party/contact_graspnet_pytorch \
        capx/third_party/robosuite \
        capx/third_party/LIBERO-PRO \
        capx/third_party/libero_dependencies/robosuite

WORKDIR /ryzers/cap-x

# Base runtime deps (the PyPI-resolvable subset of [project.dependencies]).
# Excludes torch/torchvision (kept from base) and anything CUDA-only.
# FIX: include setuptools<81 — sam3 imports pkg_resources, removed in
# setuptools>=81.
RUN pip install --no-cache-dir -c /ryzers/constraints.txt \
        "gymnasium>=0.29" "tyro>=0.8.5" "pydantic>=2.7" "rich>=13.7" \
        "scipy>=1.12" "matplotlib>=3.10,<3.11" "imageio[ffmpeg]>=0.6.0" \
        requests tqdm h5py einops decord pycocotools pyyaml omegaconf \
        open3d "robot_descriptions>=1.16.0" "yourdfpy>=0.0.56" "viser>=0.2.0" \
        "trimesh>=4.4.0" msgpack_numpy "ray==2.48.0" "mediapy>=1.2.5" \
        uvicorn fastapi "transformers<5.0" openai \
        "opencv-python-headless<4.13" "pillow>=10" cloudpickle \
        "setuptools<81" \
        "mujoco==3.5.0" "mink==1.1.0" "networkx==3.6.1" "pyglet==2.1.13" \
        "pyliblzfse==0.4.1" freetype-py six

# CaP-X overrides pyrender's stale PyOpenGL==3.1.0 metadata to PyOpenGL 3.1.6.
# pip cannot express uv's override directly, so install the exact locked
# pyrender after all of its runtime dependencies are present.
RUN pip install --no-cache-dir --no-deps "pyrender==0.1.45"

# PyRoKi (JAX-based IK / trajopt — the DEFAULT motion layer; replaces cuRobo for
# the Robosuite/LIBERO bench). CaP-X's uv.lock resolves jax/jaxlib to plain CPU
# 0.4.29 wheels (no CUDA/ROCm jax plugin), so IK runs on CPU jax exactly as
# upstream ships it. pyroki + jaxls pinned to the lockfile commits; --no-deps so
# they can't re-resolve jax off the 0.4.29 pin.
RUN pip install --no-cache-dir -c /ryzers/constraints.txt \
        "jax==0.4.29" "jaxlib==0.4.29" \
        jaxlie jax_dataclasses jaxtyping loguru termcolor \
    && pip install --no-cache-dir -c /ryzers/constraints.txt --no-deps \
        "git+https://github.com/brentyi/jaxls.git@6fe7cf9d56223736b2c0272d080ec7346203c648" \
        "git+https://github.com/chungmin99/pyroki.git@95afccc22658c461ab1042a048ae4e9c24bc2a47"

# SAM3 (pure-PyTorch, runs on ROCm torch). Weights are GATED on HuggingFace —
# request access at https://huggingface.co/facebook/sam3 and pass HF_TOKEN at
# run time (the entrypoint logs in). See README for an ungated SAM2/OWLv2
# alternative that needs no token.
RUN pip install --no-cache-dir -c /ryzers/constraints.txt \
        -e capx/third_party/sam3

# Contact-GraspNet (uynitsuj’s fork of elchun’s PyTorch port; no custom CUDA kernels).
# FIX: --no-build-isolation — its setup.py imports numpy at build time; mirrors
# CaP-X's uv `extra-build-dependencies: contact-graspnet-pytorch = ["numpy"]`.
RUN pip install --no-cache-dir -c /ryzers/constraints.txt \
        --no-build-isolation \
        -e capx/third_party/contact_graspnet_pytorch

# Simulator family (mutually exclusive — chosen by CAPX_SIM).
# All editable submodule installs use --no-build-isolation (numpy at build time).
RUN if [ "$CAPX_SIM" = "libero" ]; then \
        echo ">>> Installing LIBERO-PRO stack"; \
        pip install --no-cache-dir -c /ryzers/constraints.txt \
            "easydict==1.9" "robomimic==0.2.0" "einops==0.4.1" \
            "thop==0.1.1-2209072238" "bddl==1.0.1" "future==0.18.2" \
            "gym==0.25.2" scikit-learn && \
        pip install --no-cache-dir -c /ryzers/constraints.txt --no-build-isolation \
            -e capx/third_party/libero_dependencies/robosuite && \
        pip install --no-cache-dir -c /ryzers/constraints.txt --no-build-isolation \
            -e capx/third_party/LIBERO-PRO && \
        : "FIX: LIBERO pins future==0.18.2 whose standard_library does import imp" \
          "(removed in py3.12), which breaks pyglet->pyrender->GraspNet server." \
          "future>=1.0 drops the imp import and LIBERO still imports fine." && \
        pip install --no-cache-dir "future>=1.0" && \
        : "FIX: LIBERO prompts for a dataset path on first import via input();" \
          "pre-seed ~/.libero/config.yaml at build time (answer N) so runtime" \
          "is non-interactive." && \
        printf 'N\n' | python3 -c "import libero" || true ; \
    else \
        echo ">>> Installing Robosuite stack"; \
        pip install --no-cache-dir -c /ryzers/constraints.txt --no-build-isolation \
            -e capx/third_party/robosuite ; \
    fi

# Install the capx package itself WITHOUT deps (deps handled above; this just
# puts `capx` on the path). --no-build-isolation so it sees the base torch.
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e .

# Ungated perception path (no HF_TOKEN): a FrankaControlApi variant that segments
# via OWLv2 + SAM2 instead of the gated facebook/sam3. The leading \n is required —
# upstream's __init__.py has no trailing newline.
RUN printf '\n%s\n' \
        'register_api("FrankaControlApiSam2", lambda env: FrankaControlApi(env, use_sam3=False))' \
        >> capx/integrations/__init__.py \
    && python3 -c "from capx.integrations import list_apis; assert 'FrankaControlApiSam2' in list_apis()"
COPY franka_robosuite_cube_stack_sam2.yaml \
     /ryzers/cap-x/env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml

# Entrypoint: optional HuggingFace login (SAM3 gated weights) + sign-of-life
# test. Pass HF_TOKEN via config.yaml to enable SAM3 weight download.
COPY test.sh /ryzers/test_capx.sh
RUN chmod +x /ryzers/test_capx.sh

RUN echo -e '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true\n\
fi\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_capx.sh
