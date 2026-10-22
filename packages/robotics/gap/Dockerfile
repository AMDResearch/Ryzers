# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ARG GAP_REF=c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea
ARG OPEN_ROBOT_SKILLS_REF=752dfbd96968d4d08379f8111765efbf6457800b
ARG SAM3_REF=b26a5f330e05d321afb39d01d3d4881f258f65ff

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    JAX_PLATFORMS=cpu \
    GAP_DINO_DEVICE=cuda \
    GAP_SAM3_DEVICE=cuda \
    GAP_TOOL_BUNDLE_USE_UV=0 \
    LIBERO_CONFIG_PATH=/ryzers/libero-config \
    GAP_SKILLS_PATH=/ryzers/open-robot-skills

WORKDIR /ryzers

# Build tools plus the EGL/OpenGL and video stack used by MuJoCo/LIBERO.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential cmake pkg-config python3-dev \
        ffmpeg libegl1 libegl1-mesa-dev libgl1 libgl1-mesa-dri libglib2.0-0 \
        libgles2 libgles2-mesa-dev libglfw3 libglfw3-dev libglew-dev \
        libosmesa6 libosmesa6-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "uv==0.12.21" "setuptools<81"

# Only initialize the simulator submodules used by this image.
RUN git clone https://github.com/graph-robots/graph-as-policy.git \
        /ryzers/graph-as-policy \
    && cd /ryzers/graph-as-policy \
    && git checkout "${GAP_REF}" \
    && git submodule update --init --depth 1 \
        third_party/LIBERO-PRO \
        third_party/Variational-Automation-Benchmark \
        third_party/robosuite \
    && git clone https://github.com/graph-robots/open-robot-skills.git \
        /ryzers/open-robot-skills \
    && cd /ryzers/open-robot-skills \
    && git checkout "${OPEN_ROBOT_SKILLS_REF}"

RUN mkdir -p /ryzers/libero-config \
        /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/datasets \
    && printf '%s\n' \
        'assets: /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/libero/assets' \
        'bddl_files: /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/libero/bddl_files' \
        'benchmark_root: /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/libero' \
        'datasets: /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/datasets' \
        'init_states: /ryzers/graph-as-policy/third_party/LIBERO-PRO/libero/libero/init_files' \
        > /ryzers/libero-config/config.yaml

# Keep source changes explicit and fail when a pinned source anchor changes.
COPY apply_overrides.py /ryzers/apply_overrides.py
RUN python3 /ryzers/apply_overrides.py /ryzers/graph-as-policy

# Preserve the upstream workflow and add a PyRoKI motion variant beside it.
COPY prepare_grocery.py /ryzers/prepare_grocery.py
RUN python3 /ryzers/prepare_grocery.py

# Keep the base image's torch stack and tested numerical versions.
RUN python3 - <<'PY' > /ryzers/constraints.txt
import torch
import torchvision

print(f"torch=={torch.__version__.split('+')[0]}")
print(f"torchvision=={torchvision.__version__.split('+')[0]}")
print("numpy==1.26.4")
print("opencv-python-headless<4.13")
print("jax==0.4.29")
print("jaxlib==0.4.29")
print("setuptools<81")
print("trimesh==5.1.1")
print("yourdfpy==0.0.60")
PY

WORKDIR /ryzers/graph-as-policy

# Install the engine, simulator, and skills registry.
RUN python3 -m pip install --no-cache-dir -c /ryzers/constraints.txt \
        -e /ryzers/graph-as-policy/gap-core \
        -e /ryzers/graph-as-policy/third_party/Variational-Automation-Benchmark \
        -e /ryzers/graph-as-policy/third_party/robosuite \
        -e /ryzers/graph-as-policy \
        -e /ryzers/open-robot-skills

# PyRoKI and jaxls revisions tested with JAX 0.4.29.
RUN python3 -m pip install --no-cache-dir -c /ryzers/constraints.txt \
        "jax==0.4.29" "jaxlib==0.4.29" "jaxlie==1.5.0" \
        "jax-dataclasses==1.6.3" "jaxtyping==0.3.11" "loguru==0.7.3" \
        "termcolor==3.3.0" "pyliblzfse==0.4.1" \
    && python3 -m pip install --no-cache-dir --no-deps \
        "jaxls @ git+https://github.com/brentyi/jaxls.git@6fe7cf9d56223736b2c0272d080ec7346203c648" \
        "pyroki @ git+https://github.com/chungmin99/pyroki.git@95afccc22658c461ab1042a048ae4e9c24bc2a47"

RUN JAX_PLATFORMS=cpu python3 -c \
    "from gap.connector.ik import load_robot; load_robot('panda_description')"

RUN git clone https://github.com/facebookresearch/sam3.git /ryzers/sam3 \
    && cd /ryzers/sam3 \
    && git checkout "${SAM3_REF}" \
    && python3 - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
source = path.read_text()
old = '"numpy==1.26"'
if source.count(old) != 1:
    raise RuntimeError(f"expected one SAM3 NumPy pin, found {source.count(old)}")
path.write_text(source.replace(old, '"numpy>=1.26,<2"', 1))
PY

# Supported tool bundles share the tested torch environment.
RUN python3 -m pip install --no-cache-dir -c /ryzers/constraints.txt \
        "numpy==1.26.4" "open3d==0.20.0" "scikit-learn==1.9.1" \
        "scikit-image==0.26.0" "transformers==4.57.6" \
        "timm==1.0.30" "ftfy==6.1.1" "regex==2026.9.29" \
        "iopath==0.1.10" "huggingface_hub==0.36.2" "tqdm==4.68.4" \
        "decord2==3.4.0" "pycocotools==2.0.11" "einops==0.8.2" \
        "psutil==7.2.2" "google-genai==1.59.0" \
    && python3 -m pip install --no-cache-dir --no-deps \
        -e /ryzers/sam3

COPY test.sh /ryzers/test_gap.sh
RUN chmod +x /ryzers/test_gap.sh

EXPOSE 9432

CMD /ryzers/test_gap.sh
