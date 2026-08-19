# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic RoboTwin 2.0 simulator base image for AMD Strix Halo (gfx1151).
# Ships the SAPIEN (offscreen Vulkan ray-tracing) + mplib motion-planning closed-loop
# stack plus the sim_robotwin harness (Policy seam + interactive/sanity runners and a
# built-in scripted expert / random policy). No policy/model code lives here: any policy
# chains on top of this image (ryzers build simulation/robotwin <policy>) and drops a
# RoboTwin policy/<name> plugin selected at runtime. curobo (CUDA-only, and its NVIDIA
# Warp dependency has no ROCm backend) is replaced by mplib; the OIDN denoiser is disabled
# via scripts/patches/robotwin_rocm.patch. PyTorch3D is built from upstream (ROCm/HIP port) for
# gfx1151 to provide the point-cloud ops (envs/camera.py FPS/kNN, 3D policies). RoboTwin is
# cloned from upstream at a pinned commit (de-vendored); sim assets are third-party and
# fetched at runtime by scripts/setup_robotwin.sh, never baked here.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      mesa-vulkan-drivers libvulkan1 vulkan-tools \
      libgl1 libglx-mesa0 libegl-mesa0 libgles2 \
      libxrender1 libxext6 libsm6 libxi6 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libglib2.0-0 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# SAPIEN sim deps under a torch/numpy constraint so the base ROCm torch stays put.
# setuptools is pinned <81 because sapien/mplib import the removed pkg_resources.
# numpy is pinned to 1.26.4 last so it protects the base torch stack from transitive bumps.
RUN pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "setuptools<81" \
      "sapien==3.0.0b1" "transforms3d==0.4.2" "trimesh==4.4.3" "gymnasium==0.29.1" \
      toppra scipy "pyglet<2" termcolor moviepy av matplotlib pydantic zarr h5py lxml \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" opencv-python-headless \
      huggingface_hub && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps "mplib==0.2.1" && \
    pip install --no-cache-dir "numpy==1.26.4"

# PyTorch3D (ROCm/HIP) built from upstream at a pinned commit for gfx1151. Enables the
# point-cloud ops (pytorch3d.ops FPS/kNN used by envs/camera.py and 3D policies like DP3)
# on AMD. Upstream added ROCm support in facebookresearch/pytorch3d PR #2039: torch's
# BuildExtension auto-hipifies the .cu sources and hipcub/rocThrust ship in the base ROCm
# image, so no CUB_HOME is needed. Built with --no-build-isolation so setup.py sees the
# base ROCm torch (torch.version.hip). PIP_CONSTRAINT keeps the base torch/numpy pinned.
# The build-time check only asserts the compiled _C extension exists (no GPU at build).
ARG P3D_COMMIT=c8fcd83ff96fa0a5893c0b994f9285d7aa772540
RUN PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir ninja fvcore iopath && \
    git clone https://github.com/facebookresearch/pytorch3d /tmp/pytorch3d && \
    git -C /tmp/pytorch3d checkout ${P3D_COMMIT} && \
    ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx1151 FORCE_CUDA=1 MAX_JOBS=8 \
      PIP_CONSTRAINT=/opt/torch-constraints.txt \
      pip install --no-cache-dir --no-build-isolation /tmp/pytorch3d && \
    rm -rf /tmp/pytorch3d && \
    python -c "import os, glob, pytorch3d; so=glob.glob(os.path.dirname(pytorch3d.__file__)+'/_C*.so'); assert so, 'pytorch3d _C extension not built'; print('pytorch3d', pytorch3d.__version__, 'gfx1151 _C:', os.path.basename(so[0]))"

# De-vendored RoboTwin: clone upstream at the pinned commit into /opt/RoboTwin, then apply
# the compatibility patch there. It bundles the ROCm changes (curobo->mplib, OIDN off,
# optional open3d, expert seed pre-check off) with the model-agnostic eval-harness
# adaptations to script/eval_policy.py + envs/_base_task.py (configurable output dir /
# episode count / skip-obs, and the 4-view eval-frame compositor the harness streams). The
# pin matches the commit the patch was generated against, so it applies to a fresh clone.
ARG RT_COMMIT=bf44be51cf5717a5595ce59447f2cf5263d2aa95
COPY scripts/patches/robotwin_rocm.patch /tmp/robotwin_rocm.patch
RUN git clone https://github.com/RoboTwin-Platform/RoboTwin /opt/RoboTwin && \
    git -C /opt/RoboTwin checkout ${RT_COMMIT} && \
    git -C /opt/RoboTwin apply --whitespace=nowarn /tmp/robotwin_rocm.patch && \
    echo "applied robotwin_rocm.patch @ ${RT_COMMIT:0:8}" && \
    rm -rf /opt/RoboTwin/.git

# The sim harness (Policy seam, interactive/sanity runners, scripted-expert + random).
ENV PYTHONPATH=/opt/sim:/opt/RoboTwin
COPY lib/ /opt/sim/

RUN python -c "import torch, sapien, mplib.planner, pytorch3d; from mplib.sapien_utils import SapienPlanner; import sim_robotwin; from sim_robotwin.policy import load_policy; assert torch.version.hip, torch.__version__; print('OK torch', torch.__version__, 'sapien', sapien.__version__, 'pytorch3d', pytorch3d.__version__, '| sim_robotwin imports cleanly')"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
