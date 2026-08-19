# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic RoboTwin 2.0 simulator base for AMD Strix Halo (gfx1151): SAPIEN (offscreen
# Vulkan RT) + mplib stack + sim_robotwin harness. curobo (CUDA/Warp, no ROCm) -> mplib; OIDN
# off via robotwin_rocm.patch; pytorch3d built from source for gfx1151 (point-cloud ops).
# RoboTwin de-vendored (upstream @ pin); assets fetched at runtime, never baked.

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

# SAPIEN deps under a torch/numpy constraint to keep base ROCm torch. setuptools<81 (sapien/mplib
# use removed pkg_resources); numpy 1.26.4 pinned last to protect the base torch stack.
RUN pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "setuptools<81" \
      "sapien==3.0.0b1" "transforms3d==0.4.2" "trimesh==4.4.3" "gymnasium==0.29.1" \
      toppra scipy "pyglet<2" termcolor moviepy av matplotlib pydantic zarr h5py lxml \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" opencv-python-headless \
      huggingface_hub && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps "mplib==0.2.1" && \
    pip install --no-cache-dir "numpy==1.26.4"

# PyTorch3D (ROCm/HIP) built from source @ pin for gfx1151 (point-cloud ops FPS/kNN). torch's
# BuildExtension auto-hipifies the sources; --no-build-isolation so setup.py sees base ROCm torch.
ARG P3D_COMMIT=c8fcd83ff96fa0a5893c0b994f9285d7aa772540
RUN PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir ninja fvcore iopath && \
    git clone https://github.com/facebookresearch/pytorch3d /tmp/pytorch3d && \
    git -C /tmp/pytorch3d checkout ${P3D_COMMIT} && \
    ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx1151 FORCE_CUDA=1 MAX_JOBS=8 \
      PIP_CONSTRAINT=/opt/torch-constraints.txt \
      pip install --no-cache-dir --no-build-isolation /tmp/pytorch3d && \
    rm -rf /tmp/pytorch3d && \
    python -c "import os, glob, pytorch3d; so=glob.glob(os.path.dirname(pytorch3d.__file__)+'/_C*.so'); assert so, 'pytorch3d _C extension not built'; print('pytorch3d', pytorch3d.__version__, 'gfx1151 _C:', os.path.basename(so[0]))"

# Clone upstream RoboTwin @ pin, then apply ROCm patch (curobo->mplib, OIDN off) + harness adaptations.
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
