# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722
ARG SPCONV_TRITON_REF=63361695e891de7aff32841ebb86f2661a41b059
ENV DEBIAN_FRONTEND=noninteractive \
    LIDRA_SKIP_INIT=true \
    CONDA_PREFIX=/opt/venv \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Keep the base image's ROCm PyTorch; upstream's requirements target CUDA 12.1.
RUN pip3 install --no-cache-dir --break-system-packages \
    'hydra-core==1.3.2' 'lightning==2.3.3' 'timm==0.9.16' \
    'huggingface-hub<1.0' 'safetensors>=0.4' 'einops' 'easydict==1.13' \
    'loguru==0.7.2' 'optree==0.14.1' 'plyfile' 'trimesh' \
    'open3d==0.19.0' 'opencv-python-headless<4.12' 'xatlas==0.0.9' \
    'pymeshfix==0.17.0' 'igraph==0.11.8' 'seaborn==0.13.2' \
    'gradio==5.49.0' 'astor==0.8.1' 'imageio[ffmpeg]' 'ipython' \
    'utils3d @ git+https://github.com/EasternJournalist/utils3d@e05f53a59e0c90a3670d44f8761cc32947ec3530' \
    "spconv-triton @ git+https://github.com/L-Reichardt/spconv-triton@${SPCONV_TRITON_REF}"

# Only PyTorch3D's transforms and structures are used by the default example.
# Its optional CUDA rasterizer is not needed for mesh/PLY export.
RUN MAX_JOBS=4 FORCE_CUDA=0 pip3 install --no-cache-dir --break-system-packages --no-build-isolation --no-deps \
    'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d@75ebeeaea0908c5527e7b1e305fbc7681382db47'
RUN BUILD_NO_CUDA=1 pip3 install --no-cache-dir --break-system-packages --no-build-isolation \
    'gsplat @ git+https://github.com/nerfstudio-project/gsplat@2323de5905d5e90e035f792fe65bad0fedd413e7' \
    'MoGe @ git+https://github.com/microsoft/MoGe@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b'

WORKDIR /ryzers
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /ryzers/sam-3d-objects \
 && git -C /ryzers/sam-3d-objects checkout "${SAM3D_REF}" \
 && pip3 install --no-cache-dir --break-system-packages --no-deps -e /ryzers/sam-3d-objects
RUN pip3 install --no-cache-dir --break-system-packages 'iopath==0.1.10'
COPY patch_rocm.py /ryzers/patch_sam3d_rocm.py
COPY conditioning_cache.py /ryzers/sam-3d-objects/sam3d_objects/conditioning_cache.py
COPY backbone_rocm.py /ryzers/sam-3d-objects/sam3d_objects/backbone_rocm.py
COPY linear_rocm.py /ryzers/sam-3d-objects/sam3d_objects/linear_rocm.py
COPY sdpa_rocm.py /ryzers/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/sdpa_rocm.py
COPY attention_rocm.py /ryzers/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/attention_rocm.py
COPY norm_rocm.py /ryzers/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/modules/norm_rocm.py
COPY pointwise_rocm.py /ryzers/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/modules/pointwise_rocm.py
RUN python /ryzers/patch_sam3d_rocm.py \
 && /ryzers/sam-3d-objects/patching/hydra
# gfx1151 otherwise falls back to quadratic-memory math attention.
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True
COPY test.py /ryzers/test_sam3d.py
COPY check_numerics.py /ryzers/check_sam3d_numerics.py
COPY test_conditioning_cache.py /ryzers/test_sam3d_conditioning_cache.py
RUN chmod +x /ryzers/test_sam3d.py
CMD /ryzers/test_sam3d.py --smoke
