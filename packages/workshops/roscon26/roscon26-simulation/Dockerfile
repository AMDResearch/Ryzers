# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

# gslab needs torch >= 2.10 and genesis-world 1.4.0. The base image ships an
# older torch built against system ROCm 7.14, so the workshop gets its own venv
# with a torch from the host-matched ROCm 7.13 wheel index. Pinning the index
# per architecture means a missing wheel fails loudly instead of silently
# resolving to a PyPI CPU/CUDA build.
ARG GPU_TARGET=gfx1151
ARG TORCH_VERSION=2.11.0+rocm7.13.0
ARG GSLAB_WHEEL=gslab-0.1.2-py3-none-any.whl

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    GSLAB_VENV=/opt/gslab-venv

WORKDIR /ryzers

# Genesis pulls in system OpenGL at import: pygel3d dlopens libGL.so.1 and
# pyglet's default backend dlopens the X11/libXrender stack. PYGLET_HEADLESS
# (set below) routes pyglet to EGL instead, so only libGL + libEGL are needed
# rather than all of X11.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libegl1 libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY workshop /ryzers/notebooks/

# ---------------------------------------------------------------------------
# Isolated workshop environment.
#
# Two ROCm layers are in play. torch finds HIP through the pip `rocm_sdk`
# packages its wheel pulls from the per-architecture index. Genesis compiles
# physics kernels at runtime with `quadrants`, which instead wants a classic
# /opt/rocm layout the pip SDK does not ship. Keeping this in its own venv
# leaves the base image's torch untouched, the same way roscon26-rllearning
# isolates its ROCm JAX.
# ---------------------------------------------------------------------------
RUN python3 -m venv ${GSLAB_VENV} \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# torch FIRST, from the pinned architecture index only, so a missing
# +rocm7.13.0 wheel is an error rather than a silent PyPI fallback.
#
# Then freeze the ROCm torch stack into a constraints file. gslab only asks for
# torch>=2.10, and the next layer resolves across PyPI as well, so without this
# pip is free to "satisfy" that bound with a newer *CUDA* torch from PyPI --
# which it does, silently swapping the ROCm build and dragging in ~3 GB of
# nvidia-* wheels. Constraining it turns that into a resolution error instead.
RUN ${GSLAB_VENV}/bin/pip install --no-cache-dir \
        --index-url https://repo.amd.com/rocm/whl/${GPU_TARGET}/ \
        "torch==${TORCH_VERSION}" \
    && ${GSLAB_VENV}/bin/python -c \
      "import torch; assert torch.version.hip, 'gslab venv torch is not a ROCm/HIP build'; \
print('gslab torch', torch.__version__, 'hip', torch.version.hip)" \
    && ${GSLAB_VENV}/bin/pip freeze \
        | grep -iE '^(torch|triton|rocm|rocm-sdk-core|rocm-sdk-libraries)[-a-z0-9_]*==' \
        > /opt/gslab-constraints.txt \
    && cat /opt/gslab-constraints.txt

# Then gslab from the bundled wheel plus the notebook stack, resolving across
# PyPI and the AMD index but held to the ROCm torch by the constraints above.
# The wheel is deleted in the SAME layer so its 20 MB are not stranded.
RUN PIP_CONSTRAINT=/opt/gslab-constraints.txt ${GSLAB_VENV}/bin/pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://repo.amd.com/rocm/whl/${GPU_TARGET}/ \
        "gslab @ file:///ryzers/notebooks/wheels/${GSLAB_WHEEL}" \
        ipykernel ipywidgets plotly tqdm \
    && rm -f /ryzers/notebooks/wheels/${GSLAB_WHEEL} \
    && ${GSLAB_VENV}/bin/python -c \
      "import torch; assert torch.version.hip, 'gslab deps replaced the ROCm torch'; \
print('torch still ROCm after gslab:', torch.__version__)"

# plotly 5's labextension targets JupyterLab 3 and the 4.x server rejects it, so
# FigureWidget never renders. plotly 6+ uses anywidget instead, whose frontend
# must be visible to the system JupyterLab rather than only the kernel venv.
# --no-deps bypasses PyGEL3D's plotly<6 pin; PyGEL3D only imports plotly in
# pygel3d/jupyter_display.py, which neither Genesis nor gslab references.
RUN PIP_CONSTRAINT=/opt/gslab-constraints.txt \
      ${GSLAB_VENV}/bin/pip install --no-cache-dir anywidget narwhals \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir --no-deps --upgrade "plotly==7.1.0" \
    && python3 -m pip install --no-cache-dir --break-system-packages anywidget

# Assemble the <venv>/rocm tree quadrants needs: unversioned .so aliases for
# bare-name dlopen, plus an ld.lld of LLVM >= 19 (Ubuntu 24.04's stock lld-18
# rejects Genesis' v6 AMDGPU code objects). The script prefers
# /opt/rocm/llvm/bin/ld.lld from the base image and self-verifies, so this layer
# fails loudly if that linker is ever too old.
RUN VENV=${GSLAB_VENV} bash /ryzers/notebooks/link_rocm_tree.sh

# Route pyglet (imported transitively by Genesis) to its headless EGL backend so
# `import genesis` and offscreen camera rendering work with no X server.
ENV PYGLET_HEADLESS=1

# Register the workshop kernel. quadrants runs ${ROCM_PATH}/llvm/bin/ld.lld and
# dlopens libamdhip64.so by bare name, so the kernel bakes in ROCM_PATH and
# LD_LIBRARY_PATH pointing at the tree built above. Only those two venv-rocm
# dirs belong there -- torch finds HIP via the rocm_sdk package, not this path.
# QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES raises the compiled-kernel cache limit so
# kernels persist between runs instead of recompiling each session.
#
# LIB_TARGET is read back from the tree rather than assumed, because the
# directory is named after the rocm-sdk-libraries package that torch actually
# pulled in, which follows GPU_TARGET.
RUN LIB_DIR="$(basename "$(ls -d ${GSLAB_VENV}/rocm/lib-* | head -1)")" \
    && mkdir -p /usr/local/share/jupyter/kernels/gslab-roscon \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/gslab-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "gslab ROSCon (ROCm 7.13)",' \
      '  "language": "python",' \
      '  "env": {' \
      '    "ROCM_PATH": "/opt/gslab-venv/rocm",' \
      "    \"LD_LIBRARY_PATH\": \"/opt/gslab-venv/rocm/lib:/opt/gslab-venv/rocm/${LIB_DIR}\"," \
      '    "QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES": "2147483647",' \
      '    "PYGLET_HEADLESS": "1",' \
      '    "PYTHONDONTWRITEBYTECODE": "1"' \
      '  }' \
      '}' \
      > /usr/local/share/jupyter/kernels/gslab-roscon/kernel.json \
    && cat /usr/local/share/jupyter/kernels/gslab-roscon/kernel.json

# Import check. Build hosts have no GPU, so this is import-and-link only; the
# notebook's own environment check is the real runtime gate.
RUN ROCM_PATH=${GSLAB_VENV}/rocm \
    LD_LIBRARY_PATH=${GSLAB_VENV}/rocm/lib:$(ls -d ${GSLAB_VENV}/rocm/lib-* | head -1) \
    ${GSLAB_VENV}/bin/python -c \
      "import torch, gslab, genesis; assert torch.version.hip; \
print('gslab workshop env OK:', torch.__version__, '| genesis', genesis.__version__)"

# Copy test script
COPY test.sh /ryzers/test_roscon26_simulation.sh
RUN chmod +x /ryzers/test_roscon26_simulation.sh

WORKDIR /ryzers/notebooks

# Default command runs the test (for interactive use: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root)
CMD ["/ryzers/test_roscon26_simulation.sh"]
