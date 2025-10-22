# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

# Common scientific packages
RUN pip3 install --no-cache-dir --break-system-packages \
    pandas matplotlib seaborn pillow tqdm hiredis loguru omegaconf

# # Jupyter (Notebook + Lab) - these ones known to work with the tutorials
RUN pip3 install --no-cache-dir --break-system-packages \
    notebook==7.0.6 ipykernel==6.27.1 jupyterlab==4.2.5

# Working directory 
WORKDIR /ryzers/notebooks

COPY Tutorials /ryzers/notebooks/Tutorials
COPY Model_files /ryzers/notebooks/Model_files

# Default command
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/ryzers/notebooks", "--allow-root"]

