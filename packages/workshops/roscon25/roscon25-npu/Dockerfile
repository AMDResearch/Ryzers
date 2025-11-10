# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install jupyterlab matplotlib timm ipywidgets

COPY notebooks /ryzers/notebooks/
WORKDIR /ryzers/notebooks

RUN echo "source /opt/xilinx/xrt/setup.sh" >> ~/.bashrc && \
    echo "source /opt/ros/kilted/setup.sh" >> ~/.bashrc

# EntryPoint for jupyter lab
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'export SHELL=/bin/bash' >> /entrypoint.sh && \
    echo 'source /opt/xilinx/xrt/setup.sh' >> /entrypoint.sh && \
    echo "source /opt/ros/kilted/setup.sh" >> ~/.bashrc && \
    echo 'exec python3 -m jupyter lab --ip=0.0.0.0 --port=8888 --allow-root "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

CMD ["/bin/bash", "/entrypoint.sh"]
