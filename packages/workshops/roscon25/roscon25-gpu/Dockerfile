# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN pip install jupyterlab "numpy<2" matplotlib catkin_pkg empy ipywidgets "datasets<3"

COPY notebooks /ryzers/notebooks/

SHELL ["/bin/bash", "-lc"]

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --merge-install \
    --base-paths /ryzers/notebooks/vlm_ros \
    --build-base /ryzers/notebooks/vlm_ros/build \
    --install-base /ryzers/notebooks/vlm_ros/install

RUN python3 -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
LeRobotDataset('lerobot/svla_so101_pickplace')"

#  Vulkan
RUN apt-get update \
&& wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc \
&& wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
&& apt update \
&& apt install -y vulkan-sdk

# Make ROS environment variables available to all users and shells
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc
RUN echo "source /ryzers/notebooks/vlm_ros/install/setup.bash" >> /etc/bash.bashrc

# EntryPoint for Jupyter
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'export SHELL=/bin/bash' >> /entrypoint.sh && \
    echo 'source /opt/ros/kilted/setup.bash' >> /entrypoint.sh && \
    echo "source /ryzers/notebooks/vlm_ros/install/setup.bash" >> /entrypoint.sh && \
    echo 'exec python3 -m jupyterlab --ip=0.0.0.0 --port=8888 --allow-root "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /ryzers/notebooks

ENV TQDM_DISABLE=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_NO_PROGRESS_BAR=1

CMD ["/bin/bash", "/entrypoint.sh"]
