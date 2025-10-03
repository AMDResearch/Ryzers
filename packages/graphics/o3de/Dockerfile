# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install dependencies
RUN apt-get update && \
    apt-get install -y libevdev2 ffmpeg libyaml-cpp0.7 

# Install Vulkan
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.309-jammy.list https://packages.lunarg.com/vulkan/1.4.309/lunarg-vulkan-1.4.309-jammy.list 
RUN apt-get update 
RUN apt-get install -y libevdev2 ffmpeg libyaml-cpp0.7 vulkan-sdk


RUN getent group audio || groupadd audio
RUN usermod -aG audio root

# Necessary for visualisation on Wayland backed systems like Ubuntu
#ENV PYOPENGL_PLATFORM='glx'

RUN mkdir -p -m 0700 /run/user/0
ENV XDG_RUNTIME_DIR=/run/user/0
RUN cd /ryzers

# Disable sound in docker
RUN echo 'pcm.!default = null;' >> /root/.asoundrc

# Install O3DE
RUN wget https://o3debinaries.org/main/Latest/Linux/o3de_2409_2.deb
RUN apt install -y ./o3de_2409_2.deb
RUN rm ./o3de_2409_2.deb

COPY test.sh /ryzers/test_o3de.sh
RUN chmod +x /ryzers/test_o3de.sh

CMD /ryzers/test_o3de.sh