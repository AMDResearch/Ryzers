ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Suppress prompts in scripts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	libboost-filesystem1.74.0 \
	vulkan-tools \
	mesa-vulkan-drivers \
	git-lfs

RUN apt install -y vulkan-tools mesa-vulkan-drivers
RUN add-apt-repository ppa:deadsnakes/ppa && \
	apt update && \
	apt install -y python3.10 python3.10-venv python3.10-dev libboost-filesystem1.74.0

# Clone Ryzen AI repo and lock onto 1.6 release
RUN git clone https://github.com/amd/RyzenAI-SW
RUN cd RyzenAI-SW && \
    git checkout 45176f1 && \
    git lfs install && \
    git lfs pull

# Build CVML sample apps
RUN cd /ryzers/RyzenAI-SW/Ryzen-AI-CVML-Library/samples && \
	mkdir build-samples && \
	cmake -S $PWD -B $PWD/build-samples && \
	cmake --build $PWD/build-samples --config Release 

# Download video for testing
RUN cd /ryzers/RyzenAI-SW/Ryzen-AI-CVML-Library/samples && \
	curl -o video_call.mp4 https://videos.pexels.com/video-files/8136072/8136072-hd_1920_1080_25fps.mp4

# Copy demo files
COPY cvml_ros /ryzers/cvml_ros
COPY demo_ros.sh /ryzers/

# Cleanup
ENV SHELL=/bin/bash

COPY test.sh /ryzers/
CMD ["/ryzers/test.sh"]
