# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git \
	vim \
	wget \
	libegl1-mesa-dev \
	libgles2-mesa-dev \
	bazel-bootstrap

RUN git clone https://github.com/tonyzhaozh/act

# labmaze (a dm_control dependency) has no cp313 wheel and builds from source via
# Bazel. Its bundled abseil references @platforms//cpu:wasm32, which the platforms
# repo it pins is too old to declare. Override platforms with a modern checkout so
# the wasm32 target exists, then install labmaze before dm_control reuses it.
RUN git clone https://github.com/google-deepmind/labmaze /tmp/labmaze && \
	git clone --depth 1 https://github.com/bazelbuild/platforms /tmp/platforms && \
	echo 'build --override_repository=platforms=/tmp/platforms' >> /tmp/labmaze/.bazelrc && \
	pip install /tmp/labmaze

RUN pip install matplotlib && \
	pip install einops && \
	pip install packaging && \
	pip install h5py && \
 	pip install ipython && \
	pip install dm_control && \
	pip install pyquaternion


COPY test.sh /ryzers/
RUN chmod +x /ryzers/

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
