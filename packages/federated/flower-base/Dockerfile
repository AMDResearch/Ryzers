# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Flower CLI binaries (flower-superlink, flower-supernode, flower-superexec)
# all come from this single pip package. Torch + ROCm is provided by the
# default rocm/pytorch base image, so no torch install is needed here.
ARG FLWR_VERSION=1.30.0
RUN pip3 install --no-cache-dir --break-system-packages \
    "flwr[simulation]==${FLWR_VERSION}" \
    "flwr-datasets[vision]>=0.5.0"

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-base.sh
RUN chmod +x /ryzers/test_flower-base.sh

CMD /ryzers/test_flower-base.sh
