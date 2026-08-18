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

# Python 3.14 fix. The base image is now py3.14, but flwr-datasets 0.5.0
# hard-pins datasets<=3.1.0, which drags in dill 0.3.8. That old
# datasets/dill combo runs a fingerprinting path that trips over py3.14's
# changed pickle.Pickler._batch_setitems() signature:
#   TypeError: _batch_setitems() takes 2 positional arguments but 3 were given
# raised the moment a run loads a dataset (e.g. the ServerApp's CIFAR-10
# eval set), aborting the whole federation. datasets 4.x drops that legacy
# code path and works under py3.14. It must be installed in a SEPARATE step:
# resolving it together with flwr-datasets fails (its <=3.1.0 cap), and
# bumping flwr-datasets to 0.6.0 instead is blocked by a rich pin conflict
# with flwr 1.32.0. The residual pip "incompatible" notice about the 3.1.0
# cap is cosmetic — 0.5.0 works with datasets 4.x at runtime.
ARG DATASETS_VERSION=4.8.5
RUN pip3 install --no-cache-dir --break-system-packages \
    "datasets==${DATASETS_VERSION}"

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-base.sh
RUN chmod +x /ryzers/test_flower-base.sh

CMD /ryzers/test_flower-base.sh
