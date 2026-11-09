# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install the example Flower app (quickstart-pytorch). The pyproject.toml
# shipped here has torch dependencies stripped so we don't overwrite the
# ROCm PyTorch provided by the base image.
WORKDIR /app
COPY app/ /app/
RUN pip3 install --no-cache-dir --break-system-packages /app

# Re-assert the py3.14-compatible datasets. `pip install /app` above
# re-resolves the app's deps and, because flwr-datasets 0.5.0 hard-pins
# datasets<=3.1.0, downgrades datasets from the 4.x pinned in flower-base
# back to 3.1.0 — which reintroduces the py3.14 dill/pickle crash
# (_batch_setitems() TypeError) on the first dataset load. Keep this in
# lockstep with DATASETS_VERSION in flower-base/Dockerfile.
ARG DATASETS_VERSION=4.8.5
RUN pip3 install --no-cache-dir --break-system-packages \
    "datasets==${DATASETS_VERSION}"

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-superexec.sh
COPY run-superexec.sh /ryzers/run-superexec.sh
RUN chmod +x /ryzers/test_flower-superexec.sh /ryzers/run-superexec.sh

RUN mkdir -p /app/certificates

# When running --plugin-type clientapp this connects out to the SuperNode
# at supernode:9094; when running --plugin-type serverapp it connects out
# to the SuperLink at superlink:9091. No inbound ports are exposed.

# Default CMD is the role launcher (reads FLOWER_PLUGIN_TYPE etc. from
# config.yaml). To run the install-validation smoke test instead:
#   ryzers run /ryzers/test_flower-superexec.sh
CMD /ryzers/run-superexec.sh
