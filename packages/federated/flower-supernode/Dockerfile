# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-supernode.sh
COPY run-supernode.sh /ryzers/run-supernode.sh
RUN chmod +x /ryzers/test_flower-supernode.sh /ryzers/run-supernode.sh

RUN mkdir -p /app/certificates

# ClientAppIo port — a paired superexec (clientapp plugin) connects here.
EXPOSE 9094

# Default CMD is the role launcher (reads env vars from config.yaml).
# To run the install-validation smoke test instead:
#   ryzers run /ryzers/test_flower-supernode.sh
CMD /ryzers/run-supernode.sh
