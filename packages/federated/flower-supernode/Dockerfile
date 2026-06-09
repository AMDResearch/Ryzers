# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-supernode.sh
RUN chmod +x /ryzers/test_flower-supernode.sh

RUN mkdir -p /app/certificates

# ClientAppIo port — a paired superexec (clientapp plugin) connects here.
EXPOSE 9094

CMD /ryzers/test_flower-supernode.sh
