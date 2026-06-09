# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install openssl (used by gen-certs.sh) — base image normally has it, but be safe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /ryzers
COPY test.sh /ryzers/test_flower-superlink.sh
COPY gen-certs.sh /ryzers/gen-certs.sh
COPY run-superlink.sh /ryzers/run-superlink.sh
RUN chmod +x /ryzers/test_flower-superlink.sh /ryzers/gen-certs.sh /ryzers/run-superlink.sh

# Persisted SuperLink state and TLS certificates are expected to be
# mounted from the host (see config.yaml).
RUN mkdir -p /app/state /app/certificates

# 9091 = ExecApi (control plane: flwr run / serverapp superexec)
# 9092 = FleetApi (data plane: supernodes connect here)
# 9093 = ServerAppIo (legacy / Exec API TLS)
EXPOSE 9091 9092 9093

CMD /ryzers/test_flower-superlink.sh
