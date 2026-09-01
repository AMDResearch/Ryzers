# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CONTACT_REPO=https://github.com/Vict0rHu/CONTACT.git
ARG CONTACT_BRANCH=feat/amd-rocm
ARG CONTACT_REF=4b3496a41e1e0a839ab44b7c934374367128a736
ARG DIFFUSION_POLICY_REF=5ba07ac6661db573af695b419a7947ecb704690f

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    USE_SIM=0 \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
 && if ! getent group render >/dev/null; then groupadd --system render; fi \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "${CONTACT_BRANCH}" --single-branch \
        "${CONTACT_REPO}" /workspace/CONTACT \
 && cd /workspace/CONTACT \
 && git checkout --detach "${CONTACT_REF}"

RUN git clone https://github.com/real-stanford/diffusion_policy.git \
        /workspace/diffusion_policy \
 && cd /workspace/diffusion_policy \
 && git checkout --detach "${DIFFUSION_POLICY_REF}"

RUN python -m pip install --no-cache-dir \
        -r /workspace/CONTACT/requirements-amd.txt \
 && python -m pip install --no-cache-dir --no-deps \
        --config-settings editable_mode=compat \
        -e /workspace/diffusion_policy \
        -e /workspace/CONTACT

WORKDIR /ryzers
COPY test.sh /ryzers/test_contact.sh
RUN chmod +x /ryzers/test_contact.sh

CMD ["/ryzers/test_contact.sh"]
