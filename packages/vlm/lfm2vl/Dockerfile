# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN pip install transformers==4.57.3 accelerate pillow

COPY test.py /ryzers/test.py
COPY test.sh /ryzers/test.sh

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
