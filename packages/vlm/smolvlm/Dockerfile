# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

WORKDIR /ryzers

# Copy test script
COPY test.sh /ryzers/test_smolvlm.sh
COPY demo.sh /ryzers/demo_smolvlm.sh
RUN chmod +x /ryzers/test_smolvlm.sh /ryzers/demo_smolvlm.sh

RUN cd /ryzers

CMD /ryzers/test_smolvlm.sh
