# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

# Setup GEPA
RUN git clone https://github.com/gepa-ai/gepa && \
    cd gepa && \
    pip install -e ".[test]"

# Copy test script
COPY test.sh /ryzers/test_gepa.sh
RUN chmod +x /ryzers/test_gepa.sh

# Set entrypoint to test script
CMD ["/ryzers/test_gepa.sh"]
