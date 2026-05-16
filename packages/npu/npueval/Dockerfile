ARG BASE_IMAGE=iron
FROM ${BASE_IMAGE}

WORKDIR /ryzers

COPY test.sh /ryzers/

# Install NPUEval
RUN git clone https://github.com/amdresearch/npueval
RUN source /ryzers/setup.sh && \
    cd /ryzers/npueval && \
    python3 -m pip install -e .

# Cleanup
ENV SHELL=/bin/bash

# Make sure to source the setup script on docker start
# COPY test.sh /ryzers/
