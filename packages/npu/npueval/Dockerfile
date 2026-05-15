ARG BASE_IMAGE=iron
FROM ${BASE_IMAGE}

WORKDIR /ryzers

COPY test.sh /ryzers/

# Install NPUEval
RUN git clone https://github.com/amdresearch/npueval
# Source /ryzers/setup.sh only if it's present. The iron base provides it but
# we want this Dockerfile to also work when built directly on ryzer_env.
RUN if [ -f /ryzers/setup.sh ]; then \
        bash -lc "source /ryzers/setup.sh && cd /ryzers/npueval && python3 -m pip install -e ."; \
    else \
        cd /ryzers/npueval && python3 -m pip install -e .; \
    fi

# Cleanup
ENV SHELL=/bin/bash

# Make sure to source the setup script on docker start
# COPY test.sh /ryzers/
