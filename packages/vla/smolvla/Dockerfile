# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git vim wget

RUN git clone https://github.com/huggingface/lerobot.git && \
	cd lerobot && \
	git checkout 5c87365cc160617c45dc5d1bbb3788de010271a7 && \
	sed -i -E '/assert not torch\.isinf\((mean|std)\)\.any\(\)/ s/^([[:space:]]*)/\1#/' lerobot/common/policies/normalize.py

RUN cd lerobot && \
	pip install ".[smolvla]"

COPY test.py /ryzers/test.py
COPY test.sh /ryzers/test.sh
RUN chmod +x /ryzers/test.sh

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
