# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    libfuse2 xdg-utils libgbm-dev\
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /ryzers/lmstudio

# Download and install LM Studio
# Note: LM Studio provides an AppImage for Linux
RUN wget -O lmstudio.AppImage "https://installers.lmstudio.ai/linux/x64/0.3.29-1/LM-Studio-0.3.29-1-x64.AppImage" && \
    chmod +x lmstudio.AppImage && \
    ./lmstudio.AppImage --appimage-extract && \
    cp -r squashfs-root/* / && \
    ln -s /lm-studio /usr/local/bin/lm-studio

# Set environment variables for LM Studio
ENV LMSTUDIO_PORT=1234
ENV LMSTUDIO_HOST=0.0.0.0

# Expose ports
EXPOSE 1234 5900

# Create test script
COPY test.sh /ryzers/test_lmstudio.sh
RUN chmod +x /ryzers/test_lmstudio.sh

# Default command
CMD ["/ryzers/test_lmstudio.sh"]