#!/usr/bin/env bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

# Exercise a GPU forward/backward pass without relying on torchvision's removed
# VideoReader API.
python /ryzers/test_lerobot.py