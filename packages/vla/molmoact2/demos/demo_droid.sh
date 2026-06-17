#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 2: DROID open-loop replay. Pick a random episode from the real
# allenai/MolmoAct2-DROID-Dataset, run MolmoAct2-DROID open-loop, and write a
# scene video + GT-vs-prediction joint-angle plot under /outputs.
#
#   ryzers run /ryzers/demo_droid.sh                  # random episode
#   EPISODE=42 ryzers run /ryzers/demo_droid.sh       # fixed episode
#   SEED=7     ryzers run /ryzers/demo_droid.sh       # reproducible random pick
set -euo pipefail
exec python /ryzers/droid_openloop_demo.py
