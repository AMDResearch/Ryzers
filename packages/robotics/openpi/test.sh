#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

cd /ryzers/openpi

## Download model checkpoint
#uv run scripts/serve_policy.py --download-only policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Apply PyTorch support patch 
#cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# Convert JAX models to PyTorch
#uv run examples/convert_jax_model_to_pytorch.py \
#    --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
#    --config_name pi0_droid \
#    --output_path /root/.cache/openpi/openpi-assets/checkpoints/pi0_droid

uv run test.py