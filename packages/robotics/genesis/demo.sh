#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# needed new numpy for the drone demos
# pip install --upgrade numpy || echo "compatibility error is okay, can be ignored or the demo"...
# pip install pynput
# clear

sed -i 's/gs\.init()/gs.init(backend=gs.vulkan)/' /ryzers/Genesis/examples/tutorials/sph_liquid.py
sed -i 's/horizon = 1000/horizon = 10000/' /ryzers/Genesis/examples/tutorials/sph_liquid.py
cd Genesis/examples/tutorials
python sph_liquid.py
exit 0