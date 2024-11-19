#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# sleep to allow server to start
/usr/local/bin/ollama serve &
sleep 5

ollama pull llama3.2
python3 test_ollama.py