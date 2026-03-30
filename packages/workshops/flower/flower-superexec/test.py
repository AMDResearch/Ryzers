#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import sys

def main():
    print("Testing Flower SuperExec...")

    # Test that flwr is installed
    try:
        import flwr
        print(f"✓ Flower installed: {flwr.__version__}")
    except ImportError:
        print("✗ Flower not found")
        sys.exit(1)

    # Test that torch is available (from base image)
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        sys.exit(1)

    # Test that flower-superexec command exists
    result = subprocess.run(
        ["which", "flower-superexec"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✓ flower-superexec command found: {result.stdout.strip()}")
    else:
        print("✗ flower-superexec command not found")
        sys.exit(1)

    # Test Flower CLI for project setup
    result = subprocess.run(
        ["flwr", "--version"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✓ Flower CLI available: {result.stdout.strip()}")
    else:
        print("⚠ Flower CLI not found (optional for demo setup)")

    print("\n✓ SuperExec ryzer ready!")
    print("\nTo run the Flower federated learning demo:")
    print("  cd packages/workshops/flower/scripts")
    print("  ./setup_env.sh     # run once")
    print("  ./build_containers.sh  # run once")
    print("  ./start_demo.sh    # run demo")

    sys.exit(0)

if __name__ == "__main__":
    main()
