#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import sys

def main():
    print("Testing Flower SuperNode...")

    # Test that flwr is installed
    try:
        import flwr
        print(f"✓ Flower installed: {flwr.__version__}")
    except ImportError:
        print("✗ Flower not found")
        sys.exit(1)

    # Test that flower-supernode command exists
    result = subprocess.run(
        ["which", "flower-supernode"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✓ flower-supernode command found: {result.stdout.strip()}")
    else:
        print("✗ flower-supernode command not found")
        sys.exit(1)

    print("\n✓ SuperNode ryzer ready!")
    sys.exit(0)

if __name__ == "__main__":
    main()
