#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import sys

def run_command(cmd):
    """Run command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def main():
    print("Testing Flower Ryzer Installation...\n")

    # Test Flower CLI installed
    success, output = run_command("flwr --version")
    if success:
        print(f"✓ Flower CLI installed: {output}")
    else:
        print(f"✗ Flower CLI not found")
        sys.exit(1)

    print("\n✓ All tests passed!")
    print("\nTo run the Flower federated learning demo:")
    print("  cd packages/workshops/flower/scripts")
    print("  ./setup_env.sh     # run once")
    print("  ./build_containers.sh  # run once")
    print("  ./start_demo.sh    # run demo")

    sys.exit(0)

if __name__ == "__main__":
    main()
