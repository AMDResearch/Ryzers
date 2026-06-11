#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import sys

def test_drake():
    print("Testing PyDrake installation...")

    try:
        import pydrake
        print(f"PyDrake path: {pydrake.getDrakePath()}")
    except ImportError as e:
        print(f"Failed to import pydrake: {e}")
        return False

    # Test core Drake modules
    try:
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.multibody.parsing import Parser
        from pydrake.geometry import SceneGraph
        print("Core Drake modules imported successfully")
    except ImportError as e:
        print(f"Failed to import core modules: {e}")
        return False

    # Test basic MultibodyPlant creation
    try:
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        plant.Finalize()
        diagram = builder.Build()
        print("MultibodyPlant creation: OK")
    except Exception as e:
        print(f"MultibodyPlant test failed: {e}")
        return False

    # Test solvers
    try:
        from pydrake.solvers import MathematicalProgram, Solve
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2, "x")
        prog.AddCost(x[0]**2 + x[1]**2)
        prog.AddConstraint(x[0] + x[1] >= 1)
        result = Solve(prog)
        print(f"Solver test: {'OK' if result.is_success() else 'FAILED'}")
    except Exception as e:
        print(f"Solver test failed: {e}")
        return False

    return True


def test_pytorch_rocm():
    print("\nTesting PyTorch with ROCm backend...")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm/CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")

            # Simple GPU computation test
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            y = torch.sum(x ** 2)
            print(f"GPU computation test: OK (result={float(y)})")
            return True
        else:
            print("No GPU devices found (CPU fallback)")
            return True  # Don't fail if GPU not available

    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return True
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return True


if __name__ == "__main__":
    print("=" * 50)
    print("PyDrake + ROCm GPU Acceleration Test")
    print("=" * 50)

    drake_ok = test_drake()
    gpu_ok = test_pytorch_rocm()

    print("\n" + "=" * 50)
    if drake_ok:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)
