#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
PyDrake Demo: Simple pendulum simulation with Meshcat visualization.

Run with: python3 /ryzers/demo_pydrake.py
Then open http://localhost:7000 in a browser to view the simulation.
"""

import numpy as np
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import LogVectorOutput
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import RevoluteJoint, SpatialInertia, UnitInertia
from pydrake.geometry import Sphere
from pydrake.math import RigidTransform
from pydrake.visualization import AddDefaultVisualization


def create_pendulum():
    """Create a simple pendulum MultibodyPlant."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Pendulum parameters
    length = 1.0
    mass = 1.0
    radius = 0.05

    # Add pendulum body
    M_link = SpatialInertia(
        mass=mass,
        p_PScm_E=[0, 0, -length / 2],
        G_SP_E=UnitInertia.SolidCylinder(radius, length, [0, 0, 1])
    )
    pendulum = plant.AddRigidBody("pendulum", M_link)

    # Add revolute joint at the top
    joint = plant.AddJoint(RevoluteJoint(
        "pivot",
        plant.world_frame(),
        pendulum.body_frame(),
        [1, 0, 0],  # Rotate around X axis
    ))

    # Add visual geometry
    plant.RegisterVisualGeometry(
        pendulum,
        RigidTransform([0, 0, -length / 2]),
        Sphere(radius * 2),
        "pendulum_visual",
        [0.8, 0.2, 0.2, 1.0]  # Red color
    )

    plant.Finalize()
    return builder, plant, scene_graph


def run_simulation():
    """Run the pendulum simulation with visualization."""
    print("Creating pendulum simulation...")

    builder, plant, scene_graph = create_pendulum()

    # Add visualization
    AddDefaultVisualization(builder)

    # Add state logger
    logger = LogVectorOutput(plant.get_state_output_port(), builder)

    # Build the diagram
    diagram = builder.Build()

    # Set up simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Set initial conditions: pendulum at 45 degrees
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, [np.pi / 4])  # 45 degrees
    plant.SetVelocities(plant_context, [0.0])

    # Run simulation
    print("Running simulation for 10 seconds...")
    print("Open http://localhost:7000 in a browser to view.")

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(10.0)

    print("Simulation complete!")

    # Print final state
    log = logger.FindLog(context)
    times = log.sample_times()
    states = log.data()
    print(f"Recorded {len(times)} timesteps")
    print(f"Final angle: {states[0, -1]:.3f} rad")
    print(f"Final velocity: {states[1, -1]:.3f} rad/s")


if __name__ == "__main__":
    run_simulation()
