#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
PyDrake Hello World: Falling objects simulation with MP4 output.
Demonstrates MultibodyPlant, SceneGraph, and video rendering.
"""

import numpy as np
from pydrake.geometry import Box, Sphere, MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import VideoWriter


def add_ground(plant):
    """Add a ground plane."""
    ground_friction = CoulombFriction(static_friction=1.0, dynamic_friction=0.8)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0, 0, -0.05]),
        Box(10, 10, 0.1),
        "ground_collision",
        ground_friction,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0, 0, -0.05]),
        Box(10, 10, 0.1),
        "ground_visual",
        np.array([0.3, 0.3, 0.3, 1.0]),
    )


def add_falling_box(plant, name, size, mass, color, initial_pos):
    """Add a falling box to the simulation."""
    inertia = SpatialInertia(
        mass=mass,
        p_PScm_E=[0, 0, 0],
        G_SP_E=UnitInertia.SolidBox(size[0], size[1], size[2]),
    )
    body = plant.AddRigidBody(name, inertia)

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.7)
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        Box(*size),
        f"{name}_collision",
        friction,
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Box(*size),
        f"{name}_visual",
        np.array(color),
    )
    return body, initial_pos


def add_falling_sphere(plant, name, radius, mass, color, initial_pos):
    """Add a falling sphere to the simulation."""
    inertia = SpatialInertia(
        mass=mass,
        p_PScm_E=[0, 0, 0],
        G_SP_E=UnitInertia.SolidSphere(radius),
    )
    body = plant.AddRigidBody(name, inertia)

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.7)
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        Sphere(radius),
        f"{name}_collision",
        friction,
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Sphere(radius),
        f"{name}_visual",
        np.array(color),
    )
    return body, initial_pos


def create_simulation():
    """Create the falling objects simulation."""
    print("Creating Drake simulation...")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.002)

    # Add ground
    add_ground(plant)

    # Add falling objects with different colors and positions
    objects = []

    # Red box
    objects.append(add_falling_box(
        plant, "red_box",
        size=[0.3, 0.3, 0.3], mass=1.0,
        color=[0.9, 0.2, 0.2, 1.0],
        initial_pos=[0.0, 0.0, 2.0]
    ))

    # Blue sphere
    objects.append(add_falling_sphere(
        plant, "blue_sphere",
        radius=0.2, mass=0.8,
        color=[0.2, 0.4, 0.9, 1.0],
        initial_pos=[0.5, 0.3, 2.5]
    ))

    # Green box
    objects.append(add_falling_box(
        plant, "green_box",
        size=[0.25, 0.25, 0.25], mass=1.2,
        color=[0.2, 0.8, 0.3, 1.0],
        initial_pos=[-0.4, 0.2, 3.0]
    ))

    # Yellow sphere
    objects.append(add_falling_sphere(
        plant, "yellow_sphere",
        radius=0.15, mass=0.5,
        color=[0.9, 0.9, 0.2, 1.0],
        initial_pos=[0.2, -0.3, 3.5]
    ))

    # Purple box (tilted)
    objects.append(add_falling_box(
        plant, "purple_box",
        size=[0.35, 0.2, 0.2], mass=0.9,
        color=[0.7, 0.2, 0.8, 1.0],
        initial_pos=[-0.2, -0.4, 2.8]
    ))

    plant.Finalize()

    # Add VTK renderer with proper settings
    vtk_params = RenderEngineVtkParams()
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(vtk_params))

    # Camera positioned to view the falling objects
    # Drake camera looks along +Z in camera frame
    # Place camera behind and above, looking toward origin
    camera_pos = np.array([0.0, -5.0, 2.0])
    target_pos = np.array([0.0, 0.0, 1.5])

    # Compute rotation: camera +Z should point from camera to target
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    # World up
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    # Camera frame: X=right, Y=up, Z=forward (optical axis)
    R = np.column_stack([right, cam_up, forward])

    video_writer = VideoWriter.AddToBuilder(
        filename="/output/drake_hello_world.mp4",
        builder=builder,
        sensor_pose=RigidTransform(RotationMatrix(R), camera_pos),
        width=1280,
        height=720,
        fps=30.0,
        fov_y=1.0,
        near=0.1,
        far=50.0,
        backend="cv2",
        fourcc="mp4v",
    )

    diagram = builder.Build()

    return diagram, plant, objects, video_writer


def run_simulation(duration=5.0):
    """Run the simulation and save video."""
    diagram, plant, objects, video_writer = create_simulation()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Set initial positions with some rotation
    np.random.seed(42)
    for i, (body, initial_pos) in enumerate(objects):
        rotation = RotationMatrix(RollPitchYaw(
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3)
        ))
        X_WB = RigidTransform(rotation, initial_pos)
        plant.SetFreeBodyPose(plant_context, body, X_WB)

    print(f"Running simulation for {duration} seconds...")
    print("Recording video...")

    simulator.AdvanceTo(duration)

    video_writer.Save()

    print("Simulation complete!")
    print("Video saved to: /output/drake_hello_world.mp4")


if __name__ == "__main__":
    print("=" * 60)
    print("PyDrake Hello World - Falling Objects Simulation")
    print("=" * 60)
    run_simulation(duration=5.0)
