#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Interactive tactile sandbox: a torus, a sphere, a duck and a dragon, one per environment, moved with the keyboard
over a 20x20 taxel pad (arrows move, j/k lower/raise, n/m rotate, \\ resets, ESC quits), with a live sensor plot.
Needs a display.

    ryzers run /ryzers/demo_tactile_sandbox.py
    SENSOR=elastomer ryzers run /ryzers/demo_tactile_sandbox.py   # depth | kinematic | elastomer | proximity
"""

import os
import sys
import tempfile

if not os.environ.get("DISPLAY"):
    sys.exit("demo_tactile_sandbox.py is interactive and needs a display.")
# The viewer needs the window-system OpenGL platform, which must be settled before importing genesis.
os.environ.pop("PYOPENGL_PLATFORM", None)

# coacd must be imported before torch (pulled in by genesis): the ROCm libraries torch preloads export their own
# spdlog, which crashes the convex decomposition of the meshes otherwise.
import coacd
import matplotlib.pyplot as plt
import numpy as np
import trimesh

print("Loading Genesis...")
import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir, tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

SENSORS = ("depth", "kinematic", "elastomer", "proximity")
LABELS = ("torus", "sphere", "duck", "dragon")
GRID_SIZE = 20
PAD_SIZE = 0.15
PAD_HEIGHT = 0.05
PROBE_RADIUS = 0.004
OBJECT_SIZE = 0.08
CLEARANCE = 0.01
MAX_PENETRATION = 0.006
KEY_SPEED = 0.02
KEY_YAW_RATE = 1.0


def add_sensor(scene, kind, pad, obj, probe_grid):
    common = dict(entity_idx=pad.idx, link_idx_local=0, probe_radius=PROBE_RADIUS, draw_debug=True)
    tracked = dict(probe_local_normal=(0.0, 0.0, 1.0), track_link_idx=(obj.base_link_idx,))
    probe_flat = probe_grid.reshape(-1, 3)
    if kind == "depth":
        return scene.add_sensor(gs.sensors.ContactDepthProbe(probe_local_pos=probe_flat, **common))
    if kind == "kinematic":
        return scene.add_sensor(gs.sensors.KinematicTaxel(
            probe_local_pos=probe_flat, normal_stiffness=500.0, shear_scalar=4.0, twist_scalar=1.0,
            normal_exponent=1.5, **common))
    if kind == "elastomer":
        return scene.add_sensor(gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_grid, n_sample_points=1000, lambda_d=5000.0, lambda_s=4000.0, dilate_scale=0.1,
            normal_exponent=1.5, compressibility=0.8, debug_point_cloud_radius=0.001, **tracked, **common))
    common["probe_radius"] = PROBE_RADIUS * 5
    return scene.add_sensor(gs.sensors.ProximityTaxel(
        probe_local_pos=probe_flat, n_sample_points=4000, stiffness=40.0, shear_coupling=10.0,
        debug_point_cloud_radius=0.0005, debug_probe_sphere_opacity=0.0, **tracked, **common))


def add_plot(scene, kind, sensor, probe_grid):
    if kind == "depth":
        return scene.add_recorder(
            lambda: tuple(float(depth.max()) for depth in sensor.read()),
            gs.recorders.MPLLinePlot(title="ContactDepthProbe max depth", labels=LABELS, y_label="depth [m]",
                                     history_length=1000, hz=25),
        )
    if kind == "elastomer":
        read_field = lambda: sensor.read().reshape(len(LABELS), -1, 3).cpu().numpy()
        field = dict(title="ElastomerTaxel marker displacement", scale_factor=3.0, max_magnitude=0.004)
    else:
        read_field = lambda: tuple(t.reshape(len(LABELS), -1, 3).cpu().numpy() for t in sensor.read())
        field = {
            "kinematic": dict(title="KinematicTaxel force + twist", scale_factor=0.1, max_magnitude=0.35,
                              twist_scale_factor=0.004, twist_max_magnitude=1.0),
            "proximity": dict(title="ProximityTaxel force + twist", scale_factor=0.5, max_magnitude=2.0,
                              twist_scale_factor=0.001, twist_max_magnitude=1.0),
        }[kind]
    return scene.add_recorder(read_field, gs.recorders.MPLVectorFieldPlot(
        positions=probe_grid.reshape(-1, 3), normal=(0.0, 0.0, 1.0), subplot_titles=LABELS, hz=25, **field))


def mesh_bottom(path, scale, euler):
    """Lowest point of a mesh below its origin once scaled and rotated (euler in degrees)."""
    vertices = trimesh.load(path, force="mesh").vertices * scale
    return float((vertices @ gu.euler_to_R(np.asarray(euler, dtype=float)).T)[:, 2].min())


def main():
    kind = os.environ.get("SENSOR") or "kinematic"
    if kind not in SENSORS:
        sys.exit(f"SENSOR must be one of {', '.join(SENSORS)}, got '{kind}'.")

    coacd.set_log_level("error")
    gs.init(backend=gs.gpu, logging_level="warning")
    print(f"Backend: {gs.backend}")
    print("Building scene and compiling GPU kernels (a few minutes on the first run, faster once cached)...")

    # The objects are driven kinematically through their velocity: no gravity and no constraint forces, so they go
    # exactly where the keys send them while collision detection still gates the tactile queries.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(disable_constraint=True),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.4, -0.4, 0.42), camera_lookat=(0.0, 0.0, PAD_HEIGHT),
                                                enable_help_text=False),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    pad = scene.add_entity(
        gs.morphs.Box(size=(PAD_SIZE, PAD_SIZE, PAD_HEIGHT), pos=(0.0, 0.0, PAD_HEIGHT / 2), fixed=True),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.8, 1.0)),
    )
    torus_path = os.path.join(tempfile.gettempdir(), "tactile_torus.obj")
    trimesh.creation.torus(major_radius=1.0, minor_radius=0.5).export(torus_path)
    duck_path, dragon_path = (os.path.join(get_assets_dir(), f) for f in ("meshes/duck.obj", "meshes/dragon/dragon.obj"))

    # Place every object at its start pose before the build, so the viewer opens on it.
    def start(bottom):
        return (0.0, 0.0, PAD_HEIGHT + CLEARANCE - bottom)

    obj = scene.add_entity(
        morph=[
            gs.morphs.Mesh(file=torus_path, scale=OBJECT_SIZE / 2, convexify=False,
                           pos=start(mesh_bottom(torus_path, OBJECT_SIZE / 2, (0.0, 0.0, 0.0)))),
            gs.morphs.Sphere(radius=OBJECT_SIZE / 2, pos=start(-OBJECT_SIZE / 2)),
            gs.morphs.Mesh(file=duck_path, euler=(90.0, 0.0, 0.0), scale=0.03,
                           pos=start(mesh_bottom(duck_path, 0.03, (90.0, 0.0, 0.0)))),
            gs.morphs.Mesh(file=dragon_path, euler=(90.0, 0.0, 90.0), scale=0.001,
                           pos=start(mesh_bottom(dragon_path, 0.001, (90.0, 0.0, 90.0)))),
        ],
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )

    coords = (np.arange(GRID_SIZE) - (GRID_SIZE - 1) / 2) * (PAD_SIZE / GRID_SIZE)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    probe_grid = np.stack([xx, yy, np.full_like(xx, PAD_HEIGHT / 2)], axis=-1)
    sensor = add_sensor(scene, kind, pad, obj, probe_grid)
    add_plot(scene, kind, sensor, probe_grid)
    scene.build(n_envs=len(LABELS), env_spacing=(PAD_SIZE * 1.2, PAD_SIZE * 1.2))

    # Refine the start height on the collision geometry, which the visual mesh only approximates.
    start_pos = tensor_to_array(obj.get_pos())
    start_pos[:, 2] += PAD_HEIGHT + CLEARANCE - tensor_to_array(obj.get_AABB())[:, 0, 2]
    obj.set_pos(start_pos)
    start_quat = tensor_to_array(obj.get_quat())

    # Velocity command (x, y, z, yaw rate) set while a key is held; the viewer thread writes it, the loop reads it.
    command = np.zeros(4)
    state = {"reset": False, "quit": False}

    def hold(axis, value):
        command[axis] = value

    def flag(name):
        state[name] = True

    keybinds = [Keybind("reset", Key.BACKSLASH, callback=flag, args=("reset",)),
                Keybind("quit", Key.ESCAPE, callback=flag, args=("quit",))]
    for key, axis, value in ((Key.UP, 0, -KEY_SPEED), (Key.DOWN, 0, KEY_SPEED), (Key.LEFT, 1, -KEY_SPEED),
                             (Key.RIGHT, 1, KEY_SPEED), (Key.J, 2, -KEY_SPEED), (Key.K, 2, KEY_SPEED),
                             (Key.M, 3, -KEY_YAW_RATE), (Key.N, 3, KEY_YAW_RATE)):
        keybinds += [Keybind(f"{key.name}_press", key, KeyAction.PRESS, callback=hold, args=(axis, value)),
                     Keybind(f"{key.name}_release", key, KeyAction.RELEASE, callback=hold, args=(axis, 0.0))]
    scene.viewer.register_keybinds(*keybinds)
    print(f"{kind} sensor. Keys: arrows move, j/k lower/raise, n/m rotate, \\ resets, ESC quits.")

    lowest = start_pos[:, 2] - CLEARANCE - MAX_PENETRATION
    try:
        while not state["quit"] and scene.viewer.is_alive() and plt.get_fignums():
            if state["reset"]:
                obj.set_pos(start_pos)
                obj.set_quat(start_quat)
                state["reset"] = False
            lin_vel = np.tile(command[:3], (len(LABELS), 1))
            at_floor = tensor_to_array(obj.get_pos())[:, 2] <= lowest
            lin_vel[at_floor & (lin_vel[:, 2] < 0.0), 2] = 0.0
            # Free-joint angular velocity is expressed in the solver's (inertia-aligned) link frame.
            ang_vel = gu.inv_transform_by_quat(np.tile([0.0, 0.0, command[3]], (len(LABELS), 1)),
                                               tensor_to_array(obj.get_quat(relative=False)))
            obj.set_dofs_velocity(np.concatenate([lin_vel, ang_vel], axis=1))
            scene.step()
    except KeyboardInterrupt:
        pass

    # Stop the recorders before exiting: matplotlib closes its figures at exit before genesis records a last frame.
    # Once the user has closed the plot window, that last frame cannot be drawn anymore.
    try:
        scene.stop_recording()
    except AttributeError:
        if plt.get_fignums():
            raise


if __name__ == "__main__":
    main()
