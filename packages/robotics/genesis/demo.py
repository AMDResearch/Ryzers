import warnings
import os

# os.environ["TI_LOG_LEVEL"] = "error"
# warnings.filterwarnings("ignore")

import numpy as np
import genesis as gs

########################## init ##########################
gs.init(backend=gs.vulkan)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.0, -2, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=200,
    ),
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=False,
    ),
    show_viewer=True
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(6, 4, 2),
    lookat=(0, 0, 0.2),
    fov=30,
)

########################## build ##########################
n_envs = 9
scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

import logging
from tqdm import tqdm

# Set logger to warning to avoid log info.
gs.logger._logger.setLevel(logging.WARNING)

rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
cam.start_recording()

target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
center = np.tile(np.array([0.4, -0.2, 0.25]), [n_envs, 1])
angular_speed = np.random.uniform(-10, 10, n_envs)
r = 0.1

ee_link = robot.get_link("hand")

#for i in tqdm(range(1000), ncols=100):
i=0
while True:    
    target_pos = np.zeros([n_envs, 3])
    target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
    target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
    target_pos[:, 2] = center[:, 2]
    target_q = np.hstack([target_pos, target_quat])

    q = robot.inverse_kinematics(
        link=ee_link,
        pos=target_pos,
        quat=target_quat,
        rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
    )

    robot.set_qpos(q)
    scene.step()
    i=i+1

