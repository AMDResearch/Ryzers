# Multi-task CaP-X repository contract

This repository deploys 2 generated robot policies:

- `solver/tasks/cube_stack.py` stacks a red cube on a green cube.
- `solver/tasks/cube_lift.py` picks up the red cube and lifts it clear of the table.

The evaluator runs different simulator layouts for training and validation.
Its feedback includes deployable reward, raw environment reward, task completion,
Python tracebacks, and the tail of robot-policy output. An execution failure has
deployable reward zero even if the robot made partial progress.

You may edit any file below `solver/`, including extracting shared calculations
or safety behavior into `solver/geometry.py` and `solver/runtime.py`. Do not
special-case scenario identifiers, trial numbers, or fixed object coordinates.

Important API facts:

- `get_object_pose(..., return_bbox_extent=True)` returns flat XYZ position,
  WXYZ quaternion, and full XYZ extents.
- `sample_grasp_pose(...)` returns a flat XYZ position and WXYZ quaternion.
- `goto_pose(...)` is blocking and may raise if code keeps issuing actions after
  the simulator has already completed or terminated an episode.
- Imported helper modules cannot directly access the injected robot primitives;
  use them for calculations, path generation, and reusable control decisions.
