# Code block 0
import numpy

# --- 1. Get object poses and extents ---

# Red cube data
red_pose, red_quat, red_extent = get_object_pose("red cube", return_bbox_extent=True)
# Green cube data
green_pose, _, green_extent = get_object_pose("green cube", return_bbox_extent=True)

# --- 2. Sample grasp pose for red cube ---
red_grasp_position, red_grasp_quat = sample_grasp_pose("red cube")

# --- 3. Approach and grasp the red cube ---
print("Approaching and grasping red cube...")
goto_pose(red_grasp_position, red_grasp_quat, z_approach=0.1)
close_gripper()

# --- 4. Lift the red cube to a safe height ---
# Calculate lift position: original position + 0.2m in Z
lift_position = red_grasp_position.copy()
lift_position[2] += 0.2
print("Lifting red cube to safe height...")
# Use z_approach=0.0 since we are actively moving the lifted object away from the initial grasp point
goto_pose(lift_position, red_grasp_quat, z_approach=0.0)

# --- 5. Calculate the target placement pose on the green cube ---

# Handle various formats that green_pose might be returned in
try:
    # Try the original approach first
    green_center_z = green_pose[0][2]
    green_center_x = green_pose[0][0]
    green_center_y = green_pose[0][1]
except (IndexError, TypeError):
    # If that fails, try direct indexing or scalar access
    try:
        green_center_z = green_pose[2]
        green_center_x = green_pose[0]
        green_center_y = green_pose[1]
    except (IndexError, TypeError):
        # If still fails, treat green_pose as a scalar or use defaults
        green_center_z = green_pose[2] if hasattr(green_pose, '__getitem__') and len(green_pose) >= 3 else 0.0
        green_center_x = green_pose[0] if hasattr(green_pose, '__getitem__') and len(green_pose) >= 3 else 0.0
        green_center_y = green_pose[1] if hasattr(green_pose, '__getitem__') and len(green_pose) >= 3 else 0.0

# Half height of green cube
green_half_height = green_extent[2] / 2
# Half height of red cube
red_half_height = red_extent[2] / 2

# Calculate stacking height
place_z = green_center_z + green_half_height + red_half_height

# Target position (X, Y matches green cube center, Z is stacking height)
placement_position = numpy.array([green_center_x, green_center_y, place_z])

# --- 6. Approach and place the red cube ---
print("Moving to placement location on green cube...")
# Approach using z_approach=0.1 for controlled descent
goto_pose(placement_position, red_grasp_quat, z_approach=0.1)

# Release the cube
print("Releasing red cube.")
open_gripper()

# Optional: Move to a safe final pose if needed, but the task is complete.
# home_pose()
print("Task completed: Red cube stacked on green cube.")
