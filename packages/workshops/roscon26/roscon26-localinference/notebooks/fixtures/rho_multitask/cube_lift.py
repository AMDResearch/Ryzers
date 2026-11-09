# Code block 0
home_pose()
# Get the grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose("red cube")

# Move the robot to the grasp position
# Using a small positive z_approach ensures a controlled approach before grasping
goto_pose(grasp_position, grasp_quaternion, z_approach=0.05)

# Open the gripper just in case
open_gripper()

# Close the gripper to pick up the cube
close_gripper()

# Define a lift height
lift_height = 0.1 # meters

# Calculate the new position by lifting the grasp position along the Z-axis
# Note: We assume the Z-axis is the vertical axis in the robot's base frame for simple lifting.
lifted_position = grasp_position + numpy.array([0.0, 0.0, lift_height])

# Move the robot to the lifted position while maintaining the grasp orientation
goto_pose(lifted_position, grasp_quaternion)
