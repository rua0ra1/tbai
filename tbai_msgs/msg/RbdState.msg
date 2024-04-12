# Robot state message
# The message contains the following values:
# Base ZYX Euler angles - 3
# Base position in world frame - 3
# Joint angles - 12
# [w_x, w_y, w_z] angular velocity of the base in world frame - 3
# [v_x, v_y, v_z] linear velocity of the base in world frame - 3
# Joint velocities - 12
#
# In total, there are (3+3+12+3+3+12) = 36 values in the message.

float64[36] rbd_state