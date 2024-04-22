# Robot state message
# The message contains the following values:
# Base position in world frame - 3
# Base orientation - XYZW quaternion - 4
# Joint angles - 12
# [v_x, v_y, v_z] linear velocity of the base in world frame - 3
# [w_x, w_y, w_z] angular velocity of the base in world frame - 3
# Joint velocities - 12
#
# In total, there are (3+4+12+3+3+12) = 37 values in the message.

float64[37] rbd_state