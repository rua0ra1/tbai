# Robot state message
# The message contains the following values:
# Base orientation - ZYX Euler angles - 3
# Base position in world frame - 3
# Base angular velocity in base frame - 3
# Base linear velocity in base frame - 3
# Joint angles - 12
# Joint velocities - 12
#
# In total, there are (3+3+3+3+12+12) = 36 values in the message.

time stamp
float64[36] rbd_state