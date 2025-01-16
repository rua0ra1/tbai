# Joint state message
# The message contains the following values:
# Joint angles - 12
# Joint velocities - 12
#
# In total, there are (12+12) = 24 values in the message.
# The message also includes contact information for the feet.
# There are four feet, each contains a single contact flag
# The feet are ordered as follows: LF, RF, LH, RH

std_msgs/Header header
float64[12] joints_position
float64[12] joints_velocity

