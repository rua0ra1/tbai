# tbai_static package

## Example
```bash
# Start ROS and relevant nodes
roslaunch tbai_static simple.launch

# Change controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'SIT'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'STAND'"
```