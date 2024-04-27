# tbai_rl_perceptive package

## Example
```bash
# Start ROS and relevant nodes
roslaunch tbai_rl_perceptive simple.launch

# Change controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'RL'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'BOB'"  # RL and BOB are the same controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'STAND'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'SIT'"
```


https://github.com/lnotspotl/tbai/assets/82883398/8d296aff-8e88-4dd1-9682-4772d7b0f952

