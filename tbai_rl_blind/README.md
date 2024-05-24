# tbai_rl_blind package

## Example
```bash
# Start ROS and relevant nodes
roslaunch tbai_rl_blind simple.launch gui:=true

# Change controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'RL'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'BOB'"  # RL and BOB are the same controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'STAND'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'SIT'"
```



https://github.com/lnotspotl/tbai/assets/82883398/308b67f4-4493-4e47-8e53-8a773d556fc4

