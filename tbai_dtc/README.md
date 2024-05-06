# tbai_dtc package

## Example
```bash
# Start ROS and relevant nodes
roslaunch tbai_dtc simple.launch

# Change controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'STAND'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'DTC'"
```



https://github.com/lnotspotl/tbai/assets/82883398/19a4185f-0b82-4cac-b961-186789db3875

