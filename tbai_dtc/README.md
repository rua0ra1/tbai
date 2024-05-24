# tbai_dtc package

## Example
```bash
# Start ROS and relevant nodes
roslaunch tbai_dtc simple.launch

# Change controllers
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'STAND'"
rostopic pub /anymal_d/change_controller std_msgs/String "data: 'DTC'"
```




https://github.com/lnotspotl/tbai/assets/82883398/0ecef030-1581-4400-9e6a-9df2fadb3d3a

