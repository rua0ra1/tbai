cmake_minimum_required(VERSION 3.0.2)
project(tbai_mpc)

## C++17 is a requirement
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    tbai_core
    tbai_msgs
    ocs2_core
    ocs2_mpc
    ocs2_msgs
    ocs2_ros_interfaces
    ocs2_anymal_commands
    grid_map_core
    grid_map_ros
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES tbai_mpc 
 CATKIN_DEPENDS  
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(tbai_mpc
  src/reference/ReferenceTrajectoryGenerator.cpp
)
target_link_libraries(tbai_mpc ${catkin_LIBRARIES})