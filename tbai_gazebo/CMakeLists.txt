cmake_minimum_required(VERSION 3.0.2)
project(tbai_gazebo)

## C++17 is a requirement
set (CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  hardware_interface
  pluginlib
  roscpp
  realtime_tools
  tbai_msgs
  tbai_config
  tbai_core
)

find_package(gazebo REQUIRED)

catkin_package(
 CATKIN_DEPENDS tbai_msgs tbai_config tbai_core
 DEPENDS pinocchio
)



include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)


## Joint controller plugin
add_library(JointController src/JointController.cpp)
target_link_libraries(JointController ${catkin_LIBRARIES}) 

## RBD state plugin