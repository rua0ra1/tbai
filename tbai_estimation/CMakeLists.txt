cmake_minimum_required(VERSION 3.0.2)
project(tbai_estimation)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ocs2_legged_robot
  realtime_tools
  roscpp
  tbai_legged_common
  tf2_ros
  tbai_msgs
  sensor_msgs
  message_filters
  tbai_core
)

find_package(yaml-cpp REQUIRED)

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(pinocchio REQUIRED pinocchio)

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        ocs2_legged_robot
        tbai_legged_common
        realtime_tools
        tf2_ros
        tbai_msgs
        sensor_msgs
        message_filters
        tbai_core
        DEPENDS yaml-cpp
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # ${pinocchio_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}

)

# link_directories(
#         ${pinocchio_LIBRARY_DIRS}
# )


add_executable(state_estimate_publisher_node src/StateEstimatePublisherNode.cpp src/StateEstimatePublisher.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(state_estimate_publisher_node
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

target_compile_options(state_estimate_publisher_node PRIVATE ${OCS2_CXX_FLAGS})


