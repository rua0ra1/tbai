cmake_minimum_required(VERSION 3.0.2)
project(tbai_dtc)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  tbai_msgs
  tbai_core
  tbai_static
  ocs2_core
  ocs2_mpc
  ocs2_ros_interfaces
  ocs2_robotic_tools
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  ocs2_legged_robot
  ocs2_legged_robot_ros
)
set(Torch_DIR /home/kuba/Downloads/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME}
  CATKIN_DEPENDS 
  DEPENDS 
)

find_package(pinocchio REQUIRED)

## Build
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/DtcController.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${pinocchio_LIBRARIES}
    ${TORCH_LIBRARIES}
)
target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_CXX_FLAGS})

add_executable(simple
    src/simple.cpp
)
target_link_libraries(simple
    ${catkin_LIBRARIES} ${PROJECT_NAME} yaml-cpp
)
target_compile_options(simple PUBLIC ${OCS2_CXX_FLAGS})

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)