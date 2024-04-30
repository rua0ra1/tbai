cmake_minimum_required(VERSION 3.0.2)
project(tbai_rl)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  tf
  urdf
  kdl_parser
  robot_state_publisher
  tbai_msgs
  tbai_core
  tbai_static
  tbai_gridmap
  tbai_reference
  geometry_msgs
  visualization_msgs
)

set(Torch_DIR ${LIBTORCH_CMAKE_DIR})
find_package(Torch)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch Not Found! Move your libtorch to the ${LIBTORCH_DIR} folder. You can a symlink if you want. See the 'Installing libtorch C++' section in the README.md file.")
endif()

find_package(pinocchio REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME}
  CATKIN_DEPENDS tbai_reference tbai_gridmap
  DEPENDS Torch
)

## Build
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/BobController.cpp
  src/InverseKinematics.cpp
  src/CentralPatternGenerator.cpp
  src/Visualizers.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${pinocchio_LIBRARIES}
)