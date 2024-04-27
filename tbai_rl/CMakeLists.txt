cmake_minimum_required(VERSION 3.0.2)
project(tbai_rl)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR /home/kuba/Downloads/libtorch/share/cmake/Torch)

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

find_package(pinocchio REQUIRED)
find_package(Torch REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME}
  CATKIN_DEPENDS 
  DEPENDS 
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

add_executable(simple
    src/simple.cpp
)
target_link_libraries(simple
    ${catkin_LIBRARIES} ${PROJECT_NAME} yaml-cpp
)