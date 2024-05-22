cmake_minimum_required(VERSION 3.0.2)
project(tbai_joe)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  tbai_core
)

set(Torch_DIR ${LIBTORCH_CMAKE_DIR})
find_package(Torch)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch Not Found! Move your libtorch to the ${LIBTORCH_DIR} folder. You can a symlink if you want. See the 'Installing libtorch C++' section in the README.md file.")
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  tbai_msgs
  tbai_core
  tbai_static
  ocs2_core
  ocs2_mpc
  ocs2_robotic_tools
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  ocs2_legged_robot
  ocs2_legged_robot_ros
  ocs2_msgs
  ocs2_ros_interfaces
  tbai_reference
  tbai_gridmap
  ocs2_quadruped_interface
  ocs2_anymal_mpc
  ocs2_switched_model_interface
)

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
  src/JoeController.cpp
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