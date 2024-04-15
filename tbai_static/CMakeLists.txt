cmake_minimum_required(VERSION 3.0.2)
project(tbai_static)

## C++17 is a requirement
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  tf
  urdf
  kdl_parser
  robot_state_publisher
  tbai_msgs
  tbai_core
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME}
  CATKIN_DEPENDS 
  DEPENDS Eigen
)

## Build
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/StaticController.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

add_executable(simple
    src/simple.cpp
)
target_link_libraries(simple
    ${catkin_LIBRARIES} ${PROJECT_NAME} yaml-cpp
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)