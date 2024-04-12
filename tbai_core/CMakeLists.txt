cmake_minimum_required(VERSION 3.0.2)
project(tbai_core)

## C++17 is a requirement
add_compile_options(-std=c++17)

find_package(catkin REQUIRED
    ocs2_core
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES tbai_core
  CATKIN_DEPENDS ocs2_core
  DEPENDS Eigen
)

## Build
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)