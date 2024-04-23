cmake_minimum_required(VERSION 3.0.2)
project(tbai_mpc)

## C++17 is a requirement
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
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