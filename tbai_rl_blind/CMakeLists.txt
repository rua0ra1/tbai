cmake_minimum_required(VERSION 3.0.2)
project(tbai_rl_blind)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR /home/kuba/Downloads/libtorch/share/cmake/Torch)


find_package(catkin REQUIRED COMPONENTS
  tbai_core
  tbai_static
  tbai_rl
)

find_package(Torch REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS 
  DEPENDS 
)

add_executable(simple
    src/simple.cpp
)
target_link_libraries(simple
    ${catkin_LIBRARIES} yaml-cpp
)