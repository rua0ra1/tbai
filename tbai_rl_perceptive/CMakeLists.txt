cmake_minimum_required(VERSION 3.0.2)
project(tbai_rl_perceptive)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  tbai_core
  tbai_static
  tbai_rl
)

set(Torch_DIR ${LIBTORCH_CMAKE_DIR})
find_package(Torch)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch Not Found! Move your libtorch to the ${LIBTORCH_DIR} folder. You can a symlink if you want. See the 'Installing libtorch C++' section in the README.md file.")
endif()

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