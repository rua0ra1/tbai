cmake_minimum_required(VERSION 3.0.2)
project(tbai_context_aware)

## C++17 is a requirement
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    tbai_core
    tbai_mpc
    tbai_static
    tbai_rl
)
set(Torch_DIR ${LIBTORCH_CMAKE_DIR})
find_package(Torch)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch Not Found! Move your libtorch to the ${LIBTORCH_DIR} folder. You can a symlink if you want. See the 'Installing libtorch C++' section in the README.md file.")
endif()


catkin_package(
  INCLUDE_DIRS 
  LIBRARIES  
 CATKIN_DEPENDS  
)

include_directories(
 ${catkin_INCLUDE_DIRS}
 ${TORCH_INCLUDE_DIRS}
)

add_executable(simple
    src/simple.cpp
)
target_link_libraries(simple
    ${catkin_LIBRARIES} yaml-cpp ${TORCH_LIBRARIES}
)