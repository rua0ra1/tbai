cmake_minimum_required(VERSION 3.0.2)
project(tbai_gridmap)

## C++17 is a requirement
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    tbai_core
    grid_map_ros
    grid_map_core
)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES tbai_gridmap 
 CATKIN_DEPENDS  
 DEPENDS OpenCV
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(tbai_gridmap
    src/GridmapInterface.cpp
)
target_link_libraries(tbai_gridmap ${catkin_LIBRARIES})

install(TARGETS tbai_gridmap
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)