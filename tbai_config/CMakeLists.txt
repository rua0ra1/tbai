cmake_minimum_required(VERSION 3.0.2)
project(tbai_config)

## C++17 is a requirement
add_compile_options(-std=c++17)

## Find packages
find_package(catkin REQUIRED COMPONENTS 
    roscpp
    tbai_core
)
find_package(yaml-cpp REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp tbai_core
    DEPENDS yaml-cpp
)

## Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

## Add library
add_library(${PROJECT_NAME}
    src/YamlConfig.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

## Install

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

## Testing
catkin_add_gtest(test_config
  test/testConfig.cpp
)
target_link_libraries(test_config
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
