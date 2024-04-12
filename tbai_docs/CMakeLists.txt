cmake_minimum_required(VERSION 3.10.2)

project (tbai_docs VERSION 0.0.1)

# Modules to document
list(APPEND TBAI_MODULES
  tbai_core
  tbai_config
)

find_package(Doxygen REQUIRED)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/tools/doxygen/Doxyfile.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/tools/doxygen/Doxyfile)

set(DOXYGEN_PROJECT_NAME "TBAI")
set(DOXYGEN_PROJECT_NUMBER ${CMAKE_PROJECT_VERSION})
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output/doxygen)
set(DOXYGEN_HTML_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

# Style taken from: https://jothepro.github.io/doxygen-awesome-css/
set(DOXYGEN_CSS ${CMAKE_CURRENT_SOURCE_DIR}/tools/doxygen/style.css)

# Convert list to space-separated string for feeding to doxygen
foreach(MODULE ${TBAI_MODULES})
  set(MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../${MODULE})
  if (NOT EXISTS ${MODULE_PATH})
    message(WARNING "Could not find module ${MODULE}")
  endif()
  list(APPEND DOXYGEN_INPUT_LIST ${MODULE_PATH})
  unset(MODULE_PATH)
endforeach()
set(DOXYGEN_INPUT "${DOXYGEN_INPUT_LIST}")
string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT}")

message(STATUS "Generating ${DOXYFILE}")
configure_file(${DOXYFILE_IN} ${DOXYFILE} IMMEDIATE)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_HTML_FILE}
  BYPRODUCTS ${DOXYGEN_OUTPUT_DIR}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
  MAIN_DEPENDENCY ${DOXYFILE} ${DOXYFILE_IN}
  COMMENT "Generating C++ API documentation HTML"
  )

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_HTML_FILE})

# File inspired by: https://github.com/leggedrobotics/ocs2/blob/main/ocs2_doc/CMakeLists.txt