cmake_minimum_required(VERSION 3.5)
project(cuFFS)

# Find external packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
FIND_PACKAGE(CFITSIO REQUIRED)

# Set includes
include_directories(${PROJECT_SOURCE_DIR}/src/rotate/)
include_directories(${CFITSIO_INCLUDE_DIR})

# Enable optimization
LIST(APPEND CMAKE_C_FLAGS "-O3 -march=native")

# Compile the sources
set(SOURCES ${PROJECT_SOURCE_DIR}/src/rotate/transpose.c ${PROJECT_SOURCE_DIR}/src/rotate/fitsrotate.c)
add_executable(fitsrotate ${SOURCES})

# link
target_link_libraries(fitsrotate ${CFITSIO_LIBRARIES} m)
