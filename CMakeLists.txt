cmake_minimum_required(VERSION 3.5)
project(cuFFS)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Enable optimization
LIST(APPEND CMAKE_C_FLAGS "-O2 -march=native -Wall -Wextra -Wunreachable-code -Wswitch-default -Wstrict-prototypes -Wpointer-arith -Wshadow -Wfloat-equal -Wuninitialized -fopenmp")

###################################
#
# Build CPU rmsynthesis
#
###################################
# Set include directories for cpu rmsynthesis
include_directories(${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/)
find_package(PkgConfig REQUIRED)
pkg_search_module(LIBCONFIG REQUIRED libconfig)
include_directories(${LIBCONFIG_INCLUDE_DIRS})
link_directories(${LIBCONFIG_LIBRARY_DIRS})
find_package(CFITSIO REQUIRED)
include_directories(${CFITSIO_INCLUDE_DIR})
find_package(OpenMP REQUIRED)
# Build cpu rmsynthesis
set(CPUSOURCES ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_fileaccess.c ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_rmsf.c ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_rmsynthesis.c)
add_executable(rmsynthesis_cpu ${CPUSOURCES})
# Link cpu rmsynthesis
target_link_libraries(rmsynthesis_cpu ${LIBCONFIG_LIBRARIES} ${CFITSIO_LIBRARIES} ${OpenMP_C_FLAGS} m)
