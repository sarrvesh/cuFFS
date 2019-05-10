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

# FindHDF5 uses environment variables, so set it if needed.
if (HDF5_ROOT_DIR)
    set (ENV{HDF5_ROOT} ${HDF5_ROOT_DIR})
endif (HDF5_ROOT_DIR)

find_package(HDF5 REQUIRED C HL)
find_package(CUDA REQUIRED)
find_library(LIBCONFIG config)

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")

# Compile the sources
set(RMSYNTHESISSOURCES ${PROJECT_SOURCE_DIR}/src/rmsynthesis/rmsynthesis.c ${PROJECT_SOURCE_DIR}/src/rmsynthesis/devices.cu ${PROJECT_SOURCE_DIR}/src/rmsynthesis/fileaccess.c ${PROJECT_SOURCE_DIR}/src/rmsynthesis/inputparser.c ${PROJECT_SOURCE_DIR}/src/rmsynthesis/rmsf.c)

cuda_add_executable(rmsynthesis ${RMSYNTHESISSOURCES})

SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/rmsynthesis/fileaccess.c PROPERTIES COMPILE_FLAGS -DH5Gcreate_vers=2)

set(ROTATESOURCES ${PROJECT_SOURCE_DIR}/src/rotate/transpose.c ${PROJECT_SOURCE_DIR}/src/rotate/fitsrotate.c)
add_executable(fitsrotate ${ROTATESOURCES})

# link
target_link_libraries(fitsrotate ${CFITSIO_LIBRARIES} m)
target_link_libraries(rmsynthesis ${CFITSIO_LIBRARIES} ${LIBCONFIG} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

