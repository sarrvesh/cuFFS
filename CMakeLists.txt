cmake_minimum_required(VERSION 3.5)
project(cuFFS)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Enable optimization
LIST(APPEND CMAKE_C_FLAGS "-std=c11 -O3 -march=native -Wall")
#LIST(APPEND CMAKE_C_FLAGS "-std=c11 -ggdb3 -march=native -Wall -fopenmp")

# Find the libraries
include_directories(${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/)
find_package(LIBCONFIG REQUIRED)
include_directories(${LIBCONFIG_INCLUDES})
find_package(CFITSIO REQUIRED)
include_directories(${CFITSIO_INCLUDE_DIR})
if(HDF5_ROOT_DIR)
    set (ENV{HDF5_ROOT} ${HDF5_ROOT_DIR})
endif (HDF5_ROOT_DIR)
find_package(HDF5 REQUIRED C HL)
include_directories(${HDF5_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

###################################
#
# Build CPU rmsynthesis
#
###################################
set(CPUSOURCES 
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_fileaccess.c 
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_rmsf.c 
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_rmclean.c 
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis_cpu/cpu_rmsynthesis.c)
add_executable(rmsynthesis_cpu ${CPUSOURCES})
target_link_libraries(rmsynthesis_cpu 
    ${LIBCONFIG_LIBRARIES} 
    ${CFITSIO_LIBRARIES} 
    ${OpenMP_C_FLAGS} 
    ${CMAKE_C_FLAGS} 
    m)

###################################
#
# Build fits rotate
#
###################################
set(FITSROT 
    ${PROJECT_SOURCE_DIR}/src/rotate/transpose.c
    ${PROJECT_SOURCE_DIR}/src/rotate/fitsrotate.c)
add_executable(fitsrotate ${FITSROT})
target_link_libraries(fitsrotate 
    ${CFITSIO_LIBRARIES} 
    m 
    ${CMAKE_C_FLAGS})

###################################
#
# Build make cube
#
###################################
set(MAKECUBE ${PROJECT_SOURCE_DIR}/src/makecube/makecube.c)
add_executable(makecube ${MAKECUBE})
target_link_libraries(makecube 
    ${CFITSIO_LIBRARIES} 
    m 
    ${CMAKE_C_FLAGS})

###################################
#
# Build the GPU code
#
###################################
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_50")
set(RMSYNTHESIS 
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis/rmsynthesis.c
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis/devices.cu
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis/fileaccess.c
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis/inputparser.c
    ${PROJECT_SOURCE_DIR}/src/rmsynthesis/rmsf.c)
cuda_add_executable(rmsynthesis ${RMSYNTHESIS})
SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/rmsynthesis/fileaccess.c PROPERTIES COMPILE_FLAGS -DH5Gcreate_vers=2)
target_link_libraries(rmsynthesis 
    ${CFITSIO_LIBRARIES} 
    ${LIBCONFIG_LIBRARIES} 
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    m
    ${CMAKE_C_FLAGS})
