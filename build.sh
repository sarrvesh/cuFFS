# Rootdir of libconfig
LIB_CONFIG_PATH="/home/sarrvesh/Documents/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/home/sarrvesh/Documents/cfitsio"
#Rootdir of CUDA Toolkit
CUDA_PATH="/home/sarrvesh/Documents/cuda"
#Rootdir of HDF5
HDF5_PATH="/home/sarrvesh/Documents/hdf5"
#Rootdir for casacore
CASA_PATH="/home/sarrvesh/Documents/casacore"
# NVCC flags
NVCC_FLAGS=arch=compute_50,code=sm_50

#####################################################################
################## DO NOT EDIT BELOW THIS LINE ######################
#####################################################################

# Set C flags
C_FLAGS="-Wextra -Wall -Wunreachable-code -Wswitch-default -Wstrict-prototypes -Wpointer-arith -Wshadow -Wfloat-equal -Wuninitialized"
L_FLAGS="-O3 -march=native"

# Test if gnuplot is installed
printf "Searching for gnuplot: "
if ! gnuplot_loc="$(type -p gnuplot)" || [ -z "$gnuplot_loc" ]; then
    printf "not found\n"
    MACRO=""
else
    printf "found\n"
    MACRO="-DGNUPLOT_ENABLE"
    printf "Compiling with Gnuplot\n"
fi

printf "compiling Faraday synthesis\n"
g++ -std=c++11 -c src/synthesis/synthesis.cpp
g++ -std=c++11 -I${LIB_CONFIG_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -c src/synthesis/synth_fileaccess.cpp
g++ -std=c++11 -O3 -I${CASA_PATH}/include -L${CASA_PATH}/lib -c src/synthesis/ms_access.cpp
g++ -std=c++11 -I${LIB_CONFIG_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -I${CASA_PATH}/include -L${CASA_PATH}/lib64 -o synthesis synthesis.o synth_fileaccess.o ms_access.o -lconfig -lcasa_ms -lcasa_casa -lcasa_tables -lcasa_measures
