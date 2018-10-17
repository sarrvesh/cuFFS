# Rootdir of libconfig
LIB_CONFIG_PATH="/home/sarrvesh/Documents/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/home/sarrvesh/Documents/cfitsio"
#Rootdir of CUDA Toolkit
CUDA_PATH="/home/sarrvesh/Documents/cuda"
#Rootdir of HDF5
HDF5_PATH="/home/sarrvesh/Documents/hdf5"
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

printf "Compiling devices.cu\n"
nvcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -c src/rmsynthesis/devices.cu -lhdf5 -gencode $NVCC_FLAGS

printf "Compiling fileaccess.c\n"
gcc -Wno-unused-result $GCC_FLAGS -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -L${HDF5_PATH}/lib/ -c src/rmsynthesis/fileaccess.c -lhdf5 -lhdf5_hl

printf "Compiling inputparser.c\n"
gcc $GCC_FLAGS -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsynthesis/inputparser.c

printf "Compiling rmsf.c\n"
gcc $GCC_FLAGS -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsynthesis/rmsf.c

#printf "Compiling rmsynthesis.c\n"
gcc -DMACRO $GCC_FLAGS -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsynthesis/rmsynthesis.c

nvcc -O3 -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -L${HDF5_PATH}/lib/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -lhdf5 -lhdf5_hl -gencode $NVCC_FLAGS

printf "compiling rotate.c\n"
gcc $C_FLAGS -c src/rotate/fitsrotate.c
gcc $C_FLAGS -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/rotate/transpose.c

gcc $L_FLAGS -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ transpose.o fitsrotate.o -o fitsrot -lcfitsio -lm

printf "compiling makecube.c\n"
gcc $C_FLAGS -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/makecube/makecube.c

gcc $L_FLAGS -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ makecube.o -o makecube -lcfitsio -lm
