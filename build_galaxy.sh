# Rootdir of libconfig
LIB_CONFIG_PATH="/home/gheald/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/pawsey/cle52up04/devel/PrgEnv-gnu/5.2.82/gcc/4.9.2/ivybridge/cfitsio/3370"
# CUDA rootdir
CUDA_PATH="/opt/nvidia/cudatoolkit7.0/7.0.28-1.0502.10280.4.1"
#Rootdir of HDF5
HDF5_PATH="/opt/cray/hdf5/1.8.14/GNU/4.9"
# NVCC flags
NVCC_FLAGS=arch=compute_35,code=sm_35

#####################################################################
################## DO NOT EDIT BELOW THIS LINE ######################
#####################################################################

# Test if gnuplot is installed
printf "Searching for gnuplot: "
if ! gnuplot_loc="$(type -p gnuplot)" || [ -z "$gnuplot_loc" ]; then
    printf "not found\n"
else
    printf "found\n"
    MACRO="-DGNUPLOT_ENABLE"
    printf "Compiling with Gnuplot\n"
fi

printf "Compiling devices.cu\n"
nvcc -g -G -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/devices.cu -gencode $NVCC_FLAGS -use_fast_math

printf "Compiling fileaccess.c\n"
gcc -g -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/fileaccess.c

printf "Compiling inputparser.c\n"
gcc -g -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/inputparser.c

printf "Compiling rmsf.c\n"
gcc -g -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsf.c

printf "Compiling rmsynthesis.c\n"
gcc -g -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsynthesis.c

nvcc -g -G -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -lhdf5 -lhdf5_hl -gencode $NVCC_FLAGS -use_fast_math
