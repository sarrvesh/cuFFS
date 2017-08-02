# Rootdir of libconfig
LIB_CONFIG_PATH="/home/see041/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/apps/cfitsio/3.39"
# CUDA rootdir
CUDA_PATH="/apps/cuda/8.0.61"
#Rootdir of HDF5
HDF5_PATH="/media/sarrvesh/Work/INSTALLATIONS/hdf5"
# NVCC flags
NVCC_FLAGS=arch=compute_60,code=sm_60

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
nvcc -O4 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/devices.cu -gencode $NVCC_FLAGS -use_fast_math

printf "Compiling fileaccess.c\n"
gcc -O3 -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/fileaccess.c

printf "Compiling inputparser.c\n"
gcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/inputparser.c

printf "Compiling rmsf.c\n"
gcc -O3 -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsf.c

printf "Compiling rmsynthesis.c\n"
gcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -lhdf5 -lhdf5_hl -c src/rmsynthesis.c

nvcc -O4 -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -I${HDF5_PATH}/include/ -L${HDF5_PATH}/lib/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -lhdf5 -lhdf5_hl -gencode $NVCC_FLAGS -use_fast_math
