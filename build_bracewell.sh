# Rootdir of libconfig
LIB_CONFIG_PATH="/home/see041/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/apps/cfitsio/3.39"
# CUDA rootdir
CUDA_PATH="/apps/cuda/8.0.61"
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
nvcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/devices.cu -gencode $NVCC_FLAGS

printf "Compiling fileaccess.c\n"
gcc -O3 -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/fileaccess.c 

printf "Compiling inputparser.c\n"
gcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/inputparser.c 

printf "Compiling rmsf.c\n"
gcc -O3 -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/rmsf.c 

printf "Compiling rmsynthesis.c\n"
gcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/rmsynthesis.c 

nvcc -O3 -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -gencode $NVCC_FLAGS
