# Rootdir of libconfig
LIB_CONFIG_PATH="/data/users/punzo/sarrvesh/Libraries/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/data/users/punzo/sarrvesh/Libraries/cfitsio"
# CUDA rootdir
CUDA_PATH="/data/users/punzo/sarrvesh/CUDA/Toolkit"

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
nvcc -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/devices.cu -gencode arch=compute_30,code=sm_30

printf "Compiling fileaccess.c\n"
gcc -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/fileaccess.c 

printf "Compiling inputparser.c\n"
gcc -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/inputparser.c 

printf "Compiling rmsf.c\n"
gcc -I${CFITSIO_PATH}/include/ -L/${CFITSIO_PATH}/lib/ -c src/rmsf.c 

printf "Compiling rmsynthesis.c\n"
gcc -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -c src/rmsynthesis.c 

nvcc -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -gencode arch=compute_30,code=sm_30 
