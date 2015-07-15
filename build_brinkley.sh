# Rootdir of libconfig
LIB_CONFIG_PATH="/data/users/punzo/sarrvesh/Libraries/libconfig"
# Rootdir of cfitsio
CFITSIO_PATH="/data/users/punzo/sarrvesh/Libraries/cfitsio"

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

nvcc -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -o rmsynthesis src/rmsynthesis.c -lconfig -lcfitsio -lm $MACRO
