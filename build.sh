# Rootdir of libconfig
LIB_CONFIG_PATH="/home/sarrvesh/Documents/workEnv/Others/libconfig"

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

gcc -Wall -I${LIB_CONFIG_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -o rmsynthesis src/rmsynthesis.c -lconfig -lcfitsio -lm $MACRO
