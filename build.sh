# Rootdir of libconfig
LIB_CONFIG_PATH="/home/sarrvesh/Documents/workEnv/Others/libconfig"

#gcc -Wall -v -I${LIB_CONFIG_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -o rmsynthesis rmsynthesis.c -lconfig
gcc -Wall -I${LIB_CONFIG_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -o rmsynthesis src/rmsynthesis.c -lconfig -lcfitsio -lm
