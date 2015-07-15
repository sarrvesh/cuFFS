#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "version.h"
#include "libconfig.h"
#include "fitsio.h"

#include<cuda_runtime.h>
#include<cuda.h>

#define NO_DEVICE 0
#define N_DIMS 3
#define X_DIM 0
#define Y_DIM 1
#define Z_DIM 2

#define SUCCESS 0
#define FAILURE 1

#define FILENAME_LEN        256
#define STRING_BUF_LEN      256
#define DEFAULT_OUT_PREFIX  "output_"
#define SCREEN_WIDTH        40
#define FILE_READONLY       "r"
#define FILE_READWRITE      "w"

#define LIGHTSPEED 299792458

/* Structure to store the input options */
struct optionsList {
    char *qCubeName;
    char *uCubeName;
    char *freqFileName;
    char *outPrefix;
    
    double phiMin, dPhi;
    int nPhi;
};

/* Structure to store all information related to RM Synthesis */
struct parList {
    fitsfile *qFile, *uFile;
    FILE *freq;
    
    int qAxisNum, uAxisNum;
    int qAxisLen1, qAxisLen2, qAxisLen3;
    int uAxisLen1, uAxisLen2, uAxisLen3;
    
    float *freqList;
    float *lambda2;
    float lambda20;
    
    float *phiAxis;
    float *rmsf, *rmsfReal, *rmsfImag;
    
    float *qImageArray, *uImageArray;
};

/* Structure to store useful GPU device information */
struct deviceInfoList {
    int deviceID;
    long unsigned int globalMem, constantMem, sharedMemPerBlock;
    //int nMP, nCudaCorePerMP, nCudaCores;
    int maxThreadPerMP;
    int maxThreadPerBlock;
    int threadBlockSize[N_DIMS];
};
