#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include"version.h"
#include"libconfig.h"
#include"fitsio.h"

#define SUCCESS 0
#define FAILURE 1

#define FILENAME_LEN        256
#define DEFAULT_OUT_PREFIX  "output_"
#define SCREEN_WIDTH        40
#define FILE_READONLY       "r"
#define FILE_READWRITE      "w"

#define LIGHTSPEED 299792458

/* Structure to store the input options */
struct optionsList {
    const char *qCubeName;
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
    
    double *freqList;
    double *lambda2;
    double lambda20;
    
    double *phiAxis;
    double *rmsf, *rmsfReal, *rmsfImag;
    
    double *qImageArray, *uImageArray;
};
