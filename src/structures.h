#include "fitsio.h"
#include "constants.h"

#ifdef __cplusplus
extern "C"
#endif

/* Structure to store the input options */
struct optionsList {
    char *qCubeName;
    char *uCubeName;
    char *freqFileName;
    char *outPrefix;
    char *imageMask;
    int isImageMaskDefined;
    
    int plotRMSF;
    double phiMin, dPhi;
    int nPhi;
};

/* Structure to store all information related to RM Synthesis */
struct parList {
    fitsfile *qFile, *uFile;
    fitsfile *maskFile;
    FILE *freq;
    
    int qAxisNum, uAxisNum;
    int qAxisLen1, qAxisLen2, qAxisLen3;
    int uAxisLen1, uAxisLen2, uAxisLen3;
    int maskAxisLen1, maskAxisLen2;
    
    float *freqList;
    float *lambda2;
    float lambda20;
    
    float *phiAxis;
    float *rmsf, *rmsfReal, *rmsfImag;

    float *qPhi, *uPhi;

    float *maskArray;
};

/* Structure to store useful GPU device information */
struct deviceInfoList {
    int deviceID;
    long unsigned int globalMem, constantMem, sharedMemPerBlock;
    int maxThreadPerMP;
    int maxThreadPerBlock;
    int threadBlockSize[N_DIMS];
};
