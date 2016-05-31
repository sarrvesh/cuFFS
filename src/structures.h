/******************************************************************************
structures.h
Copyright (C) 2016  {fullname}

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Correspondence concerning RMSynth_GPU should be addressed to: 
sarrvesh.ss@gmail.com

******************************************************************************/
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
    float crval1, crval2;
    float crpix1, crpix2;
    float cdelt1, cdelt2;
    char ctype1[CTYPE_LEN], ctype2[CTYPE_LEN];
    
    float *freqList;
    float *lambda2;
    float lambda20;
    
    float *phiAxis;
    float *rmsf, *rmsfReal, *rmsfImag;

    float *maskArray;
};

/* Structure to store useful GPU device information */
struct deviceInfoList {
    int deviceID;
    long unsigned int globalMem, constantMem, sharedMemPerBlock;
    int maxThreadPerMP;
    int maxThreadPerBlock;
    int threadBlockSize[N_DIMS];
    int warpSize;
    int nSM;
};
