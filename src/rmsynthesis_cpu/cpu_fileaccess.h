/******************************************************************************
cpu_fileaccess.h
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
#ifndef CPU_FILEACCESS_H
#define CPU_FILEACCESS_H

#include "fitsio.h"

#define CTYPE_LEN 10
#define CUBE_DIM 3

/* Structure to store the input options */
struct optionsList {
   char *qCubeName;
   char *uCubeName;
   char *freqFileName;
   char *outPrefix;
   
   double phiMin, dPhi;
   int nPhi;

   int doRMClean;
   int nIter;
   double threshold;
   double gain;
   
   int nThreads;
};

/* Structure to store all information related to RM Synthesis */
struct parList {
    fitsfile *qFile, *uFile;

    FILE *freq;

    int qAxisNum, uAxisNum;
    int qAxisLen1, qAxisLen2, qAxisLen3;
    int uAxisLen1, uAxisLen2, uAxisLen3;
    float crval1, crval2, crval3;
    float crpix1, crpix2, crpix3;
    float cdelt1, cdelt2, cdelt3;
    char ctype1[CTYPE_LEN], ctype2[CTYPE_LEN], ctype3[CTYPE_LEN];

    float *freqList;
    float *lambda2;
    float lambda20;

    float *phiAxis;
    float *rmsf, *rmsfReal, *rmsfImag;
    float K;
    
    float *phiAxisDouble;
    float *rmsfDouble, *rmsfRealDouble, *rmsfImagDouble;
};

struct optionsList parseInput(char *parsetFileName);
void checkInputFiles(struct optionsList *inOptions, struct parList *params);
int getFitsHeader(struct optionsList *inOptions, struct parList *params);
void checkFitsError(int status);
void makeOutputFitsImages(struct optionsList *inOptions, struct parList *params);
void printOptions(struct optionsList inOptions, struct parList params);
int getFreqList(struct optionsList *inOptions, struct parList *params);
void writeOutputToDisk(struct optionsList *inOptions, struct parList *params, 
                       float *array, long nOutElements, char filenamefull[]);
void freeStructures(struct optionsList *inOptions, struct parList *params);
#endif
