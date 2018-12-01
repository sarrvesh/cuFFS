/******************************************************************************
inputparser.c
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

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "libconfig.h"
#include "fitsio.h"
#include "cpu_fileaccess.h"

#define DEFAULT_OUT_PREFIX "output_"
#define SCREEN_WIDTH 40
#define LIGHTSPEED 299792458.

/*************************************************************
*
* Parse the input file and extract the relevant keywords
*
*************************************************************/
struct optionsList parseInput(char *parsetFileName) {
   config_t cfg;
   struct optionsList inOptions;
   const char *str;
   
   /* Initialize configurations */
   config_init(&cfg);
   
   /* Read in the configuration file */
   if(!config_read_file(&cfg, parsetFileName)) {
      printf("ERROR: Error reading parset file %s\n\n", 
             config_error_text(&cfg));
      config_destroy(&cfg);
      exit(1);
   }
   
   /* Get the number of threads to use */
   if(! config_lookup_int(&cfg, "nThreads", &inOptions.nThreads)) {
        printf("INFO: 'nThreads' undefined in parset\n");
        printf("INFO: Continuing with nThreads=1.\n");
        inOptions.nThreads = 1;
    }
    if(inOptions.nThreads <= 0) {
       printf("Error: nThreads cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(0);
    }
    
    /* Get the number of nPhi */
    if(! config_lookup_int(&cfg, "nPhi", &inOptions.nPhi)) {
        printf("Error: 'nPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(inOptions.nPhi <= 0) {
       printf("Error: nPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(1);
    }
    
    /* Get Faraday depth */
    if(! config_lookup_float(&cfg, "phiMin", &inOptions.phiMin)) {
        printf("Error: 'phiMin' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    /* Get number of output phi planes */
    if(! config_lookup_float(&cfg, "dPhi", &inOptions.dPhi)) {
        printf("Error: 'dPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(inOptions.dPhi <= 0) {
       printf("Error: dPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(1);
    }
    
    /* Get prefix for output files */
    if(config_lookup_string(&cfg, "outPrefix", &str)) {
        inOptions.outPrefix = malloc(strlen(str)+1);
        strcpy(inOptions.outPrefix, str);
    }
    else {
        printf("INFO: 'outPrefix' is not defined. Defaulting to %s\n\n", 
                DEFAULT_OUT_PREFIX);
        inOptions.outPrefix = malloc(strlen(DEFAULT_OUT_PREFIX)+1);
        strcpy(inOptions.outPrefix, DEFAULT_OUT_PREFIX);
    }
    
    /* Get the name of the frequency file */
    if(config_lookup_string(&cfg, "freqFileName", &str)) {
        inOptions.freqFileName = malloc(strlen(str)+1);
        strcpy(inOptions.freqFileName, str);
    }
    else {
        printf("Error: 'freqFileName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    /* Get the names of fits files */
    if(config_lookup_string(&cfg, "qCubeName", &str)) {
        inOptions.qCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.qCubeName, str);
    }
    else {
        printf("Error: 'qCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    if(config_lookup_string(&cfg, "uCubeName", &str)) {
        inOptions.uCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.uCubeName, str);
    }
    else {
        printf("Error: 'uCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(1);
    }
    
    /* Get all RM Clean related parameters */
    if(! config_lookup_bool(&cfg, "doRMClean", &inOptions.doRMClean)) {
       printf("WARN: 'doRMClean' undefined in parset.\n");
       printf("INFO: Will not apply RM CLEAN.");
    }
    else {
       /* Read nIter and threshold from the parset */
       if(! config_lookup_int(&cfg, "nIter", &inOptions.nIter)) {
          printf("WARN: 'nIter' undefined in parset. Defaulting to 0\n");
          inOptions.nIter = 0;
       }
       if(! config_lookup_float(&cfg, "threshold", &inOptions.threshold)) {
          printf("WARN: 'threshold' undefined in parset. Defaulting to 0\n");
          inOptions.threshold = 0;
       }
    }
    if(inOptions.doRMClean) {
       printf("WARN: Support for RM CLEAN is yet to be implemented.\n");
    }
    
    config_destroy(&cfg);
    return(inOptions);    
}

/*************************************************************
*
* Check Fitsio error and exit if required.
*
*************************************************************/
void checkFitsError(int status) {
    if(status) {
        printf("ERROR:");
        fits_report_error(stdout, status);
        printf("\n");
        exit(1);
    }
}

/*************************************************************
*
* Check of the input files are open-able
*
*************************************************************/
void checkInputFiles(struct optionsList *inOptions, struct parList *params) {
   int fitsStatus = 0;
   
   /* Check if all the input fits files are accessible */
   fits_open_file(&(params->qFile), inOptions->qCubeName, READONLY, &fitsStatus);
   fits_open_file(&(params->uFile), inOptions->uCubeName, READONLY, &fitsStatus);
   checkFitsError(fitsStatus);

   /* Check if you can open the frequency file */
   params->freq = fopen(inOptions->freqFileName, "r");
   if(params->freq == NULL) {
      printf("Error: Unable to open the frequency file\n\n");
      exit(1);
   }
}

/*************************************************************
*
* Read header information from the fits files
*
*************************************************************/
int getFitsHeader(struct optionsList *inOptions, struct parList *params) {
    int fitsStatus = 0;
    char fitsComment[FLEN_COMMENT];
    
    /* Remember that the input fits images are not rotated in the cpu version.*/
    /* Frequency is the third axis */
    /* RA is the first */
    /* Dec is the second */
    
    /* Get the image dimensions from the Q cube */
    fits_read_key(params->qFile, TINT, "NAXIS", &params->qAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS1", &params->qAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS2", &params->qAxisLen2, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TINT, "NAXIS3", &params->qAxisLen3, 
      fitsComment, &fitsStatus);
    /* Get the image dimensions from the U cube */
    fits_read_key(params->uFile, TINT, "NAXIS", &params->uAxisNum, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS1", &params->uAxisLen1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS2", &params->uAxisLen2, 
      fitsComment, &fitsStatus);
    fits_read_key(params->uFile, TINT, "NAXIS3", &params->uAxisLen3,
      fitsComment, &fitsStatus);
    /* Get WCS information */
    fits_read_key(params->qFile, TFLOAT, "CRVAL1", &params->crval1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRVAL2", &params->crval2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRVAL3", &params->crval3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX1", &params->crpix1, 
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX2", &params->crpix2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CRPIX3", &params->crpix3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT1", &params->cdelt1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT2", &params->cdelt2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TFLOAT, "CDELT3", &params->cdelt3,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE1", &params->ctype1,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE2", &params->ctype2,
      fitsComment, &fitsStatus);
    fits_read_key(params->qFile, TSTRING, "CTYPE3", &params->ctype3,
      fitsComment, &fitsStatus);
    
    return(fitsStatus);
}

/*************************************************************
*
* Write the content of output cubes in memory maps to FITS cubes
*
*************************************************************/
void writeOutputToDisk(struct optionsList *inOptions, struct parList *params, 
                       float *array, long nOutElements, char filenamefull[]) {
   int stat = 0;
   char fComment[] = "  ";
   fitsfile *file;
   long naxis[CUBE_DIM];
   int tempVar = 1;
   long fPixel[CUBE_DIM];
   
   /* Create the output fits file */
   fits_create_file(&file, filenamefull, &stat);
   checkFitsError(stat);
   
   /* What are the output cube sizes? */
   naxis[0] = params->qAxisLen1;
   naxis[1] = params->qAxisLen2;
   naxis[2] = inOptions->nPhi;
   
   /* Create the fits header */
   fits_create_img(file, FLOAT_IMG, CUBE_DIM, naxis, &stat);
   checkFitsError(stat);
   
   /* Set the relevant fits keys */
   fits_write_key(file, TSTRING, "BUNIT", "JY/BEAM", fComment, &stat);
   fits_write_key(file, TDOUBLE, "CRVAL1", &params->crval1, fComment, &stat);
   fits_write_key(file, TDOUBLE, "CDELT1", &params->cdelt1, fComment, &stat);
   fits_write_key(file, TFLOAT, "CRPIX1", &params->crpix1, fComment, &stat);
   fits_write_key(file, TSTRING, "CTYPE1", &params->ctype1, fComment, &stat);
   fits_write_key(file, TFLOAT, "CRVAL2", &params->crval2, fComment, &stat);
   fits_write_key(file, TFLOAT, "CDELT2", &params->cdelt2, fComment, &stat);
   fits_write_key(file, TFLOAT, "CRPIX2", &params->crpix2, fComment, &stat);
   fits_write_key(file, TSTRING, "CTYPE2", params->ctype2, fComment, &stat);
   fits_write_key(file, TFLOAT, "CRVAL3", &inOptions->phiMin, fComment, &stat);
   fits_write_key(file, TFLOAT, "CDELT3", &inOptions->dPhi, fComment, &stat);
   fits_write_key(file, TFLOAT, "CRPIX3", &tempVar, fComment, &stat);
   fits_write_key(file, TSTRING, "CTYPE3", "PHI", fComment, &stat);
   
   /* Write array to data */
   fPixel[0] = fPixel[1] = fPixel[2] = 1;
   fits_write_pix(file, TFLOAT, fPixel, nOutElements, array, &stat);
   fits_close_file(file, &stat);
   checkFitsError(stat);
}

/*************************************************************
*
* Read some useful information to stdout
*
*************************************************************/
void printOptions(struct optionsList inOptions, struct parList params) {
   int i;
   
   printf("\n");
   for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
   printf("\n");
   printf("Q Cube: %s\n", inOptions.qCubeName);
    printf("U Cube: %s\n", inOptions.uCubeName);
    printf("phi min: %.2f\n", inOptions.phiMin);
    printf("# of phi planes: %d\n", inOptions.nPhi);
    printf("delta phi: %.2lf\n", inOptions.dPhi);
    printf("%d threads will be used\n", inOptions.nThreads);
    printf("\n");
    printf("Input dimension: %d x %d x %d\n", params.qAxisLen1,
                                              params.qAxisLen2,
                                              params.qAxisLen3);
    printf("Output dimension: %d x %d x %d\n", params.qAxisLen1,
                                               params.qAxisLen2,
                                               inOptions.nPhi);
    for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
    printf("\n");
}

/*************************************************************
*
* Comparison function used by quick sort.
*
*************************************************************/
int compFunc(const void * a, const void * b) {
   return ( *(double*)a - *(double*)b );
}

/*************************************************************
*
* Read the frequency information from the input file.
* Note that the file has already been opened and the file 
* pointer is in parList.freq. The freq values will be written 
* to parList.freqList. While we are at it, 
*    - compute the \lambda^2 values are write it to parList.lambda2.
*    - write the median \lambda^2 value to parList.lambda20
*
*************************************************************/
int getFreqList(struct optionsList *inOptions, struct parList *params) {
   int i;
   float tempFloat;
   double *tempArray;
   
   params->freqList = calloc(params->qAxisLen3, sizeof(params->freqList));
    if(params->freqList == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list\n\n");
        return 1;
    }
    for(i=0; i<params->qAxisLen3; i++) {
        fscanf(params->freq, "%f", &params->freqList[i]);
        if(feof(params->freq)) {
            printf("Error: Frequency values and fits frames don't match\n");
            return 1;
        }
    }
    fscanf(params->freq, "%f", &tempFloat);
    if(! feof(params->freq)) {
        printf("Error: More frequency values present than fits frames\n\n");
        return 1;
    }
    
    /* Inaddition also compute the \lambda^2 values */
    params->lambda2 = calloc(params->qAxisLen3, sizeof(params->lambda2));
    if(params->lambda2 == NULL) {
        printf("Error: Mem alloc failed while reading in frequency list\n\n");
        return 1;
    }
    params->lambda20 = 0.0;
    for(i=0; i<params->qAxisLen3; i++)
        params->lambda2[i] = (LIGHTSPEED / params->freqList[i]) * 
                             (LIGHTSPEED / params->freqList[i]);
    
    /* Also, conpute the median lambda2 value */
    tempArray = calloc(params->qAxisLen3, sizeof(tempArray));
    for(i=0; i<params->qAxisLen3; i++) { tempArray[i] = params->lambda2[i]; }
    qsort(tempArray, params->qAxisLen3, sizeof(tempArray), compFunc);
    params->lambda20 = tempArray[params->qAxisLen3/2];
    free(tempArray);
    
    /* Close the frequency file */
    fclose(params->freq);
    
    return 0;   
}

/*************************************************************
*
* Free all alloced pointers in parList and optionsList
*
*************************************************************/
void freeStructures(struct optionsList *inOptions, struct parList *params) {
   free(inOptions->outPrefix); 
   free(inOptions->freqFileName);
   free(inOptions->qCubeName); 
   free(inOptions->uCubeName);
   
   free(params->freqList);
   free(params->lambda2);
   free(params->rmsf); 
   free(params->rmsfReal); 
   free(params->rmsfImag);
   free(params->phiAxis); 
   free(params->phiAxisDouble);
   free(params->rmsfDouble); 
   free(params->rmsfRealDouble); 
   free(params->rmsfImagDouble);
}
