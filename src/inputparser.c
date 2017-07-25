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
#include "libconfig.h"
#include "structures.h"
#include<stdlib.h>
#include<string.h>

#include "inputparser.h"

#define FITS_STR "FITS"
#define HDF5_STR "HDF5"

/*************************************************************
*
* Parse the input file and extract the relevant keywords
*
*************************************************************/
struct optionsList parseInput(char *parsetFileName) {
    config_t cfg;
    struct optionsList inOptions;
    const char *str;
    char *tempStr;
    
    /* Initialize configuration */
    config_init(&cfg);
    
    /* Read in the configuration file */
    if(!config_read_file(&cfg, parsetFileName)) {
        printf("Error: Error reading parset file. %s\n\n", 
               config_error_text(&cfg));
        config_destroy(&cfg);
        exit(FAILURE);
    }
    
    /* Get the input file format */
    if(config_lookup_string(&cfg, "fileFormat", &str)) {
        tempStr = malloc(strlen(str)+1);
        strcpy(tempStr, str);
    }
    else {
        printf("Error: 'fileFormat' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(((strcasecmp(tempStr, FITS_STR)!=SUCCESS) &&
         strcasecmp(tempStr, HDF5_STR)!=SUCCESS)) {
        printf("Error: 'fileFormat' has to be FITS or HDF5\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    } // Note strcasecmp is not standard C
    if(strcasecmp(tempStr, HDF5_STR)==SUCCESS) {
        inOptions.fileFormat = HDF5;
        printf("ERROR: HDF5 is not supported in this version\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    else { inOptions.fileFormat = FITS; }
    free(tempStr);
    
    /* Get the names of fits files */
    if(config_lookup_string(&cfg, "qCubeName", &str)) {
        inOptions.qCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.qCubeName, str);
    }
    else {
        printf("Error: 'qCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(config_lookup_string(&cfg, "uCubeName", &str)) {
        inOptions.uCubeName = malloc(strlen(str)+1);
        strcpy(inOptions.uCubeName, str);
    }
    else {
        printf("Error: 'uCubeName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    
    /* Get the name of the frequency file */
    if(config_lookup_string(&cfg, "freqFileName", &str)) {
        inOptions.freqFileName = malloc(strlen(str)+1);
        strcpy(inOptions.freqFileName, str);
    }
    else {
        printf("Error: 'freqFileName' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }

    /* Check if an image mask is defined */
    if(! config_lookup_string(&cfg, "imageMask", &str)) {
        printf("INFO: Image mask not specified\n\n");
        inOptions.isImageMaskDefined = FALSE;
    }
    else {
        inOptions.imageMask = malloc(strlen(str)+1);
        strcpy(inOptions.imageMask, str);
        inOptions.isImageMaskDefined = TRUE;
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
    
    /* Get Faraday depth */
    if(! config_lookup_float(&cfg, "phiMin", &inOptions.phiMin)) {
        printf("Error: 'phiMin' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    /* Get number of output phi planes */
    if(! config_lookup_float(&cfg, "dPhi", &inOptions.dPhi)) {
        printf("Error: 'dPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(inOptions.dPhi <= ZERO) {
       printf("Error: dPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(FAILURE);
    }
    if(! config_lookup_int(&cfg, "nPhi", &inOptions.nPhi)) {
        printf("Error: 'nPhi' undefined in parset\n\n");
        config_destroy(&cfg);
        exit(FAILURE);
    }
    if(inOptions.nPhi <= ZERO) {
       printf("Error: nPhi cannot be less than 0\n\n");
       config_destroy(&cfg);
       exit(FAILURE);
    }
    if(! config_lookup_bool(&cfg, "plotRMSF", &inOptions.plotRMSF)) {
        printf("INFO: 'plotRMSF' undefined in parset\n");
        inOptions.plotRMSF = FALSE;
    }
    if(! config_lookup_int(&cfg, "nGPU", &inOptions.nGPU)) {
        printf("INFO: 'nGPU' undefined in parset. Will use 1 device.\n");
        inOptions.nGPU = 1;
    }
    
    config_destroy(&cfg);
    return(inOptions);
}

/*************************************************************
*
* Print parsed input to screen
*
*************************************************************/
void printOptions(struct optionsList inOptions, struct parList params) {
    int i;
    
    printf("\n");
    for(i=0; i<SCREEN_WIDTH; i++) { printf("#"); }
    printf("\n");
    printf("Q Cube: %s\n", inOptions.qCubeName);
    printf("U Cube: %s\n", inOptions.uCubeName);
    if(inOptions.isImageMaskDefined == TRUE) {
        printf("Image mask: %s\n", inOptions.imageMask);
        printf("\n");
    }
    printf("phi min: %.2f\n", inOptions.phiMin);
    printf("# of phi planes: %d\n", inOptions.nPhi);
    printf("delta phi: %.2lf\n", inOptions.dPhi);
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
