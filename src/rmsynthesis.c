/******************************************************************************
rmsynthesis.c: A GPU based implementation of RM Synthesis.
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
#include<time.h>

#include "structures.h"
#include "constants.h"
#include "version.h"
#include "devices.h"
#include "fileaccess.h"
#include "inputparser.h"
#include "rmsf.h"

/*************************************************************
*
* Main code
*
*************************************************************/
int main(int argc, char *argv[]) {
    /* Host Variable declaration */
    char *parsetFileName = argv[1];
    struct optionsList inOptions;
    struct parList params;
    int fitsStatus;
    int status;
    int nDevices;
    int selectedDevice;
    struct deviceInfoList *gpuList;
    struct deviceInfoList selectedDeviceInfo;
    int i, j, k;
    long *fPixel;
    LONGLONG nElements;
    float *qImageArray, *uImageArray;
    size_t size;
    clock_t startTime, endTime;
    double cpuTime;
    
    /* Start the clock */
    startTime = clock();

    /* Device variable declaration */
    float *d_qImageArray, *d_uImageArray;
    
    printf("\nRM Synthesis v%s", VERSION_STR);
    printf("\nWritten by Sarrvesh S. Sridhar\n");
    
    /* Verify command line input */
    if(argc!=NUM_INPUTS) {
        printf("\nERROR: Invalid command line input. Terminating Execution!");
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        return(FAILURE);
    } 
    if(strcmp(parsetFileName, "-h") == 0) {
        /* Print help and exit */
        printf("\nUsage: %s <parset filename>\n\n", argv[0]);
        return(SUCCESS);
    }
    
    /* Parse the input file */
    printf("\nINFO: Parsing input file %s", parsetFileName);
    inOptions = parseInput(parsetFileName);
    
    /* Print parset input options to screen */
    printOptions(inOptions);
    
    /* Retreive information about all connected GPU devices */
    gpuList = getDeviceInformation(&nDevices);

    /* Select the best device */
    selectedDevice = getBestDevice(gpuList, nDevices);
    cudaSetDevice(selectedDevice);
    checkCudaError();

    /* Copy the device info for the best device */
    selectedDeviceInfo = copySelectedDeviceInfo(gpuList, selectedDevice);
    free(gpuList);
    
    /* Open the input files */
    printf("\nINFO: Accessing the input files");
    printf("\nWARN: Assuming NAXIS3 is the frequency axis");
    fitsStatus = SUCCESS;
    puts(inOptions.qCubeName);
    fits_open_file(&params.qFile, inOptions.qCubeName, READONLY, &fitsStatus);
    fits_open_file(&params.uFile, inOptions.uCubeName, READONLY, &fitsStatus);
    checkFitsError(fitsStatus);
    params.freq = fopen(inOptions.freqFileName, FILE_READONLY);
    if(params.freq == NULL) {
        printf("\nError: Unable to open the frequency file\n\n");
        return(FAILURE);
    }
    if(inOptions.isImageMaskDefined == TRUE) {
        printf("\nINFO: Accessing the input mask %s", inOptions.imageMask);
        fitsStatus = SUCCESS;
        fits_open_file(&params.maskFile, inOptions.imageMask, READONLY, 
                       &fitsStatus);
        checkFitsError(status);
    }
    
    /* Gather information from input image fits header */
    status = getFitsHeader(&inOptions, &params);
    checkFitsError(fitsStatus);
    
    /* Read frequency list */
    if(getFreqList(&inOptions, &params))
        return(FAILURE);
    
    /* Find median lambda20 */
    getMedianLambda20(&params);
    
    /* Generate RMSF */
    printf("\nINFO: Computing RMSF");
    if(generateRMSF(&inOptions, &params)) {
        printf("\nError: Mem alloc failed while generating RMSF");
        return(FAILURE);
    }    
    /* Write RMSF to disk */
    if(writeRMSF(inOptions, params)) {
        printf("\nError: Unable to write RMSF to disk\n\n");
        return(FAILURE);
    }
    /* Plot RMSF */
    #ifdef GNUPLOT_ENABLE
    if(inOptions.plotRMSF == TRUE) {
        printf("\nINFO: Plotting RMSF with gnuplot");
        if(plotRMSF(inOptions)) {
            printf("\nError: Unable to plot RMSF\n\n");
            return(FAILURE);
        }
    }
    #endif
    
    /* Start RM Synthesis */
    printf("\nINFO: Starting RM Synthesis");
    doRMSynthesis(&inOptions, &params, selectedDeviceInfo);

    /* Free up all allocated memory */
    
    /* Estimate the execution time */
    endTime = clock();
    cpuTime = (double)(endTime - startTime)/CLOCKS_PER_SEC;
    printf("\nINFO: Total CPU time: %lf minutes", cpuTime/SEC_PER_MIN);
    printf("\n\n");
    return(SUCCESS);
}
