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
#include<string.h>

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
    int cpuTime, hours, mins, secs;
    
    /* Start the clock */
    startTime = clock();
    
    printf("\n");
    printf("RM Synthesis v%s\n", VERSION_STR);
    printf("Written by Sarrvesh S. Sridhar\n");
    
    /* Verify command line input */
    if(argc!=NUM_INPUTS) {
        printf("ERROR: Invalid command line input. Terminating Execution!\n");
        printf("Usage: %s <parset filename>\n\n", argv[0]);
        return(FAILURE);
    } 
    if(strcmp(parsetFileName, "-h") == 0) {
        /* Print help and exit */
        printf("Usage: %s <parset filename>\n\n", argv[0]);
        return(SUCCESS);
    }
    
    /* Parse the input file */
    printf("INFO: Parsing input file %s\n", parsetFileName);
    inOptions = parseInput(parsetFileName);
    printOptions(inOptions, params);
    
    /* Check input files */
    printf("INFO: Checking input files\n");
    checkInputFiles(&inOptions, &params);
    
    /* Retreive information about all connected GPU devices */
    /* Find the best device to use */
    gpuList = getDeviceInformation(&nDevices);
    selectedDevice = getBestDevice(gpuList, nDevices);
    cudaSetDevice(selectedDevice);
    checkCudaError();
    /* Copy the device info for the best device */
    selectedDeviceInfo = copySelectedDeviceInfo(gpuList, selectedDevice);
    free(gpuList);
    
    /* Gather information from input fits header and setup output images */
    if(inOptions.fileFormat) {
       fitsStatus = getFitsHeader(&inOptions, &params);
       checkFitsError(fitsStatus);
       makeOutputFitsImages(&inOptions, &params);
    }
    else { /* Gather information from input hdf5 images */ }
    
    /* Read frequency list */
    if(getFreqList(&inOptions, &params)) { return(FAILURE); }
    
    /* Find median lambda20 */
    getMedianLambda20(&params);
    
    /* Generate RMSF */
    printf("INFO: Computing RMSF\n");
    if(generateRMSF(&inOptions, &params)) {
        printf("Error: Mem alloc failed while generating RMSF\n");
        return(FAILURE);
    }
    /* Write RMSF to disk */
    if(writeRMSF(inOptions, params)) {
        printf("Error: Unable to write RMSF to disk\n\n");
        return(FAILURE);
    }
    /* Plot RMSF */
    #ifdef GNUPLOT_ENABLE
    if(inOptions.plotRMSF == TRUE) {
        printf("INFO: Plotting RMSF with gnuplot\n");
        if(plotRMSF(inOptions)) {
            printf("Error: Unable to plot RMSF\n\n");
            return(FAILURE);
        }
    }
    #endif
    
    /* Start RM Synthesis */
    printf("INFO: Starting RM Synthesis\n");
    doRMSynthesis(&inOptions, &params, selectedDeviceInfo);

    /* Free up all allocated memory */
    free(params.rmsf);
    free(params.rmsfReal);
    free(params.rmsfImag);
    free(params.phiAxis);
    free(params.freqList);
    free(params.lambda2);
    free(inOptions.qCubeName);
    free(inOptions.uCubeName);
    free(inOptions.freqFileName);
    free(inOptions.outPrefix);
    if(inOptions.isImageMaskDefined == TRUE) { free(inOptions.imageMask); }

    /* Close all open files */
    fits_close_file(params.qFile, &fitsStatus);
    fits_close_file(params.uFile, &fitsStatus);
    fits_close_file(params.qDirty, &fitsStatus);
    fits_close_file(params.uDirty, &fitsStatus);
    fits_close_file(params.pDirty, &fitsStatus);
    if(inOptions.isImageMaskDefined == TRUE) { fits_close_file(params.maskFile, &fitsStatus); }
    checkFitsError(fitsStatus);
    
    /* Estimate the execution time */
    endTime = clock();
    cpuTime = (int)(endTime - startTime)/CLOCKS_PER_SEC;
    hours   = (int)cpuTime/SEC_PER_HOUR;
    mins    = (int)(cpuTime%SEC_PER_HOUR)/SEC_PER_MIN;
    secs    = (int)(cpuTime%SEC_PER_HOUR)%SEC_PER_MIN;
    printf("INFO: Total CPU time: %d:%d:%d\n", hours, mins, secs);
    printf("\n");
    return(SUCCESS);
}
