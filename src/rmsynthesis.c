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
    struct timeInfoList t;
    unsigned int hours, mins, secs;
    
    /* Initialize the clock variables */
    t.cpuTime = 0; t.msRead = 0;
    t.msWrite = 0; t.msProc = 0;
    t.msX = 0;
    
    /* Start the clock */
    t.startTime = clock();
    
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
    
    /* Check input files */
    printf("INFO: Checking input files\n");
    checkInputFiles(&inOptions, &params);
    
    /* Retreive information about all connected GPU devices */
    /* Find the best device to use */
    gpuList = getDeviceInformation(&nDevices);
    selectedDevice = getBestDevice(gpuList, nDevices);
    printf("INFO: Selected device %d\n", selectedDevice);
    cudaSetDevice(selectedDevice);
    checkCudaError();
    /* Copy the device info for the best device */
    selectedDeviceInfo = copySelectedDeviceInfo(gpuList, selectedDevice);
    free(gpuList);
    
    /* Gather information from input fits header and setup output images */
    t.startRead = clock();
    switch(inOptions.fileFormat) {
       case FITS:
          fitsStatus = getFitsHeader(&inOptions, &params);
          checkFitsError(fitsStatus);
          makeOutputFitsImages(&inOptions, &params);
          break;
       case HDF5:
          getHDF5Header(&inOptions, &params);
          makeOutputHDF5Images(&inOptions, &params);
          break;
       default:
          // Control should never reach this point.
          printf("ERROR: Encountered an unknown error.\n");
          printf("ERROR: Contact Sarrvesh if you see this.");
          exit(FAILURE);
          break;
    }
    t.stopRead = clock();
    t.msRead += ((unsigned int)(t.stopRead - t.startRead))/CLOCKS_PER_SEC;

    /* Print some useful information */
    t.startWrite = clock();
    printOptions(inOptions, params);
    t.stopWrite = clock();
    t.msWrite += ((unsigned int)(t.stopWrite - t.startWrite))/CLOCKS_PER_SEC;
    
    /* Read frequency list */
    t.startRead = clock();
    if(getFreqList(&inOptions, &params)) { return(FAILURE); }
    t.stopRead = clock();
    t.msRead += ((unsigned int)(t.stopRead - t.startRead))/CLOCKS_PER_SEC;
    
    /* Find median lambda20 */
    t.startProc = clock();
    getMedianLambda20(&params);
    
    /* Generate RMSF */
    printf("INFO: Computing RMSF\n");
    if(generateRMSF(&inOptions, &params)) {
        printf("Error: Mem alloc failed while generating RMSF\n");
        return(FAILURE);
    }
    t.stopProc = clock();
    t.msProc += ((unsigned int)(t.stopProc - t.startProc))/CLOCKS_PER_SEC;
    
    /* Write RMSF to disk */
    t.startWrite = clock();
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
    t.stopWrite = clock();
    t.msWrite += ((unsigned int)(t.stopWrite - t.startWrite))/CLOCKS_PER_SEC;
    
    /* Start RM Synthesis */
    printf("INFO: Starting RM Synthesis\n");

    doRMSynthesis(&inOptions, &params, selectedDeviceInfo, &t);

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

    /* Close all open files */
    switch(inOptions.fileFormat) {
       case FITS:
          fits_close_file(params.qFile, &fitsStatus);
          fits_close_file(params.uFile, &fitsStatus);
          fits_close_file(params.qDirty, &fitsStatus);
          fits_close_file(params.uDirty, &fitsStatus);
          fits_close_file(params.pDirty, &fitsStatus);
          checkFitsError(fitsStatus);
          break;
       case HDF5:
          H5Fclose(params.qFileh5);
          H5Fclose(params.uFileh5);
          break;
       default:
          // Control should never reach this point
          printf("ERROR: Encountered an unknown error.\n");
          printf("ERROR: Contact Sarrvesh if you see this.");
          exit(FAILURE);
    }
    
    /* Estimate the execution time */
    t.endTime = clock();
    t.cpuTime = (unsigned int)(t.endTime - t.startTime)/CLOCKS_PER_SEC;
    hours   = (unsigned int)t.cpuTime/SEC_PER_HOUR;
    mins    = (unsigned int)(t.cpuTime%SEC_PER_HOUR)/SEC_PER_MIN;
    secs    = (unsigned int)(t.cpuTime%SEC_PER_HOUR)%SEC_PER_MIN;
    /* Write timing information to stdout */
    printf("INFO: Timing Information\n");
    printf("   Input read time: %0.3f s\n", t.msRead);
    printf("   Compute time: %0.3f s\n", t.msProc);
    printf("   Output write time: %0.3f s\n", t.msWrite);
    printf("   D2H Transfer time: %0.3f s\n", t.msX);
    printf("INFO: Total execution time: %d:%d:%d\n", hours, mins, secs);
    printf("\n");
    cudaDeviceReset();
    return(SUCCESS);
}
