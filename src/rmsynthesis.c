/******************************************************************************

A GPU Based implementation of RM Synthesis

Version: 0.1
Last edited: March 31, 2016

Version history:
================
v0.1    Assume FITS cubes have at least 3 frames with frequency being the 
         3rd axis. Also, implicitly assumes each freq channel has equal weight.

******************************************************************************/

#include<stdio.h>

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
    int i, j, k;
    long *fPixel;
    LONGLONG nElements;
    float *qImageArray, *uImageArray;
    size_t size;

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
        printf("\nINFO: Accessing the input image mask %s", inOptions.imageMask);
        fitsStatus = SUCCESS;
        fits_open_file(&params.maskFile, inOptions.imageMask, READONLY, &fitsStatus);
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
    
    /* Now that everything is set, do some memory checks */ 
    printf("\nINFO: Size of input Q/U channel: %0.3f KiB", 
      sizeof(qImageArray)*params.qAxisLen1*params.qAxisLen2/KILO);
    printf("\nINFO: Size of output Q/U cube: %0.3f MiB", 
      sizeof(qImageArray)*params.qAxisLen1*params.qAxisLen2*inOptions.nPhi/MEGA);
    printf("\nINFO: Available memory on GPU: %0.3f MiB", 
      gpuList[selectedDevice].globalMem/MEGA);
    if(sizeof(qImageArray)*params.qAxisLen1*params.qAxisLen2*inOptions.nPhi >
      gpuList[selectedDevice].globalMem) {
        printf("\nERROR: Insufficient memory on device! Try reducing nPhi\n\n");
        return(FAILURE);
    }
    printf("\nINFO: Starting RM Synthesis");
    printf("\nINFO: Processing channel by channel");
    doRMSynthesis(&inOptions, &params);

    /* Free up all allocated memory */
    free(gpuList);

    printf("\n\n");
    return(SUCCESS);
}
