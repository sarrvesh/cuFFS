extern "C" {
#include<cuda_runtime.h>
#include<cuda.h>
#include "structures.h"
#include "constants.h"
#include "devices.h"
}

/*************************************************************
*
* Check for valid CUDA supported devices. If detected, 
*  print useful device information
*
*************************************************************/
extern "C"
struct deviceInfoList * getDeviceInformation(int *nDevices) {
    int dev;
    cudaError_t errorID;
    int deviceCount = NO_DEVICE;
    struct cudaDeviceProp deviceProp;
    struct deviceInfoList *gpuList;
    
    /* Check for valid devices */
    cudaDeviceReset();
    errorID = cudaGetDeviceCount(&deviceCount);
    if(errorID != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n%s", (int)errorID, 
               cudaGetErrorString(errorID));
        exit(FAILURE);
    }
    if(deviceCount == NO_DEVICE) {
        printf("\nError: Could not detect CUDA supported GPU(s)\n\n");
        exit(FAILURE);
    }
    printf("\nINFO: Detected %d CUDA-supported GPU(s)\n", deviceCount);
    *nDevices = deviceCount;

    /* Store useful information about each GPU in a structure array */
    gpuList = (deviceInfoList *)malloc(deviceCount * sizeof(struct deviceInfoList));
    for(dev=0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaGetDeviceProperties(&deviceProp, dev);
        gpuList[dev].deviceID    = dev;
        gpuList[dev].globalMem   = deviceProp.totalGlobalMem;
        gpuList[dev].constantMem = deviceProp.totalConstMem;
        gpuList[dev].sharedMemPerBlock = deviceProp.sharedMemPerBlock;
        gpuList[dev].maxThreadPerMP = deviceProp.maxThreadsPerMultiProcessor;
        gpuList[dev].maxThreadPerBlock = deviceProp.maxThreadsPerBlock;
        gpuList[dev].threadBlockSize[0] = deviceProp.maxThreadsDim[0];
        gpuList[dev].threadBlockSize[1] = deviceProp.maxThreadsDim[1];
        gpuList[dev].threadBlockSize[2] = deviceProp.maxThreadsDim[2];
        /* Print device info */
        printf("\nDevice %d: %s (version: %d.%d)", dev, deviceProp.name, 
                                                        deviceProp.major, 
                                                        deviceProp.minor);
        printf("\n\tGlobal memory: %f MB", gpuList[dev].globalMem/MEGA);
        printf("\n\tShared memory: %f kB", gpuList[dev].sharedMemPerBlock/KILO);
        printf("\n\tMax threads per block: %d", gpuList[dev].maxThreadPerBlock);
        printf("\n\tMax threads per MP: %d", gpuList[dev].maxThreadPerMP);
        printf("\n\tProcessor count: %d", deviceProp.multiProcessorCount);
        printf("\n\tMax thread dim: (%d, %d, %d)", deviceProp.maxThreadsDim[0], 
                                                   deviceProp.maxThreadsDim[1], 
                                                   deviceProp.maxThreadsDim[2]);
    }
    printf("\n");
    return(gpuList);
}

/*************************************************************
*
* Select the best GPU device
*
*************************************************************/
extern "C"
int getBestDevice(struct deviceInfoList *gpuList, int nDevices) {
    int dev=0;
    int i, maxMem;
    if(nDevices == 1) { dev = 0; }
    else {
        maxMem = gpuList[dev].globalMem;
        for(i=1; i<nDevices; i++) {
            if(maxMem < gpuList[i].globalMem) { 
                maxMem = gpuList[i].globalMem;
                dev = i;
            }
            else { continue; }
        }
    }
    return dev;
}

/*************************************************************
*
* GPU accelerated RM Synthesis function
*
*************************************************************/
extern "C"
int doRMSynthesis(struct optionsList *inOptions, struct parList *params) {
    long *fPixel;
    LONGLONG nElements;
    float *qImageArray, *uImageArray;
    float *d_qImageArray, *d_uImageArray;
    int i, j;
    size_t size;
    int status = 0;
    
    /* First setup fitsio access variables */
    fPixel = (long *)calloc(params->qAxisNum, sizeof(fPixel));
    for(i=1; i<=params->qAxisNum; i++) { fPixel[i-1] = 1; }
    nElements = params->qAxisLen1 * params->qAxisLen2;
    qImageArray = (float *)calloc(nElements, sizeof(qImageArray));
    uImageArray = (float *)calloc(nElements, sizeof(uImageArray));
    for(j=1; j<=params->qAxisLen3; j++) {
       /* Read in this Q and U channel */
       fPixel[2] = j;
       fits_read_pix(params->qFile, TFLOAT, fPixel, nElements, NULL, qImageArray, NULL, &status);
       fits_read_pix(params->uFile, TFLOAT, fPixel, nElements, NULL, uImageArray, NULL, &status);
       /* Copy the read in channel maps to GPU */
       size = nElements*sizeof(d_qImageArray);
       cudaMalloc(&d_qImageArray, size);
       cudaMalloc(&d_uImageArray, size);
       cudaMemcpy(d_qImageArray, qImageArray, size, cudaMemcpyHostToDevice);
       cudaMemcpy(d_uImageArray, uImageArray, size, cudaMemcpyHostToDevice);
       /* Launch kernels to do RM Synthesis */
       /* Copy the results to host */
       /* Free the allocated device memory */
       cudaFree(d_qImageArray);
       cudaFree(d_uImageArray);
    }
    if(status) {
        fits_report_error(stdout, status);
        return(FAILURE);
    }
    free(fPixel);
    free(qImageArray);
    free(uImageArray);

    return(SUCCESS);
}
