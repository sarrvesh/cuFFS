/******************************************************************************
devices.cu
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
extern "C" {
#include<cuda_runtime.h>
#include<cuda.h>
#include "structures.h"
#include "constants.h"
#include "devices.h"
#include "fileaccess.h"
__global__ void computeQ(float *d_qImageArray, float *d_uImageArray, 
                         float *d_qPhi, float *d_phiAxis, int nPhi, 
                         int nElements, float dlambda2);
__global__ void computeU(float *d_qImageArray, float *d_uImageArray, 
                         float *d_uPhi, float *d_phiAxis, int nPhi, 
                         int nElements, float dlambda2);
__global__ void initializeQUP(float *d_qPhi, float *d_uPhi, 
                              float *d_pPhi, int nPhi);
__global__ void computeP(float *d_qPhi, float *d_uPhi, float *d_pPhi);
void getGpuAllocForRMSynth(int *blockSize, int *threadSize, int nPhi,
                           struct deviceInfoList selectedDeviceInfo);
}

/*************************************************************
*
* Check if CUDA ERROR flag has been set. If raised, print 
*   error message to stdout and exit.
*
*************************************************************/
extern "C"
void checkCudaError() {
    cudaError_t errorID = cudaGetLastError();
    if(errorID != cudaSuccess) {
        printf("\nERROR: %s", cudaGetErrorString(errorID));
        exit(FAILURE);
    }
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
    int deviceCount = NO_DEVICE;
    struct cudaDeviceProp deviceProp;
    struct deviceInfoList *gpuList;
    
    /* Check for valid devices */
    cudaDeviceReset();
    cudaGetDeviceCount(&deviceCount);
    checkCudaError();
    if(deviceCount == NO_DEVICE) {
        printf("\nError: Could not detect CUDA supported GPU(s)\n\n");
        exit(FAILURE);
    }
    printf("\nINFO: Detected %d CUDA-supported GPU(s)\n", deviceCount);
    *nDevices = deviceCount;

    /* Store useful information about each GPU in a structure array */
    gpuList = (deviceInfoList *)malloc(deviceCount * 
      sizeof(struct deviceInfoList));
    for(dev=0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaGetDeviceProperties(&deviceProp, dev);
        checkCudaError();
        gpuList[dev].deviceID    = dev;
        gpuList[dev].globalMem   = deviceProp.totalGlobalMem;
        gpuList[dev].constantMem = deviceProp.totalConstMem;
        gpuList[dev].sharedMemPerBlock = deviceProp.sharedMemPerBlock;
        gpuList[dev].maxThreadPerMP = deviceProp.maxThreadsPerMultiProcessor;
        gpuList[dev].maxThreadPerBlock = deviceProp.maxThreadsPerBlock;
        gpuList[dev].threadBlockSize[0] = deviceProp.maxThreadsDim[0];
        gpuList[dev].threadBlockSize[1] = deviceProp.maxThreadsDim[1];
        gpuList[dev].threadBlockSize[2] = deviceProp.maxThreadsDim[2];
        gpuList[dev].warpSize           = deviceProp.warpSize;
        gpuList[dev].nSM                = deviceProp.multiProcessorCount;
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
* Copy GPU device information of selectedDevice from gpuList 
*  to selectedDevice
*
*************************************************************/
extern "C"
struct deviceInfoList copySelectedDeviceInfo(struct deviceInfoList *gpuList, 
                                             int selectedDevice) {
    int i = selectedDevice;
    struct deviceInfoList selectedDeviceInfo;
    selectedDeviceInfo.deviceID           = gpuList[i].deviceID;
    selectedDeviceInfo.globalMem          = gpuList[i].globalMem;
    selectedDeviceInfo.constantMem        = gpuList[i].constantMem;
    selectedDeviceInfo.sharedMemPerBlock  = gpuList[i].sharedMemPerBlock;
    selectedDeviceInfo.maxThreadPerMP     = gpuList[i].maxThreadPerMP;
    selectedDeviceInfo.maxThreadPerBlock  = gpuList[i].maxThreadPerBlock;
    selectedDeviceInfo.threadBlockSize[0] = gpuList[i].threadBlockSize[0];
    selectedDeviceInfo.threadBlockSize[1] = gpuList[i].threadBlockSize[1];
    selectedDeviceInfo.threadBlockSize[2] = gpuList[i].threadBlockSize[2];
    selectedDeviceInfo.warpSize           = gpuList[i].warpSize;
    selectedDeviceInfo.nSM                = gpuList[i].nSM;
    return selectedDeviceInfo;
}

/*************************************************************
*
* GPU accelerated RM Synthesis function
*
*************************************************************/
extern "C"
int doRMSynthesis(struct optionsList *inOptions, struct parList *params,
                  struct deviceInfoList selectedDeviceInfo) {
    int i, j; 
    float *lambdaDiff2, *d_lambdaDiff2;
    size_t size;
    float *qImageArray, *uImageArray;
    float *d_qImageArray, *d_uImageArray;
    float *d_qPhi, *d_uPhi, *d_pPhi;
    float *d_phiAxis;
    int initThreadSize, initBlockSize;
    int calcThreadSize, calcBlockSize;
    cudaEvent_t startEvent, stopEvent;
    float millisec = 0.;
    long inc[] = {1,1,1};
    long fPixel[params->qAxisLen3], lPixel[params->qAxisLen3];
    int fitsStatus = 0;
    
    /* Initialize CUDA events to measure time */
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    /* Compute \lambda^2 - \lambda^2_0 once */
    lambdaDiff2 = (float *)calloc(params->qAxisLen3, sizeof(lambdaDiff2));
    if(lambdaDiff2 == NULL) {
        printf("ERROR: Mem alloc failed for lambdaDiff2\n\n");
        return(FAILURE);
    }
    for(i=0;i<params->qAxisLen3;i++)
        lambdaDiff2[i] = 2.0*(params->lambda2[i]-params->lambda20);
    
    /* Allocate input arrays on CPU */
    qImageArray = (float *)calloc(params->qAxisLen3, sizeof(qImageArray));
    uImageArray = (float *)calloc(params->qAxisLen3, sizeof(uImageArray));
    /* Allocate and initialize input arrays on GPU */
    size = sizeof(d_qImageArray)*params->qAxisLen3;
    cudaMalloc(&d_qImageArray, size);
    cudaMalloc(&d_uImageArray, size);
    cudaMalloc(&d_lambdaDiff2, size);
    cudaMemcpy(d_lambdaDiff2, lambdaDiff2, size, cudaMemcpyHostToDevice);
    /* Allocate and initialize output arrays on GPU */
    size = sizeof(d_qPhi)*inOptions->nPhi;
    cudaMalloc(&d_qPhi, size); 
    cudaMalloc(&d_uPhi, size);
    cudaMalloc(&d_pPhi, size);
    cudaMalloc(&d_phiAxis, size);
    cudaMemcpy(d_phiAxis, params->phiAxis, size, cudaMemcpyHostToDevice);
    checkCudaError();

    /* Start the clock */
    cudaEventRecord(startEvent);

    /* Determine what the appropriate block and grid sizes are */
    initThreadSize = selectedDeviceInfo.warpSize;
    initBlockSize  = inOptions->nPhi/initThreadSize + 1;

    /* Since we are reading the entire frequency axis at once, 
       fPixel[2] = 1 and lPixel[2] = {NAXIS3}+1 */
    fPixel[2] = 1; lPixel[2] = params->qAxisLen3;
    
    /* Process each line of sight individually */
    size = sizeof(d_qImageArray)*params->qAxisLen3;
    for(i=1; i<=params->qAxisLen1; i++) {
        fPixel[1] = i; lPixel[1] = i;
        for(j=1; j<=params->qAxisLen2; j++) {
            printf("INFO: Processing i=%d j=%d\n", i, j);
            fPixel[0] = j; lPixel[0] = j;
            
            /* Set Q/U/P accumulator output arrays on GPU to 0 */
            initializeQUP<<<initBlockSize, initThreadSize>>>(d_qPhi, d_uPhi,
                                                   d_pPhi, inOptions->nPhi);
    
            /* Read this line of sight from Q and U array */
            fits_read_subset(params->qFile, TFLOAT, fPixel, lPixel, inc, 
                             NULL, qImageArray, NULL, &fitsStatus);
            /*fits_read_subset(params->uFile, TFLOAT, fPixel, lPixel, inc, 
                             NULL, &uImageArray, NULL, &fitsStatus);
            checkFitsError(fitsStatus);*/
            
            /* Move Q(lambda) and U(lambda) to device */
            /*cudaMemcpy(d_qImageArray, qImageArray, size,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_uImageArray, uImageArray, size,
                       cudaMemcpyHostToDevice);*/
            
            /* Launch kernels to compute Q(\phi), U(\phi), and P(\phi) */
            
            /* Move Q(\phi), U(\phi) and P(\phi) to host */
            checkCudaError();
        }
    }
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&millisec, startEvent, stopEvent);
    printf("INFO: Time to process the cubes: %0.2f ms.\n", millisec);
    
    return(SUCCESS);
}

/*************************************************************
*
* Device code to compute Q(\phi)
*
*************************************************************/
extern "C"
__global__ void computeQ(float *d_qImageArray, float *d_uImageArray, 
                         float *d_qPhi, float *d_phiAxis, int nPhi, 
                         int nElements, float dlambda2) {
    int i;
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int offset = index*nElements;
    const float thisPhi = d_phiAxis[index];    
    const float thisCos = cosf(2*thisPhi*dlambda2);
    const float thisSin = sinf(2*thisPhi*dlambda2);

    if(index < nPhi) {
        /* For each element in Q, compute Q(thisPhi) and add it to Q(phi) */
        for(i=0; i<nElements; i++)
            d_qPhi[offset+i] += d_qImageArray[i]*thisCos - 
                                d_uImageArray[i]*thisSin;
    }
}

/*************************************************************
*
* Device code to compute U(\phi)
*
*************************************************************/
extern "C"
__global__ void computeU(float *d_qImageArray, float *d_uImageArray, 
                         float *d_uPhi, float *d_phiAxis, int nPhi, 
                         int nElements, float dlambda2) {
    int i;
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int offset = index*nElements;
    const float thisPhi = d_phiAxis[index];
    const float thisCos = cosf(2*thisPhi*dlambda2);
    const float thisSin = sinf(2*thisPhi*dlambda2);

    if(index < nPhi) {
        /* For each element in U, compute U(thisPhi) and add it to U(phi) */
        for(i=0; i<nElements; i++) 
            d_uPhi[offset+i] += d_uImageArray[i]*thisCos - 
                                d_qImageArray[i]*thisSin;
    }
}

/*************************************************************
*
* Initialize Q(\phi) and U(\phi)
*
*************************************************************/
extern "C"
__global__ void initializeQUP(float *d_qPhi, float *d_uPhi, 
                              float *d_pPhi, int nPhi) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index < nPhi) {
        d_qPhi[index] = 0.0;
        d_uPhi[index] = 0.0;
        d_pPhi[index] = 0.0;
    }
}

/*************************************************************
*
* Estimate the optimal number of block and thread size 
*  for RM Synthesis.
*
*************************************************************/
extern "C"
void getGpuAllocForRMSynth(int *blockSize, int *threadSize, int nPhi,
                           struct deviceInfoList selectedDeviceInfo) {
    *threadSize = selectedDeviceInfo.warpSize;
    if(!(*blockSize = nPhi/(*threadSize))) { *blockSize = 1; }
}

/*************************************************************
*
* Estimate the optimal number of block, thread and memory to
*  compute P from Q and U images/cubes.
*
*************************************************************/
extern "C"
void getGpuAllocForP(int *blockSize, int *threadSize, long *nFrames, 
                     int nImRows, int nRowElements, 
                     struct deviceInfoList selectedDeviceInfo) {
    long totalThreads;

    /* How many phi frames can be stored in gpu at a time */
    *nFrames = (int)(selectedDeviceInfo.globalMem % 
              (3*nImRows*nRowElements*sizeof(float)));
    
    /* Determine the thread and block size */
    totalThreads = *nFrames;
    if(totalThreads <= selectedDeviceInfo.maxThreadPerBlock) {
        *threadSize = totalThreads;
        *blockSize = 1;
    }
    else {
        *threadSize = selectedDeviceInfo.maxThreadPerBlock;
        *blockSize = (totalThreads % selectedDeviceInfo.maxThreadPerBlock) + 1;
    }
}
