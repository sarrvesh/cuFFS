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
__global__ void computeQUP(float *d_qImageArray, float *d_uImageArray, int nLOS, 
                           int nChan, float K, float *d_qPhi, float *d_uPhi, 
                           float *d_pPhi, float *d_phiAxis, int nPhi, 
                           float *d_lambdaDiff2);
}

/*************************************************************
*
* Enable host memory mapping 
*
*************************************************************/
extern "C"
void setMemMapFlag() {
    /* Enable host memory mapping */
    cudaSetDeviceFlags(cudaDeviceMapHost);
    checkCudaError();
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
    int i, j, k; 
    int inputIdx, outputIdx;
    float *lambdaDiff2, *d_lambdaDiff2;
    size_t size;
    float *qImageArray, *uImageArray;
    float *d_qImageArray, *d_uImageArray;
    float *d_qPhi, *d_uPhi, *d_pPhi;
    float *qPhi, *uPhi, *pPhi;
    float *d_phiAxis;
    dim3 calcThreadSize, calcBlockSize;
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t tStart, tStop;
    float millisec = 0., totMilliSec = 0.;
    long fPixel[params->qAxisLen3];
    int fitsStatus = 0;
    
    /* Initialize CUDA events to measure time */
    cudaEventCreate(&startEvent); cudaEventCreate(&tStart);
    cudaEventCreate(&stopEvent);  cudaEventCreate(&tStop);

    /* Compute \lambda^2 - \lambda^2_0 once. Common for all threads */
    lambdaDiff2 = (float *)calloc(params->qAxisLen3, sizeof(lambdaDiff2));
    if(lambdaDiff2 == NULL) {
        printf("ERROR: Mem alloc failed for lambdaDiff2\n\n");
        return(FAILURE);
    }
    for(i=0;i<params->qAxisLen3;i++)
        lambdaDiff2[i] = 2.0*(params->lambda2[i]-params->lambda20);
    size = sizeof(d_lambdaDiff2)*params->qAxisLen3;
    cudaMalloc(&d_lambdaDiff2, size);
    cudaMemcpy(d_lambdaDiff2, lambdaDiff2, size, cudaMemcpyHostToDevice);
    
    /* Allocate input arrays on CPU */
    size = params->qAxisLen3*params->qAxisLen2*sizeof(qImageArray);
    cudaHostAlloc( (void**)&qImageArray, size,
                 cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc( (void**)&uImageArray, size,
                 cudaHostAllocWriteCombined | cudaHostAllocMapped);
    /* Get pointers to input arrays on GPU */
    cudaHostGetDevicePointer(&d_qImageArray, qImageArray, 0);
    cudaHostGetDevicePointer(&d_uImageArray, uImageArray, 0);
    /* Allocate output arrays on CPU */
    size = sizeof(qPhi)*inOptions->nPhi*params->qAxisLen2;
    cudaHostAlloc( (void**)&qPhi, size, cudaHostAllocMapped);
    cudaHostAlloc( (void**)&uPhi, size, cudaHostAllocMapped);
    cudaHostAlloc( (void**)&pPhi, size, cudaHostAllocMapped);
    /* Get pointers to output arrays on GPU */
    cudaHostGetDevicePointer(&d_qPhi, qPhi, 0);
    cudaHostGetDevicePointer(&d_uPhi, uPhi, 0);
    cudaHostGetDevicePointer(&d_pPhi, pPhi, 0);
    checkCudaError();
    
    /* Allocate and transfer phi axis info. Common for all threads */
    cudaMalloc(&d_phiAxis, size);
    cudaMemcpy(d_phiAxis, params->phiAxis, size, cudaMemcpyHostToDevice);
    checkCudaError();

    /* Start the clock */
    cudaEventRecord(startEvent);

    /* Determine what the appropriate block and grid sizes are */
    calcThreadSize.x = selectedDeviceInfo.warpSize;
    calcBlockSize.y  = params->qAxisLen2;
    calcBlockSize.x  = inOptions->nPhi/calcThreadSize.x + 1;
    printf("INFO: Launching %dx%d blocks each with %d threads\n", 
            calcBlockSize.x, calcBlockSize.y, calcThreadSize.x);

    /* Process each line of sight individually */
    for(i=1; i<=params->qAxisLen1; i++) {
        fPixel[1] = i;
        /* Read all lines of sight in each row separately */
        cudaEventRecord(tStart);
        for(j=1; j<=params->qAxisLen2; j++) {
            fPixel[0] = j;
            for(k=1; k<=params->qAxisLen3; k++) {
               fPixel[2] = k;
               inputIdx = (j-1)*params->qAxisLen2 + (k-1);
               fits_read_pix(params->qFile, TFLOAT, fPixel, 1, NULL, 
                             &(qImageArray[inputIdx]), NULL, &fitsStatus);
               fits_read_pix(params->uFile, TFLOAT, fPixel, 1, NULL,
                             &(uImageArray[inputIdx]), NULL, &fitsStatus);
               checkFitsError(fitsStatus);
            }
        }
        cudaEventRecord(tStop);
        cudaEventSynchronize(tStop);
        cudaEventElapsedTime(&millisec, tStart, tStop);
        printf("INFO: %0.2f ms to read fits data\n", millisec);
            
        /* Launch kernels to compute Q(\phi), U(\phi), and P(\phi) */
        cudaEventRecord(tStart);
        computeQUP<<<calcBlockSize, calcThreadSize>>>(d_qImageArray, d_uImageArray, 
                         params->qAxisLen2, params->qAxisLen3, params->K, d_qPhi,
                         d_uPhi, d_pPhi, d_phiAxis, inOptions->nPhi, d_lambdaDiff2);
        cudaThreadSynchronize();
        cudaEventRecord(tStop);
        cudaEventSynchronize(tStop);
        cudaEventElapsedTime(&millisec, tStart, tStop);
        totMilliSec += millisec;
        printf("INFO: Took %0.2f ms to process a row\n", millisec);

        /* Move Q(\phi), U(\phi) and P(\phi) to host */
    }
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&millisec, startEvent, stopEvent);
    printf("INFO: Time to process the cubes: %0.2f ms.\n", millisec);
    printf("INFO: Time spent on GPU compute: %0.2f ms.\n", totMilliSec);

    /* Free all the allocated memory */
    cudaFreeHost(qImageArray); cudaFreeHost(uImageArray);
    
    return(SUCCESS);
}

/*************************************************************
*
* Device code to compute Q(\phi)
*
*************************************************************/
extern "C"
__global__ void computeQUP(float *d_qImageArray, float *d_uImageArray, int nLOS, 
                           int nChan, float K, float *d_qPhi, float *d_uPhi, 
                           float *d_pPhi, float *d_phiAxis, int nPhi, 
                           float *d_lambdaDiff2) {
    int i;
    float myphi;
    /* xIndex tells me what my phi is */
    const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    /* yIndex tells me which LOS I am */
    const int yIndex = blockIdx.y*nPhi;
    float qPhi, uPhi, pPhi;

    if(xIndex < nPhi) {
        myphi = d_phiAxis[xIndex];
        /* qPhi and uPhi are accumulators. So initialize to 0 */
        qPhi = 0.0; uPhi = 0.0;
        for(i=0; i<nChan; i++) {
            qPhi += d_qImageArray[yIndex+i]*cosf(myphi*d_lambdaDiff2[yIndex+i]) + 
                    d_uImageArray[yIndex+i]*sinf(myphi*d_lambdaDiff2[yIndex+i]);
            uPhi += d_uImageArray[yIndex+i]*cosf(myphi*d_lambdaDiff2[yIndex+i]) -
                    d_qImageArray[yIndex+i]*cosf(myphi*d_lambdaDiff2[yIndex+i]);
        }
        pPhi = sqrt(qPhi*qPhi + uPhi*uPhi);

        d_qPhi[xIndex] = K*qPhi;
        d_uPhi[xIndex] = K*uPhi;
        d_pPhi[xIndex] = K*pPhi;
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
