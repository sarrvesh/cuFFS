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
                         int nElements, float lambda2, float lambda20);
__global__ void computeU(float *d_qImageArray, float *d_uImageArray, 
                         float *d_uPhi, float *d_phiAxis, int nPhi, 
                         int nElements, float lambda2, float lambda20);
__global__ void initializeQU(float *d_array, int nElements, int nPhi);
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
    long *fPixel;
    LONGLONG nImElements, nCubeElements;
    float *qImageArray, *uImageArray;
    float *d_qImageArray, *d_uImageArray;
    float *d_phiAxis;
    float *d_qPhi, *d_uPhi;
    float *qPhi;
    int i, j;
    size_t size, imSize, cubeSize;
    int status = 0;
    int threadSize, blockSize;
    int nImRows, nRowElements;
    long nFrames;    

    /* Copy the phi array to GPU */
    size = sizeof(d_phiAxis)*inOptions->nPhi;
    cudaMalloc(&d_phiAxis, size);
    cudaMemcpy(d_phiAxis, params->phiAxis, size, cudaMemcpyHostToDevice);
    checkCudaError();

    /* Estimate the sizes of the input and output images */
    nImElements = params->qAxisLen1 * params->qAxisLen2;
    nCubeElements = params->qAxisLen1*params->qAxisLen2*inOptions->nPhi;
    imSize = sizeof(float)*nImElements;
    cubeSize = sizeof(float)*nCubeElements;
    
    /* Check if output fits inside GPU memory */
    printf("INFO: Size of input Q/U channel: %0.3f kiB\n", imSize/KILO);
    printf("INFO: Size of output Q and U cube: %0.3f MiB\n", 2.0*cubeSize/MEGA);
    printf("INFO: Available memory on GPU: %0.3f MiB\n", 
           selectedDeviceInfo.globalMem/MEGA);
    if(selectedDeviceInfo.globalMem < cubeSize) {
        printf("ERROR: Insufficient memory on device! Try reducing nPhi\n\n");
        return(FAILURE);
    }
    
    /* Allocate memory on device for the input Q and U images */
    cudaMalloc(&d_qImageArray, imSize);
    cudaMalloc(&d_uImageArray, imSize);
    checkCudaError();
    
    /* Allocate memory for output \phi cube */
    cudaMalloc(&d_qPhi, cubeSize);
    cudaMalloc(&d_uPhi, cubeSize);
    checkCudaError();

    /* Initialize output cubes to 0. */
    initializeQU<<<1,inOptions->nPhi>>>(d_qPhi, nImElements, inOptions->nPhi);
    initializeQU<<<1,inOptions->nPhi>>>(d_uPhi, nImElements, inOptions->nPhi);
    checkCudaError();
    
    /* Allocate memory for output cube */
    qPhi = (float *)calloc(nCubeElements, sizeof(qPhi));
    if(qPhi == NULL) {
        printf("ERROR: Unable to allocate memory for output\n");
        return(FAILURE);
    }
    
    /* Setup fitsio access variables */
    fPixel = (long *)calloc(params->qAxisNum, sizeof(fPixel));
    for(i=1; i<=params->qAxisNum; i++) { fPixel[i-1] = 1; }
    qImageArray = (float *)calloc(nImElements, sizeof(qImageArray));
    uImageArray = (float *)calloc(nImElements, sizeof(uImageArray));

    /* Get the optimal thread and block size */
    getGpuAllocForRMSynth(&blockSize, &threadSize, inOptions->nPhi, 
                          selectedDeviceInfo);
    printf("\nINFO: Launching kernels with %d blocks with %d threads each",
           blockSize, threadSize);

    for(j=1; j<=params->qAxisLen3; j++) {
       /* Read in this Q and U channel */
       fPixel[2] = j;
       fits_read_pix(params->qFile, TFLOAT, fPixel, nImElements, NULL,
         qImageArray, NULL, &status);
       fits_read_pix(params->uFile, TFLOAT, fPixel, nImElements, NULL,
         uImageArray, NULL, &status);
       checkFitsError(status);
       /* Copy the read in channel maps to GPU */
       cudaMemcpy(d_qImageArray, qImageArray, imSize, cudaMemcpyHostToDevice);
       cudaMemcpy(d_uImageArray, uImageArray, imSize, cudaMemcpyHostToDevice);
       checkCudaError();
       /* Launch kernels to do RM Synthesis */
       computeQ<<<blockSize, threadSize>>>(d_qImageArray, d_uImageArray, d_qPhi,
         d_phiAxis, inOptions->nPhi, nImElements, params->lambda2[j-1], 
         params->lambda20);
       computeU<<<blockSize, threadSize>>>(d_qImageArray, d_uImageArray, d_uPhi,
         d_phiAxis, inOptions->nPhi, nImElements, params->lambda2[j-1], 
         params->lambda20);
       checkCudaError();
    }
    
    /* Move the computed Q(phi) to host */
    cudaMemcpy(qPhi, d_qPhi, cubeSize, cudaMemcpyDeviceToHost);
    cudaFree(d_qPhi);
    checkCudaError();
    /* Write the Q cube to disk */
    writePolCubeToDisk(qPhi, DIRTY_Q, inOptions, params);
    /* Move the computed U(\phi) to host */
    cudaMemcpy(qPhi, d_uPhi, cubeSize, cudaMemcpyDeviceToHost);
    cudaFree(d_uPhi);
    checkCudaError();
    /* Write the U cube to disk */
    writePolCubeToDisk(qPhi, DIRTY_U, inOptions, params);
    free(qPhi);

    /* Free remaining allocated mem on device */
    cudaFree(d_phiAxis);
    cudaFree(d_qImageArray);
    cudaFree(d_uImageArray);

    /* Free remaining allocated mem on host */
    free(fPixel);
    free(qImageArray);
    free(uImageArray);

    /* Compute P cube. Q, U and P will all might not fit in the device
       global memory. Need a clever way to manage memory and threads. */
    nRowElements = params->qAxisLen1;
    nImRows = params->qAxisLen2 * inOptions->nPhi;
    getGpuAllocForP(&blockSize, &threadSize, &nFrames, nImRows, nRowElements, 
                    selectedDeviceInfo);

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
                         int nElements, float lambda2, float lambda20) {
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float thisPhi = d_phiAxis[index];    
    float thisCos = cosf(2*thisPhi*(lambda2-lambda20));
    float thisSin = sinf(2*thisPhi*(lambda2-lambda20));

    if(index < nPhi) {
        /* For each element in Q, compute Q(thisPhi) and add it to Q(phi) */
        for(i=0; i<nElements; i++)
            d_qPhi[index*nElements+i] += d_qImageArray[i]*thisCos - 
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
                         int nElements, float lambda2, float lambda20) {
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float thisPhi = d_phiAxis[index];
    float thisCos = cosf(2*thisPhi*(lambda2-lambda20));
    float thisSin = sinf(2*thisPhi*(lambda2-lambda20));

    if(index < nPhi) {
        /* For each element in U, compute U(thisPhi) and add it to U(phi) */
        for(i=0; i<nElements; i++) 
            d_uPhi[index*nElements+i] += d_uImageArray[i]*thisCos - 
                                         d_qImageArray[i]*thisSin;
    }
}

/*************************************************************
*
* Initialize Q(\phi) and U(\phi)
*
*************************************************************/
extern "C"
__global__ void initializeQU(float *d_array, int nElements, int nPhi) {
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index < nPhi) {
        for(i=0; i<nElements; i++)
            d_array[index*nElements+i] = 0.0;
    }
}

/*************************************************************
*
* Estimate the optimal number of block and thread size 
*  for RM Synthesis.
* Conditions:
*    1. # of kernel calls should be equal to or greater than nPhi
*    2. For good occupancy, thread and block sizes should be integer 
*       multiples of warp size and # of SMs.
*    3. Each kernel takes care of a single phi plane.
*    ==> minimize threadSize*blockSize - nPhi
*    ==> min(M*nSM*N*warpSize - nPhi) such that M \geq N
*
*************************************************************/
extern "C"
void getGpuAllocForRMSynth(int *blockSize, int *threadSize, int nPhi,
                           struct deviceInfoList selectedDeviceInfo) {
    int M, N;
    N = 1;
    M = nPhi/(N*selectedDeviceInfo.warpSize) + 1;
    *blockSize = M*selectedDeviceInfo.nSM;
    *threadSize = N*selectedDeviceInfo.warpSize;
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
