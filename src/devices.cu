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
#include<time.h>
#include "structures.h"
#include "constants.h"
#include "devices.h"
#include "fileaccess.h"
__global__ void computeQUP_fits(float *d_qImageArray, float *d_uImageArray, 
                           int nChan, int nPhi, float K, float *d_qPhi,  
                           float *d_uPhi, float *d_pPhi, float *d_phiAxis, 
                           float *d_lambdaDiff2);
__global__ void computeQUP_hdf5(float *d_qImageArray, float *d_uImageArray, int nLOS,
                           int nChan, float K, float *d_qPhi, float *d_uPhi,
                           float *d_pPhi, float *d_phiAxis, int nPhi,
                           float *d_lambdaDiff2);
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
    printf("INFO: Detected %d CUDA-supported GPU(s)\n", deviceCount);
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
        /*** COMMENTED OUT FOR NOW. TOO MUCH INFORMATION.
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
        ***/
    }
    //printf("\n");
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
* Device code to compute Q(\phi) for HDF5 mode
* 
* In HDF5 mode, d_?ImageArray are such that the LOS varies 
* fast than the frequency channel. Which means that each kernel
* has to do strided read and write. 
*
* threadIdx.x and blockIdx.x tell us which phi to process
* blockIdx.y tells us which LOS to process
*
*************************************************************/
extern "C"
__global__ void computeQUP_hdf5(float *d_qImageArray, float *d_uImageArray, int nLOS,
                           int nChan, float K, float *d_qPhi, float *d_uPhi,
                           float *d_pPhi, float *d_phiAxis, int nPhi,
                           float *d_lambdaDiff2) {
    int i, readIdx, writeIdx;
    float myphi, mylambdaDiff2;
    /* xIndex tells me what my phi is */
    const int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    /* yIndex tells me which LOS I am */
    const int yIndex = blockIdx.y;
    float qPhi, uPhi, pPhi;
    float sinVal, cosVal;

    if(xIndex < nPhi) {
        myphi = d_phiAxis[xIndex];
        /* qPhi and uPhi are accumulators. So initialize to 0 */
        qPhi = 0.0; uPhi = 0.0;
        for(i=0; i<nChan; i++) {
            readIdx = yIndex + i*nLOS;
            mylambdaDiff2 = d_lambdaDiff2[i];
            sinVal = sinf(myphi*mylambdaDiff2);
            cosVal = cosf(myphi*mylambdaDiff2);
            qPhi += d_qImageArray[readIdx]*cosVal +
                    d_uImageArray[readIdx]*sinVal;
            uPhi += d_uImageArray[readIdx]*cosVal -
                    d_qImageArray[readIdx]*sinVal;
        }
        pPhi = sqrt(qPhi*qPhi + uPhi*uPhi);

        writeIdx = xIndex*nLOS + yIndex;
        d_qPhi[writeIdx] = K*qPhi;
        d_uPhi[writeIdx] = K*uPhi;
        d_pPhi[writeIdx] = K*pPhi;
    }
}

/*************************************************************
*
* Device code to compute Q(\phi)
*
* threadIdx.x and blockIdx.x tell us which phi to process
* blockIdx.y tells us which LOS to process
*
*************************************************************/
extern "C"
__global__ void computeQUP_fits(float *d_qImageArray, float *d_uImageArray, 
                           int nChan, int nPhi, float K, float *d_qPhi,  
                           float *d_uPhi, float *d_pPhi, float *d_phiAxis, 
                           float *d_lambdaDiff2) {
    int i, readIdx, writeIdx;
    float myphi, mylambdaDiff2;
    /* xIndex tells me what my phi is */
    const int xIndex = blockIdx.y*blockDim.x + threadIdx.x;
    /* yIndex tells me which LOS I am */
    const int yIndex = blockIdx.x;
    float qPhi, uPhi, pPhi;
    float sinVal, cosVal;

    if(xIndex < nPhi) {
        myphi = d_phiAxis[xIndex];
        /* qPhi and uPhi are accumulators. So initialize to 0 */
        qPhi = 0.0; uPhi = 0.0;
        for(i=0; i<nChan; i++) {
            mylambdaDiff2 = d_lambdaDiff2[i];
            sinVal = sinf(myphi*mylambdaDiff2);
            cosVal = cosf(myphi*mylambdaDiff2);
            readIdx = yIndex*nChan + i;
            qPhi += d_qImageArray[readIdx]*cosVal + 
                    d_uImageArray[readIdx]*sinVal;
            uPhi += d_uImageArray[readIdx]*cosVal -
                    d_qImageArray[readIdx]*sinVal;
        }
        pPhi = sqrt(qPhi*qPhi + uPhi*uPhi);

        writeIdx = yIndex*nPhi + xIndex;
        d_qPhi[writeIdx] = K*qPhi;
        d_uPhi[writeIdx] = K*uPhi;
        d_pPhi[writeIdx] = K*pPhi;
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
