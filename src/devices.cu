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
__global__ void computeQUP_fits(float *d_qImageArray, float *d_uImageArray, int nLOS, 
                           int nChan, float K, float *d_qPhi, float *d_uPhi, 
                           float *d_pPhi, float *d_phiAxis, int nPhi, 
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
* GPU accelerated RM Synthesis function
*
*************************************************************/
extern "C"
int doRMSynthesis(struct optionsList *inOptions, struct parList *params,
                  struct deviceInfoList selectedDeviceInfo) {
    int i, j; 
    float *lambdaDiff2, *d_lambdaDiff2;
    float *qImageArray, *uImageArray;
    float *d_qImageArray, *d_uImageArray;
    float *d_qPhi, *d_uPhi, *d_pPhi;
    float *qPhi, *uPhi, *pPhi;
    float *d_phiAxis;
    dim3 calcThreadSize, calcBlockSize;
    long *fPixel;
    int fitsStatus = 0;
    long nInElements, nOutElements;
    hid_t qDataspace, uDataspace;
    hid_t qOutDataspace, uOutDataspace, pOutDataspace;
    hid_t qDataset, uDataset;
    hid_t qOutDataset, uOutDataset, pOutDataset;
    hid_t qMemspace, uMemspace;
    hid_t qOutMemspace, uOutMemspace, pOutMemspace;
    herr_t h5ErrorQ, h5ErrorU, h5ErrorP;
    herr_t qerror, uerror, perror;
    hsize_t offsetIn[N_DIMS], countIn[N_DIMS], dimIn;
    hsize_t offsetOut[N_DIMS], countOut[N_DIMS], dimOut;
    clock_t startRead, stopRead;
    clock_t startWrite, stopWrite;
    clock_t startProc, stopProc;
    clock_t startX, stopX;
    float msRead=0, msWrite=0, msProc=0, msX=0, msTemp=0;
    
    /* Set mode-specific configuration */
    switch(inOptions->fileFormat) {
       case FITS:
          /* For FITS, set some pixel access limits */
          fPixel = (long *)calloc(params->qAxisNum, sizeof(*fPixel));
          fPixel[0] = 1; fPixel[1] = 1;
          /* Determine what the appropriate block and grid sizes are */
          calcThreadSize.x = selectedDeviceInfo.warpSize;
          calcBlockSize.y  = params->qAxisLen2;
          calcBlockSize.x  = inOptions->nPhi/calcThreadSize.x + 1;
          printf("INFO: Launching %dx%d blocks each with %d threads\n",
                  calcBlockSize.x, calcBlockSize.y, calcThreadSize.x);
          break;
       case HDF5:
          /* For HDF5, set up the hyperslab and data subset for input */
          dimIn = params->qAxisLen2 * params->qAxisLen3;
          qDataset   = H5Dopen2(params->qFileh5, PRIMARYDATA, H5P_DEFAULT);
          qDataspace = H5Dget_space(qDataset);
          uDataset   = H5Dopen2(params->uFileh5, PRIMARYDATA, H5P_DEFAULT);
          uDataspace = H5Dget_space(uDataset);
          countIn[0] = params->qAxisLen3;
          countIn[1] = 1; countIn[2] = params->qAxisLen2;
          offsetIn[0] = 0; offsetIn[1] = 0; offsetIn[2] = 0;
          qMemspace = H5Screate_simple(1, &dimIn, NULL);
          uMemspace = H5Screate_simple(1, &dimIn, NULL);
          if( qDataset<0 || uDataset<0 || qDataspace<0 || uDataspace<0 || qMemspace<0 || uMemspace<0 )
          { printf("\nError: HDF5 allocation failed\n"); }

          /* Set up the hyperslab and data subset for output */
          dimOut = params->qAxisLen2 * inOptions->nPhi;
          qOutDataset   = H5Dopen2(params->qDirtyH5, PRIMARYDATA, H5P_DEFAULT);
          qOutDataspace = H5Dget_space(qOutDataset);
          uOutDataset   = H5Dopen2(params->uDirtyH5, PRIMARYDATA, H5P_DEFAULT);
          uOutDataspace = H5Dget_space(uOutDataset);
          pOutDataset   = H5Dopen2(params->uDirtyH5, PRIMARYDATA, H5P_DEFAULT);
          pOutDataspace = H5Dget_space(uOutDataset);
          countOut[0] = inOptions->nPhi;
          countOut[1] = 1; countOut[2] = params->qAxisLen2;
          offsetOut[0] = 0; offsetOut[1] = 0; offsetOut[2] = 0;
          qOutMemspace = H5Screate_simple(1, &dimOut, NULL);
          uOutMemspace = H5Screate_simple(1, &dimOut, NULL);
          pOutMemspace = H5Screate_simple(1, &dimOut, NULL);
          if( qOutDataset<0 || uOutDataset<0 || pOutDataset<0 ||
              qOutDataspace<0 || uOutDataspace<0 || pOutDataspace<0 ||
              qOutMemspace<0 || uOutMemspace<0 || pOutMemspace<0 ) {
             printf("\nError: HDF5 output allocation failed\n");
             exit(FAILURE);
          }

          /* Determine what the appropriate block and grid sizes are */
          calcThreadSize.x = selectedDeviceInfo.warpSize;
          calcBlockSize.y  = params->qAxisLen2;
          calcBlockSize.x  = inOptions->nPhi/calcThreadSize.x + 1;
          printf("INFO: Launching %dx%d blocks each with %d threads\n",
                 calcBlockSize.x, calcBlockSize.y, calcThreadSize.x);
          break;
    }
    
    /* Allocate memory on the host */
    nInElements = params->qAxisLen2 * params->qAxisLen3;
    nOutElements= inOptions->nPhi * params->qAxisLen2;
    lambdaDiff2 = (float *)calloc(params->qAxisLen3, sizeof(*lambdaDiff2));
    qImageArray = (float *)calloc(nInElements, sizeof(*qImageArray));
    uImageArray = (float *)calloc(nInElements, sizeof(*uImageArray));
    qPhi = (float *)calloc(nOutElements, sizeof(*qPhi));
    uPhi = (float *)calloc(nOutElements, sizeof(*uPhi));
    pPhi = (float *)calloc(nOutElements, sizeof(*pPhi));
    if(lambdaDiff2 == NULL || qImageArray == NULL || uImageArray == NULL ||
       qPhi == NULL || uPhi == NULL || pPhi == NULL) {
       printf("ERROR: Unable to allocate memory on host\n");
       exit(FAILURE);    
    }
    
    /* Allocate memory on the device */
    cudaMalloc(&d_lambdaDiff2, sizeof(*lambdaDiff2)*params->qAxisLen3);
    cudaMalloc(&d_phiAxis, sizeof(*(params->phiAxis))*inOptions->nPhi);
    cudaMalloc(&d_qImageArray, nInElements*sizeof(*qImageArray));
    cudaMalloc(&d_uImageArray, nInElements*sizeof(*uImageArray));
    cudaMalloc(&d_qPhi, nOutElements*sizeof(*qPhi));
    cudaMalloc(&d_uPhi, nOutElements*sizeof(*uPhi));
    cudaMalloc(&d_pPhi, nOutElements*sizeof(*pPhi));
    checkCudaError();

    /* Compute \lambda^2 - \lambda^2_0 once. Common for all threads */
    for(i=0;i<params->qAxisLen3;i++)
        lambdaDiff2[i] = 2.0*(params->lambda2[i]-params->lambda20);
    cudaMemcpy(d_lambdaDiff2, lambdaDiff2, 
               sizeof(*lambdaDiff2)*params->qAxisLen3, cudaMemcpyHostToDevice);
    checkCudaError();
    
    /* Allocate and transfer phi axis info. Common for all threads */
    cudaMemcpy(d_phiAxis, params->phiAxis, 
               sizeof(*(params->phiAxis))*inOptions->nPhi, 
               cudaMemcpyHostToDevice);
    checkCudaError();

    /* Process each line of sight individually */
    //cudaEventRecord(totStart);
    for(j=1; j<=params->qAxisLen1; j++) {
       /* Read one frame at a time. In the original cube, this is 
          all sightlines in one DEC row */
       startRead = clock();
       switch(inOptions->fileFormat) {
          case FITS:
             fPixel[2] = j;
             fits_read_pix(params->qFile, TFLOAT, fPixel, nInElements, NULL, 
                           qImageArray, NULL, &fitsStatus);
             fits_read_pix(params->uFile, TFLOAT, fPixel, nInElements, NULL,
                           uImageArray, NULL, &fitsStatus);
             checkFitsError(fitsStatus);
             break;
          case HDF5:
             offsetIn[1] = j-1;
             qerror = H5Sselect_hyperslab(qDataspace, H5S_SELECT_SET, offsetIn, 
                                    NULL, countIn, NULL);
             uerror = H5Sselect_hyperslab(uDataspace, H5S_SELECT_SET, offsetIn, 
                                    NULL, countIn, NULL);
             h5ErrorQ = H5Dread(qDataset, H5T_NATIVE_FLOAT, qMemspace, 
                                   qDataspace, H5P_DEFAULT, qImageArray);
             h5ErrorU = H5Dread(uDataset, H5T_NATIVE_FLOAT, uMemspace, 
                                   uDataspace, H5P_DEFAULT, uImageArray);
             if(h5ErrorQ<0 || h5ErrorU<0 || qerror<0 || uerror<0 ) {
                printf("\nError: Unable to read input data cubes\n\n");
                exit(FAILURE);
             }
             break;
       }
       stopRead = clock();
       msTemp = ((float)(stopRead - startRead))/CLOCKS_PER_SEC;
       msRead += msTemp;

       /* Transfer input images to device */
       startX = clock();
       cudaMemcpy(d_qImageArray, qImageArray, 
                  nInElements*sizeof(*qImageArray),
                  cudaMemcpyHostToDevice);
       cudaMemcpy(d_uImageArray, uImageArray, 
                  nInElements*sizeof(*qImageArray),
                  cudaMemcpyHostToDevice);
       stopX = clock();
       msTemp = ((float)(stopX - startX))/CLOCKS_PER_SEC;
       msX += msTemp;
 
       /* Launch kernels to compute Q(\phi), U(\phi), and P(\phi) */
       startProc = clock();
       switch(inOptions->fileFormat) {
       case FITS:
          computeQUP_fits<<<calcBlockSize, calcThreadSize>>>(d_qImageArray, d_uImageArray, 
                         params->qAxisLen2, params->qAxisLen3, params->K, d_qPhi,
                         d_uPhi, d_pPhi, d_phiAxis, inOptions->nPhi, d_lambdaDiff2);
          break;
       case HDF5:
          computeQUP_hdf5<<<calcBlockSize, calcThreadSize>>>(d_qImageArray, d_uImageArray,
                         params->qAxisLen2, params->qAxisLen3, params->K, d_qPhi,
                         d_uPhi, d_pPhi, d_phiAxis, inOptions->nPhi, d_lambdaDiff2);
          break;
       }
       stopProc = clock();
       msTemp = ((float)(stopProc - startProc))/CLOCKS_PER_SEC;
       msProc += msTemp;

       /* Move Q(\phi), U(\phi) and P(\phi) to host */
       startX = clock();
       cudaMemcpy(d_qPhi, qPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
       cudaMemcpy(d_uPhi, uPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
       cudaMemcpy(d_pPhi, pPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
       stopX = clock();
       msTemp = ((float)(stopX - startX))/CLOCKS_PER_SEC;
       msX += msTemp;

       /* Write the output cubes to disk */
       startWrite = clock();
       switch(inOptions->fileFormat) {
          case FITS:
             fits_write_pix(params->qDirty, TFLOAT, fPixel, nOutElements, qPhi, &fitsStatus);
             fits_write_pix(params->uDirty, TFLOAT, fPixel, nOutElements, uPhi, &fitsStatus);
             fits_write_pix(params->pDirty, TFLOAT, fPixel, nOutElements, pPhi, &fitsStatus);
             checkFitsError(fitsStatus);
             break;
          case HDF5:
             offsetOut[1] = j-1;
             qerror = H5Sselect_hyperslab(qOutDataspace, H5S_SELECT_SET, 
                                          offsetOut, NULL, countOut, NULL);
             uerror = H5Sselect_hyperslab(uOutDataspace, H5S_SELECT_SET, 
                                          offsetOut, NULL, countOut, NULL);
             perror = H5Sselect_hyperslab(pOutDataspace, H5S_SELECT_SET, 
                                          offsetOut, NULL, countOut, NULL);
             h5ErrorQ = H5Dwrite(qOutDataset, H5T_NATIVE_FLOAT, qOutMemspace, 
                                 qOutDataspace, H5P_DEFAULT, qPhi);
             h5ErrorU = H5Dwrite(uOutDataset, H5T_NATIVE_FLOAT, uOutMemspace, 
                                 uOutDataspace, H5P_DEFAULT, uPhi);
             h5ErrorP = H5Dwrite(pOutDataset, H5T_NATIVE_FLOAT, pOutMemspace, 
                                 pOutDataspace, H5P_DEFAULT, pPhi);
             if(h5ErrorQ<0 || h5ErrorU || h5ErrorP<0 ||
                qerror<0 || uerror<0 || perror<0 ) {
                printf("\nError: Unable to write output data cubes\n\n");
                exit(FAILURE);
             }
             break;
       }
       stopWrite = clock();
       msTemp = ((float)(stopWrite - startWrite))/CLOCKS_PER_SEC;
       msWrite += msTemp;
    }

    /* Free all the allocated memory */
    free(qImageArray); free(uImageArray);
    cudaFree(d_qImageArray); cudaFree(d_uImageArray);
    free(qPhi); free(uPhi); free(pPhi);
    cudaFreeHost(d_qPhi); cudaFreeHost(d_uPhi); cudaFreeHost(d_pPhi);
    free(lambdaDiff2); cudaFree(d_lambdaDiff2);
    cudaFree(d_phiAxis);
    switch(inOptions->fileFormat) {
    case FITS:
       free(fPixel);
       break;
    case HDF5:
       H5Sclose(qMemspace);  H5Sclose(uMemspace);
       H5Sclose(qDataspace); H5Sclose(uDataspace);
       H5Dclose(qDataset);   H5Dclose(uDataset);
       H5Sclose(qOutMemspace);  H5Sclose(uOutMemspace);
       H5Sclose(qOutDataspace); H5Sclose(uOutDataspace);
       H5Dclose(qOutDataset);   H5Dclose(uOutDataset);
       H5Sclose(pOutMemspace); H5Sclose(pOutDataspace);
       H5Dclose(pOutDataset);
       H5Fclose(params->qDirtyH5);
       H5Fclose(params->uDirtyH5);
       H5Fclose(params->pDirtyH5);
       break;
    }

    /* Write timing information to stdout */
    printf("INFO: Timing Information\n");
    printf("   Input read time: %0.3f s\n", msRead);
    printf("   Compute time: %0.3f s\n", msProc);
    printf("   Output write time: %0.3f s\n", msWrite);
    printf("   D2H Transfer time: %0.3f s\n", msX);

    return(SUCCESS);
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
*************************************************************/
extern "C"
__global__ void computeQUP_fits(float *d_qImageArray, float *d_uImageArray, int nLOS, 
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
    float sinVal, cosVal;

    if(xIndex < nPhi) {
        myphi = d_phiAxis[xIndex];
        /* qPhi and uPhi are accumulators. So initialize to 0 */
        qPhi = 0.0; uPhi = 0.0;
        for(i=0; i<nChan; i++) {
            sinVal = sinf(myphi*d_lambdaDiff2[yIndex+i]);
            cosVal = cosf(myphi*d_lambdaDiff2[yIndex+i]);
            qPhi += d_qImageArray[yIndex+i]*cosVal + 
                    d_uImageArray[yIndex+i]*sinVal;
            uPhi += d_uImageArray[yIndex+i]*cosVal -
                    d_qImageArray[yIndex+i]*sinVal;
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
