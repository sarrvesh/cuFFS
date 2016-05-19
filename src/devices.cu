extern "C" {
#include<cuda_runtime.h>
#include<cuda.h>
#include "structures.h"
#include "constants.h"
#include "devices.h"
__global__ void computeQ(float *d_qImageArray, float *d_uImageArray, float *d_qPhi, float *d_phiAxis, int nPhi, int nElements, float lambda2, float lambda20);
__global__ void computeU(float *d_qImageArray, float *d_uImageArray, float *d_uPhi, float *d_phiAxis, int nPhi, int nElements, float lambda2, float lambda20);
__global__ void initializeQU(float *d_array, int nElements, int nPhi);
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
    float *d_phiAxis;
    float *d_qPhi, *d_uPhi;
    int i, j;
    size_t size;
    int status = 0;
    cudaError_t errorID;

    /* Copy the phi array to GPU */
    size = sizeof(d_phiAxis)*inOptions->nPhi;
    errorID = cudaMalloc(&d_phiAxis, size);
    cudaMemcpy(d_phiAxis, params->phiAxis, size, cudaMemcpyHostToDevice);
    checkCudaError();

    /**********************************************************
     *                    COMPUTE Q(\PHI)
     **********************************************************/
    /* Allocate memory for Q(\phi) cube and initialize to 0. */
    nElements = params->qAxisLen1 * params->qAxisLen2;
    size = sizeof(d_qPhi)*params->qAxisLen1*params->qAxisLen2*inOptions->nPhi;
    cudaMalloc(&d_qPhi, size);
    checkCudaError();
    initializeQU<<<1,inOptions->nPhi>>>(d_qPhi, nElements, inOptions->nPhi);
    checkCudaError();
    /* Setup fitsio access variables */
    fPixel = (long *)calloc(params->qAxisNum, sizeof(fPixel));
    for(i=1; i<=params->qAxisNum; i++) { fPixel[i-1] = 1; }
    nElements = params->qAxisLen1 * params->qAxisLen2;
    qImageArray = (float *)calloc(nElements, sizeof(qImageArray));
    uImageArray = (float *)calloc(nElements, sizeof(uImageArray));
    for(j=1; j<=params->qAxisLen3; j++) {
       /* Read in this Q and U channel */
       fPixel[2] = j;
       fits_read_pix(params->qFile, TFLOAT, fPixel, nElements, NULL,
         qImageArray, NULL, &status);
       fits_read_pix(params->uFile, TFLOAT, fPixel, nElements, NULL,
         uImageArray, NULL, &status);
       /* Copy the read in channel maps to GPU */
       size = nElements*sizeof(d_qImageArray);
       cudaMalloc(&d_qImageArray, size);
       cudaMalloc(&d_uImageArray, size);
       cudaMemcpy(d_qImageArray, qImageArray, size, cudaMemcpyHostToDevice);
       cudaMemcpy(d_uImageArray, uImageArray, size, cudaMemcpyHostToDevice);
       checkCudaError();
       /* Launch kernels to do RM Synthesis */
       /* Note that the number of threads launched MUST BE EQUAL TO OR GREATER
          than the number of phi planes. Assume for now that <<<>>> is int and
          not dim3. If this assumption is changed, index computation in
          kernels must be changed */
       computeQ<<<1,inOptions->nPhi>>>(d_qImageArray, d_uImageArray, d_qPhi,
         d_phiAxis, inOptions->nPhi, nElements, params->lambda2[j-1], 
         params->lambda20);
       checkCudaError();
       /* Free the allocated device memory */
       cudaFree(d_qImageArray);
       cudaFree(d_uImageArray);
    }
    /* Move the computed Q(phi) to host */
    nElements = params->qAxisLen1*params->qAxisLen2*inOptions->nPhi;
    size = nElements * sizeof(float);
    params->qPhi = (float *)calloc(params->qAxisLen1*params->qAxisLen2*inOptions->nPhi, sizeof(params->qPhi));
    cudaMemcpy(params->qPhi, d_qPhi, size, cudaMemcpyDeviceToHost);
    cudaFree(d_qPhi);
    checkCudaError();

    /**********************************************************
     *                    COMPUTE U(\PHI)
     **********************************************************/
    /* Allocate memory for U(\phi) cube and initialize to 0. */
    size = sizeof(d_uPhi)*params->qAxisLen1*params->qAxisLen2*inOptions->nPhi;
    cudaMalloc(&d_uPhi, size);
    checkCudaError();
    initializeQU<<<1,inOptions->nPhi>>>(d_uPhi, nElements, inOptions->nPhi);
    checkCudaError();
    /* Setup fitsio access variables */
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
       if(status) {
           fits_report_error(stdout, status);
           exit(FAILURE);
       }
       /* Copy the read in channel maps to GPU */
       size = nElements*sizeof(d_qImageArray);
       cudaMalloc(&d_qImageArray, size);
       cudaMalloc(&d_uImageArray, size);
       cudaMemcpy(d_qImageArray, qImageArray, size, cudaMemcpyHostToDevice);
       cudaMemcpy(d_uImageArray, uImageArray, size, cudaMemcpyHostToDevice);
       checkCudaError();
       /* Launch kernels to do RM Synthesis */
       computeU<<<1,inOptions->nPhi>>>(d_qImageArray, d_uImageArray, d_uPhi, d_phiAxis, inOptions->nPhi, nElements, params->lambda2[j-1], params->lambda20);
       checkCudaError();
       /* Free the allocatd device memory */
       cudaFree(d_qImageArray);
       cudaFree(d_uImageArray);
    }
    /* Move the computed U(phi) to host */
    nElements = params->qAxisLen1*params->qAxisLen2*inOptions->nPhi;
    size = nElements * sizeof(float);
    params->uPhi = (float *)calloc(params->qAxisLen1*params->qAxisLen2*inOptions->nPhi, sizeof(params->uPhi));
    cudaMemcpy(params->uPhi, d_uPhi, size, cudaMemcpyDeviceToHost);
    cudaFree(d_uPhi);
    checkCudaError();

    /* Free remaining allocated mem on device */
    cudaFree(d_phiAxis);

    /* Free remaining allocated mem on host */
    free(fPixel);
    free(qImageArray);
    free(uImageArray);

    return(SUCCESS);
}

/*************************************************************
*
* Device code to compute Q(\phi)
*
*************************************************************/
extern "C"
__global__ void computeQ(float *d_qImageArray, float *d_uImageArray, float *d_qPhi, float *d_phiAxis, int nPhi, int nElements, float lambda2, float lambda20) {
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float thisPhi = d_phiAxis[index];    
    float thisCos = cosf(2*thisPhi*(lambda2-lambda20));
    float thisSin = sinf(2*thisPhi*(lambda2-lambda20));

    if(index < nPhi) {
        /* For each element in Q, compute Q(thisPhi) and add it to Q(phi) */
        for(i=0; i<nElements; i++) 
            d_qPhi[index*nElements+i] += d_qImageArray[i]*thisCos - d_uImageArray[i]*thisSin;
    }
}

/*************************************************************
*
* Device code to compute U(\phi)
*
*************************************************************/
extern "C"
__global__ void computeU(float *d_qImageArray, float *d_uImageArray, float *d_uPhi, float *d_phiAxis, int nPhi, int nElements, float lambda2, float lambda20) {
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float thisPhi = d_phiAxis[index];
    float thisCos = cosf(2*thisPhi*(lambda2-lambda20));
    float thisSin = sinf(2*thisPhi*(lambda2-lambda20));

    if(index < nPhi) {
        /* For each element in U, compute U(thisPhi) and add it to U(phi) */
        for(i=0; i<nElements; i++) 
            d_uPhi[index*nElements+i] += d_uImageArray[i]*thisCos - d_qImageArray[i]*thisSin;
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
