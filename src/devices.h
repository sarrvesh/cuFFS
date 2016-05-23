#ifndef DEVICES_H
#define DEVICES_H

#ifdef __cplusplus
extern "C"
#endif

struct deviceInfoList * getDeviceInformation(int *nDevices);
int doRMSynthesis(struct optionsList *inOptions, struct parList *params,
                  struct deviceInfoList *gpuList, int deviceID);
int getBestDevice(struct deviceInfoList *gpuList, int nDevices);
struct deviceInfoList copySelectedDeviceInfo(struct deviceInfoList *gpuList,  
                                             int selectedDevice);
void checkCudaError(void);
void getGpuAllocForP(int *blockSize, int *threadSize, int *nImRows, 
                     int *nRowElements, struct deviceInfoList *gpuList,
                     int deviceID);

#endif
