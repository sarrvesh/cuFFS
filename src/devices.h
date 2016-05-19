#ifndef DEVICES_H
#define DEVICES_H

#ifdef __cplusplus
extern "C"
#endif

struct deviceInfoList * getDeviceInformation(int *nDevices);
int doRMSynthesis(struct optionsList *inOptions, struct parList *params);
int getBestDevice(struct deviceInfoList *gpuList, int nDevices);
void checkCudaError(void);

#endif
