#ifndef RMSF_H
#define RMSF_H

#ifdef __cplusplus
extern "C"
#endif

int generateRMSF(struct optionsList *inOptions, struct parList *params);
int compFunc(const void * a, const void * b);
void getMedianLambda20(struct parList *params);
int writeRMSF(struct optionsList inOptions, struct parList params);
int plotRMSF(struct optionsList inOptions);

#endif
