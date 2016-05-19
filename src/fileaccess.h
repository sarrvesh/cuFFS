#ifndef FILEACCESS_H
#define FILEACCESS_H

#ifdef __cplusplus
extern "C"
#endif

int getFitsHeader(struct optionsList *inOptions, struct parList *params);
int getFreqList(struct optionsList *inOptions, struct parList *params);
int getImageMask(struct optionsList *inOptions, struct parList *params);
void checkFitsError(int status);

#endif
