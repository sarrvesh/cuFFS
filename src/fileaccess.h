#ifndef FILEACCESS_H
#define FILEACCESS_H

#ifdef __cplusplus
extern "C"
#endif

int getFitsHeader(struct optionsList *inOptions, struct parList *params);
int getFreqList(struct optionsList *inOptions, struct parList *params);
int getImageMask(struct optionsList *inOptions, struct parList *params);
void checkFitsError(int status);
int writePolCubeToDisk(float *fitsCube, char *fileName, 
                    struct optionsList *inOptions, struct parList *params);

/* Define the output file names here */
#define DIRTY_P "dirtyP.fits"
#define DIRTY_Q "dirtyQ.fits"
#define DIRTY_U "dirtyU.fits"

#endif
