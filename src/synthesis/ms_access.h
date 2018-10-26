#ifndef SYNTH_MS_ACCESS_H
#define SYNTH_MS_ACCESS_H

#include<string>
#include "synth_fileaccess.h"

struct structHeader {
   size_t nSPW;
   size_t chanPerSPW;
   double *chanFreq;
   double chanWidth;
   size_t nVisRows;
   double uMaxM, uMinM, vMaxM, vMinM, wMaxM, wMinM;
   double maxUVWm, minUVWm;
   double pointRa, pointDec;
   double pointRaDeg, pointDecDeg;
   std::string coordStr;
   int corrType;
};

int getMsHeader(struct optionsList inOptions, struct structHeader *msHeader); 
int getUvRange(struct optionsList inOptions, struct structHeader *msHeader);

#endif
