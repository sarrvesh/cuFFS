#ifndef SYNTHESIS_FILEACCESS_H
#define SYNTHESIS_FILEACCESS_H

/* Structure to store input options */
struct optionsList {
   char *msName;
   int imsize;
   double uvmin, uvmax;
   double cellsize;
   int hardware;
   int invertmode;
};

struct optionsList parseInput(char *parsetFileName);
void printUserInfo(struct optionsList inOptions, 
                   struct structHeader msHeader);

#endif
